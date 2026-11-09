"""RFT(거절 샘플링 미세조정): base가 스스로 푼 '맞은 응답'을 지도학습한다 (LoRA, bf16, 24GB).

학습 데이터는 build_rft.py가 만든 jsonl(문항, 이미지 경로, 응답). 프롬프트는 평가와 같은 mmmu_pro_cot, 해상도도 평가와 같다.
응답 부분에만 손실을 준다 (프롬프트와 이미지 토큰은 -100). 마이크로 배치 1, 그래디언트 누적, 체크포인트마다 LoRA 저장.
GRPO(train_grpo.py)와 같은 LoRA 설정(r64/alpha128, LLM의 q,k,v,o,gate,up,down만)을 쓴다. 학습률은 GRPO(2e-5)보다 낮게 시작한다(추측값).

  python music_ft/train_rft.py --data data/rft/rft_dv.jsonl --output-dir results/rft_dv1 --max-steps 2 --limit 16   # 스모크
  python music_ft/train_rft.py --data data/rft/rft_dv.jsonl --output-dir results/rft_dv1 --epochs 1 --resume       # 본 학습
이후: python music_ft/merge_lora.py --lora results/rft_dv1/final_lora --out results/rft_dv1/merged 로 합쳐서 같은 평가를 돌린다.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_grpo import MODEL_REPO, MODEL_REVISION, build_lora_config, jpeg_q95  # noqa: E402
from prompting import MAX_PIXELS, MIN_PIXELS, build_mmmu_pro_cot_text  # noqa: E402


def make_labels(input_ids, prompt_len):
    """프롬프트 부분은 -100(손실 제외), 응답 부분은 그대로. 길이가 프롬프트보다 길어야 학습할 것이 있다."""
    if len(input_ids) <= prompt_len:
        raise ValueError("응답이 비어 있다 (프롬프트 길이 이하)")
    return [-100] * prompt_len + list(input_ids[prompt_len:])


def response_only_loss(hidden, lm_head, labels, num_items_in_batch=None, grad_accum=1):
    """응답 위치의 어휘 확률만 계산해서 손실을 구한다 (전체 위치를 계산하면 15만 단어 x 수천 토큰의 fp32 확률이 24GB를 넘긴다).
    HF의 기본 손실과 같은 값이다: 위치 t의 은닉 상태가 토큰 t+1을 예측하고, 응답 토큰의 평균 교차 엔트로피.
    Trainer가 num_items_in_batch(누적 배치 전체의 응답 토큰 수)를 주면 합을 그 수로 나눠서(누적 간 토큰 평균), 없으면 평균 / 누적 횟수."""
    import torch

    first = int((labels[0] != -100).nonzero()[0])
    logits = lm_head(hidden[:, first - 1:-1, :]).float()
    target = labels[:, first:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                           ignore_index=-100, reduction="sum")
    if num_items_in_batch is not None:
        return ce / num_items_in_batch
    return ce / (target != -100).sum() / grad_accum


class RFTCollator:
    """행 하나를 모델 입력으로 만든다 (마이크로 배치 1). 이미지는 학습 데이터 준비 때와 같이 JPEG q95로 한 번 거친다."""

    def __init__(self, processor, jpeg_dir):
        self.processor = processor
        self.jpeg_dir = Path(jpeg_dir)

    def _process(self, text, images):
        try:
            out = self.processor(text=[text], images=images, return_tensors="pt", return_mm_token_type_ids=True)
        except (TypeError, ValueError):
            out = self.processor(text=[text], images=images, return_tensors="pt")
        out = dict(out)
        out.pop("token_type_ids", None)
        return out

    def __call__(self, features):
        from PIL import Image

        if len(features) != 1:
            raise ValueError("마이크로 배치 크기는 1이어야 한다 (이미지 수가 달라 패딩이 복잡해진다)")
        row = features[0]
        text, image_order, _ = build_mmmu_pro_cot_text(row)
        images = []
        for n in image_order:
            src = Path(row["data_dir"]) / row["image_paths"][f"image_{n}"]
            dst = self.jpeg_dir / f"{row['id']}_{src.stem}.jpg"
            jpeg_q95(src, dst)
            images.append(Image.open(dst).convert("RGB"))
        content = [{"type": "text", "text": text}] + [{"type": "image"} for _ in images]
        user = {"role": "user", "content": content}
        prompt_text = self.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(
            [user, {"role": "assistant", "content": [{"type": "text", "text": row["response"]}]}], tokenize=False)
        if not full_text.startswith(prompt_text):
            raise ValueError("채팅 템플릿의 프롬프트 부분이 전체 대화의 앞부분과 다르다 (손실 마스크를 만들 수 없다)")
        full = self._process(full_text, images)
        prompt_len = self._process(prompt_text, images)["input_ids"].shape[1]
        import torch

        labels = torch.tensor([make_labels(full["input_ids"][0].tolist(), prompt_len)], dtype=torch.long)
        full["labels"] = labels
        return full


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="build_rft.py가 만든 jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="스모크 시험용 (기본: 에폭 기준)")
    parser.add_argument("--grad-accum", type=int, default=16, help="최적화 한 번에 쓰는 응답 수 (마이크로 배치 1)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="디버그용: 앞 N행만")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf-loss", action="store_true", help="응답 위치만 계산하는 손실 대신 HF 기본 손실을 쓴다 (비교용, 메모리를 더 쓴다)")
    args = parser.parse_args()

    import torch
    from peft import get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent / "lab"))
    from envinfo import snapshot

    snapshot(output_dir, extra={"tool": "train_rft", "args": {k: str(v) for k, v in vars(args).items()}})

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"학습 응답 {len(rows)}개 | 출처 {sorted({r['source'] for r in rows})}")

    processor = AutoProcessor.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    ip = processor.image_processor                       # 평가와 같은 이미지 해상도 (train_grpo.py와 같은 방식)
    ip.min_pixels, ip.max_pixels = MIN_PIXELS, MAX_PIXELS
    ip.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_REPO, revision=MODEL_REVISION, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_steps=3, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=args.save_total_limit, report_to="none",
        remove_unused_columns=False, seed=args.seed, dataloader_num_workers=0)
    class ResponseOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            causal = model.get_base_model() if hasattr(model, "get_base_model") else model
            hidden = causal.model(**inputs).last_hidden_state
            loss = response_only_loss(hidden, causal.lm_head, labels, num_items_in_batch, self.args.gradient_accumulation_steps)
            return (loss, None) if return_outputs else loss

    trainer_cls = Trainer if args.hf_loss else ResponseOnlyTrainer
    trainer = trainer_cls(model=model, args=train_args, train_dataset=rows, data_collator=RFTCollator(processor, output_dir / "jpeg"))

    last = get_last_checkpoint(str(output_dir)) if args.resume else None
    print("이어서 학습:" if last else "처음부터 학습", last or "")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output_dir / "final_lora"))
    print("완료:", output_dir / "final_lora")


if __name__ == "__main__":
    main()
