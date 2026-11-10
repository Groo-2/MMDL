"""DPO(선호 학습): 맞고 끝난 응답은 확률을 올리고, 끝나지 않았거나 틀린 응답은 내린다 (LoRA, bf16, 24GB).

RFT가 '맞은 응답만' 학습해서 막혔을 때 결론을 못 내리고 재고를 이어가는 루프가 늘었다(잘림+오류 89 -> 209). 나쁜 응답(특히 잘린 루프 응답)을
명시적으로 학습 신호로 쓴다. 기준 모델은 어댑터를 끈 base(`disable_adapter`)라 별도 사본이 필요 없다.
손실 = -logsigmoid(beta * ((log pi(c) - log ref(c)) - (log pi(r) - log ref(r)))) + sft_alpha * (chosen의 토큰당 NLL)
  (두 번째 항은 좋은 응답의 확률이 같이 떨어지는 현상을 막는다. RPO)
응답 위치의 어휘 확률만 계산한다(train_rft.response_only_ce). 프롬프트·해상도는 평가와 같다.

  python music_ft/train_dpo.py --data data/dpo/dpo_dv.jsonl --output-dir results/dpo_smoke --max-steps 2 --limit 16 --grad-accum 4   # 스모크
  python music_ft/train_dpo.py --data data/dpo/dpo_dv.jsonl --output-dir results/dpo_dv1 --epochs 1 --resume                          # 본 학습
이후: python music_ft/merge_lora.py --lora results/dpo_dv1/final_lora --out results/dpo_dv1/merged --low-memory
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_grpo import MODEL_REPO, MODEL_REVISION, build_lora_config, jpeg_q95  # noqa: E402
from prompting import MAX_PIXELS, MIN_PIXELS, build_mmmu_pro_cot_text  # noqa: E402
from train_rft import RFTCollator, make_labels, response_only_ce  # noqa: E402


def dpo_loss(logp_c, logp_r, ref_c, ref_r, nll_c_per_token, beta=0.1, sft_alpha=1.0):
    """(손실, 보상 마진, 선호를 맞게 가리는가). logp_*는 응답 토큰 로그확률의 합."""
    import torch

    margin = beta * ((logp_c - ref_c) - (logp_r - ref_r))
    loss = -torch.nn.functional.logsigmoid(margin) + sft_alpha * nll_c_per_token
    return loss, margin.detach(), (margin.detach() > 0).float()


class DPOCollator(RFTCollator):
    """쌍 하나(마이크로 배치 1)를 chosen/rejected 두 모델 입력으로 만든다. 이미지는 두 번 처리한다(작은 비용)."""

    def _one(self, row, response):
        from PIL import Image
        import torch

        text, image_order, _ = build_mmmu_pro_cot_text(row)
        images = []
        for n in image_order:
            src = Path(row["data_dir"]) / row["image_paths"][f"image_{n}"]
            dst = self.jpeg_dir / f"{row['id']}_{src.stem}.jpg"
            jpeg_q95(src, dst)
            images.append(Image.open(dst).convert("RGB"))
        user = {"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image"} for _ in images]}
        prompt_text = self.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(
            [user, {"role": "assistant", "content": [{"type": "text", "text": response}]}], tokenize=False)
        if not full_text.startswith(prompt_text):
            raise ValueError("채팅 템플릿의 프롬프트 부분이 전체 대화의 앞부분과 다르다")
        full = self._process(full_text, images)
        prompt_len = self._process(prompt_text, images)["input_ids"].shape[1]
        full["labels"] = torch.tensor([make_labels(full["input_ids"][0].tolist(), prompt_len)], dtype=torch.long)
        return full

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("마이크로 배치 크기는 1이어야 한다")
        row = features[0]
        return {"chosen": self._one(row, row["chosen"]), "rejected": self._one(row, row["rejected"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="build_dpo.py가 만든 jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--grad-accum", type=int, default=16, help="최적화 한 번에 쓰는 쌍 수 (마이크로 배치 1)")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-alpha", type=float, default=1.0, help="좋은 응답의 토큰당 NLL 가중치 (0이면 순수 DPO)")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent / "lab"))
    from envinfo import snapshot

    snapshot(output_dir, extra={"tool": "train_dpo", "args": {k: str(v) for k, v in vars(args).items()}})

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    kinds = {}
    for r in rows:
        kinds[r["rejected_kind"]] = kinds.get(r["rejected_kind"], 0) + 1
    print(f"학습 쌍 {len(rows)}개 | 나쁜 응답 종류 {kinds}")

    processor = AutoProcessor.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    ip = processor.image_processor
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

    class DPOTrainer(Trainer):
        _acc, _margin = [], []

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            causal = model.get_base_model() if hasattr(model, "get_base_model") else model

            def logp(batch):                         # (응답 로그확률 합, 응답 토큰 수)
                b = dict(batch)
                labels = b.pop("labels")
                hidden = causal.model(**b).last_hidden_state
                ce, n = response_only_ce(hidden, causal.lm_head, labels)
                return -ce, n

            c, r = inputs["chosen"], inputs["rejected"]
            with torch.no_grad(), model.disable_adapter():           # 기준 = 어댑터를 끈 base
                ref_c, _ = logp(c)
                ref_r, _ = logp(r)
            lp_c, n_c = logp(c)
            lp_r, _ = logp(r)
            loss, margin, correct = dpo_loss(lp_c, lp_r, ref_c, ref_r, -lp_c / n_c, args.beta, args.sft_alpha)
            self._acc.append(float(correct)); self._margin.append(float(margin))
            return (loss, None) if return_outputs else loss          # Trainer가 누적 횟수로 나눈다 (RFT와 같은 경로)

        def log(self, logs, *a, **k):
            if self._acc:
                logs = dict(logs, pref_acc=sum(self._acc) / len(self._acc), reward_margin=sum(self._margin) / len(self._margin))
                self._acc.clear(); self._margin.clear()
            super().log(logs, *a, **k)

    trainer = DPOTrainer(model=model, args=train_args, train_dataset=rows, data_collator=DPOCollator(processor, output_dir / "jpeg"))
    last = get_last_checkpoint(str(output_dir)) if args.resume else None
    print("이어서 학습:" if last else "처음부터 학습", last or "")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output_dir / "final_lora"))
    print("완료:", output_dir / "final_lora")


if __name__ == "__main__":
    main()
