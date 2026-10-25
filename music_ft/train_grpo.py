"""SSMR-Bench Train_visual로 Qwen3-VL-4B-Instruct를 GRPO(LoRA) 학습한다. Colab A100 40GB 기준.

SSMR-Bench 논문(§3.1)의 설정을 따른다: rollout 8, KL 없음(beta=0), 정답 보상만.
논문과 다른 점은 README/계획서 참고 (동결 프롬프트, rule 파서 보상, LoRA, 작은 스텝당 문항 수).

프롬프트는 mmmu_eval/prompting.py의 build_mmmu_pro_cot_text()를 그대로 쓴다. 평가와 같은 문구이고,
평가의 OpenAI chat 형식(텍스트 먼저, 이미지는 등장 순서대로 뒤에)과 같은 순서로 메시지를 만든다.

사용 예 (Colab)
  python train_grpo.py --data-dir /content/ssmr --output-dir /content/drive/MyDrive/mmmu_project/grpo_run1 \
      --max-steps 10 --prompts-per-step 16          # 파일럿
  python train_grpo.py ... --max-steps 60 --resume    # 본 학습 (끊기면 --resume으로 이어서)
"""

import argparse
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
def _mmmu_eval_dir():
    here = HERE
    for cand in (here.parent / "baseline" / "mmmu_eval", here.parent / "mmmu_eval"):
        if cand.is_dir():
            return str(cand)
    raise SystemExit("mmmu_eval 폴더를 찾지 못했다 (../baseline/mmmu_eval 또는 ../mmmu_eval)")

sys.path.insert(0, _mmmu_eval_dir())

from prompting import MAX_PIXELS, MIN_PIXELS, build_mmmu_pro_cot_text  # noqa: E402
from reward import accuracy_reward  # noqa: E402

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

# 동결된 평가 sampling (mmmu_eval/run_mmmu_eval.py)
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 1.5
SEED = 42

NUM_GENERATIONS = 8  # 논문: rollout 8


class Heartbeat:
    """한 스텝이 끝나기 전에는 tqdm이 갱신되지 않으므로, micro-batch 진행을 주기적으로 출력한다."""

    def __new__(cls):
        import time
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self):
                self.t0 = time.time()
                self.micro = 0

            def on_step_begin(self, args, state, control, **kwargs):
                # 이전 스텝의 학습이 잡아 둔 캐시 메모리를 돌려줘야 vLLM이 깨어날 때(wake_up) 메모리 부족이 나지 않는다
                import gc
                import torch

                gc.collect()
                torch.cuda.empty_cache()
                self.micro = 0
                self.step_t0 = time.time()
                print(f"[heartbeat] step {state.global_step + 1} 시작 (rollout 포함, 경과 {time.time() - self.t0:.0f}s)", flush=True)

            def on_substep_end(self, args, state, control, **kwargs):
                self.micro += 1
                if self.micro % 8 == 0:
                    print(f"[heartbeat] step {state.global_step + 1}: micro-batch {self.micro}/{args.gradient_accumulation_steps}, "
                          f"스텝 경과 {time.time() - self.step_t0:.0f}s", flush=True)

            def on_step_end(self, args, state, control, **kwargs):
                print(f"[heartbeat] step {state.global_step} 완료, 스텝 시간 {time.time() - self.step_t0:.0f}s", flush=True)

        return _Callback()


def jpeg_q95(src, dst):
    """평가 파이프라인(prompting.image_to_data_url)과 같이 RGB → JPEG quality 95로 한 번 거친다."""
    from PIL import Image

    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)


def load_examples(data_dir, jpeg_dir, jsonl_name, limit=None):
    examples = []
    with (data_dir / jsonl_name).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text, image_order, _ = build_mmmu_pro_cot_text(row)

            image_files = []
            for n in image_order:
                src = data_dir / row["image_paths"][f"image_{n}"]
                dst = jpeg_dir / (src.stem + ".jpg")
                jpeg_q95(src, dst)
                image_files.append(str(dst))

            # 평가와 같은 순서: 텍스트 먼저, 이미지는 그 뒤에 등장 순서대로
            content = [{"type": "text", "text": text}] + [{"type": "image"} for _ in image_files]
            examples.append({
                "prompt": [{"role": "user", "content": content}],
                "images": image_files,
                "answer": row["answer"],
                "id": row["id"],
                "category": row["category"],
            })
            if limit and len(examples) >= limit:
                break
    return examples


def build_dataset(examples):
    from datasets import Dataset, Image, Sequence

    ds = Dataset.from_list(examples)
    return ds.cast_column("images", Sequence(Image()))  # 경로 → 접근할 때 PIL로 디코딩


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="build_ssmr.py 출력 폴더(dedupe 후 train.dedup.jsonl 사용)")
    parser.add_argument("--output-dir", required=True, help="Drive 경로 권장. 체크포인트가 여기에 저장된다")
    parser.add_argument("--train-file", default="train.dedup.jsonl")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--prompts-per-step", type=int, default=8, help="스텝당 문항 수 (논문 128)")
    parser.add_argument("--micro-batch", type=int, default=2, help="GPU 한 번에 처리하는 completion 수")
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--vllm-mem", type=float, default=0.35)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="디버그용: 앞 N문항만")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completions_per_step = args.prompts_per_step * NUM_GENERATIONS
    if completions_per_step % args.micro_batch:
        raise SystemExit("prompts-per-step × 8 이 micro-batch의 배수여야 한다")
    grad_accum = completions_per_step // args.micro_batch

    examples = load_examples(data_dir, Path("/content/ssmr_jpeg") if Path("/content").exists() else output_dir / "jpeg",
                             args.train_file, args.limit)
    dataset = build_dataset(examples)
    print(f"학습 문항 {len(dataset)}개, 스텝당 {args.prompts_per_step}문항 × {NUM_GENERATIONS} = {completions_per_step} completion")

    processor = AutoProcessor.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    # 평가(vLLM)와 같은 이미지 해상도. qwen-vl-finetune/data_processor.py와 같은 방식으로 설정한다.
    # TRL은 prompt_ids를 왼쪽 패딩하지만 mm_token_type_ids는 이 프로세서로 다시 만든다. Qwen3-VL 토크나이저의 기본 패딩은
    # 오른쪽이라, 길이가 다른 프롬프트(이미지 1장 vs 5장)가 한 배치에 있으면 이미지 토큰 표시가 밀려서
    # get_rope_index에서 shape mismatch가 난다. 왼쪽으로 맞춘다.
    processor.tokenizer.padding_side = "left"
    ip = processor.image_processor
    ip.min_pixels, ip.max_pixels = MIN_PIXELS, MAX_PIXELS
    ip.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

    # 해상도 설정이 실제로 먹는지 모델을 올리기 전에 확인한다 (작은 악보도 평가처럼 약 980토큰이어야 한다)
    from PIL import Image as PILImage

    probe = ip(images=[PILImage.new("RGB", (201, 66), "white")], return_tensors="np")
    t, h, w = probe["image_grid_thw"][0]
    probe_tokens = int(t * h * w) // (ip.merge_size ** 2)
    print(f"해상도 점검: 201x66 이미지 → 이미지 토큰 {probe_tokens} (기대 약 {MIN_PIXELS // 1024})")
    if not MIN_PIXELS // 1024 <= probe_tokens <= MAX_PIXELS // 1024:
        raise SystemExit("min_pixels/max_pixels가 image processor에 적용되지 않았다. 평가와 해상도가 달라진다.")

    # Qwen3-VL 비전 패치 임베딩은 원래 구현(conv3d)을 그대로 쓴다. cuDNN이 깨져 있으면 첫 rollout을 돌리기 전에 멈춘다.
    # (환경 문제는 diag_cudnn.py로 진단한다. 모델 구현을 바꾸거나 cuDNN을 끄지 않는다.)
    probe_x = torch.randn(2, 3, 2, 16, 16, device="cuda", dtype=torch.bfloat16)
    probe_conv = torch.nn.Conv3d(3, 8, (2, 16, 16), (2, 16, 16)).cuda().to(torch.bfloat16)
    try:
        probe_conv(probe_x)
    except RuntimeError as e:
        raise SystemExit(f"conv3d(cuDNN) 점검 실패: {str(e)[:120]}\n→ python diag_cudnn.py 로 원인을 확인하세요.")
    print("conv3d 점검: cuDNN 정상")
    del probe_x, probe_conv

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
    )

    # LLM의 attention/MLP에만 LoRA. vision tower와 merger는 건드리지 않는다.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        task_type="CAUSAL_LM",
    )

    config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        seed=SEED,
        bf16=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        # 논문: rollout 8, KL 없음, 정답 보상만
        num_generations=NUM_GENERATIONS,
        beta=0.0,
        loss_type="grpo",
        # 동결 sampling (평가와 동일)
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=1.0,
        generation_kwargs={"presence_penalty": PRESENCE_PENALTY},
        max_completion_length=args.max_completion_length,
        # vLLM colocate: 학습 모델과 같은 GPU에서 rollout
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=args.max_completion_length + 6144,  # 가장 긴 프롬프트 약 5.5k + completion
        # TRL 기본값(sequence_mask)은 vLLM과 학습 모델의 logprob 차이를 시퀀스 전체 곱으로 보정해서, 긴 응답에서
        # 비율이 1e-22까지 떨어져 손실과 grad_norm이 0이 된다(2026-10-25 파일럿 확인). 논문도 이런 보정은 쓰지 않았다.
        # 끄면 old logprob를 다시 계산하는 forward도 생략되어 스텝이 빨라진다.
        vllm_importance_sampling_correction=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        log_completions=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=accuracy_reward,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[Heartbeat()],
    )

    # --resume: 체크포인트가 있으면 이어서, 없으면 처음부터 (HF는 True를 주고 체크포인트가 없으면 에러를 낸다)
    last = get_last_checkpoint(str(output_dir)) if args.resume else None
    print("이어서 학습:" if last else "처음부터 학습", last or "")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(output_dir / "final_lora"))
    print("완료:", output_dir / "final_lora")


if __name__ == "__main__":
    main()
