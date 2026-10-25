"""LoRA 어댑터를 bf16 base(Qwen3-VL-4B-Instruct @ ebb281e)에 merge해서 평가용 체크포인트를 만든다.

평가는 동결 커맨드에서 --model_path만 이 폴더로 바꿔서 돌린다 (양자화 없음, bf16).

  python merge_lora.py --lora /content/drive/MyDrive/mmmu_project/grpo_run1/final_lora \
                       --out  /content/drive/MyDrive/mmmu_project/grpo_run1/merged
"""

import argparse
from pathlib import Path

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, args.lora)
    model = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(MODEL_REPO, revision=MODEL_REVISION).save_pretrained(out)

    dtypes = {str(p.dtype) for p in model.parameters()}
    print("저장:", out, "| dtype:", dtypes)
    assert dtypes == {"torch.bfloat16"}, "bf16이 아닌 파라미터가 섞여 있다"


if __name__ == "__main__":
    main()
