"""LoRA 어댑터를 bf16 base(Qwen3-VL-4B-Instruct @ ebb281e)에 merge해서 평가용 체크포인트를 만든다.

평가는 동결 커맨드에서 모델 경로만 이 폴더로 바꿔서 돌린다 (양자화 없음, bf16).

  python music_ft/merge_lora.py --lora results/grpo_run1/final_lora --out results/grpo_run1/merged
"""

import argparse
from pathlib import Path

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def merge(lora, out, base=MODEL_REPO, revision=MODEL_REVISION, dtype=None, processor_repo=None):
    """어댑터를 base에 합쳐 저장하고, 적용됐는지 확인한 뒤 merged 모델을 돌려준다."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = dtype or torch.bfloat16
    kwargs = {"revision": revision} if revision else {}

    model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype, **kwargs)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model = PeftModel.from_pretrained(model, lora)
    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    model = model.merge_and_unload()

    # 어댑터가 실제로 적용됐는지 확인한다 (키 이름이 어긋나면 조용히 base와 똑같은 모델이 만들어진다)
    after = model.state_dict()
    changed = [k for k in after if k in before and not torch.equal(after[k], before[k])]
    print(f"LoRA 파라미터 {n_lora}개, merge로 값이 바뀐 가중치 {len(changed)}개")
    if n_lora == 0 or not changed:
        raise SystemExit("LoRA가 적용되지 않았다. 어댑터 키 이름과 base 모델 구조가 맞는지 확인하세요.")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(processor_repo or base, **kwargs).save_pretrained(out)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = merge(args.lora, args.out)

    import torch

    dtypes = {str(p.dtype) for p in model.parameters()}
    print("저장:", args.out, "| dtype:", dtypes)
    assert dtypes == {str(torch.bfloat16).replace("torch.", "torch.")}, "bf16이 아닌 파라미터가 섞여 있다"


if __name__ == "__main__":
    main()
