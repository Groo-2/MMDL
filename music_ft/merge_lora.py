"""LoRA 어댑터를 bf16 base(Qwen3-VL-4B-Instruct @ ebb281e)에 merge해서 평가용 체크포인트를 만든다.

평가는 동결 커맨드에서 모델 경로만 이 폴더로 바꿔서 돌린다 (양자화 없음, bf16).

  python music_ft/merge_lora.py --lora results/grpo_run1/final_lora --out results/grpo_run1/merged
"""

import argparse
from pathlib import Path

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def merge(lora, out, base=MODEL_REPO, revision=MODEL_REVISION, dtype=None, processor_repo=None, low_memory=False):
    """어댑터를 base에 합쳐 저장하고, 적용됐는지 확인한 뒤 merged 모델을 돌려준다."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = dtype or torch.bfloat16
    kwargs = {"revision": revision} if revision else {}

    model = Qwen3VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype, **kwargs)

    # 어댑터가 실제로 적용됐는지 확인한다 (키 이름이 어긋나면 조용히 base와 똑같은 모델이 만들어진다)
    if low_memory:
        # 전체 가중치를 복사해 두면 CPU 메모리가 2배(약 18GB)라 15GB WSL에서 죽는다(2026-11-10 확인). 모듈마다 그 가중치만 복사해 비교한다.
        from peft.tuners.lora import LoraLayer

        model = PeftModel.from_pretrained(model, lora)
        n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
        n_changed = 0
        for module in model.modules():
            if isinstance(module, LoraLayer):
                before = module.base_layer.weight.detach().clone()
                module.merge()
                n_changed += int(not torch.equal(before, module.base_layer.weight))
                del before
        model = model.merge_and_unload()          # 이미 합친 모듈은 건너뛰고 래퍼만 걷어 낸다
    else:
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model = PeftModel.from_pretrained(model, lora)
        n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
        model = model.merge_and_unload()
        after = model.state_dict()
        n_changed = len([k for k in after if k in before and not torch.equal(after[k], before[k])])
        del before
    print(f"LoRA 파라미터 {n_lora}개, merge로 값이 바뀐 가중치 {n_changed}개")
    if n_lora == 0 or not n_changed:
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
    parser.add_argument("--low-memory", action="store_true",
                        help="전체 가중치를 복사하지 않고 모듈별로 검사 (CPU 메모리가 15GB 안팎일 때)")
    args = parser.parse_args()

    model = merge(args.lora, args.out, low_memory=args.low_memory)

    import torch

    dtypes = {str(p.dtype) for p in model.parameters()}
    print("저장:", args.out, "| dtype:", dtypes)
    assert dtypes == {str(torch.bfloat16).replace("torch.", "torch.")}, "bf16이 아닌 파라미터가 섞여 있다"


if __name__ == "__main__":
    main()
