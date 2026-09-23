# MMDL - Qwen3-VL-4B MMMU Evaluation

## Result

- Model: Qwen/Qwen3-VL-4B-Instruct
- Dataset: MMMU validation
- Number of subjects: 30
- Samples per subject: 30
- Total samples: 900
- Correct: 562
- Accuracy: 62.44%

## Model

Qwen/Qwen3-VL-4B-Instruct

Revision:

ebb281ec70b05090aa6165b016eac8ec08e71b17

## Dataset

MMMU/MMMU

Revision:

98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68

Split:

validation

## Generation Settings

- temperature: 0.7
- top_p: 0.8
- top_k: 20
- repetition_penalty: 1.0
- presence_penalty: 1.5
- seed: 3407
- max_new_tokens: 4096
- dtype: bfloat16

## Environment

- GPU: NVIDIA RTX 4090
- vLLM: 0.27.1
- Python: 3.10
- Ubuntu 22.04 / WSL2

## Repository Structure

```text
MMDL/
├── assignment/
│   └── assignment1.md
├── code/
│   └── run_mmmu_reasoning_4096.py
├── results/
│   ├── subject_scores.csv
│   └── summary.json
└── README.md

##  Start vLLM Server

export VLLM_USE_V2_MODEL_RUNNER=0

vllm serve ~/mmdl/models/Qwen3-VL-4B-Instruct \
  --served-model-name qwen3-vl-4b \
  --dtype bfloat16 \
  --max-model-len 9048 \
  --gpu-memory-utilization 0.90 \
  --port 8000

## Run Evaluation

in another terminal

export VLLM_USE_V2_MODEL_RUNNER=0

vllm serve ~/mmdl/models/Qwen3-VL-4B-Instruct \
  --served-model-name qwen3-vl-4b \
  --dtype bfloat16 \
  --max-model-len 9048 \
  --gpu-memory-utilization 0.90 \
  --port 8000

## Output

The evaluation generates:

predictions.jsonl
subject_scores.csv
summary.json

The final baseline score used in this project is:

562 / 900 = 62.44%


