"""평가 벤치마크 정의: MMMU validation(기본), MMMU test, MMMU-Pro(standard 10/4지선다, vision).

모든 벤치마크가 같은 평가 코드(프롬프트 방식, sampling, 파서, Llama judge, max_new_tokens 32,768, 해상도)를 쓰고
바뀌는 것은 데이터와 (vision의 경우) 프롬프트뿐이다. 기본값은 지금까지 모든 결과를 만든 MMMU validation이다.

데이터 (모두 revision을 고정한다)
  mmmu_val              MMMU/MMMU 과목별 config, validation 900문항 (과목당 30)
  mmmu_test             MMMU/MMMU 과목별 config, test 10,500문항. 정답은 2026-02에 공개되어 고정한 revision에 들어 있다
  mmmu_pro_standard10   MMMU/MMMU_Pro "standard (10 options)" test 1,730문항 (한 config에 전 과목, subject 열로 나눈다)
  mmmu_pro_standard4    MMMU/MMMU_Pro "standard (4 options)" test 1,730문항 (선택지 수는 2~9로 다양하다)
  mmmu_pro_vision       MMMU/MMMU_Pro "vision" test 1,730문항. 질문과 선택지가 스크린샷 한 장에 들어 있다

MMMU-Pro 공식 점수는 standard(10지선다)와 vision 점수의 평균이다. 이 코드는 각각을 따로 채점한다.
채점은 validation과 같은 방식(Answer: 줄 → Llama judge → 오답, 무작위 추측 없음)이라 공식 평가 코드의 점수와
조금 다를 수 있다. 같은 방식으로 base와 비교하는 것이 목적이다.

전체 정확도의 정의: validation은 과목당 30문항이라 과목 평균(macro)과 전체 평균(micro)이 같다. test와 MMMU-Pro는
과목별 문항 수가 달라서 공식 정의대로 **전체 문항 평균(micro)** 을 쓴다. 과목 평균(macro)도 summary.json에 함께 기록한다.
"""

from dataclasses import dataclass

MMMU_REPO = "MMMU/MMMU"
MMMU_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"   # validation 평가에 쓴 것과 같은 revision

PRO_REPO = "MMMU/MMMU_Pro"
PRO_REVISION = "0d7426df4ccb3d8704a992fe6850abee7c262127"    # 2026-10-05

PROMPT_STANDARD = "mmmu_pro_cot"
PROMPT_VISION = "mmmu_pro_cot_vision"

# MMMU test 과목별 문항 수 (합계 10,500). 로드할 때 이 값과 다르면 중단한다.
MMMU_TEST_COUNTS = {
    "Accounting": 380, "Agriculture": 287, "Architecture_and_Engineering": 551, "Art": 231, "Art_Theory": 429,
    "Basic_Medical_Science": 326, "Biology": 345, "Chemistry": 603, "Clinical_Medicine": 325,
    "Computer_Science": 371, "Design": 169, "Diagnostics_and_Laboratory_Medicine": 162, "Economics": 267,
    "Electronics": 256, "Energy_and_Power": 432, "Finance": 355, "Geography": 565, "History": 278,
    "Literature": 112, "Manage": 245, "Marketing": 181, "Materials": 458, "Math": 505,
    "Mechanical_Engineering": 429, "Music": 334, "Pharmacy": 430, "Physics": 408, "Psychology": 305,
    "Public_Health": 509, "Sociology": 252,
}

# MMMU-Pro 과목별 문항 수 (합계 1,730). 세 설정이 같은 문항을 다른 모양으로 담고 있어 같은 값이다.
PRO_COUNTS = {
    "Accounting": 58, "Agriculture": 60, "Architecture_and_Engineering": 60, "Art": 53, "Art_Theory": 55,
    "Basic_Medical_Science": 52, "Biology": 59, "Chemistry": 60, "Clinical_Medicine": 59, "Computer_Science": 60,
    "Design": 60, "Diagnostics_and_Laboratory_Medicine": 60, "Economics": 59, "Electronics": 60,
    "Energy_and_Power": 58, "Finance": 60, "Geography": 52, "History": 56, "Literature": 52, "Manage": 50,
    "Marketing": 59, "Materials": 60, "Math": 60, "Mechanical_Engineering": 59, "Music": 60, "Pharmacy": 57,
    "Physics": 60, "Psychology": 60, "Public_Health": 58, "Sociology": 54,
}

VAL_COUNTS = {subject: 30 for subject in MMMU_TEST_COUNTS}

# vision에는 질문 글이 따로 없다(스크린샷 안에 있다). judge가 보는 'Question' 칸에 넣는 자리표시 문구.
VISION_QUESTION_PLACEHOLDER = "[The question and its options are shown in the image.]"


@dataclass(frozen=True)
class Benchmark:
    name: str
    repo: str
    revision: str
    split: str
    config: object        # None이면 과목마다 config(MMMU), 문자열이면 한 config에 전 과목(MMMU-Pro)
    prompt: str           # 기본 프롬프트 변형
    prompts_allowed: tuple
    overall: str          # 전체 정확도 정의: "macro" 또는 "micro"
    counts: dict          # 과목 -> 문항 수
    results_subdir: object  # 결과 폴더 하위 이름. None이면 기존 경로(results/<태그>/run_NN)를 그대로 쓴다
    kind: str             # "mmmu" 또는 "pro_standard" 또는 "pro_vision"

    @property
    def expected_total(self):
        return sum(self.counts.values())

    def normalize(self, sample):
        """데이터 행을 평가 코드가 기대하는 모양(question_type, question)으로 맞춘다. MMMU 행은 그대로 둔다."""

        if self.kind == "mmmu":
            return sample

        out = dict(sample)
        out["question_type"] = "multiple-choice"   # MMMU-Pro는 객관식만 있다

        if self.kind == "pro_vision":
            out["question"] = VISION_QUESTION_PLACEHOLDER

        return out


BENCHMARKS = {
    "mmmu_val": Benchmark(
        name="mmmu_val", repo=MMMU_REPO, revision=MMMU_REVISION, split="validation", config=None,
        prompt=PROMPT_STANDARD, prompts_allowed=(PROMPT_STANDARD, "github_cot"), overall="macro",
        counts=VAL_COUNTS, results_subdir=None, kind="mmmu"),
    "mmmu_test": Benchmark(
        name="mmmu_test", repo=MMMU_REPO, revision=MMMU_REVISION, split="test", config=None,
        prompt=PROMPT_STANDARD, prompts_allowed=(PROMPT_STANDARD, "github_cot"), overall="micro",
        counts=MMMU_TEST_COUNTS, results_subdir="mmmu_test", kind="mmmu"),
    "mmmu_pro_standard10": Benchmark(
        name="mmmu_pro_standard10", repo=PRO_REPO, revision=PRO_REVISION, split="test",
        config="standard (10 options)", prompt=PROMPT_STANDARD, prompts_allowed=(PROMPT_STANDARD,),
        overall="micro", counts=PRO_COUNTS, results_subdir="mmmu_pro_standard10", kind="pro_standard"),
    "mmmu_pro_standard4": Benchmark(
        name="mmmu_pro_standard4", repo=PRO_REPO, revision=PRO_REVISION, split="test",
        config="standard (4 options)", prompt=PROMPT_STANDARD, prompts_allowed=(PROMPT_STANDARD,),
        overall="micro", counts=PRO_COUNTS, results_subdir="mmmu_pro_standard4", kind="pro_standard"),
    "mmmu_pro_vision": Benchmark(
        name="mmmu_pro_vision", repo=PRO_REPO, revision=PRO_REVISION, split="test", config="vision",
        prompt=PROMPT_VISION, prompts_allowed=(PROMPT_VISION,), overall="micro", counts=PRO_COUNTS,
        results_subdir="mmmu_pro_vision", kind="pro_vision"),
}

DEFAULT_BENCHMARK = "mmmu_val"
