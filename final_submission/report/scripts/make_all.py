#!/usr/bin/env python3
"""발표자료용 표(md/csv/tex)와 그림(png/pdf)을 만든다.

  python make_all.py --work <결과를 푼 폴더> --out <발표자료 폴더> [--base130 <base 최종본 폴더>]

--work 아래에 다음 폴더가 있어야 한다 (각 결과 묶음을 ~/Downloads의 tar로 풀어 둔 것):
  base_cu129, dpo_dv1_orig, dpo_dv1_re_cu129, grpo_run3, grpo_run3_c5, rft_dv1, rft_dv2_loop4, dpo_dv1_loop4,
  team_cu130, dvmx1_cu130, dv1_re_cu130, base_cu130_v1, team_orig, team_train
base의 cu130 값은 기본으로 base_cu130_v1(요청 오류 8건 재시도 전)을 쓴다. 최종본을 받으면 --base130으로 그 폴더를 지정해서 다시 실행한다.

수치는 원본 예측(final_predictions.jsonl)과 요약(summary_runs.json)에서 직접 계산한다.
기록에만 있고 원본 파일이 없는 값(GRPO run1/run2 요약, 4과목 진단의 일부, 악보 홀드아웃)은 HARDCODED 블록에 출처와 함께 적었다.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ---------------------------------------------------------------- 스타일
for _p in ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
    if Path(_p).exists():
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False})
C = {"base": "#7f7f7f", "grpo": "#D55E00", "rft": "#E69F00", "dpo": "#0072B2", "dvmx": "#009E73", "team": "#CC79A7"}

# ---------------------------------------------------------------- HARDCODED (기록 기준, 출처 표기)
# reports/ssmr_grpo_run1/README.md (실습실 이전 PC, WSL2, cu129, transformers 5.19.0)
HIST_README = {
    "grpo_run1": {"seeds": [55.00, 55.78], "trunc": [66, 55], "music": None, "note": "E1, 140스텝, 잘림 보상 0"},
    "grpo_run2": {"seeds": [60.22, 59.78], "trunc": [118, 128], "music": 25.0, "note": "E1'', 잘림 제외+KL 0.04, 60스텝"},
}
# 실습실 로그 화면(2026-11-15~17)에서 옮긴 악보 홀드아웃(MusiXQA holdout 400, 8,192토큰, cu129 mmdl). 팀원 GRPO는 서버 기동 실패로 미측정.
HOLDOUT = {  # 모델: [(seed, 전체, 잘림, 끝난문항정확도, Clef, KeySignature, MeasureCount, PitchReading)]
    "base": [(42, .527, 123, .7617, .98, .47, .49, .17), (43, .527, 125, .7673, .99, .49, .46, .17)],
    "dpo_dv1_re": [(42, .627, 36, .6896, .96, .57, .57, .41), (43, .620, 40, .6889, .96, .56, .58, .38)],
    "dpo_dvmx1": [(42, .650, 11, .6684, .97, .58, .54, .51), (43, .680, 12, .7010, .94, .62, .59, .57)],
    "team": [(42, .728, 0, .7275, .97, .55, .60, .79), (43, .718, 0, .7175, .96, .55, .60, .76)],  # 2026-11-17 재실행(첫 시도는 GPU 메모리를 다른 작업이 점유해 서버 기동 실패)
}
# 4과목 진단(Electronics, Materials, Mechanical, Music; 120문항, seed 42)의 루프(잘림+오류). base와 rft_dv1은 2026-11-11 기록값.
LOOP4_RECORD = {"base": 31, "rft_dv1": 72}

NAMES = {"base": "base", "team": "MusiXQA GRPO\n(팀원)", "dv1_re": "DPO\n(DeepVision)", "dvmx1": "DPO\n(DeepVision\n+MusiXQA)"}
NAMES1 = {"base": "base", "team": "MusiXQA GRPO(팀원)", "dv1_re": "DPO(DeepVision)", "dvmx1": "DPO(DeepVision+MusiXQA)"}
COL = {"base": C["base"], "team": C["team"], "dv1_re": C["dpo"], "dvmx1": C["dvmx"]}


# ---------------------------------------------------------------- 데이터 읽기
def load_run(d):
    rows = {}
    for line in open(Path(d) / "final_predictions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def load_model(d):
    d = Path(d)
    runs = sorted(p for p in d.glob("run_*") if (p / "final_predictions.jsonl").exists())
    sr = json.load(open(d / "summary_runs.json"))
    return {"dir": d, "runs": [load_run(p) for p in runs], "sr": sr}


def acc_list(m):
    return [x["overall"]["acc"] * 100 for x in m["sr"]["per_run"]]


def trunc_list(m):
    return [x["overall"]["truncated"] for x in m["sr"]["per_run"]]


def err_list(m):
    return [x["overall"]["errors"] for x in m["sr"]["per_run"]]


def music_list(m):
    return [round(x["music"]["acc"] * 30) for x in m["sr"]["per_run"]]


def avg2(m):
    return float(np.mean(acc_list(m)))


def per_q(m):
    """문항별 평균 정답률 {id: 평균}, 과목 {id: subject}."""
    ids = sorted(m["runs"][0])
    arr = np.array([[1.0 if r[i]["correct"] else 0.0 for i in ids] for r in m["runs"]]).mean(axis=0)
    subj = [m["runs"][0][i]["subject"] for i in ids]
    return ids, arr, subj


def paired(a, b, n_boot=10000, seed=0):
    """b - a 의 문항 단위 paired bootstrap (전체, Music, 과목별). 단위는 %p."""
    ia, xa, subj = per_q(a)
    ib, xb, _ = per_q(b)
    assert ia == ib
    d = (xb - xa) * 100
    rng = np.random.default_rng(seed)

    def ci(idx):
        sub = d[idx]
        boots = sub[rng.integers(0, len(sub), size=(n_boot, len(sub)))].mean(axis=1)
        return float(sub.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    out = {"전체": ci(np.arange(len(d)))}
    subs = sorted(set(subj))
    for s in subs:
        out[s] = ci(np.array([k for k, v in enumerate(subj) if v == s]))
    return out


def decomposition(base, model):
    """seed별로 (둘 다 끝남 / base만 안 끝남 / 모델만 안 끝남 / 둘 다 안 끝남)의 정답 수 변화(모델 - base)를 평균. 끝남 = finish_reason == stop."""
    acc = defaultdict(list)
    for rb, rm in zip(base["runs"], model["runs"]):
        g = defaultdict(lambda: [0, 0, 0])
        for i in rb:
            tb, tm = rb[i]["finish_reason"] != "stop", rm[i]["finish_reason"] != "stop"
            k = "both_done" if not tb and not tm else "base_loop" if tb and not tm else "model_loop" if tm and not tb else "both_loop"
            g[k][0] += 1
            g[k][1] += bool(rb[i]["correct"])
            g[k][2] += bool(rm[i]["correct"])
        for k in ("both_done", "base_loop", "model_loop", "both_loop"):
            acc[k].append(g[k])
    res = {}
    for k, v in acc.items():
        res[k] = {"n": float(np.mean([x[0] for x in v])), "base": float(np.mean([x[1] for x in v])),
                  "model": float(np.mean([x[2] for x in v]))}
        res[k]["delta"] = res[k]["model"] - res[k]["base"]
    return res


# ---------------------------------------------------------------- 표 출력
TABLE_CAP = {"table7_environments": 48, "table3_subjects_cu130": 40, "table9_hyperparameters": 44, "table5_all_experiments": 30, "table2_paired_cu130": 40, "table4_decomposition": 34}


def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
            .replace("#", r"\#").replace("≥", r"$\geq$").replace("→", r"$\to$").replace("-", "-").replace("±", r"$\pm$")
            .replace("**", "").replace("\n", " "))


def _units(t):
    """글자 폭 추정: 영문·숫자 1, 한글 등 2."""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in t)


def _wrap(text, cap):
    """cap(글자 폭 단위)을 넘으면 공백에서, 공백이 없으면 글자 단위로 줄바꿈한다."""
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = word if not cur else cur + " " + word
            if _units(cand) <= cap:
                cur = cand
                continue
            if cur:
                lines.append(cur)
                cur = ""
            while _units(word) > cap:
                k, w = 0, 0
                while k < len(word) and w + _units(word[k]) <= cap:
                    w += _units(word[k])
                    k += 1
                lines.append(word[:k])
                word = word[k:]
            cur = word
        lines.append(cur)
    return lines


def render_table_image(out, name, caption, header, rows, note=None, cap=26, fs=8.0):
    """표를 그림(png, pdf)으로 저장한다. **로 표시된 셀은 강조(연한 노랑 배경)한다."""
    from matplotlib.patches import Rectangle

    clean = lambda c: str(c).replace("**", "")  # noqa: E731
    bold = lambda c: str(c).startswith("**")  # noqa: E731
    ncol = len(header)
    wrapped = [[_wrap(clean(c), cap) for c in r] for r in rows]
    head = [_wrap(h, cap) for h in header]
    unit = fs * 0.52 / 72.0                      # 폭 단위 1칸(inch)
    colw = []
    for j in range(ncol):
        m = max([_units(l) for l in head[j]] + [_units(l) for r in wrapped for l in r[j]] + [4])
        colw.append(m * unit + 0.22)
    line_h = fs * 1.45 / 72.0
    pad = 0.09
    head_h = max(len(h) for h in head) * line_h + pad * 2
    rh = [max(len(c) for c in r) * line_h + pad * 2 for r in wrapped]
    width = sum(colw)
    cap_lines = _wrap(caption.replace("**", ""), max(24, int(width / unit * 0.82)))
    note_lines = _wrap(note, max(30, int(width / unit))) if note else []
    top = len(cap_lines) * (line_h + 0.03) + 0.18
    bottom = (len(note_lines) * (line_h + 0.02) + 0.18) if note_lines else 0.08
    height = top + head_h + sum(rh) + bottom
    fig = plt.figure(figsize=(width + 0.1, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width + 0.1)
    ax.set_ylim(height, 0)
    ax.axis("off")
    for i, l in enumerate(cap_lines):
        ax.text(0.04, 0.1 + i * (line_h + 0.03), l, fontsize=fs + 1.2, va="top", ha="left", fontweight="bold")
    y = top
    ax.add_patch(Rectangle((0.05, y), width, head_h, fc="#1f3a5f", ec="none"))
    x = 0.05
    for j, h in enumerate(head):
        for k, l in enumerate(h):
            ax.text(x + colw[j] / 2, y + pad + k * line_h, l, fontsize=fs, color="white", va="top", ha="center")
        x += colw[j]
    y += head_h
    for i, r in enumerate(wrapped):
        raw = rows[i]
        if i % 2:
            ax.add_patch(Rectangle((0.05, y), width, rh[i], fc="#f3f5f8", ec="none"))
        x = 0.05
        for j, cell in enumerate(r):
            if bold(raw[j]):
                ax.add_patch(Rectangle((x, y), colw[j], rh[i], fc="#fff3c4", ec="none"))
            left = j == 0 or max(_units(l) for l in cell) > 18
            for k, l in enumerate(cell):
                if left:
                    ax.text(x + 0.08, y + pad + k * line_h, l, fontsize=fs, va="top", ha="left")
                else:
                    ax.text(x + colw[j] / 2, y + pad + k * line_h, l, fontsize=fs, va="top", ha="center")
            x += colw[j]
        y += rh[i]
        ax.plot([0.05, 0.05 + width], [y, y], color="#d0d5dd", lw=0.5)
    ax.plot([0.05, 0.05 + width], [top, top], color="#1f3a5f", lw=1.2)
    ax.plot([0.05, 0.05 + width], [y, y], color="#1f3a5f", lw=1.2)
    for i, l in enumerate(note_lines):
        ax.text(0.05, y + 0.08 + i * (line_h + 0.02), l, fontsize=fs - 0.8, va="top", ha="left", style="italic", color="#444444")
    d = Path(out) / "table_images"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=250)
    fig.savefig(d / f"{name}.pdf")
    plt.close(fig)


def write_table(out, name, caption, header, rows, note=None):
    t = Path(out) / "tables"
    t.mkdir(parents=True, exist_ok=True)
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", f"*{note}*"]
    (t / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(t / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([str(c).replace("**", "") for c in r])
    colspec = "l" + "r" * (len(header) - 1)
    tex = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{tab:{name}}}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(tex_escape(h) for h in header) + " \\\\", "\\midrule"]
    tex += [" & ".join(tex_escape(c) for c in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    if note:
        tex += [f"\\\\[2pt]{{\\footnotesize {tex_escape(note)}}}"]
    tex += ["\\end{table}"]
    (t / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    render_table_image(out, name, caption, header, rows, note, cap=TABLE_CAP.get(name, 26))


def save_fig(fig, out, name):
    f = Path(out) / "figures"
    f.mkdir(parents=True, exist_ok=True)
    fig.savefig(f / f"{name}.png", bbox_inches="tight")
    fig.savefig(f / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def pm(a, b=None, nd=2):
    return f"{a:+.{nd}f}" if b is None else f"{a:+.{nd}f} [{b[0]:+.{nd}f}, {b[1]:+.{nd}f}]"


# ---------------------------------------------------------------- 본문
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base130", default=None, help="base의 cu130 최종본 폴더 (기본: work/base_cu130_v1)")
    a = ap.parse_args()
    W, OUT = Path(a.work), Path(a.out)
    (OUT / "data").mkdir(parents=True, exist_ok=True)

    M130 = {"base": load_model(a.base130 or W / "base_cu130_v1"), "team": load_model(W / "team_cu130"),
            "dv1_re": load_model(W / "dv1_re_cu130"), "dvmx1": load_model(W / "dvmx1_cu130")}
    base129 = load_model(W / "base_cu129" / "baseline" / "full")
    dv1_orig = load_model(W / "dpo_dv1_orig")
    dv1_re129 = load_model(W / "dpo_dv1_re_cu129" / "full")
    grpo3 = load_model(W / "grpo_run3" / "full")
    c5 = load_model(W / "grpo_run3_c5")
    rft1 = load_model(W / "rft_dv1")
    order = ["base", "team", "dv1_re", "dvmx1"]

    # ===== 표 1: cu130 주요 결과
    rows = []
    for k in order:
        m = M130[k]
        s = m["sr"]
        ac, tr, er, mu = acc_list(m), trunc_list(m), err_list(m), music_list(m)
        rows.append([NAMES1[k], f"{ac[0]:.2f} ({round(ac[0] * 9)})", f"{ac[1]:.2f} ({round(ac[1] * 9)})", f"**{np.mean(ac):.2f}**",
                     f"{tr[0]} / {tr[1]}", f"{er[0]} / {er[1]}", f"{mu[0]} / {mu[1]}", f"{np.mean(mu) / 30 * 100:.1f}"])
    write_table(OUT, "table1_main_cu130", "표 1. MMMU validation 900문항 결과 (cu130, 같은 환경, seed 42·43)",
                ["모델", "seed 42 정확도 % (정답 수)", "seed 43 정확도 % (정답 수)", "Avg@2 %", "잘림 (42 / 43)", "요청 오류 (42 / 43)",
                 "Music 정답 (42 / 43, 30문항)", "Music Avg@2 %"],
                rows, ("base는 요청 오류 재시도 후 최종본이다." if a.base130 else "base는 요청 오류 재시도 전(v1) 값이다(최종본은 --base130으로 지정).") + " 잘림은 32,768토큰 안에 끝나지 않은 응답 수.")

    # ===== 표 2: paired bootstrap
    P = {}
    for k in ("team", "dv1_re", "dvmx1"):
        P[("base", k)] = paired(M130["base"], M130[k])
    P[("dv1_re", "dvmx1")] = paired(M130["dv1_re"], M130["dvmx1"])
    P[("team", "dvmx1")] = paired(M130["team"], M130["dvmx1"])
    P[("dv1_re", "team")] = paired(M130["dv1_re"], M130["team"])
    rows = []
    for (x, y), p in P.items():
        o, mu = p["전체"], p["Music"]
        sig = lambda t: " *" if (t[1] > 0 or t[2] < 0) else ""  # noqa: E731
        rows.append([f"{NAMES1[x]} → {NAMES1[y]}", f"{o[0]:+.2f} [{o[1]:+.2f}, {o[2]:+.2f}]{sig(o)}", f"{mu[0]:+.1f} [{mu[1]:+.1f}, {mu[2]:+.1f}]{sig(mu)}"])
    write_table(OUT, "table2_paired_cu130", "표 2. 문항 단위 paired bootstrap (95% 신뢰구간, %p, cu130)",
                ["비교 (A → B)", "MMMU 전체 차이(B-A)", "Music 차이(B-A)"], rows,
                "* = 신뢰구간이 0을 포함하지 않음. 문항을 10,000번 재표집. 각 문항의 값은 두 seed의 평균 정답률.")

    # ===== 표 3: 30과목
    subs = sorted(M130["base"]["sr"]["per_subject_mean_acc"], key=lambda s: -(M130["dvmx1"]["sr"]["per_subject_mean_acc"][s] - M130["base"]["sr"]["per_subject_mean_acc"][s]))
    rows = []
    for s in subs:
        v = {k: M130[k]["sr"]["per_subject_mean_acc"][s] * 100 for k in order}
        rows.append([s] + [f"{v[k]:.1f}" for k in order] + [f"{v['dvmx1'] - v['base']:+.1f}"])
    mean = {k: np.mean(list(M130[k]["sr"]["per_subject_mean_acc"].values())) * 100 for k in order}
    rows.append(["**평균(30과목)**"] + [f"**{mean[k]:.2f}**" for k in order] + [f"**{mean['dvmx1'] - mean['base']:+.2f}**"])
    write_table(OUT, "table3_subjects_cu130", "표 3. 과목별 정확도 (%, 2 seed 평균, cu130)",
                ["과목"] + [NAMES1[k] for k in order] + ["DPO(DM)-base"], rows,
                "과목당 30문항이라 과목 단위 차이는 노이즈가 크다. 정렬은 DPO(DeepVision+MusiXQA)-base 큰 순.")

    # ===== 표 4: 정확도 변화의 분해
    D = {k: decomposition(M130["base"], M130[k]) for k in ("team", "dv1_re", "dvmx1")}
    rows = []
    lab = {"both_done": "둘 다 끝낸 문항", "base_loop": "base만 안 끝낸 문항(루프 해소)", "model_loop": "모델만 안 끝낸 문항", "both_loop": "둘 다 안 끝낸 문항"}
    for g in ("both_done", "base_loop", "model_loop", "both_loop"):
        r = [lab[g], f"{D['dvmx1'][g]['n']:.0f}"]
        for k in ("team", "dv1_re", "dvmx1"):
            d = D[k][g]
            r.append(f"{d['n']:.0f}: {d['base']:.0f} → {d['model']:.0f} ({d['delta']:+.1f})")
        rows.append([r[0], r[2], r[3], r[4]])
    tot = [f"{sum(D[k][g]['delta'] for g in D[k]):+.1f}" for k in ("team", "dv1_re", "dvmx1")]
    rows.append(["**합계(정답 수 변화)**"] + [f"**{t}**" for t in tot])
    write_table(OUT, "table4_decomposition", "표 4. base 대비 정답 수 변화의 분해 (cu130, seed별 base와 짝지어 2 seed 평균, 형식: 문항 수: base 정답 → 모델 정답 (변화))",
                ["문항 그룹", "MusiXQA GRPO(팀원)", "DPO(DeepVision)", "DPO(DeepVision+MusiXQA)"], rows,
                "'끝남' = finish_reason이 stop. 합계의 대부분이 'base만 안 끝낸 문항'에서 나온다 = 이득의 주된 출처는 루프 해소.")

    # ===== 표 5: 전체 실험 이력
    hist = []  # (이름, 환경, seeds, trunc, music%, 비고, 계열, 단일seed)
    b = base129
    hist.append(("base", "cu129 (이전 PC)", acc_list(b), trunc_list(b), np.mean(music_list(b)) / 30 * 100, "기준", "base", False))
    hist.append(("GRPO run1 (E1)", "cu129 (이전 PC)", HIST_README["grpo_run1"]["seeds"], HIST_README["grpo_run1"]["trunc"], np.nan, HIST_README["grpo_run1"]["note"], "grpo", False))
    hist.append(("GRPO run2 (E1'')", "cu129 (이전 PC)", HIST_README["grpo_run2"]["seeds"], HIST_README["grpo_run2"]["trunc"], HIST_README["grpo_run2"]["music"], HIST_README["grpo_run2"]["note"], "grpo", False))
    hist.append(("GRPO run3 (E1''')", "cu129 (이전 PC)", acc_list(grpo3), trunc_list(grpo3), np.mean(music_list(grpo3)) / 30 * 100, "스텝당 32문항 × 15스텝", "grpo", False))
    hist.append(("GRPO run3 5스텝", "cu129 (이전 PC)", acc_list(c5), trunc_list(c5), music_list(c5)[0] / 30 * 100, "체크포인트, seed 42만", "grpo", True))
    hist.append(("RFT dv1", "cu129 (이전 PC)", acc_list(rft1), trunc_list(rft1), music_list(rft1)[0] / 30 * 100, "DeepVision 맞은 응답만, seed 42만, 오류 45건 포함", "rft", True))
    hist.append(("DPO dv1 (원본)", "cu129 (이전 PC)", acc_list(dv1_orig), trunc_list(dv1_orig), np.mean(music_list(dv1_orig)) / 30 * 100, "어댑터 전송 중 손상", "dpo", False))
    hist.append(("DPO dv1_re (재학습)", "cu129 (새 머신)", acc_list(dv1_re129), trunc_list(dv1_re129), np.mean(music_list(dv1_re129)) / 30 * 100, "같은 쌍·같은 설정", "dpo", False))
    for k, nm, fam in (("base", "base", "base"), ("team", "MusiXQA GRPO(팀원)", "team"), ("dv1_re", "DPO dv1_re", "dpo"), ("dvmx1", "DPO dv1+MusiXQA (제출)", "dvmx")):
        m = M130[k]
        hist.append((nm, "cu130 (새 머신)", acc_list(m), trunc_list(m), np.mean(music_list(m)) / 30 * 100, ("" if a.base130 else "base는 재시도 전") if k == "base" else "", fam, False))
    rows = []
    for nm, env, seeds, tr, mu, note, fam, single in hist:
        sd = " / ".join(f"{x:.2f}" for x in seeds)
        rows.append([nm, env, sd, f"**{np.mean(seeds):.2f}**" if not single else f"{np.mean(seeds):.2f} (1회)",
                     " / ".join(str(x) for x in tr), "-" if np.isnan(mu) else f"{mu:.1f}", note])
    write_table(OUT, "table5_all_experiments", "표 5. 전체 실험 이력 (MMMU validation 900문항)",
                ["실험", "평가 환경", "seed별 정확도 %", "Avg@2 %", "잘림", "Music %", "비고"], rows,
                "환경이 다른 값끼리는 직접 비교하지 말고 같은 환경의 base와 비교한다. GRPO run1·run2의 요약은 reports/ssmr_grpo_run1/README.md의 기록값.")

    # ===== 표 6: 악보 홀드아웃
    rows = []
    for k, label in (("base", "base"), ("team", "MusiXQA GRPO(팀원)"), ("dpo_dv1_re", "DPO(DeepVision)"), ("dpo_dvmx1", "DPO(DeepVision+MusiXQA)")):
        arr = np.array([r[1:] for r in HOLDOUT[k]])
        mean = arr.mean(axis=0)
        rows.append([label, f"{mean[0] * 100:.1f}", f"{mean[1]:.0f}", f"{mean[2] * 100:.1f}", f"{mean[3] * 100:.0f}", f"{mean[4] * 100:.0f}", f"{mean[5] * 100:.0f}", f"{mean[6] * 100:.0f}"])
    write_table(OUT, "table6_holdout", "표 6. MusiXQA 악보 홀드아웃 400문항 (학습에 쓰지 않음, 8,192토큰, cu129, 2 seed 평균)",
                ["모델", "전체 %", "잘림", "끝낸 문항의 정확도 %", "Clef %", "KeySignature %", "MeasureCount %", "PitchReading %"], rows,
                "끝낸 문항의 정확도는 잘리지 않은 응답만의 정확도(선택 효과가 있어 문항 구성이 다르다).")

    # ===== 표 7: 환경
    def pkgs(snap):
        s = json.load(open(snap))
        fr = {re.split(r"==| @ ", x)[0].lower().replace("_", "-"): x for x in s["pip_freeze"]}

        def g(n):
            x = fr.get(n, "-")
            return x.split("==")[1] if "==" in x else ("0.27.1+cu129 (GitHub 휠)" if "@" in x else "-")
        t = s.get("torch", {})
        return {"GPU/드라이버": s.get("nvidia_smi", "-").replace("NVIDIA GeForce ", "").replace(", 24564 MiB", ""),
                "OS": s.get("platform", "-").split("-with")[0], "torch": t.get("version", "-") if isinstance(t, dict) else "-",
                "vllm": g("vllm"), "transformers": g("transformers"), "datasets": g("datasets"), "trl": g("trl"), "peft": g("peft"), "torchcodec": g("torchcodec")}
    envs = {"이전 PC (cu129)": pkgs(W / "base_cu129" / "baseline" / "full" / "env_snapshot.json"),
            "새 머신 cu129": pkgs(W / "dpo_dv1_re_cu129" / "full" / "env_snapshot.json"),
            "새 머신 cu130": pkgs(W / "dv1_re_cu130" / "env_snapshot.json"),
            "팀원 (cu130)": pkgs(W / "team_orig" / "musixqa_runb_base" / "env_snapshot.json")}
    keys = list(next(iter(envs.values())))
    rows = [[k] + [envs[e][k] for e in envs] for k in keys]
    write_table(OUT, "table7_environments", "표 7. 평가 환경 비교", ["항목"] + list(envs), rows,
                "새 머신 cu130 열은 dpo_dv1_re의 cu130 평가 기록. vllm은 cu129에서 GitHub 릴리스 휠(+cu129), cu130에서 PyPI 기본.")

    # ===== 표 8: 환경 강건성(dpo_dv1_re를 cu129와 cu130에서)
    rob = []
    for s, k in ((42, 0), (43, 1)):
        x, y = dv1_re129["runs"][k], M130["dv1_re"]["runs"][k]
        ids = sorted(set(x) & set(y))
        same = sum(1 for i in ids if bool(x[i]["correct"]) == bool(y[i]["correct"]))
        only129 = sum(1 for i in ids if x[i]["correct"] and not y[i]["correct"])
        only130 = sum(1 for i in ids if y[i]["correct"] and not x[i]["correct"])
        rob.append((s, same / len(ids) * 100, only129, only130, sum(1 for i in ids if x[i]["correct"]), sum(1 for i in ids if y[i]["correct"])))
    seedvar = {}
    for k in ("dv1_re", "base"):
        r0, r1 = M130[k]["runs"]
        ids = sorted(set(r0) & set(r1))
        seedvar[k] = sum(1 for i in ids if bool(r0[i]["correct"]) == bool(r1[i]["correct"])) / len(ids) * 100
    rows = [[f"seed {s}: cu129 vs cu130 (같은 모델·같은 seed)", f"{c1} vs {c2}", f"{same:.1f}", f"{o1} / {o2}"] for s, same, o1, o2, c1, c2 in rob]
    rows.append(["cu130에서 seed 42 vs 43 (DPO dv1_re)", "-", f"{seedvar['dv1_re']:.1f}", "-"])
    rows.append(["cu130에서 seed 42 vs 43 (base)", "-", f"{seedvar['base']:.1f}", "-"])
    write_table(OUT, "table8_env_robustness", "표 8. 환경 강건성: DPO(DeepVision) 모델을 cu129와 cu130에서 같은 seed로 평가",
                ["비교", "정답 수", "같은 정오 비율 %", "한쪽만 정답 (cu129 / cu130)"], rows,
                "정답 수는 거의 같고 문항별 정오는 약 12%가 다르다. 이 정도는 같은 환경에서 seed만 바꿨을 때의 차이(약 14~15%)와 같은 수준이다.")

    # ===== 표 9: 하이퍼파라미터(수기)
    hp = [
        ["공통", "Qwen3-VL-4B-Instruct @ ebb281e, bf16", "LoRA r=64, α=128, LLM의 q,k,v,o,gate,up,down(비전 타워 제외), 학습 가능 파라미터 132,120,576개(fp32 어댑터 약 504MiB)", "-", "-"],
        ["GRPO run1 (E1)", "SSMR-Bench Train_visual 1장 문항 중 통과율 1/8~7/8, 1,120문항", "GRPO, rollout 8, KL 없음(β=0), 정답 보상만, 스텝당 8문항 × 140스텝", "lr 2e-5 상수, max_completion 2048, T 0.7 / top_p 0.8 / top_k 20, grad accum 64, HF generate rollout", "9.9시간"],
        ["GRPO run2 (E1'')", "같음", "GRPO, 잘림 응답을 손실에서 제외(--mask-truncated), β=0.04, 60스텝", "max_completion 2048, 나머지 run1과 같음", "15.4시간"],
        ["GRPO run3 (E1''')", "같음", "run2와 같고 스텝당 32문항 × 15스텝(총 480문항 동일)", "나머지 같음", "16.3시간"],
        ["RFT dv1", "DeepVision 6,000문항 중 통과율 1~7/8인 1,897문항에서 base가 맞고 끝낸 응답 3,326개", "지도학습(교차엔트로피, 응답 위치만)", "lr 1e-5, grad accum 16, 1에폭", "-"],
        ["RFT dv2", "같음", "같음", "lr 3e-6", "-"],
        ["DPO dv1", "DeepVision 쌍 1,897개 (잘린 응답 1,271 / 틀린 응답 626)", "DPO + 좋은 응답 NLL 보조항", "lr 5e-6 cosine, warmup 3스텝, β=0.1, NLL 가중치 0.1, 기준=어댑터를 끈 base, grad accum 16쌍, 1에폭, seed 42", "4,998초 / 재학습 4,684초(119스텝)"],
        ["DPO dv1+MusiXQA (제출)", "DeepVision 쌍 1,897 + MusiXQA 쌍 1,052 (PitchReading 466, KeySignature 470, MeasureCount 116) = 2,949쌍", "DPO + NLL 보조항 (설정은 DPO dv1과 같음)", "185스텝", "8,321초"],
        ["MusiXQA GRPO(팀원)", "MusiXQA 객관식(팀원 변환)", "GRPO 40스텝, 정답 보상", "lr 2e-5 상수, 장치당 batch 1, LoRA r64/α128(같은 구성), rollout 수·보상 세부는 기록에서 확인 못함", "-"],
    ]
    write_table(OUT, "table9_hyperparameters", "표 9. 실험별 데이터와 주요 파라미터", ["실험", "학습 데이터", "방법", "주요 파라미터", "학습 시간"], hp,
                "GRPO run1의 설정은 reports/ssmr_grpo_run1/README.md, DPO는 reports/dpo_dv1/README.md, 팀원 설정은 trainer_state.json에서 읽은 값.")

    # ================================================================= 그림
    # ----- 그림 1: 주요 결과
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for j, (title, getter, ylo, yhi, unit) in enumerate((("MMMU validation 전체 (900문항)", lambda m: acc_list(m), 58, 68, "정확도 (%)"),
                                                         ("MMMU Music (30문항)", lambda m: [x / 30 * 100 for x in music_list(m)], 0, 56, "정확도 (%)"))):
        for i, k in enumerate(order):
            v = getter(M130[k])
            ax[j].bar(i, np.mean(v), color=COL[k], width=0.62, alpha=0.9)
            ax[j].scatter([i - 0.12, i + 0.12], v, color="white", edgecolor="black", s=14, zorder=3, linewidth=0.7)
            top = max(max(v), np.mean(v))
            ax[j].text(i, top + (yhi - ylo) * 0.03, f"{np.mean(v):.1f}" if j == 1 else f"{np.mean(v):.2f}", ha="center", va="bottom", fontsize=8)
        ax[j].set_xticks(range(len(order)))
        ax[j].set_xticklabels([NAMES[k] for k in order], fontsize=7.5)
        ax[j].set_ylim(ylo, yhi)
        ax[j].set_ylabel(unit)
        ax[j].set_title(title)
        ax[j].axhline(np.mean(getter(M130["base"])), color="gray", ls="--", lw=0.8)
    fig.suptitle("그림 1. 같은 환경(cu130)에서 측정한 모델별 성능 (막대: 2 seed 평균, 점: seed별 값)", y=1.02, fontsize=10)
    save_fig(fig, OUT, "fig1_main_results")

    # ----- 그림 2: base 대비 차이 (forest)
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
    for j, (key, title, xl) in enumerate((("전체", "MMMU 전체: base 대비 차이", (-1, 9)), ("Music", "MMMU Music: base 대비 차이", (-15, 55)))):
        for i, k in enumerate(("team", "dv1_re", "dvmx1")):
            m, lo, hi = P[("base", k)][key]
            y = 2 - i
            ax[j].errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=COL[k], capsize=3, ms=6)
            ax[j].text(m, y + 0.26, f"{m:+.1f}  [{lo:+.1f}, {hi:+.1f}]", ha="center", va="bottom", fontsize=7.5)
        ax[j].axvline(0, color="black", lw=0.8)
        ax[j].set_ylim(-0.6, 2.8)
        ax[j].set_xlim(*xl)
        ax[j].set_xlabel("차이 (%p, 95% 신뢰구간)")
        ax[j].set_title(title)
    ax[0].set_yticks([2, 1, 0])
    ax[0].set_yticklabels([NAMES1[k] for k in ("team", "dv1_re", "dvmx1")], fontsize=8)
    fig.suptitle("그림 2. base 대비 정답률 차이와 문항 단위 paired bootstrap 95% 신뢰구간 (cu130)", y=1.04, fontsize=10)
    save_fig(fig, OUT, "fig2_paired_difference")

    # ----- 그림 3: 정확도 vs 루프
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    pts = []
    for nm, env, seeds, tr, mu, note, fam, single in hist:
        pts.append((nm, env, np.mean(seeds), np.mean(tr), fam, single))
    off = {5: (-9, -11), 7: (-3, 8), 8: (7, -13), 10: (7, 5), 11: (-14, 1)}      # 겹치는 번호의 위치만 조정(기본 오른쪽 위)
    for idx, (nm, env, acc, tr, fam, single) in enumerate(pts, 1):
        mk = "s" if "cu130" in env else "o"
        ax.scatter(tr, acc, s=62, marker=mk, facecolor="white" if single else C[fam], edgecolor=C[fam], linewidth=1.4, zorder=3)
        dx, dy = off.get(idx, (6, 5))
        ax.annotate(str(idx), (tr, acc), xytext=(dx, dy), textcoords="offset points", fontsize=8, fontweight="normal")
    legend = "\n".join(f"{i}. {nm.replace(' (제출)', ' [제출]')} ({'cu130' if 'cu130' in env else 'cu129'})" for i, (nm, env, *_r) in enumerate(pts, 1))
    fig.text(0.93, 0.5, legend, va="center", ha="left", fontsize=7.2, family=plt.rcParams["font.family"])
    ax.set_xlabel("32,768토큰 안에 끝나지 않은 응답 수 (900문항 중, 2 seed 평균)")
    ax.set_ylabel("MMMU 전체 정확도 (%)")
    ax.set_title("그림 3. 루프(잘림)와 정확도의 관계 (원: cu129, 사각: cu130, 빈 점: seed 1회)", loc="left")
    ax.invert_xaxis()
    save_fig(fig, OUT, "fig3_accuracy_vs_loops")

    # ----- 그림 4: 정확도 변화의 분해
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    cats = [("both_done", "둘 다 끝낸 문항", "#bdbdbd"), ("base_loop", "base만 안 끝낸 문항(루프 해소)", "#2ca02c"), ("model_loop", "모델만 안 끝낸 문항", "#d62728"), ("both_loop", "둘 다 안 끝낸 문항", "#8c564b")]
    bottoms_pos = np.zeros(3)
    bottoms_neg = np.zeros(3)
    for g, label, col in cats:
        vals = np.array([D[k][g]["delta"] for k in ("team", "dv1_re", "dvmx1")]) / 9.0
        bot = np.where(vals >= 0, bottoms_pos, bottoms_neg)
        ax.bar(range(3), vals, bottom=bot, color=col, width=0.55, label=label)
        bottoms_pos += np.where(vals >= 0, vals, 0)
        bottoms_neg += np.where(vals < 0, vals, 0)
    for i, k in enumerate(("team", "dv1_re", "dvmx1")):
        tot = sum(D[k][g]["delta"] for g in D[k]) / 9.0
        ax.text(i, bottoms_pos[i] + 0.12, f"합계 {tot:+.1f}%p", ha="center", fontsize=8.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(bottoms_neg.min() - 0.4, bottoms_pos.max() + 0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAMES[k] for k in ("team", "dv1_re", "dvmx1")], fontsize=8)
    ax.set_ylabel("base 대비 정확도 변화 (%p)")
    ax.set_title("그림 4. 정확도 향상의 출처 분해 (cu130, 같은 seed끼리 짝지음)", pad=10)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_fig(fig, OUT, "fig4_gain_decomposition")

    # ----- 그림 5: 과목별 히트맵
    sig = {}
    mat = []
    names_sorted = sorted([s for s in P[("base", "dvmx1")] if s not in ("전체", "Music")], key=lambda s: -P[("base", "dvmx1")][s][0])
    names_sorted = ["Music"] + names_sorted
    for s in names_sorted:
        row = []
        for k in ("team", "dv1_re", "dvmx1"):
            m, lo, hi = P[("base", k)][s]
            row.append(m)
            sig[(s, k)] = lo > 0 or hi < 0
        mat.append(row)
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(4.8, 8.6))
    im = ax.imshow(mat, cmap="RdBu", vmin=-30, vmax=30, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["MusiXQA\nGRPO(팀원)", "DPO\n(DeepVision)", "DPO\n(DV+MusiXQA)"], fontsize=7.5)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=7)
    ax.xaxis.tick_top()
    ax.grid(False)
    for i, s in enumerate(names_sorted):
        for j, k in enumerate(("team", "dv1_re", "dvmx1")):
            ax.text(j, i, f"{mat[i, j]:+.0f}{'*' if sig[(s, k)] else ''}", ha="center", va="center", fontsize=6.5, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("base 대비 정확도 차이 (%p)")
    ax.set_title("그림 5. 과목별 base 대비 차이 (cu130)\n* = 95% 신뢰구간이 0을 포함하지 않음 (과목당 30문항)", fontsize=9, pad=28)
    save_fig(fig, OUT, "fig5_subject_heatmap")

    # ----- 그림 6: 실험 이력
    short = ["base", "GRPO\nrun1", "GRPO\nrun2", "GRPO\nrun3", "GRPO run3\n5스텝", "RFT\ndv1", "DPO\ndv1(원본)", "DPO\ndv1_re", "base", "GRPO\n(팀원)", "DPO\ndv1_re", "DPO\ndv1+MX\n(제출)"]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    xs = []
    xi = 0
    prev_env = None
    for n_i, (nm, env, seeds, tr, mu, note, fam, single) in enumerate(hist):
        if prev_env is not None and env.split(" (")[0] != prev_env:
            xi += 0.8
        prev_env = env.split(" (")[0]
        ax.bar(xi, np.mean(seeds), color="white" if single else C[fam], edgecolor=C[fam], hatch="//" if single else None, width=0.72, linewidth=1.2)
        if len(seeds) > 1:
            ax.scatter([xi - 0.1, xi + 0.1], seeds, color="white", edgecolor="black", s=9, zorder=3, linewidth=0.6)
        ax.text(xi, max(max(seeds), np.mean(seeds)) + 0.35, f"{np.mean(seeds):.1f}", ha="center", fontsize=7.5)
        xs.append((xi, short[n_i]))
        xi += 1
    ax.set_xticks([x for x, _ in xs])
    ax.set_xticklabels([n for _, n in xs], fontsize=7)
    ax.axhline(avg2(base129), color="gray", ls="--", lw=0.8)
    ax.axhline(avg2(M130["base"]), color="gray", ls=":", lw=0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="gray", ls="--", lw=0.9, label=f"cu129 환경의 base ({avg2(base129):.1f})"),
                       Line2D([0], [0], color="gray", ls=":", lw=0.9, label=f"cu130 환경의 base ({avg2(M130['base']):.1f})")], loc="upper left", fontsize=7.5, ncol=2)
    ax.set_ylim(50, 70)
    ax.set_ylabel("MMMU 전체 Avg@2 (%)")
    ax.set_title("그림 6. 전체 실험의 MMMU 정확도 (왼쪽 군 cu129, 오른쪽 군 cu130, 빗금은 seed 1회, 점은 seed별 값)", fontsize=9.5)
    save_fig(fig, OUT, "fig6_experiment_history")

    # ----- 그림 7: 악보 홀드아웃
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.3), gridspec_kw={"width_ratios": [2.1, 1, 1]})
    cats = ["Clef", "Key\nSignature", "Measure\nCount", "Pitch\nReading", "전체"]
    mods = [("base", "base", C["base"]), ("team", "MusiXQA GRPO(팀원)", C["team"]), ("dpo_dv1_re", "DPO(DeepVision)", C["dpo"]), ("dpo_dvmx1", "DPO(DeepVision+MusiXQA)", C["dvmx"])]
    for i, (k, lab_, col) in enumerate(mods):
        arr = np.array([r[1:] for r in HOLDOUT[k]]).mean(axis=0) * np.array([100, 1, 100, 100, 100, 100, 100])
        vals = [arr[3], arr[4], arr[5], arr[6], arr[0]]
        ax[0].bar(np.arange(5) + (i - 1.5) * 0.2, vals, width=0.19, color=col, label=lab_)
    ax[0].set_xticks(range(5))
    ax[0].set_xticklabels(cats, fontsize=8)
    ax[0].set_ylabel("정확도 (%)")
    ax[0].set_title("(a) 유형별 정확도")
    ax[0].legend(fontsize=7)
    for i, (k, lab_, col) in enumerate(mods):
        arr = np.array([r[1:] for r in HOLDOUT[k]]).mean(axis=0)
        ax[1].bar(i, arr[1], color=col)
        ax[1].text(i, arr[1] + 2, f"{arr[1]:.0f}", ha="center", fontsize=8)
        ax[2].bar(i, arr[2] * 100, color=col)
        ax[2].text(i, arr[2] * 100 + 0.6, f"{arr[2] * 100:.1f}", ha="center", fontsize=8)
    for j, t in ((1, "(b) 잘림 (400문항 중)"), (2, "(c) 끝까지 답한 문항의 정확도 (%)")):
        ax[j].set_xticks(range(4))
        ax[j].set_xticklabels(["base", "GRPO\n(팀원)", "DPO\n(DV)", "DPO\n(DV+MX)"], fontsize=7)
        ax[j].set_title(t, fontsize=9)
    ax[2].set_ylim(50, 80)
    fig.suptitle("그림 7. MusiXQA 악보 홀드아웃 400문항 (학습에 쓰지 않음, 8,192토큰, 2 seed 평균, cu129)", y=1.04, fontsize=10)
    save_fig(fig, OUT, "fig7_holdout")

    # ----- 그림 8: 환경 강건성
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ys = [(f"seed {s}", c1, c2) for s, _, _, _, c1, c2 in rob]
    for i, (lab_, c1, c2) in enumerate(ys):
        ax[0].bar(i - 0.17, c1, width=0.32, color=C["dpo"], label="cu129" if i == 0 else None)
        ax[0].bar(i + 0.17, c2, width=0.32, color="#56B4E9", label="cu130" if i == 0 else None)
        ax[0].text(i - 0.17, c1 + 1, str(c1), ha="center", fontsize=8)
        ax[0].text(i + 0.17, c2 + 1, str(c2), ha="center", fontsize=8)
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels([y[0] for y in ys])
    ax[0].set_ylim(540, 600)
    ax[0].set_ylabel("정답 수 (900문항 중)")
    ax[0].set_title("(a) DPO(DeepVision)의 정답 수")
    ax[0].legend(fontsize=8)
    labs = ["129 vs 130\nseed 42", "129 vs 130\nseed 43", "seed 42 vs 43\nDPO (cu130)", "seed 42 vs 43\nbase (cu130)"]
    vals = [rob[0][1], rob[1][1], seedvar["dv1_re"], seedvar["base"]]
    cols = ["#0072B2", "#0072B2", "#999999", "#999999"]
    ax[1].bar(range(4), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(labs, fontsize=7)
    ax[1].set_ylim(80, 92)
    ax[1].set_ylabel("같은 정오로 채점된 문항 비율 (%)")
    ax[1].set_title("(b) 문항 단위 일치도 (파랑: 환경만 다름, 회색: seed만 다름)")
    fig.suptitle("그림 8. 환경(cu129와 cu130)을 바꿔도 점수는 같고, 문항 단위 차이는 seed 변동과 같은 수준", y=1.04, fontsize=10)
    save_fig(fig, OUT, "fig8_env_robustness")

    # ----- 그림 9: GRPO 학습 동역학 (우리 run1 vs 팀원)
    ours = []
    for line in open(OUT / "data" / "training_curve_grpo_run1.txt", encoding="utf-8"):      # GRPO run1 140스텝 학습 지표 (ssmr 브랜치 reports/ssmr_grpo_run1/training_curve.txt의 사본)
        p = line.split()
        if p and p[0].isdigit() and len(p) >= 8:
            ours.append([float(x) for x in p[:8]])
    ours = np.array(ours)
    st = json.load(open(W / "team_train" / "checkpoint-40" / "trainer_state.json"))["log_history"]
    team = np.array([[e["step"], e["rewards/accuracy_reward/mean"], e["completions/clipped_ratio"], e["completions/mean_terminated_length"], e["frac_reward_zero_std"]] for e in st])

    def smooth(y, w=5):
        k = np.ones(w) / w
        return np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), k, mode="valid")[:len(y)]
    fig, ax = plt.subplots(1, 4, figsize=(11.2, 2.9))
    spec = [(1, 1, "(a) 학습 보상(정답률)"), (2, 2, "(b) 잘린 응답 비율"), (3, 3, "(c) 끝낸 응답의 길이 (토큰)"), (4, 4, "(d) 신호 없는 그룹 비율")]
    for j, (oc, tc, title) in enumerate(spec):
        ax[j].plot(ours[:, 0], ours[:, oc], color=C["grpo"], alpha=0.25, lw=0.8)
        ax[j].plot(ours[:, 0], smooth(ours[:, oc]), color=C["grpo"], lw=1.6, label="우리 GRPO run1 (SSMR, 140스텝)")
        ax[j].plot(team[:, 0], team[:, tc], color=C["team"], alpha=0.25, lw=0.8)
        ax[j].plot(team[:, 0], smooth(team[:, tc]), color=C["team"], lw=1.6, label="팀원 GRPO (MusiXQA, 40스텝)")
        ax[j].set_title(title, fontsize=9)
        ax[j].set_xlabel("스텝")
    ax[0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("그림 9. GRPO 학습 중 지표 (두 실험 모두 응답이 짧아지고 학습 신호가 사라짐; 굵은 선은 5스텝 이동평균)", y=1.05, fontsize=10)
    save_fig(fig, OUT, "fig9_grpo_training_dynamics")

    # ----- 그림 10: RFT vs DPO 루프 (4과목 진단)
    rft2 = json.load(open(W / "rft_dv2_loop4" / "summary_runs.json"))["per_run"][0]["overall"]
    dpo4 = json.load(open(W / "dpo_dv1_loop4" / "summary_runs.json"))["per_run"][0]["overall"]
    loops = {"base": LOOP4_RECORD["base"], "RFT dv1\n(lr 1e-5)": LOOP4_RECORD["rft_dv1"], "RFT dv2\n(lr 3e-6)": rft2["truncated"] + rft2["errors"], "DPO dv1": dpo4["truncated"] + dpo4["errors"]}
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    cols = [C["base"], C["rft"], C["rft"], C["dpo"]]
    ax.bar(range(4), list(loops.values()), color=cols, width=0.6)
    for i, v in enumerate(loops.values()):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(loops.keys()), fontsize=8)
    ax.set_ylabel("루프(잘림+오류) 수, 120문항 중")
    ax.set_title("그림 10. 4과목 진단: RFT는 루프를 늘리고 DPO는 줄임", fontsize=9)
    save_fig(fig, OUT, "fig10_rft_vs_dpo_loops")

    # ----- 그림 11: DPO 데이터·학습 파이프라인 (도식)
    fig, ax = plt.subplots(figsize=(10.4, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")
    boxes = [(1, 14, 17, 12, "① 학습 풀\nDeepVision 수학 12,000\nMusiXQA 악보 3,200\n(MMMU와 중복 제거)", "#f2f2f2"),
             (22, 14, 18, 12, "② base로 문항당 8번 풀이\n최대 2,048토큰\nT 0.7, top_p 0.8, top_k 20", "#dbeafe"),
             (44, 14, 17, 12, "③ 통과율 1/8~7/8만 사용\n(0/8, 8/8은 신호 없음)", "#dbeafe"),
             (65, 14, 17, 12, "④ 쌍 만들기\n좋은 응답: 맞고 끝난 것\n나쁜 응답: 잘린 응답 우선,\n없으면 틀린 응답", "#d1fae5"),
             (86, 14, 13, 12, "⑤ DPO 학습\nLoRA r64, β 0.1\nlr 5e-6, 1에폭", "#fde68a")]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=c, ec="#555", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.4)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2] + 0.6
        x1 = boxes[i + 1][0] - 0.6
        ax.add_patch(FancyArrowPatch((x0, 20), (x1, 20), arrowstyle="-|>", mutation_scale=11, color="#444", lw=1.2))
    ax.text(50, 8.0, r"$\mathcal{L}=-\log\sigma\left(\beta\left[\log\frac{\pi(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\log\frac{\pi(y_l)}{\pi_{\mathrm{ref}}(y_l)}\right]\right)+0.1\cdot \mathrm{NLL}(y_w)$", ha="center", fontsize=10)
    ax.text(50, 2.2, "y_w = 좋은 응답, y_l = 나쁜 응답, π_ref = 어댑터를 끈 base. 응답 위치의 토큰 확률만 계산한다.", ha="center", fontsize=7.6, style="italic")
    ax.text(50, 33.0, "그림 11. 제출 모델(DPO)의 데이터 구성과 학습 파이프라인", ha="center", fontsize=10)
    save_fig(fig, OUT, "fig11_dpo_pipeline")

    # ================================================================= data/ (재사용 가능한 요약 CSV)
    with open(OUT / "data" / "per_subject_cu130.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{k}_acc" for k in order] + [f"{k}_minus_base_pp" for k in ("team", "dv1_re", "dvmx1")] + [f"{k}_ci_lo" for k in ("team", "dv1_re", "dvmx1")] + [f"{k}_ci_hi" for k in ("team", "dv1_re", "dvmx1")])
        for s in sorted(M130["base"]["sr"]["per_subject_mean_acc"]):
            w.writerow([s] + [round(M130[k]["sr"]["per_subject_mean_acc"][s] * 100, 2) for k in order] + [round(P[("base", k)][s][0], 2) for k in ("team", "dv1_re", "dvmx1")]
                       + [round(P[("base", k)][s][1], 2) for k in ("team", "dv1_re", "dvmx1")] + [round(P[("base", k)][s][2], 2) for k in ("team", "dv1_re", "dvmx1")])
    with open(OUT / "data" / "holdout_musixqa.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "overall", "truncated", "acc_among_finished", "Clef", "KeySignature", "MeasureCount", "PitchReading", "source"])
        for k, rs in HOLDOUT.items():
            for r in rs:
                w.writerow([k, *r, "실습실 로그 화면(2026-11-15~17), cu129 mmdl"])
    print("base(cu130) Avg@2:", round(avg2(M130["base"]), 2), "| 검증: base→dvmx1 전체", [round(x, 2) for x in P[("base", "dvmx1")]["전체"]], "Music", [round(x, 1) for x in P[("base", "dvmx1")]["Music"]])
    print("분해(dvmx1):", {g: round(D["dvmx1"][g]["delta"], 1) for g in D["dvmx1"]})
    print("완료")


if __name__ == "__main__":
    main()
