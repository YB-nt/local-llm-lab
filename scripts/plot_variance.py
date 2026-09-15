"""
img/assets/*.png 생성 스크립트

    uv run python scripts/plot_variance.py \
        --csv results/bench.csv --model-id Qwen/Qwen2.5-0.5B-Instruct --dtype fp16 \
        --latest-commit --since 2026-09-13T19:00:00 \
        --metric decode_tok_s --mode summary \
        --output img/assets/latest_summary.png
    uv run python scripts/plot_variance.py \
        --csv results/bench.csv --model-id Qwen/Qwen2.5-0.5B-Instruct --dtype fp16 \
        --latest-commit --since 2026-09-13T19:00:00 \
        --metric decode_tok_s --mode sessions \
        --output img/assets/latest_sessions.png
        
    uv run python scripts/plot_variance.py \
        --csv results/bench.csv --model-id Qwen/Qwen2.5-0.5B-Instruct --dtype fp16 \
        --latest-commit --since 2026-09-13T19:00:00 \
        --metric decode_tok_s --mode sessions \
        --cond no_warmup \
        --output img/assets/latest_no_warmup_sessions.png
    
    uv run python scripts/plot_variance.py \
        --csv results/bench.csv --model-id Qwen/Qwen2.5-0.5B-Instruct --dtype fp16 \
        --cond no_warmup --group-by notes --mode summary \
        --metric decode_tok_s \
        --latest-commit --since 2026-09-13T20:14:00
        --output img/assets/quiet_env_comparison.png
        
    uv run python scripts/plot_variance.py \
        --csv results/bench.csv --model-id Qwen/Qwen2.5-0.5B-Instruct --dtype fp16 \
        --cond no_warmup --group-by notes --mode sessions \
        --latest-commit --since 2026-09-13T20:14:00 \
        --metric decode_tok_s \
        --output img/assets/quiet_env_comparison.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROMPT_ORDER = ["p16", "p64", "p256", "p1024"]

COND_COLORS = {
    "baseline": "#888888",
    "no_sync": "#4a90d9",
    "no_warmup": "#e8933b",
}
FALLBACK_COLORS = ["#7b4fa3", "#5aa469", "#c65170", "#3d8f8f", "#d9a441", "#4fa3a3"]


def load_filtered_df(
    csv_path: str, model_id: str, dtype: str, since: str | None, latest_commit: bool,
    notes_contains: str | None = None, notes_excludes: str | None = None,
) -> pd.DataFrame:
    """model_id/dtype로 거른 뒤, since/latest_commit/notes 필터를 순서대로 적용한다."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.loc[(df["model_id"] == model_id) & (df["dtype"] == dtype)].copy()
    df["notes"] = df["notes"].fillna("")

    if df.empty:
        return df

    if since is not None:
        cutoff = pd.Timestamp(since)
        if cutoff.tzinfo is None and df["timestamp"].dt.tz is not None:
            cutoff = cutoff.tz_localize(df["timestamp"].dt.tz)
        before = len(df)
        df = df.loc[df["timestamp"] >= cutoff]
        print(f"[info] --since {since} 적용: {before} → {len(df)}행")

    if latest_commit and not df.empty:
        latest_row = df.sort_values("timestamp").iloc[-1]
        commit = latest_row["git_commit"]
        before = len(df)
        df = df.loc[df["git_commit"] == commit]
        print(f"[info] --latest-commit ({commit}) 적용: {before} → {len(df)}행")

    if notes_contains:
        before = len(df)
        df = df.loc[df["notes"].str.contains(notes_contains, na=False)]
        print(f"[info] --notes-contains '{notes_contains}' 적용: {before} → {len(df)}행")

    if notes_excludes:
        before = len(df)
        df = df.loc[~df["notes"].str.contains(notes_excludes, na=False)]
        print(f"[info] --notes-excludes '{notes_excludes}' 적용: {before} → {len(df)}행")

    return df


# 런타임 간 '구성'이 달라 같은 축에 올리면 안 되는 지표.
# 값은 (차단할지, 사유). 근거는 docs/bench-schema.md의 해당 컬럼 항목.
CROSS_RUNTIME_UNSAFE: dict[str, tuple[bool, str]] = {
    "ttft_ms": (
        True,
        (
            "HF는 max_new_tokens=1 별도 호출의 wall-clock(prefill + 디코딩 1스텝 + 프레임워크 "
            "오버헤드)이고 Ollama는 prompt_eval_duration(prefill만)이다. 같은 축에 올리면 "
            "정의 차이를 엔진 성능 차이로 오독한다 — TTFT는 런타임 내부의 프롬프트 길이 "
            "스케일링만 본다."
        ),
    ),
    "peak_rss_gb": (
        False,
        (
            "HF는 ru_maxrss(프로세스 시작 이후 누적 최대), Ollama는 생성 구간 샘플링 "
            "최댓값이다. 절대값 직접 비교는 피하고 runtime_alloc_gb와 함께 읽을 것."
        ),
    ),
}


def assert_metric_comparable(df: pd.DataFrame, metric: str) -> None:
    """여러 런타임이 섞인 데이터로 비교 불가 지표를 그리는 것을 막는다.

    Args:
        df (pd.DataFrame): 필터가 모두 적용된 데이터프레임.
        metric (str): 그릴 지표 컬럼명.

    Returns:
        None

    Raises:
        ValueError: metric이 런타임 간 비교 불가이고 df에 런타임이 2개 이상일 때.

    Note:
        - 데이터가 한 런타임뿐이면 아무 것도 하지 않는다. HF 단독 데이터에는 영향 없음.
        - 차단(ValueError) 대신 경고만 하는 지표도 있다 — CROSS_RUNTIME_UNSAFE의 첫 번째 값.
        - 이 가드가 필요한 이유: --metric 기본값이 ttft_ms이고 Ollama 행도 cond=baseline
          이라, 가드가 없으면 기본 호출에서 두 런타임이 조용히 한 그래프에 섞인다.
    """
    if metric not in CROSS_RUNTIME_UNSAFE:
        return

    runtimes = sorted(df["runtime"].dropna().unique().tolist())
    if len(runtimes) < 2:
        return

    block, reason = CROSS_RUNTIME_UNSAFE[metric]
    message = f"{metric}: 서로 다른 런타임 {runtimes}이 한 그래프에 섞였습니다.\n  {reason}"

    if block:
        raise ValueError(
            f"{message}\n  --runtime {runtimes[0]} 처럼 하나만 지정해서 다시 실행하세요."
        )
    print(f"[warn] {message}")


def discover_groups(df: pd.DataFrame, group_col: str) -> list[str]:
    """group_col 기준으로 존재하는 그룹 값을 전부 찾는다. notes 빈 문자열은 '(no tag)'로 표시."""
    if group_col == "notes":
        labels = df["notes"].replace("", "(no tag)")
        return sorted(labels.unique().tolist())
    return sorted(df[group_col].dropna().unique().tolist())


def load_sessions(df: pd.DataFrame, group_col: str, group_value: str, metric: str) -> dict[str, list[float]]:
    """그룹에 맞는 행만 골라 {session_id: [prompt_order 순 metric값]} 형태로 묶는다."""
    if group_col == "notes":
        target = "" if group_value == "(no tag)" else group_value
        subset = df.loc[df["notes"] == target]
    else:
        subset = df.loc[df[group_col] == group_value]

    if subset.empty:
        print(f"[warn] 조건에 맞는 행이 없습니다: {group_col}={group_value}")
        return {}

    sessions: dict[str, list[float]] = {}
    for session_id, group in subset.groupby("session_id"):
        ordered = group.set_index("prompt_id").reindex(PROMPT_ORDER)[metric]
        if ordered.isna().any():
            missing = ordered[ordered.isna()].index.tolist()
            print(f"[skip] session={session_id}: 누락된 prompt_id {missing}")
            continue
        sessions[str(session_id)] = ordered.tolist()

    return sessions


def color_for(group_value: str, idx: int) -> str:
    if group_value in COND_COLORS:
        return COND_COLORS[group_value]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def color_for(cond: str, idx: int) -> str:
    if cond in COND_COLORS:
        return COND_COLORS[cond]
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def plot_sessions_mode(
    sessions_by_cond: dict[str, dict[str, list[float]]],
    metric: str, highlight: list[str], output_path: str, title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    for idx, (cond, sessions) in enumerate(sessions_by_cond.items()):
        base_color = color_for(cond, idx)
        for session_id, values in sessions.items():
            is_highlighted = any(session_id.startswith(h) for h in highlight)
            if is_highlighted:
                ax.plot(PROMPT_ORDER, values, marker="o", linewidth=2.6, color="#d9534f",
                        zorder=6, label=f"{cond} / {session_id[:8]} (highlighted)")
            else:
                ax.plot(PROMPT_ORDER, values, marker="o", linewidth=1.2, alpha=0.6,
                        color=base_color, zorder=3, label=f"{cond} / {session_id[:8]}")

    _finish(ax, fig, metric, title, output_path)


def plot_summary_mode(
    sessions_by_cond: dict[str, dict[str, list[float]]],
    metric: str, output_path: str, title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    for idx, (cond, sessions) in enumerate(sessions_by_cond.items()):
        if not sessions:
            continue
        color = color_for(cond, idx)
        matrix = np.array(list(sessions.values()))  # shape: (n_sessions, n_prompts)
        mean = matrix.mean(axis=0)
        lo, hi = matrix.min(axis=0), matrix.max(axis=0)

        ax.plot(PROMPT_ORDER, mean, marker="o", linewidth=2.2, color=color,
                label=f"{cond} (n={len(sessions)}, mean)")
        ax.fill_between(PROMPT_ORDER, lo, hi, color=color, alpha=0.15)

    _finish(ax, fig, metric, title, output_path)


def _finish(ax, fig, metric: str, title: str, output_path: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("prompt")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"saved: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/bench.csv")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument(
        "--cond", default="all",
        help='콤마로 구분된 조건 목록, 예: baseline,no_warmup. "all"(기본값)이면 '
             "필터링된 데이터에 존재하는 모든 조건을 포함 (사전 필터로만 사용됨)",
    )
    parser.add_argument(
        "--runtime", default="all",
        help='콤마로 구분된 런타임 목록, 예: hf 또는 hf,ollama. "all"(기본값)이면 필터 없음. '
             "ttft_ms는 런타임 간 정의가 달라 두 개 이상 섞이면 오류로 막는다",
    )
    parser.add_argument(
        "--group-by", default="cond", choices=["cond", "notes"],
        help="선/평균선을 무엇 기준으로 나눌지. cond(기본): baseline/no_sync/no_warmup별로 색 구분. "
             "notes: notes 태그별로 색 구분 (예: quiet_env 배치 vs 기존 데이터를 같은 cond 안에서 비교)",
    )
    parser.add_argument(
        "--metric", default="ttft_ms",
        choices=["ttft_ms", "decode_tok_s", "total_s", "peak_rss_gb"],
    )
    parser.add_argument(
        "--mode", default="sessions", choices=["sessions", "summary"],
        help="sessions: 세션 개별 선 (이상치 탐색용) / summary: 그룹별 평균+범위 밴드 (개요용)",
    )
    parser.add_argument(
        "--highlight", default="",
        help="sessions 모드에서 굵은 빨간 선으로 강조할 session_id 접두사, 콤마로 여러 개",
    )
    parser.add_argument(
        "--since", default=None,
        help="이 시각(ISO8601, 예: 2026-09-13T19:00:00) 이후의 데이터만 사용",
    )
    parser.add_argument(
        "--latest-commit", action="store_true",
        help="필터링된 데이터 중 가장 최근 timestamp의 git_commit만 사용 "
             "(코드 수정 전/후 데이터가 섞이는 것을 방지)",
    )
    parser.add_argument(
        "--notes-contains", default=None,
        help="notes 컬럼에 이 부분 문자열이 포함된 행만 남김 (예: quiet_env)",
    )
    parser.add_argument(
        "--notes-excludes", default=None,
        help="notes 컬럼에 이 부분 문자열이 포함된 행은 제외",
    )
    parser.add_argument("--output", default="docs/assets/comparison.png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    df = load_filtered_df(
        args.csv, args.model_id, args.dtype, args.since, args.latest_commit,
        args.notes_contains, args.notes_excludes,
    )
    if df.empty:
        raise ValueError(
            f"필터 후 남은 행이 없습니다 (model_id={args.model_id}, dtype={args.dtype}, "
            f"since={args.since}, latest_commit={args.latest_commit}, "
            f"notes_contains={args.notes_contains}, notes_excludes={args.notes_excludes})."
        )

    # --runtime도 --cond와 같은 성격의 사전 필터
    if args.runtime != "all":
        runtime_list = [r.strip() for r in args.runtime.split(",") if r.strip()]
        before = len(df)
        df = df.loc[df["runtime"].isin(runtime_list)]
        print(f"[info] --runtime {runtime_list} 적용: {before} → {len(df)}행")
        if df.empty:
            raise ValueError(f"runtime={runtime_list} 필터 후 남은 행이 없습니다.")

    # --cond는 group-by 값과 무관하게 항상 사전 필터로 적용
    if args.cond != "all":
        cond_list = [c.strip() for c in args.cond.split(",") if c.strip()]
        before = len(df)
        df = df.loc[df["cond"].isin(cond_list)]
        print(f"[info] --cond {cond_list} 적용: {before} → {len(df)}행")
        if df.empty:
            raise ValueError(f"cond={cond_list} 필터 후 남은 행이 없습니다.")

    # 모든 필터가 끝난 뒤에 검사한다 — --runtime으로 이미 하나만 남았으면 통과해야 하므로
    assert_metric_comparable(df, args.metric)

    groups = discover_groups(df, args.group_by)
    print(f"[info] --group-by {args.group_by} 기준 발견된 그룹: {groups}")

    highlight = [h.strip() for h in args.highlight.split(",") if h.strip()]

    sessions_by_group = {g: load_sessions(df, args.group_by, g, args.metric) for g in groups}

    # 런타임을 제목에 넣어 PNG만 봐도 어느 런타임 데이터인지 알 수 있게 한다
    runtimes = sorted(df["runtime"].dropna().unique().tolist())
    runtime_label = "+".join(runtimes) if runtimes else "?"
    title = (
        f"{args.metric} — {' vs '.join(groups)} "
        f"({runtime_label}, {args.model_id}, {args.dtype})"
    )

    if args.mode == "summary":
        plot_summary_mode(sessions_by_group, args.metric, args.output, title)
    else:
        plot_sessions_mode(sessions_by_group, args.metric, highlight, args.output, title)


if __name__ == "__main__":
    main()