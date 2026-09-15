"""
hf_bench.py — HF 런타임 벤치마크 하네스.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import re
import resource
import subprocess
import time
import uuid
from typing import NamedTuple
from zoneinfo import ZoneInfo

import torch
from dotenv import load_dotenv
from loguru import logger
from schema import FIELDNAMES
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as runtime_version

load_dotenv()  # load .env file


class PromptSet(NamedTuple):
    """load_prompts()의 반환 타입. active_prompts는 이미 필터링이 끝난 상태."""

    system_prompt: str
    active_prompts: list[dict]


class GitState(NamedTuple):
    """get_git_state()의 반환 타입."""

    commit: str
    dirty: bool


def load_prompts(path: str) -> PromptSet:
    """prompts.json을 읽어 활성 프롬프트만 걸러 반환한다.

    Args:
        path (str): prompts.json 경로.

    Returns:
        PromptSet: system_prompt와 active=true인 프롬프트 목록.

    Raises:
        ValueError: 활성 프롬프트가 0개이거나, id가 중복될 때.

    Note:
        - 0개인 경우를 조용히 넘기지 않고 예외로 막는다 — 벤치마크가
          "성공적으로 끝났지만 CSV가 비어있는" 상황을 방지하기 위함.
        - id 중복 검사는 prompts.json의 freeze 정책(추가만 가능)이
          깨지지 않았는지 확인하는 안전장치.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    system_prompt = data["_meta"]["system_prompt"]["text"]

    active_prompts = [p for p in data["prompts"] if p.get("active")]

    if len(active_prompts) < 1:
        raise ValueError(f"{path}: no active prompts found")

    ids = [p["id"] for p in active_prompts]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"{path}: duplicate prompt ids found: {duplicates}")

    return PromptSet(system_prompt=system_prompt, active_prompts=active_prompts)


def load_model(model_id: str, dtype: str, device: str):
    """HF 모델과 토크나이저를 로드한다.

    Args:
        model_id (str): Hugging Face Hub 레포 식별자 (예: "Qwen/Qwen2.5-0.5B-Instruct").
        dtype (str): "fp16" 또는 "fp32". torch.dtype으로 내부 변환됨.
        device (str): "mps" | "cpu" | "cuda".

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]: 로드되어 eval 모드로
            전환된 모델과 토크나이저.

    Note:
        - MPS 백엔드가 일부 연산에서 fp32를 fp16으로 조용히 캐스팅할 수 있다.
          debug 로깅으로 실제 dtype이 요청값과 일치하는지 확인할 것.
    """

    DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32}
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE_MAP[dtype])
        .to(device)
        .eval()
    )

    logger.debug(f"model dtype: {next(model.parameters()).dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


HOST_LABEL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hostlabel.sh")


def get_host_label() -> str:
    """CSV의 host 컬럼에 쓸 익명 머신 라벨을 구한다.

    Returns:
        str: 공백·쉼표가 `_`로 치환된 단일 CSV 토큰 (예: "apple-m4-16gb").

    Note:
        - 해석 순서: BENCH_HOST 환경변수 → bench/hostlabel.sh 실행 결과 → "unknown-host".
        - socket.gethostname()을 쓰지 않는다. 실제 hostname에는 사용자 실명이
          들어갈 수 있고(예: "YooYeongBin-Macmini.local") 이 레포는 제출 시점에
          public으로 전환된다.
        - bench-memory.sh와 **같은 스크립트**를 호출한다. 라벨 생성 로직을 파이썬으로
          재구현하면 두 런타임의 host 값이 미묘하게 달라져 런타임 간 조인이 깨진다.
        - 실패 시 hostname으로 폴백하지 않는다. 익명화가 조용히 풀리는 것보다
          "unknown-host"가 낫다 — 경고는 로그로 남긴다.
    """
    label = os.getenv("BENCH_HOST", "").strip()

    if not label:
        try:
            label = subprocess.run(
                [HOST_LABEL_SCRIPT], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning(
                f"{HOST_LABEL_SCRIPT} 실행 실패 ({exc}) — host='unknown-host'로 기록한다. "
                "BENCH_HOST 환경변수로 직접 지정할 수 있다."
            )
            label = ""

    if not label:
        label = "unknown-host"

    return re.sub(r"[\s,]+", "_", label)


def get_git_state() -> GitState:
    """현재 커밋 해시와 dirty 여부를 조회한다.

    Returns:
        GitState: 커밋 SHA와 dirty 플래그.

    Note:
        - git 레포가 아니거나 git이 설치 안 된 환경에서도 예외 없이
          동작해야 한다 (commit="nogit", dirty=False로 대체).
        - `git diff --quiet`는 변경 없음=0, 변경 있음=1을 반환하는
          정상적인 두 상태이지 실패가 아니므로 check=False로 직접 분기한다.
        - 이 함수는 실행 전체에서 한 번만 호출한다 (루프 밖).
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "nogit"

    if commit == "nogit":
        dirty = False
    else:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            check=False,
        )
        dirty = result.returncode != 0

    return GitState(commit=commit, dirty=dirty)


def warmup(model, tokenizer, device: str = "mps") -> None:
    """측정 루프 진입 전 1회 실행되는 워밍업. 결과는 버린다.

    Args:
        model: load_model()이 반환한 모델.
        tokenizer: load_model()이 반환한 토크나이저.
        device (str): "mps"일 때만 synchronize()를 호출.

    Returns:
        None

    Note:
        - max_new_tokens는 1보다 크게 주어 prefill과 decode 커널을 모두 최소 한 번씩 태운다.
        - 세션당 1회로 충분한지(shape별 재컴파일이 무시할 수준인지)는 1-3 검증 실험으로 확인하였다.
    """
    logger.info("running warmup pass")

    messages = [
        {"role": "user", "content": "warmup"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model.generate(
        **inputs,
        max_new_tokens=4,
        min_new_tokens=4,
        do_sample=False,
    )
    if device == "mps":
        torch.mps.synchronize()

    logger.info("warmup complete !")


def measure_ttft(model, inputs, sync: bool = True) -> float:
    """TTFT(첫 토큰까지의 시간)를 max_new_tokens=1 런으로 근사한다.

    Args:
        model: 추론에 사용할 모델.
        inputs: 이미 토큰화되어 model.device로 옮겨진 입력.
                run_one_prompt에서 만든 것을 그대로 받으며,
                이 함수 안에 재토큰화하지 않는다.
        sync (bool): True면 generate() 앞뒤로 torch.mps.synchronize()를
            호출한다. 1-3 검증 실험에서 False로 대조군을 만든다.

    Returns:
        float: 소요 시간(초).
    """

    if sync:
        torch.mps.synchronize()

    start = time.perf_counter()
    model.generate(
        **inputs,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
    )
    if sync:
        torch.mps.synchronize()

    end = time.perf_counter()

    return end - start


def measure_generation(model, inputs, n_tokens: int, sync: bool = True):
    """min_new_tokens == max_new_tokens == n_tokens로 전체 생성을 실행한다.

    Args:
        model: 추론에 사용할 모델.
        inputs: 이미 토큰화되어 model.device로 옮겨진 입력.
                run_one_prompt에서 만든 것을 그대로 받으며,
                이 함수 안에서 재토큰화하지 않는다.
        n_tokens (int): 강제할 생성 토큰 수.
        sync (bool): True면 generate() 앞뒤로 torch.mps.synchronize()를 호출.

    Returns:
        tuple[float, int]: (소요 시간(초), 실측 생성 토큰 수).

    Note:
        - 반환하는 토큰 수가 n_tokens와 다르면 그 차이
          자체가 이상 신호이니 로그로 남겨둘 가치가 있다.
    """

    if sync:
        torch.mps.synchronize()

    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=n_tokens,
        min_new_tokens=n_tokens,
        do_sample=False,
    )
    if sync:
        torch.mps.synchronize()
    end = time.perf_counter()

    running_time = end - start

    gen_len = outputs.shape[1] - inputs["input_ids"].shape[1]
    # model.generate() -> return 2shape tensor // len() -> 1shape

    return running_time, gen_len


def measure_memory():
    """호출 시점의 메모리 스냅샷을 읽는다. 시간 측정이 아니다.

    Returns:
        tuple[float, float]: (peak_rss_gb, runtime_alloc_gb).

        - peak_rss_gb는 resource.getrusage().ru_maxrss 기반 누적 최고치.
        - runtime_alloc_gb는 torch.mps.driver_allocated_memory() 기반
            MPS 드라이버 총 할당량.

    Note:
        - 반드시 measure_generation()이 완전히 끝난 뒤(동기화 이후)
            호출한다.
        - 시간 측정 구간(start~end) 안에서 호출하면 안 된다 — 이 함수
            자체의 시스템 콜 비용이 decode_tok_s 계산용 시간에 섞인다.
        - peak_rss_gb는 프로세스 시작 이후 누적 최고치라 한번 오르면
            내려가지 않는다. "그 시점까지의 누적 최고 메모리"로 해석한다.
    """
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    runtime_alloc_gb = torch.mps.driver_allocated_memory() / (1024**3)

    return peak_rss_gb, runtime_alloc_gb


def run_one_prompt(
    model,
    tokenizer,
    prompt: dict,
    system_prompt: str,
    gen_tokens: int,
    n_ctx: int,
    sync: bool,
) -> dict:
    """프롬프트 하나를 측정해 CSV 한 행에 들어갈 dict를 만든다.

    Args:
        model: 추론에 사용할 모델.
        tokenizer: 추론에 사용할 토크나이저.
        prompt (dict): prompts.json의 프롬프트 항목 (id, text 등).
        system_prompt (str): 모든 프롬프트에 공통으로 적용할 시스템 프롬프트.
        gen_tokens (int): 강제할 생성 토큰 수 (CLI --gen-tokens).
        n_ctx (int): 토큰화 시 truncation 기준 최대 길이 (CLI --n-ctx).
        sync (bool): 하위 measure_* 함수에 그대로 전달.

    Returns:
        dict: prompt_id, prompt_tokens, gen_tokens, ttft_ms, decode_tok_s,
            - total_s, peak_rss_gb, runtime_alloc_gb 8개 키.
            - run_id/git_commit/host 등 런 전체 공통값은 여기서 채우지 않고 main()에서 합친다.

    Note:
        - decode_tok_s 계산의 분자는 (실측 gen_tokens - 1) — TTFT가 이미
          잰 첫 토큰을 이중으로 세지 않기 위함.
    """
    return_dict = {}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt["text"]},
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        chat_prompt, return_tensors="pt", truncation=True, max_length=n_ctx
    ).to(model.device)
    ttft = measure_ttft(model, inputs, sync=sync)
    total, real_gen_tokens = measure_generation(
        model=model, inputs=inputs, n_tokens=gen_tokens, sync=sync
    )
    decode_tok_s = (real_gen_tokens - 1) / (total - ttft)
    peak_rss_gb, runtime_alloc_gb = measure_memory()

    return_dict["prompt_id"] = prompt["id"]
    return_dict["prompt_tokens"] = inputs["input_ids"].shape[1]
    return_dict["gen_tokens"] = gen_tokens
    return_dict["ttft_ms"] = ttft * 1000
    return_dict["decode_tok_s"] = decode_tok_s
    return_dict["total_s"] = total
    return_dict["peak_rss_gb"] = peak_rss_gb
    return_dict["runtime_alloc_gb"] = runtime_alloc_gb

    return return_dict


def build_notes(user_notes: str) -> str:
    """notes 컬럼 값을 만든다.

    Args:
        user_notes (str): CLI --notes로 받은 문자열. `;`로 구분된 key=value 토큰.

    Returns:
        str: 정규화된 notes 문자열.

    Note:
        - 규약은 `;`로 구분된 토큰이며 **쉼표·개행 금지**다. CSV 쿼팅 없이
          안전하게 append하기 위한 제약으로, bench-memory.sh도 같은 규약을 쓴다.
        - 런타임에 의해 결정되는 사실(ttft_ms의 측정 방식 등)은 여기 넣지 않는다.
          `runtime` 컬럼으로 이미 판별되므로 중복이다.
    """
    tokens = [t.strip() for t in user_notes.split(";") if t.strip()]
    return ";".join(re.sub(r"[,\r\n]+", "_", t) for t in tokens)


def write_row(row: dict, csv_path: str, fieldnames: list[str] = FIELDNAMES) -> None:
    """완성된 한 행을 CSV에 append한다.

    Args:
        row (dict): fieldnames와 동일한 키 집합을 가진 완전한 행.
        csv_path (str): 출력 CSV 경로.
        fieldnames (list[str]): 컬럼 순서. schema.py의 FIELDNAMES를 기본으로 쓴다.

    Returns:
        None

    Raises:
        ValueError: row의 키 개수가 fieldnames와 다를 때.

    Note:
        - 헤더 중복 방지는 "파일 존재 여부"가 아니라 "파일이 존재하고
          내용이 0바이트보다 큰지"로 판단한다.
    """
    if len(row) != len(fieldnames):
        raise ValueError("Row data has not been filled out properly.")

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prompts-path", default="bench/prompts.json")
    parser.add_argument("--output", default="results/bench.csv")
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument(
        "--notes",
        default="",
        help="notes 컬럼에 넣을 태그. ';'로 구분된 key=value 형식, 쉼표 금지 "
        "(예: 'quiet_env;batch=2')",
    )
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")

    return parser


def main() -> None:
    """Note:
    - 호출 순서: load_prompts → load_model → get_git_state →
        (선택) warmup → active_prompts 순회하며 run_one_prompt + write_row.
    - run_id와 timestamp는 행마다 새로 생성한다
    - 진행 상황 로그는 프롬프트 하나 처리 후 즉시 남긴다
    """

    args = build_arg_parser().parse_args()

    timezone_setting = os.getenv("TIMEZONE", "UTC")
    try:
        tz = ZoneInfo(timezone_setting)
    except Exception:
        tz = datetime.UTC

    system_prompt, active_prompts = load_prompts(args.prompts_path)
    model, tokenizer = load_model(args.model_id, args.dtype, args.device)

    session_id = str(uuid.uuid4())
    cond = "no_sync" if args.no_sync else "no_warmup" if args.no_warmup else "baseline" # no_sync / no_warmup 모두 선택은 불가능함
    git_commit, dirty_flag = get_git_state()
    host = get_host_label()
    user_os = platform.system()
    notes = build_notes(args.notes)
    sync = not (args.no_sync)

    logger.info(
        f"start: model={args.model_id} dtype={args.dtype} device={args.device} sync={sync}"
    )
    if not args.no_warmup:
        warmup(model, tokenizer, args.device)

    for i, active_prompt in enumerate(active_prompts, start=1):
        row = run_one_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=active_prompt,
            system_prompt=system_prompt,
            gen_tokens=args.gen_tokens,
            n_ctx=args.n_ctx,
            sync=sync,
        )
        row["run_id"] = uuid.uuid4()
        row['session_id'] = session_id
        row["timestamp"] = datetime.datetime.now(tz).isoformat()
        row["git_commit"] = git_commit
        row["dirty_flag"] = dirty_flag
        row["cond"] = cond
        row["runtime"] = "hf"
        row["runtime_version"] = runtime_version
        row["model_id"] = args.model_id
        row["quant"] = None
        row["dtype"] = args.dtype
        row["device"] = args.device
        row["n_ctx"] = args.n_ctx
        row["host"] = host
        row["os"] = user_os
        row["notes"] = notes

        write_row(row, args.output, FIELDNAMES)
        logger.info(
            f"[{i}/{len(active_prompts)}] {row['prompt_id']:<8} "
            f"ttft={row['ttft_ms']:>7.1f}ms  "
            f"decode={row['decode_tok_s']:>6.1f}tok/s  "
            f"total={row['total_s']:>5.2f}s  "
            f"mem={row['peak_rss_gb']:>5.2f}GB"
        )

    logger.info(f"done: {len(active_prompts)} rows written to {args.output}")


if __name__ == "__main__":
    main()
