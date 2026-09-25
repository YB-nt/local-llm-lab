"""
check_model_drift.py - bench/models.yaml에 핀된 revision/digest가 아직 유효한지 확인한다.

읽기 전용이다. models.yaml을 절대 쓰지 않는다. 드리프트는 "재검증할 시점"이라는
신호일 뿐 "데이터가 무효"라는 뜻이 아니다 - 자동 수정하지 않고 보고만 한다
(notes/model-identity-spec.md §4).

Usage:
    uv run python scripts/check_model_drift.py
    uv run python scripts/check_model_drift.py --model-key qwen2.5-0.5b-instruct

Exit code: 드리프트 건수 (CI/cron 게이트로 쓸 수 있다). 네트워크 실패는 드리프트로
세지 않는다 - [SKIP]으로 보고하고 건너뛴다.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_PATH = REPO_ROOT / "bench" / "models.yaml"
DEFAULT_HOST = "http://localhost:11434"


def load_models(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if k != "schema_version"}


def http_get_json(url: str, timeout: int) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "local-llm-lab-drift-check"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def check_hf_sha(hf_repo: str) -> str | None:
    data = http_get_json(f"https://huggingface.co/api/models/{hf_repo}", timeout=10)
    return data.get("sha")


def check_ollama_tags(host: str) -> dict[str, str]:
    data = http_get_json(f"{host}/api/tags", timeout=5)
    return {m["name"]: m.get("digest") for m in data.get("models", []) if m.get("name")}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--model-key")
    p.add_argument("--models-path", default=str(DEFAULT_MODELS_PATH))
    p.add_argument("--host", default=DEFAULT_HOST)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    models_path = Path(args.models_path)
    if not models_path.exists():
        print(f"ERROR: not found: {models_path}", file=sys.stderr)
        return 1

    models = load_models(models_path)
    if args.model_key:
        if args.model_key not in models:
            print(f"ERROR: model-key '{args.model_key}'가 {models_path}에 없습니다.", file=sys.stderr)
            return 1
        models = {args.model_key: models[args.model_key]}

    ollama_tags: dict[str, str] | None
    try:
        ollama_tags = check_ollama_tags(args.host)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[SKIP] Ollama 접속 실패 ({args.host}): {e}", file=sys.stderr)
        ollama_tags = None

    drift_count = 0
    hf_sha_cache: dict[str, str | None] = {}

    for model_key, entry in models.items():
        source = entry.get("source") or {}
        hf_repo = source.get("hf_repo")
        recorded_source_rev = source.get("hf_revision")

        live_sha: str | None = None
        if hf_repo:
            if hf_repo not in hf_sha_cache:
                try:
                    hf_sha_cache[hf_repo] = check_hf_sha(hf_repo)
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    print(f"[SKIP] {model_key}: HF API 접근 실패 ({hf_repo}): {e}", file=sys.stderr)
                    hf_sha_cache[hf_repo] = None
                except json.JSONDecodeError as e:
                    print(f"[SKIP] {model_key}: HF API 응답 파싱 실패 ({hf_repo}): {e}", file=sys.stderr)
                    hf_sha_cache[hf_repo] = None
            live_sha = hf_sha_cache[hf_repo]

        if live_sha and recorded_source_rev and live_sha != recorded_source_rev:
            drift_count += 1
            print(
                f"[DRIFT] {model_key}: source.hf_revision recorded={recorded_source_rev} "
                f"live={live_sha}"
            )

        for variant_name, variant in (entry.get("variants") or {}).items():
            runtimes = variant.get("runtimes") or {}

            hf_rt = runtimes.get("hf")
            if hf_rt and hf_rt.get("revision") and live_sha and hf_rt["revision"] != live_sha:
                drift_count += 1
                print(
                    f"[DRIFT] {model_key}/{variant_name} (hf): "
                    f"recorded={hf_rt['revision']} live={live_sha}"
                )

            ollama_rt = runtimes.get("ollama")
            if ollama_rt and ollama_rt.get("digest"):
                tag = ollama_rt.get("model_id")
                if ollama_tags is None:
                    continue
                live_digest = ollama_tags.get(tag)
                if live_digest is None:
                    print(
                        f"[WARN] {model_key}/{variant_name} (ollama): "
                        f"태그 '{tag}'가 로컬에 없음 (pull 안 됨 또는 삭제됨)"
                    )
                elif live_digest != ollama_rt["digest"]:
                    drift_count += 1
                    print(
                        f"[DRIFT] {model_key}/{variant_name} (ollama): "
                        f"recorded={ollama_rt['digest']} live={live_digest}"
                    )

    if drift_count > 0:
        print(f"\n{drift_count} drift(s). 재검증: uv run python scripts/capture_model_identity.py ...")

    return drift_count


if __name__ == "__main__":
    sys.exit(main())
