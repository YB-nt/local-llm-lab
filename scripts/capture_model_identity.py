"""
capture_model_identity.py - bench/models.yaml에 채울 증거(evidence)를 생성한다.

목적: model_id 문자열만으로는 HF와 Ollama가 같은 가중치를 돌렸는지 판정할 수 없다
(notes/model-identity-spec.md §0). 이 스크립트는 그 판정에 쓸 증거를 기계적으로
수집해 bench/models.yaml에 병합한다. relation/note/derived_from은 사람이 정하는
값이라 절대 건드리지 않는다 - 기계가 쓰는 건 evidence 블록과 pin 필드
(source.hf_revision, runtimes.hf.revision, runtimes.ollama.digest/quant/dtype)뿐이다.

Usage:
    uv run python scripts/capture_model_identity.py \
        --model-key qwen2.5-0.5b-instruct --variant fp16 \
        --hf-repo Qwen/Qwen2.5-0.5B-Instruct --hf-dtype fp16 \
        --ollama-model qwen2.5:0.5b-instruct-fp16

    # 재검증 (이미 기록된 hf_repo/model_id를 그대로 쓴다 - CLI 인자 불필요)
    uv run python scripts/capture_model_identity.py \
        --model-key qwen2.5-0.5b-instruct --variant fp16
"""

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_PATH = REPO_ROOT / "bench" / "models.yaml"
DEFAULT_PROMPTS_PATH = REPO_ROOT / "bench" / "prompts.json"
DEFAULT_HOST = "http://localhost:11434"

# spec §3-2 매핑표는 추정이었다 - "왼쪽 HF 필드는 실측으로 확정된 값(24/896/14/2/4864/1e6),
# 오른쪽 Ollama 필드명은 추측"이라는 전제 자체를 검증 없이 하드코딩하면 안 된다는 지적을
# 받고, 이름이 아니라 **값**으로 매핑을 역산해 확정했다. 방법:
#
#   qwen2.5:0.5b-instruct-fp16의 /api/show model_info에서 general.architecture="qwen2"
#   접두사를 가진 숫자 필드 8개를 전부 나열한 뒤, HF config.json의 확정값과 일치하는
#   키를 값으로 검색했다 (스크립트: 이 파일 git 이력의 커밋 메시지 참고, 또는 아래 재현 절차).
#
#     기대값(HF config.json)         -> 값이 일치하는 model_info 키 (유일하게 1개씩 매칭됨)
#     num_hidden_layers    = 24      -> qwen2.block_count
#     hidden_size          = 896     -> qwen2.embedding_length
#     num_attention_heads  = 14      -> qwen2.attention.head_count
#     num_key_value_heads  = 2       -> qwen2.attention.head_count_kv
#     intermediate_size    = 4864    -> qwen2.feed_forward_length
#     rope_theta           = 1000000 -> qwen2.rope.freq_base
#
#   qwen2.* 접두사를 가진 숫자 필드는 이 6개 외에 context_length(32768)와
#   layer_norm_rms_epsilon(1e-06)뿐이라 값 충돌(같은 값을 가진 후보가 2개 이상)이 없었다 -
#   즉 이름이 아니라 값으로 확정한 매핑이고, 우연히 spec 표와 결과가 같았을 뿐이다.
#
#   재현: uv run python -c "
#     import json, urllib.request
#     req = urllib.request.Request('http://localhost:11434/api/show',
#         data=json.dumps({'model': 'qwen2.5:0.5b-instruct-fp16'}).encode(),
#         headers={'Content-Type': 'application/json'}, method='POST')
#     mi = json.load(urllib.request.urlopen(req))['model_info']
#     arch = mi['general.architecture']
#     for k, v in mi.items():
#         if k.startswith(arch + '.') and isinstance(v, (int, float)): print(k, v)"
#
# 이 표는 qwen2 아키텍처에서만 값으로 확정된 것이다 - 다른 아키텍처를 캡처할 때는
# general.architecture가 바뀌므로, 값 매칭이 여기서도 유일하게 성립하는지 같은 방법으로
# 먼저 재확인할 것. 런타임에 매번 값으로 재검색하지 않고 이름 매핑을 고정해서 쓰는 이유는:
# 값으로만 매칭하면 "이름은 맞는데 값이 실제로 어긋난" 진짜 드리프트(예: block_count가
# 24가 아니라 23으로 나오는 corrupted 케이스)를 값이 안 나온다는 이유로 match:false가
# 아니라 unavailable로 오분류하게 되어, 이 도구의 목적(불일치 검출)과 충돌한다.
ARCH_FIELD_MAP = [
    ("num_hidden_layers", "block_count"),
    ("hidden_size", "embedding_length"),
    ("num_attention_heads", "attention.head_count"),
    ("num_key_value_heads", "attention.head_count_kv"),
    ("intermediate_size", "feed_forward_length"),
    ("rope_theta", "rope.freq_base"),
]


# ------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": "1.0.0"}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {"schema_version": "1.0.0"}


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)


def git_short_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def load_prompts(path: Path) -> tuple[str, dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    system_prompt = data["_meta"]["system_prompt"]["text"]
    prompts = {p["id"]: p["text"] for p in data["prompts"]}
    return system_prompt, prompts


def get_prompt_text(prompts: dict[str, str], prompt_id: str) -> str:
    if prompt_id not in prompts:
        raise SystemExit(f"ERROR: prompts.json에 '{prompt_id}'가 없습니다.")
    return prompts[prompt_id]


def http_get_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "local-llm-lab"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def http_post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_hf_config(hf_repo: str, sha: str) -> dict:
    """§3-1: API의 config 필드는 축약본이라 부족하다. sha에 핀한 원본 config.json을 받는다."""
    url = f"https://huggingface.co/{hf_repo}/resolve/{sha}/config.json"
    try:
        return http_get_json(url)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SystemExit(f"ERROR: config.json fetch 실패 ({url}): {e}") from e


def ollama_show(host: str, model: str) -> dict:
    try:
        return http_post_json(f"{host}/api/show", {"model": model})
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"ERROR: /api/show 실패 (model={model}): {e}. "
            f"로컬에 pull 되어 있는지 확인하세요 (ollama pull {model})."
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SystemExit(f"ERROR: Ollama에 연결할 수 없습니다 ({host}): {e}") from e


def ollama_tag_digest(host: str, model: str) -> str | None:
    tags = http_get_json(f"{host}/api/tags")
    for m in tags.get("models", []):
        if m.get("name") == model:
            return m.get("digest")
    return None


def ollama_generate(host: str, body: dict) -> dict:
    resp = http_post_json(f"{host}/api/generate", body)
    if not resp.get("done"):
        raise SystemExit(f"ERROR: /api/generate 실패: {resp.get('error', resp)}")
    return resp


# ------------------------------------------------------------------------
# Ollama quant/dtype 정규화 - bench-memory.sh의 로직과 동일 (spec §2-1, docs/bench-schema.md §2)
# ------------------------------------------------------------------------
def normalize_quant_dtype(quantization_level: str) -> tuple[str, str]:
    """Returns (dtype, quant) - 정확히 하나만 채워진다."""
    ql = quantization_level
    if ql == "F16":
        return "fp16", ""
    if ql == "F32":
        return "fp32", ""
    if ql == "Q8_0":
        return "", "q8_0"
    if ql == "Q4_0":
        return "", "q4_0"
    # 그 외 (Q4_K_M 등): 선두 Q만 소문자화
    quant = "q" + ql[1:] if ql.startswith("Q") else ql.lower()
    return "", quant


# ------------------------------------------------------------------------
# 증거 수집
# ------------------------------------------------------------------------
def compare_architecture(hf_config: dict, model_info: dict) -> dict:
    arch = model_info.get("general.architecture")
    if not arch:
        return {
            "match": False,
            "compared": {},
            "unavailable_fields": ["general.architecture"],
        }
    compared = {}
    unavailable = []
    for hf_key, ollama_suffix in ARCH_FIELD_MAP:
        ollama_key = f"{arch}.{ollama_suffix}"
        hf_val = hf_config.get(hf_key)
        ol_val = model_info.get(ollama_key)
        if hf_val is None or ol_val is None:
            unavailable.append(hf_key)
            continue
        if isinstance(hf_val, (int, float)) and isinstance(ol_val, (int, float)):
            match = float(hf_val) == float(ol_val)
        else:
            match = hf_val == ol_val
        compared[hf_key] = {
            "hf": hf_val,
            "ollama": ol_val,
            "ollama_field": ollama_key,
            "match": match,
        }
    all_match = all(c["match"] for c in compared.values())
    return {
        "match": all_match and not unavailable,
        "compared": compared,
        "unavailable_fields": unavailable,
    }


def compare_tokenization(
    tokenizer, system_prompt: str, prompt_text: str, prompt_id: str,
    host: str, ollama_model: str, num_ctx: int,
) -> dict:
    """spec §3-3: p1024에서 805==805가 이미 확인된 값 - 재현 안 되면 그 자체가 신호."""
    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    hf_count = len(tokenizer(chat)["input_ids"])

    body = {
        "model": ollama_model,
        "prompt": prompt_text,
        "system": system_prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {"num_ctx": num_ctx, "num_predict": 1},
    }
    resp = ollama_generate(host, body)
    ollama_count = resp.get("prompt_eval_count", 0)

    return {
        "match": hf_count == ollama_count,
        "prompt_id": prompt_id,
        "hf": hf_count,
        "ollama": ollama_count,
    }


def compare_greedy_prefix(
    hf_model, hf_tokenizer, device: str, system_prompt: str, prompt_text: str,
    prompt_id: str, host: str, ollama_model: str, num_ctx: int, n: int,
) -> dict:
    """spec §3-4: 완전 일치를 요구하지 않는다 - 일치한 접두사 길이를 기록한다.

    ⚠ Ollama 기본 샘플링 파라미터는 HF의 순수 그리디와 다르다 (특히 repeat_penalty).
    options에 temperature/top_k/top_p/repeat_penalty/seed를 명시해야 갈라지지 않는다.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]
    chat = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = hf_tokenizer(chat, return_tensors="pt").to(device)

    import torch

    outputs = hf_model.generate(
        **inputs, max_new_tokens=n, min_new_tokens=n, do_sample=False
    )
    if device == "mps":
        torch.mps.synchronize()
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    hf_text = hf_tokenizer.decode(gen_ids, skip_special_tokens=True)

    body = {
        "model": ollama_model,
        "prompt": prompt_text,
        "system": system_prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_ctx": num_ctx,
            "num_predict": n,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "repeat_penalty": 1.0,
            "seed": 0,
        },
    }
    resp = ollama_generate(host, body)
    ollama_text = resp.get("response", "")

    matched_prefix_chars = 0
    for a, b in zip(hf_text, ollama_text):
        if a != b:
            break
        matched_prefix_chars += 1

    return {
        "applicable": True,
        "prompt_id": prompt_id,
        "n": n,
        "match": hf_text == ollama_text,
        "matched_prefix_chars": matched_prefix_chars,
        "hf_text": hf_text,
        "ollama_text": ollama_text,
    }


# ------------------------------------------------------------------------
# models.yaml 병합 (§3-5 쓰기 규칙)
# ------------------------------------------------------------------------
def ensure_source(entry: dict, hf_repo_cli: str | None, hf_revision_cli: str | None, api: HfApi) -> tuple[dict, str]:
    source = entry.setdefault("source", {})
    if "hf_repo" in source:
        if hf_repo_cli and hf_repo_cli != source["hf_repo"]:
            raise SystemExit(
                f"ERROR: source.hf_repo가 이미 '{source['hf_repo']}'로 고정되어 있습니다 "
                f"(요청: '{hf_repo_cli}'). hf_repo 변경은 identity 문제이므로 사람이 결정합니다."
            )
        hf_repo = source["hf_repo"]
    else:
        if not hf_repo_cli:
            raise SystemExit(
                "ERROR: 새 model-key입니다. --hf-repo가 필요합니다 (예: Qwen/Qwen2.5-0.5B-Instruct)."
            )
        hf_repo = hf_repo_cli
        source["hf_repo"] = hf_repo

    info = api.model_info(hf_repo, revision=hf_revision_cli) if hf_revision_cli else api.model_info(hf_repo)
    sha = info.sha
    if not sha:
        raise SystemExit(f"ERROR: HfApi().model_info('{hf_repo}')가 sha를 반환하지 않았습니다.")
    source["hf_revision"] = sha  # pin 필드 - 매 실행 갱신
    if info.safetensors and info.safetensors.parameters:
        dtypes = sorted(info.safetensors.parameters.keys())
        source["native_dtype"] = "+".join(d.lower() for d in dtypes)
        source["parameters"] = info.safetensors.total
    return source, sha


def ensure_hf_runtime(runtimes: dict, hf_repo: str, hf_dtype_cli: str | None, sha: str) -> dict | None:
    if hf_dtype_cli:
        hf_rt = runtimes.get("hf")
        if hf_rt is None:
            hf_rt = {"model_id": hf_repo, "dtype": hf_dtype_cli}
            runtimes["hf"] = hf_rt
        else:
            if hf_rt.get("model_id") != hf_repo:
                raise SystemExit(
                    f"ERROR: 기존 runtimes.hf.model_id='{hf_rt.get('model_id')}' vs '{hf_repo}'. "
                    "다른 repo면 새 variant를 만드세요."
                )
            if hf_rt.get("dtype") != hf_dtype_cli:
                raise SystemExit(
                    f"ERROR: 기존 runtimes.hf.dtype='{hf_rt.get('dtype')}' vs 요청 '{hf_dtype_cli}'. "
                    "dtype이 다르면 새 variant를 만드세요."
                )
        hf_rt["revision"] = sha  # pin 필드 - 매 실행 갱신
        return hf_rt
    if "hf" in runtimes:
        runtimes["hf"]["revision"] = sha
        return runtimes["hf"]
    return None


def ensure_ollama_runtime(runtimes: dict, ollama_model_cli: str | None, host: str) -> dict:
    ol_rt = runtimes.get("ollama")
    if ollama_model_cli:
        if ol_rt is None:
            ol_rt = {"model_id": ollama_model_cli}
            runtimes["ollama"] = ol_rt
        elif ol_rt.get("model_id") != ollama_model_cli:
            raise SystemExit(
                f"ERROR: 기존 runtimes.ollama.model_id='{ol_rt.get('model_id')}' vs 요청 "
                f"'{ollama_model_cli}'. 다른 태그면 새 variant를 만드세요."
            )
    elif ol_rt is None:
        raise SystemExit("ERROR: --ollama-model이 필요합니다 (아직 기록된 값이 없음).")

    model_id = ol_rt["model_id"]
    show = ollama_show(host, model_id)
    ql = (show.get("details") or {}).get("quantization_level", "")
    if ql:
        dtype, quant = normalize_quant_dtype(ql)
        if dtype:
            ol_rt["dtype"] = dtype
            ol_rt.pop("quant", None)
        else:
            ol_rt["quant"] = quant
            ol_rt.pop("dtype", None)
    digest = ollama_tag_digest(host, model_id)
    if digest:
        ol_rt["digest"] = digest  # pin 필드 - 매 실행 갱신
    return ol_rt


# ------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--model-key", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--models-path", default=str(DEFAULT_MODELS_PATH))
    p.add_argument("--prompts-path", default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--hf-repo", help="새 model-key를 만들 때만 필요")
    p.add_argument("--hf-revision", help="생략하면 현재 HEAD sha로 해석")
    p.add_argument("--hf-dtype", choices=["fp16", "fp32"], help="이 variant에 HF 실측을 연결할 때")
    p.add_argument("--ollama-model", help="이 variant의 Ollama 태그 (처음 만들 때 필요)")
    p.add_argument("--device", default="mps")
    p.add_argument("--num-ctx", type=int, default=2048)
    p.add_argument("--tokenization-prompt-id", default="p1024")
    p.add_argument("--greedy-prompt-id", default="p16")
    p.add_argument("--greedy-n", type=int, default=16)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    models_path = Path(args.models_path)
    prompts_path = Path(args.prompts_path)

    data = load_yaml(models_path)
    entry = data.setdefault(args.model_key, {})
    api = HfApi()
    source, sha = ensure_source(entry, args.hf_repo, args.hf_revision, api)

    variants = entry.setdefault("variants", {})
    is_new_variant = args.variant not in variants
    v = variants.setdefault(args.variant, {})
    if is_new_variant:
        v["relation"] = None
        print(
            f"[WARN] '{args.model_key}/{args.variant}'는 새 variant입니다. "
            f"relation을 사람이 {models_path}에 직접 채워야 합니다 (same_weights | derived_from).",
            file=sys.stderr,
        )

    runtimes = v.setdefault("runtimes", {})
    hf_rt = ensure_hf_runtime(runtimes, source["hf_repo"], args.hf_dtype, sha)
    ollama_rt = ensure_ollama_runtime(runtimes, args.ollama_model, args.host)
    ollama_model_id = ollama_rt["model_id"]

    system_prompt, prompts = load_prompts(prompts_path)

    print(f"[1/3] architecture: HF config.json@{sha[:12]} vs {ollama_model_id} model_info")
    hf_config = fetch_hf_config(source["hf_repo"], sha)
    show = ollama_show(args.host, ollama_model_id)
    model_info = show.get("model_info") or {}
    architecture_evidence = compare_architecture(hf_config, model_info)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source["hf_repo"], revision=sha)

    print(f"[2/3] tokenization: prompt_id={args.tokenization_prompt_id}")
    tok_prompt_text = get_prompt_text(prompts, args.tokenization_prompt_id)
    tokenization_evidence = compare_tokenization(
        tokenizer, system_prompt, tok_prompt_text, args.tokenization_prompt_id,
        args.host, ollama_model_id, args.num_ctx,
    )

    relation = v.get("relation")
    if relation != "same_weights":
        if relation == "derived_from":
            reason = "양자화 쌍은 출력이 갈라지는 것이 정상"
        elif relation is None:
            reason = "relation 미지정 - 사람이 models.yaml에 relation을 채운 뒤 재실행하세요"
        else:
            reason = f"relation='{relation}' - same_weights 아님"
        print(f"[3/3] greedy_prefix: 건너뜀 ({reason})")
        greedy_evidence = {"applicable": False, "reason": reason}
    elif hf_rt is None:
        reason = "runtimes.hf 정보 없음 (--hf-dtype 필요)"
        print(f"[3/3] greedy_prefix: 건너뜀 ({reason})")
        greedy_evidence = {"applicable": False, "reason": reason}
    else:
        print(f"[3/3] greedy_prefix: prompt_id={args.greedy_prompt_id} n={args.greedy_n} (HF 모델 로드 중...)")
        import torch
        from transformers import AutoModelForCausalLM

        dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
        hf_model = (
            AutoModelForCausalLM.from_pretrained(
                source["hf_repo"], revision=sha, torch_dtype=dtype_map[hf_rt["dtype"]]
            )
            .to(args.device)
            .eval()
        )
        greedy_prompt_text = get_prompt_text(prompts, args.greedy_prompt_id)
        greedy_evidence = compare_greedy_prefix(
            hf_model, tokenizer, args.device, system_prompt, greedy_prompt_text,
            args.greedy_prompt_id, args.host, ollama_model_id, args.num_ctx, args.greedy_n,
        )
        del hf_model

    v["evidence"] = {
        "verified_at": datetime.now().astimezone().isoformat(),
        "captured_by": f"scripts/capture_model_identity.py@{git_short_sha()}",
        "architecture": architecture_evidence,
        "tokenization": tokenization_evidence,
        "greedy_prefix": greedy_evidence,
    }

    save_yaml(models_path, data)
    print(f"\nOK: {args.model_key}/{args.variant} evidence -> {models_path}")
    print(f"  architecture : match={architecture_evidence['match']}")
    print(f"  tokenization : match={tokenization_evidence['match']} "
          f"(hf={tokenization_evidence['hf']} ollama={tokenization_evidence['ollama']})")
    if greedy_evidence.get("applicable"):
        print(f"  greedy_prefix: match={greedy_evidence['match']} "
              f"matched_prefix_chars={greedy_evidence['matched_prefix_chars']}")
    else:
        print(f"  greedy_prefix: applicable=false ({greedy_evidence['reason']})")


if __name__ == "__main__":
    main()
