import argparse
import json
import sys

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT = "Hello, how are you?"

PROMPT_PATH = "bench/prompts.json"


def setup_logger(verbose: bool = False) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>",
        level="DEBUG" if verbose else "INFO",
        colorize=True,
    )


def run_decoding_experiment(
    model,
    tokenizer,
    prompt: str,
    *,
    system_prompt: str,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 128,
    use_cache: bool = True,
    seed: int | None = None,
) -> str:
    """디코딩 파라미터 하나를 적용해 텍스트를 생성하고 결과를 반환한다."""
    if seed is not None:
        torch.manual_seed(seed)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # instruct 모델의 경우 chat_template 형식이 필요함
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=use_cache,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    formatted_outputs = outputs[0][input_length:]
    return tokenizer.decode(formatted_outputs, skip_special_tokens=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HF decoding parameter experiment")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logger(args.verbose)

    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    system_prompt = data["_meta"]["system_prompt"]["text"]

    logger.info(f"model={args.model} dtype={args.dtype} device={args.device}")

    if args.device == "mps" and args.dtype == "bfloat16":
        logger.warning(
            "MPS + bfloat16은 커널 폴백 가능. 속도 이상 시 float16으로 재측정"
        )

    # 토크나이저를 먼저 로드한다. 모델명이 틀렸을 때 몇 초 만에 실패한다.
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(args.device)

    logger.info(f"loaded: device={next(model.parameters()).device} dtype={model.dtype}")
    logger.debug(f"generation_config: {model.generation_config}")

    for i in range(args.repeat):
        outputs = run_decoding_experiment(
            model,
            tokenizer,
            args.prompt,
            system_prompt=system_prompt,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            use_cache=not args.no_cache,
            seed=args.seed,
        )

        logger.info(f"run {i + 1}/{args.repeat}")
        print(outputs)
        print()


if __name__ == "__main__":
    main()
