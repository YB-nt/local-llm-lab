# local-llm-lab

로컬 LLM(Ollama) 실험 및 벤치마크 작업 공간.

## 구조

```
local-llm-lab/
├── scripts/      실행 스크립트
│   └── bench-memory.sh          모델 로드 전/후 메모리 + 생성 처리량 기록
├── notes/        해설·문서
│   └── bench-memory.explained.md
└── benchmarks/   벤치 결과 (스크립트가 자동 생성/append)
    ├── bench-results.csv
    └── bench-memory.log
```

## 사용

```bash
# 기본 (3회 생성, 5m keep-alive)
./scripts/bench-memory.sh llama3.1:8b-instruct-q4_K_M

# 반복 횟수·컨텍스트·keep-alive 조정
./scripts/bench-memory.sh qwen2.5:14b -n 5 -c 8192 -k 10m

# 도움말
./scripts/bench-memory.sh -h
```

결과는 실행 위치와 무관하게 항상 `benchmarks/`에 쌓입니다. 자세한 동작은
[`notes/bench-memory.explained.md`](notes/bench-memory.explained.md) 참고.

- 플랫폼: macOS
- 의존성: `bash`, `curl`, `jq`, 실행 중인 Ollama 서버(`ollama serve` / Ollama.app)
