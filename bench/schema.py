"""
bench.csv 컬럼 정의.
 
이 파일은 docs/bench-schema.md(사람이 읽는 컬럼정의서)의 코드 쪽 대응물이다.
컬럼을 추가/변경/삭제할 때는 이 파일과 bench-schema.md를 함께 수정한다 —
한쪽만 고치면 문서와 코드가 어긋난다.
"""
 
SCHEMA_VERSION = "1.0.1"  # bench-schema.md의 버전과 맞출 것
 
# 컬럼정의서 1절 순서 그대로. 순서를 바꾸면 CSV 헤더 순서도 바뀌므로,
# 이미 데이터가 쌓인 bench.csv가 있다면 순서 변경 시 기존 파일과의 호환성을 확인할 것.
FIELDNAMES: list[str] = [
    # 실행 식별
    "run_id",
    "timestamp",
    "git_commit",
    "dirty_flag",
    # 실행 환경
    "runtime",
    "runtime_version",
    "model_id",
    "quant",
    "dtype",
    "device",
    "n_ctx",
    # 프롬프트 / 토큰
    "prompt_id",
    "prompt_tokens",
    "gen_tokens",
    # 성능 지표
    "ttft_ms",
    "decode_tok_s",
    "total_s",
    # 메모리
    "peak_rss_gb",
    "runtime_alloc_gb",
    # 메타
    "host",
    "os",
    "notes",
]
 