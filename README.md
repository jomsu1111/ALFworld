<<<<<<< HEAD
# AFLworld
Llama-3-8b with ALFworld
=======
# ALFWorld + Llama-3-8B Zero-shot Baseline

이 프로젝트는 ALFWorld 텍스트 환경에서 `meta-llama/Meta-Llama-3-8B-Instruct`를 사용해 zero-shot baseline을 측정하기 위한 최소 실행 코드입니다.

공식 ALFWorld 코드의 환경 인터페이스(`get_environment`, `reset`, `step`, `admissible_commands`)를 그대로 따릅니다.
참고: https://github.com/alfworld/alfworld

## 1) 권장 환경

- Python `3.10` 또는 `3.11` 권장
- CUDA GPU 권장 (8B 모델 추론)

## 2) 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3) ALFWorld 데이터 준비

```bash
alfworld-data
```

데이터 설치 후 `ALFWORLD_DATA` 환경변수가 설정되어야 합니다.

```bash
export ALFWORLD_DATA=/path/to/alfworld/data
```

## 4) Llama-3 접근 토큰 설정

Hugging Face에서 `Meta-Llama-3-8B-Instruct` 접근 권한을 받은 토큰이 필요합니다.

```bash
export HF_TOKEN=hf_xxx
```

## 5) Zero-shot 실행

스모크 테스트(1~2개 에피소드)로 먼저 동작 확인:

```bash
python run_alfworld_llama3_zeroshot.py \
  --split eval_id \
  --episodes 2 \
  --max-steps 30 \
  --load-in-4bit
```

정식 baseline(예: 20 episodes):

ID split (`valid_seen`) 평가:

```bash
python run_alfworld_llama3_zeroshot.py \
  --config configs/alfworld_llama3_zeroshot.yaml \
  --split eval_id \
  --episodes 20 \
  --max-steps 50 \
  --load-in-4bit
```

OOD split (`valid_unseen`) 평가:

```bash
python run_alfworld_llama3_zeroshot.py \
  --config configs/alfworld_llama3_zeroshot.yaml \
  --split eval_ood \
  --episodes 20 \
  --max-steps 50 \
  --load-in-4bit
```

## 6) 결과

실행이 끝나면 `outputs/*.json`에 아래 요약이 저장됩니다.

- `success_rate`
- `avg_score`
- `avg_steps`
- 에피소드별 `success/score/steps`

## 7) 메모

- 모델 출력이 action 후보와 정확히 일치하지 않아도, 스크립트에서 후보 리스트 기반으로 정규화/매칭합니다.
- 메모리가 부족하면 `--load-in-4bit`를 유지하고 `--episodes`를 줄여 먼저 동작 검증하세요.

## 8) 코드 구조

- `run_alfworld_llama3_zeroshot.py`: 실행 진입점(인자 파싱, 요약 저장)
- `alfworld_utils.py`: 시드/설정 로딩/경로 검증/파이썬 3.13 호환 패치
- `model_client.py`: Llama 정책 및 모델 로드 캐시(`get_llama_action_policy`)
- `env_runner.py`: ALFWorld 환경 초기화 및 에피소드 실행 루프
>>>>>>> 7f3a161 (alfworld llama3 runner + colab notebook)
