# ALFWorld + Llama-3-8B Few-shot Runner

ALFWorld 텍스트 환경에서 `meta-llama/Meta-Llama-3-8B-Instruct`로 평가를 수행하는 실행 코드입니다.
현재 정책은 **ReAct few-shot 기반 단일 액션 선택**으로 구성되어 있으며, subgoal/reflexion 로직은 제거된 상태입니다.

참고: https://github.com/alfworld/alfworld

## 1) 권장 환경

- Python `3.10` 또는 `3.11`
- CUDA GPU 권장 (8B 모델 추론)

## 2) 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3) 환경 변수

```bash
export ALFWORLD_DATA=/path/to/alfworld/data
export HF_TOKEN=hf_xxx
```

## 4) 실행

스모크 테스트:

```bash
python run_alfworld_llama3_zeroshot.py \
  --split eval_id \
  --episodes 2 \
  --max-steps 30
```

일반 실행:

```bash
python run_alfworld_llama3_zeroshot.py \
  --split eval_id \
  --episodes 20 \
  --max-steps 50 \
  --prompting-mode react
```

주요 인자:

- `--config`: 설정 파일 경로 (기본: `configs/alfworld_llama3_zeroshot.yaml`)
- `--split`: `eval_id` 또는 `eval_ood`
- `--episodes`: 평가 에피소드 수
- `--max-steps`: 에피소드당 최대 스텝
- `--model-id`: Hugging Face 모델 ID
- `--hf-token`: HF 토큰 (미지정 시 `HF_TOKEN` 환경변수 사용)
- `--load-in-4bit / --no-load-in-4bit`: 4bit 양자화 사용 여부
- `--temperature`, `--top-p`, `--max-new-tokens`
- `--history-window`: 프롬프트 state summary 생성 시 참고할 최근 trajectory 길이
- `--output-dir`: 결과 JSON 저장 디렉토리

## 5) 정책 개요

`model_client.py`의 `LlamaActionPolicy`는 아래 방식으로 동작합니다.

- 태스크 타입별 few-shot 예시 1개(`*_0` 원문)를 프롬프트에 포함
  - `pick_and_place_simple`
  - `pick_clean_then_place_in_recep`
  - `pick_heat_then_place_in_recep`
  - `pick_cool_then_place_in_recep`
  - `pick_two_obj_and_place`
  - `look_at_obj_in_light`
- 최근 trajectory는 긴 원문 대신 요약 형태로 주입
  - `Visited: ...`
  - `Holding: ...`
  - `Opened: ...`
- 모델 출력에서 `Thought`/`Action`을 파싱하고,
- 최종 액션은 `admissible_commands`에 맞춰 정규화 매칭합니다.

## 6) 결과 파일

실행 후 `outputs/*.json`에 저장됩니다.

상위 필드:

- `timestamp`
- `model_id`
- `split`
- `episodes`
- `max_steps`
- `prompting_mode`
- `success_rate`
- `avg_score`
- `avg_steps`
- `results` (episode별 상세)

episode별 필드:

- `episode`
- `success`
- `won`
- `score`
- `steps`

## 7) 파일 구조

- `run_alfworld_llama3_zeroshot.py`
  - CLI 엔트리포인트
  - 환경/정책 초기화, 전체 평가 실행, 결과 저장

- `env_runner.py`
  - ALFWorld 환경 생성(`build_env`)
  - 에피소드 루프(`run_episodes`)
  - 태스크 타입 추론 및 스텝별 action 실행/집계

- `model_client.py`
  - Llama 모델/토크나이저 로딩
  - few-shot 기반 프롬프트 구성
  - 모델 출력 파싱 및 admissible action 매칭

- `alfworld_utils.py`
  - 시드 고정, YAML 로딩, 경로 검증, split 매핑 유틸
