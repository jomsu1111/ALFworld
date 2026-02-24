# ALFWorld + Llama-3-8B Runner

이 프로젝트는 ALFWorld 텍스트 환경에서 `meta-llama/Meta-Llama-3-8B-Instruct`를 사용해 평가를 수행하는 실행 코드입니다.
기본 실행은 ReAct 프롬프팅과 Reflexion 메모리(옵션)를 지원합니다.

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

## 4) 실행 예시

스모크 테스트:

```bash
python run_alfworld_llama3_zeroshot.py \
  --split eval_id \
  --episodes 2 \
  --max-steps 30
```

ReAct + Reflexion 본 실행:

```bash
python run_alfworld_llama3_zeroshot.py \
  --split eval_id \
  --episodes 20 \
  --max-steps 50 \
  --prompting-mode react \
  --use-reflexion \
  --reflection-window 4
```

## 5) 결과

실행 후 `outputs/*.json`에 저장됩니다.

- `success_rate`
- `avg_score`
- `avg_steps`
- 에피소드별 `task_type/success/score/steps/reflections`

## 6) Python 파일 역할

- `run_alfworld_llama3_zeroshot.py`
  - CLI 엔트리포인트입니다.
  - 실험 인자를 파싱하고(config/split/steps/prompting mode/reflexion 등), 환경과 정책을 생성한 뒤 전체 에피소드를 실행합니다.
  - 최종 요약 JSON을 `outputs/`에 저장합니다.

- `model_client.py`
  - Llama 모델/토크나이저 로딩과 캐시를 담당합니다.
  - action 선택 정책(`LlamaActionPolicy`)이 들어있습니다.
  - ReAct 형식 프롬프트, 6개 ALFWorld 태스크별 few-shot 예시, PDDL 스타일 행동 가이드, Reflexion 메모리 주입을 구성합니다.
  - 모델 출력에서 admissible command 하나로 정규화/매칭합니다.

- `env_runner.py`
  - ALFWorld 환경 생성(`build_env`)과 에피소드 루프(`run_episodes`)를 담당합니다.
  - 매 스텝에서 admissible action을 가져와 정책에 질의하고, `env.step`을 수행합니다.
  - `infos`/observation 기반으로 태스크 타입을 추론하고, 실패/정체 신호를 감지해 reflection 메모리를 누적합니다.
  - 에피소드 결과를 집계해 상위 실행기로 반환합니다.

- `alfworld_utils.py`
  - 공통 유틸 모음입니다.
  - 시드 고정, YAML 설정 로딩, 데이터 경로 검증, split 이름 매핑을 처리합니다.
  - Python 3.13 환경에서 textworld 호환 패치(`apply_textworld_py313_compat`)를 제공합니다.
