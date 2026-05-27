#!/usr/bin/env bash
# =============================================================================
#  GUSEOK(구석) 실종자 수색 AI 서버 — 배포 부트스트랩 스크립트
#  대상: 네이버 클라우드 L4 GPU 서버 (gp1l4-g3), Ubuntu 24.04
#  사용법:
#     1) 아래 [사용자 설정] 변수만 본인 값으로 수정
#     2) chmod +x setup_guseok_server.sh
#     3) ./setup_guseok_server.sh
# =============================================================================
set -eo pipefail   # -u 는 conda activate 와 충돌해서 일부러 뺌

# ─────────────────────────────────────────────────────────────────────────────
#  [사용자 설정] — 여기만 본인 값으로 바꾸세요
# ─────────────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/guseok-lab/guseok-ai"   # ← 본인 GitHub 레포 주소
REPO_BRANCH="main"                                     # ← 브랜치명
# 설치 기준 경로: NCP 등 일반 VM 은 "$HOME", RunPod 은 영구디스크 "/workspace" 권장
INSTALL_BASE="/workspace"                               # ← RunPod: /workspace / NCP: $HOME
PROJECT_DIR="$INSTALL_BASE/guseok"                     # 코드 받을 위치
CONDA_ENV="guseok"                                     # conda 환경 이름
PY_VER="3.10"                                          # 파이썬 버전
TORCH_CUDA="cu124"   # 이미지가 CUDA 12.4.1 → cu124 / 11.8 → cu118 / 12.1~12.3 → cu121
APP_PORT="8000"                                        # FastAPI(uvicorn) 포트
APP_MODULE="app:app"                                   # app.py 안의 FastAPI 인스턴스명

# 환경변수 (백엔드/LLM) — 실제 값으로 채우세요
SPRING_SERVER_URL=""                                   # ← Spring 백엔드 주소
GOOGLE_API_KEY=""                                      # ← Gemini 폴백용 키 (없으면 비워둬도 됨)

# ─────────────────────────────────────────────────────────────────────────────
C_G='\033[1;32m'; C_Y='\033[1;33m'; C_R='\033[1;31m'; C_0='\033[0m'
step(){ echo -e "\n${C_G}== $* ==${C_0}"; }
warn(){ echo -e "${C_Y}[!] $*${C_0}"; }

# root 계정이면 sudo 불필요 (RunPod 등 컨테이너 환경 대응)
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# ─────────────────────────────────────────────────────────────────────────────
#  1. GPU / 드라이버 확인 (NCP 가 미리 설치해서 줌 — 우리는 확인만)
# ─────────────────────────────────────────────────────────────────────────────
step "1. GPU 드라이버 확인 (nvidia-smi)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo -e "${C_R}nvidia-smi 가 없습니다. NCP GPU 서버가 맞는지, 드라이버가 설치됐는지 확인하세요.${C_0}"
  exit 1
fi
nvidia-smi
echo
warn "위에 표시된 'CUDA Version' 이 ${TORCH_CUDA} 와 안 맞으면, 스크립트 상단 TORCH_CUDA 를 고치고 다시 실행하세요."
echo "  (CUDA 12.1~12.3 → cu121 / 12.4 이상 → cu124 / 11.8 → cu118)"
sleep 2

# ─────────────────────────────────────────────────────────────────────────────
#  2. 시스템 패키지
# ─────────────────────────────────────────────────────────────────────────────
step "2. 시스템 패키지 설치"
$SUDO apt-get update -y
$SUDO apt-get install -y \
  git wget curl build-essential \
  ffmpeg libgl1 libglib2.0-0 libsm6 libxext6   # opencv / 영상 디코딩 의존성

# ─────────────────────────────────────────────────────────────────────────────
#  3. Miniconda 설치 (없을 때만)
# ─────────────────────────────────────────────────────────────────────────────
step "3. Miniconda 확인/설치"
MINICONDA="$INSTALL_BASE/miniconda3"
if [ ! -d "$MINICONDA" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINICONDA"
  rm -f /tmp/miniconda.sh
else
  echo "이미 설치됨: $MINICONDA"
fi
# shellcheck disable=SC1091
source "$MINICONDA/etc/profile.d/conda.sh"

# ─────────────────────────────────────────────────────────────────────────────
#  4. conda 환경 생성/활성화
# ─────────────────────────────────────────────────────────────────────────────
step "4. conda 환경 ($CONDA_ENV / python $PY_VER)"
if ! conda env list | grep -qE "^${CONDA_ENV}\s"; then
  conda create -y -n "$CONDA_ENV" python="$PY_VER"
else
  echo "이미 존재: $CONDA_ENV"
fi
conda activate "$CONDA_ENV"
python -m pip install --upgrade pip

# ─────────────────────────────────────────────────────────────────────────────
#  5. PyTorch (CUDA 맞춤 wheel) — 가장 먼저, 명시적으로
# ─────────────────────────────────────────────────────────────────────────────
step "5. PyTorch 설치 ($TORCH_CUDA)"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# reid/opencv 스택 호환을 위해 numpy 1.x 고정
pip install "numpy<2"

# ─────────────────────────────────────────────────────────────────────────────
#  6. 프로젝트 코드 clone (또는 갱신)
# ─────────────────────────────────────────────────────────────────────────────
step "6. 프로젝트 코드 받기"
if [ ! -d "$PROJECT_DIR/.git" ]; then
  git clone -b "$REPO_BRANCH" "$REPO_URL" "$PROJECT_DIR"
else
  echo "이미 clone 됨 → git pull"
  git -C "$PROJECT_DIR" pull
fi
cd "$PROJECT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
#  7. 프로젝트 의존성 (requirements.txt 에서 torch 계열은 제외하고 설치)
#      → 위에서 깐 CUDA torch 가 CPU 버전으로 덮어써지는 걸 방지
# ─────────────────────────────────────────────────────────────────────────────
step "7. requirements.txt 설치 (torch 제외)"
if [ -f requirements.txt ]; then
  grep -ivE '^(torch|torchvision|torchaudio)([=<>!~[:space:]]|$)' requirements.txt > /tmp/req_notorch.txt || cp requirements.txt /tmp/req_notorch.txt
  pip install -r /tmp/req_notorch.txt
else
  warn "requirements.txt 가 없습니다. 레포 구조 확인 필요."
fi

# ─────────────────────────────────────────────────────────────────────────────
#  8. torchreid GitHub 소스 빌드
# ─────────────────────────────────────────────────────────────────────────────
step "8. torchreid 소스 빌드"
REID_DIR="$INSTALL_BASE/deep-person-reid"
if [ ! -d "$REID_DIR/.git" ]; then
  git clone https://github.com/KaiyangZhou/deep-person-reid.git "$REID_DIR"
fi
cd "$REID_DIR"
pip install Cython          # 확장 빌드용 (혹시 빠졌을 때 대비)
pip install -r requirements.txt
# 최신 setuptools 에선 setup.py develop 이 막힐 수 있어 -e 우선
pip install -e . || python setup.py develop

# ─────────────────────────────────────────────────────────────────────────────
#  9. 환경변수 파일 작성 (.env) + 셸 자동 로딩
# ─────────────────────────────────────────────────────────────────────────────
step "9. 환경변수 설정"
ENV_FILE="$PROJECT_DIR/.env"
cat > "$ENV_FILE" <<EOF
SPRING_SERVER_URL=${SPRING_SERVER_URL}
GOOGLE_API_KEY=${GOOGLE_API_KEY}
EOF
echo "작성됨: $ENV_FILE"

# ─────────────────────────────────────────────────────────────────────────────
#  10. 설치 검증
# ─────────────────────────────────────────────────────────────────────────────
step "10. 설치 검증"
cd "$PROJECT_DIR"
python - <<'PY'
import importlib, sys
def chk(name, attr=None):
    try:
        m = importlib.import_module(name)
        print(f"  [OK] {name} {getattr(m,'__version__','')}")
        return m
    except Exception as e:
        print(f"  [FAIL] {name}: {e}"); return None
import torch
print("  [OK] torch", torch.__version__)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
else:
    print("  [경고] CUDA 인식 실패 — TORCH_CUDA 버전 확인 필요")
chk("torchvision"); chk("ultralytics"); chk("torchreid")
chk("cv2"); chk("scipy"); chk("requests")
try:
    from google import genai; print("  [OK] google.genai")
except Exception as e:
    print("  [info] google.genai 미설치/미설정:", e)
PY

# ─────────────────────────────────────────────────────────────────────────────
#  완료 안내
# ─────────────────────────────────────────────────────────────────────────────
cat <<EOF

${C_G}========================================================${C_0}
 설치 완료!

 [수동 실행 — 테스트용]
   source ${MINICONDA}/etc/profile.d/conda.sh
   conda activate ${CONDA_ENV}
   cd ${PROJECT_DIR}
   set -a && source .env && set +a
   uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${APP_PORT}

 [백그라운드 상시 실행]
   - NCP(일반 VM): 아래 11번 systemd 블록 주석 해제 후 사용
   - RunPod 등 컨테이너: systemd 없음 → tmux 사용
       tmux new -s guseok
       conda activate ${CONDA_ENV} && cd ${PROJECT_DIR}
       set -a && source .env && set +a
       uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${APP_PORT}
       (Ctrl+B 누르고 D 로 빠져나오면 백그라운드 유지)

 [포트 열기]
   - NCP: ACG 인바운드에 TCP ${APP_PORT} (백엔드 IP) 추가
   - RunPod: Pod 생성 시 HTTP Port 에 ${APP_PORT} 노출
             → 백엔드는 https://{POD_ID}-${APP_PORT}.proxy.runpod.net 로 호출
${C_G}========================================================${C_0}
EOF

# ─────────────────────────────────────────────────────────────────────────────
#  11. (선택) systemd 서비스 등록 — 로그아웃/재부팅에도 서버 유지
#      필요하면 아래 주석(#)을 풀고 다시 실행하거나, 명령만 따로 실행하세요.
# ─────────────────────────────────────────────────────────────────────────────
# sudo tee /etc/systemd/system/guseok.service >/dev/null <<EOF
# [Unit]
# Description=GUSEOK Missing Person Search API
# After=network.target
#
# [Service]
# User=$USER
# WorkingDirectory=${PROJECT_DIR}
# EnvironmentFile=${PROJECT_DIR}/.env
# ExecStart=${MINICONDA}/envs/${CONDA_ENV}/bin/uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${APP_PORT}
# Restart=on-failure
# RestartSec=5
#
# [Install]
# WantedBy=multi-user.target
# EOF
# sudo systemctl daemon-reload
# sudo systemctl enable --now guseok
# sudo systemctl status guseok --no-pager
