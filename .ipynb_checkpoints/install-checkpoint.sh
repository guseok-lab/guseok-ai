#!/usr/bin/env bash
# GUSEOK AI 분석 서버 - 설치 스크립트
# 사용법: bash install.sh
set -e

echo "[1/2] 기본 패키지 설치..."
pip install -r requirements.txt

echo "[2/2] Re-ID(torchreid) GitHub 소스 빌드..."
if [ ! -d "deep-person-reid" ]; then
    git clone https://github.com/KaiyangZhou/deep-person-reid.git
fi
cd deep-person-reid
pip install --no-build-isolation -e .
cd ..

echo "설치 확인..."
python3 -c "import torchreid; print('torchreid OK')"
python3 -c "import fastapi, ultralytics, cv2, torch; print('나머지 패키지 OK')"

echo ""
echo "설치 완료. 실행:"
echo "  export GOOGLE_API_KEY=...   # 제미나이 키"
echo "  export SPRING_SERVER_URL=http://백엔드주소:8080"
echo "  uvicorn app:app --host 0.0.0.0 --port 8000"