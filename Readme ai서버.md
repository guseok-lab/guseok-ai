# GUSEOK(구석) AI 분석 서버 - 배포 안내

## 파일 구성
- `app.py`        : FastAPI 서버 (백엔드가 호출하는 진입점)
- `analyzer.py`   : 분석 코어 (FEM + Re-ID + 겹침/중복 필터)
- `utils.py`      : 모델 로드, 백엔드 통신, 다운로드/업로드
- `custom_tracker.yaml` : ByteTrack ID 안정화 설정
- `models/`       : YOLO/FEM 가중치 (frontal, top, fem)

## 설치

### 1) 기본 패키지
```bash
pip install fastapi uvicorn ultralytics torch torchvision \
            opencv-python scipy pillow requests google-genai
```

### 2) Re-ID 라이브러리 (OSNet / torchreid) — GitHub에서 직접 빌드
torchreid 는 pip 일반 설치가 아니라 소스에서 빌드해야 합니다.
```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install tensorboard cython
pip install --no-build-isolation -e .
cd ..
```
설치 확인:
```bash
python3 -c "import torchreid; print('torchreid OK')"
```

## 환경변수 (코드에 키 하드코딩 금지!)
```bash
export GOOGLE_API_KEY="실제_제미나이_키"
export SPRING_SERVER_URL="http://백엔드주소:8080"
```

## 실행
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 동작
1. 백엔드 -> `POST http://<AI서버>:8000/analyze`  body: `{"searchId": 1}`
2. AI 서버가 `GET /api/v1/searches/{id}`로 타겟 정보 조회
3. 영상/사진 다운로드 -> 분석 -> 결과 스냅샷 생성
4. 결과 스냅샷 OCI 업로드 (※ utils.upload_result_image 구현 필요)
5. `POST /api/files/callback`으로 결과 보고

## ⚠️ 배포 후 백엔드와 확정할 부분 (코드에 TODO 표시됨)
1. **영상/스트림 URL 필드명** — detail 응답에서 영상은 어느 필드?
   (app.py: `detail.get("videoUrl")`, LIVE는 `detail.get("streamUrl")`)
2. **결과 이미지 OCI 업로드 방식** — AI서버 직접 업로드 vs 백엔드 presigned URL
   (utils.py: `upload_result_image`)
3. **드론/정면 판단** — searchMode 외에 영상이 드론인지 정면인지 구분 기준
   (app.py: `is_drone`)