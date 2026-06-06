"""
app.py
================
GUSEOK(구석) - AI 분석 서버 (FastAPI)

백엔드(Spring)가 분석을 요청하면, 이 서버가:
  1) GET /api/v1/searches/{searchId} 로 타겟 정보 조회
  2) 영상/스트림 + 기준 사진 다운로드
  3) appearance 텍스트 -> 색/성별/머리 조건으로 변환 (LLM)
  4) analyzer.run_analysis 로 분석
  5) 결과 스냅샷 OCI 업로드 후 POST /api/files/callback 으로 보고

실행:
  uvicorn app:app --host 0.0.0.0 --port 8000
  (GPU 서버에 올린 뒤, 이 서버 주소를 백엔드에 알려주면 연동 시작)

백엔드는 분석이 필요할 때:
  POST http://<AI서버주소>:8000/analyze   body: {"searchId": 1}
"""

import os
import threading

from fastapi import FastAPI
from pydantic import BaseModel

from utils import (
    get_search_detail, send_callback_report,
    get_mission_command, download_file, upload_result_image,
    MissionParseError,
)
import analyzer

app = FastAPI(title="GUSEOK AI 분석 서버")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")   # ⚠️ 환경변수로 주입 (코드에 하드코딩 금지)
WORK_DIR = "./work"   # 다운로드한 영상/사진 임시 저장

# 드론 서버(영상 스트림 소스). 백엔드가 full streamUrl 을 주지 않고 droneId 만 줄 때
# 여기에 /video/{drone_id} 를 붙여 스트림 URL 을 조립한다.
# 형태: http://168.107.63.33:5001/video/{drone_id}
DRONE_SERVER_URL = os.environ.get("DRONE_SERVER_URL", "http://168.107.63.33:5001").rstrip("/")


@app.on_event("startup")
def _startup():
    # 서버 시작 시 모델을 미리 로드 (첫 요청 지연 방지)
    analyzer.load_models()
    os.makedirs(WORK_DIR, exist_ok=True)


class AnalyzeRequest(BaseModel):
    searchId: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """백엔드가 호출. 분석은 백그라운드로 돌리고 즉시 접수 응답."""
    threading.Thread(target=_run_job, args=(req.searchId,), daemon=True).start()
    return {"success": True, "data": {"searchId": req.searchId, "status": "ACCEPTED"}}


def _resolve_stream_url(detail):
    """
    LIVE(드론) 스트림 URL 결정.
      1) 백엔드가 full streamUrl 을 주면 그대로 사용
      2) droneId 만 주면 드론서버 URL(/video/{drone_id}) 로 조립
    필드명을 백엔드가 어떻게 줄지 아직 불확실하므로 흔한 후보들을 모두 시도한다.
    """
    # 1) full URL 후보
    for key in ("streamUrl", "stream_url", "videoUrl", "video_url"):
        url = detail.get(key)
        if url and str(url).startswith("http"):
            return url
    # 2) droneId 후보 -> 드론서버 URL 조립
    for key in ("droneId", "drone_id", "droneld"):  # "id"는 searchId 오인 위험으로 제외
        drone_id = detail.get(key)
        if drone_id is not None and str(drone_id) != "":
            return f"{DRONE_SERVER_URL}/video/{drone_id}"
    # 3) [fallback] 드론서버에서 활성 드론 자동 발견 (드론 1대 데모 전제)
    try:
        if (detail.get("searchMode") or "").upper() in ("LIVE", "DRONE"):
            import requests as _rq
            _ds = _rq.get(f"{DRONE_SERVER_URL}/drones", timeout=5).json()
            if isinstance(_ds, dict):
                _ds = _ds.get("drones") or _ds.get("data") or []
            if _ds:
                _d0 = _ds[0]
                _did = _d0 if not isinstance(_d0, dict) else (_d0.get("droneId") or _d0.get("drone_id") or _d0.get("id"))
                if _did is not None and str(_did) != "":
                    print(f"[작업] droneId 미제공 -> 드론서버 자동 발견: {_did}")
                    return f"{DRONE_SERVER_URL}/video/{_did}"
            print("[작업] 드론서버에 활성 드론 없음 -> 스트림 결정 실패")
    except Exception as _de:
        print(f"[작업] 드론 자동 발견 실패: {_de}")
    return None


def _run_job(search_id):
    """실제 분석 작업 (백그라운드 스레드)."""
    try:
        # 1) 타겟 정보 조회
        detail = get_search_detail(search_id)
        if not detail:
            send_callback_report(search_id, [], status="FAILED")
            return

        appearance  = detail.get("appearance", "") or ""
        gender_kr   = detail.get("gender", "") or ""
        search_mode = (detail.get("searchMode") or "VIDEO").upper()   # VIDEO | LIVE
        target_img_url = detail.get("targetImageUrl")

        # 2) appearance 텍스트 -> 색/성별/머리 조건 (규칙 파서 우선, 실패 시 LLM)
        #    해석 자체가 불가능하면(빈 입력/파싱·LLM 모두 실패) MissionParseError -> FAILED 콜백
        text = f"성별: {gender_kr}, 옷차림: {appearance}"
        try:
            mission_data = get_mission_command(text, api_key=GOOGLE_API_KEY)
        except MissionParseError as e:
            print(f"[작업] 인상착의 해석 실패 (search_id={search_id}): {e}")
            send_callback_report(search_id, [], status="FAILED", error_message=str(e))
            return

        # 3) 영상 소스 결정
        #    LIVE : 드론 스트림 (full streamUrl 또는 droneId 로 조립)
        #    VIDEO: 업로드된 영상 URL 을 받아 다운로드해서 분석
        # 영상 타입 판단 — 백엔드가 cameraView 또는 isDrone 필드로 알려줌, 없으면 기본 드론뷰
        camera_view = (detail.get("cameraView") or "").upper()
        if camera_view:
            is_drone = camera_view in ("TOP", "DRONE", "AERIAL")
        elif "isDrone" in detail:
            is_drone = bool(detail.get("isDrone"))
        else:
            is_drone = None   # 필드 없으면 영상 받고 자동 판별
        if search_mode in ("LIVE", "DRONE"):
            video_source = _resolve_stream_url(detail)
            if video_source:
                print(f"[작업] LIVE 스트림 소스: {video_source}")
        else:
            video_urls = detail.get("videoUrls") or ([detail["videoUrl"]] if detail.get("videoUrl") else [])
            video_url = video_urls[0] if video_urls else None
            video_source = download_file(video_url, os.path.join(WORK_DIR, f"video_{search_id}.mp4")) \
                if video_url else None

        if not video_source:
            print(f"[작업] 영상 소스를 찾지 못함 (search_id={search_id}, mode={search_mode}, detail keys={list(detail.keys())})")
            send_callback_report(search_id, [], status="FAILED",
                                 error_message="영상/스트림 소스를 찾지 못했습니다.")
            return

        # 3-b) 시점 자동 판별 (cameraView/isDrone 안 왔을 때만)
        if is_drone is None:
            if search_mode in ("LIVE", "DRONE"):
                is_drone = True   # 드론 스트림
            else:
                _view = analyzer.classify_view(video_source)
                if _view is None:
                    is_drone = True
                    print(f"[시점] 자동 판별 실패 -> 기본 드론뷰 (search_id={search_id})")
                else:
                    is_drone = (_view == "top")
                    print(f"[시점] 자동 판별: {_view} -> is_drone={is_drone} (search_id={search_id})")

        # 4) 기준 사진 다운로드 (있으면 Re-ID 작동)
        query_photo_path = None
        if target_img_url:
            query_photo_path = download_file(
                target_img_url, os.path.join(WORK_DIR, f"target_{search_id}.jpg"))
            # HEIC 등 cv2가 못 읽는 포맷 -> JPG 변환
            if query_photo_path:
                import cv2 as _cv2
                if _cv2.imread(query_photo_path) is None:
                    try:
                        from pillow_heif import register_heif_opener
                        register_heif_opener()
                        from PIL import Image as _PILImage
                        _PILImage.open(query_photo_path).convert("RGB").save(query_photo_path, "JPEG", quality=95)
                        print("[작업] 기준사진 HEIC -> JPG 변환 완료")
                    except Exception as _ce:
                        print(f"[작업] 기준사진 읽기 불가 -> 사진 없이 진행: {_ce}")
                        query_photo_path = None

        # 5) 분석 실행
        print(f"[작업] 분석 시작 (search_id={search_id}, mode={search_mode}, mission={mission_data})")
        results = analyzer.run_analysis(
            search_id=search_id,
            video_source=video_source,
            mission_data=mission_data,
            query_photo_path=query_photo_path,
            is_drone=is_drone,
        )

        # 6) 결과 스냅샷 OCI 업로드 -> matchedImageUrl 채우기
        for res in results:
            res["matchedImageUrl"] = upload_result_image(res["img_path"], search_id)

        # 7) 콜백 보고
        status = "COMPLETED"
        send_callback_report(search_id, results, status=status)
        print(f"[작업] 완료 (search_id={search_id}, 결과 {len(results)}건)")

    except Exception as e:
        print(f"[작업] 예외 발생 (search_id={search_id}): {e}")
        try:
            send_callback_report(search_id, [], status="FAILED", error_message=str(e))
        except Exception:
            pass