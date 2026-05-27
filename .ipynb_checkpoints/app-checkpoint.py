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
        #    VIDEO: 업로드된 영상 URL 을 받아 다운로드해서 분석
        #    LIVE : 드론 스트림 URL 을 직접 분석
        #    ⚠️ 영상/스트림 URL 을 detail 의 어느 필드에서 받는지는 배포 후 백엔드와 확정 필요.
        #       (예: detail["videoUrl"], detail["streamUrl"])
        if search_mode == "LIVE":
            video_source = detail.get("streamUrl")
            is_drone = True
        else:
            video_url = detail.get("videoUrl")
            video_source = download_file(video_url, os.path.join(WORK_DIR, f"video_{search_id}.mp4")) \
                if video_url else None
            is_drone = True   # ⚠️ 영상이 드론/정면인지 판단 기준 확정 필요 (지금은 드론 가정)

        if not video_source:
            print(f"[작업] 영상 소스를 찾지 못함 (search_id={search_id})")
            send_callback_report(search_id, [], status="FAILED")
            return

        # 4) 기준 사진 다운로드 (있으면 Re-ID 작동)
        query_photo_path = None
        if target_img_url:
            query_photo_path = download_file(
                target_img_url, os.path.join(WORK_DIR, f"target_{search_id}.jpg"))

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