"""
analyzer.py
================
GUSEOK(구석) - AI 분석 코어

기존 test.py 의 분석 로직(FEM 색/머리 필터 + Re-ID + 겹침/중복 제거)을
input() 없이 함수 하나(run_analysis)로 호출할 수 있게 정리한 모듈.

- 모델은 서버 시작 시 한 번만 로드 (load_models)
- run_analysis(...) 가 분석을 돌리고 결과 리스트를 반환
- 백엔드 연동(FastAPI)은 app.py 에서 이 함수를 호출

[B' 패치]
- LIVE 드론 스트림은 끝이 없으므로, 분석 루프에 최대 시간 한도(30분)를 둔다.
- 스트림이 30분 전에 자연 종료되거나(드론 착륙·연결 끊김) 에러로 끊겨도
  누적된 결과로 마무리 후처리 + 콜백이 도록 try/except 로 감싼다.
- 일반 영상 파일은 그 전에 자연 종료되므로 영향 없음(안전 한도 역할).
"""

import os
import time
import cv2
import torch
from ultralytics import YOLO
from scipy.spatial.distance import cosine

from utils import (
    load_fem_model, predict_attributes,
    load_reid_model, extract_reid_feature,
)

# =====================================================================
# 경로 / 임계값 설정
# =====================================================================
CROWDHUMAN_YOLO_PATH = './models/frontal/best.pt'   # 정면/CCTV 용
TOPVIEW_YOLO_PATH    = './models/top/best.pt'        # 드론/사선 용
FEM_WEIGHT_PATH      = './models/fem/best.pt'
TRACKER_CFG          = './custom_tracker.yaml'
SNAPSHOT_DIR         = './results/snapshots'

YOLO_CONF_THRESHOLD  = 0.35
REID_SIM_THRESHOLD   = 0.65
FEM_ATTR_THRESHOLD   = 0.85
DEBUG_MODE           = False   # 서버에서는 기본 False (필요시 True)

GENDER_MARGIN     = 0.70
GENDER_MIN_FRAMES = 5
HAIR_MARGIN       = 0.55
GENDER_SOFT       = 0.50
MAX_ASPECT_RATIO  = 4.0
DEDUP_SIM_THRESHOLD = 0.80
CHECK_MULTI_PERSON  = True
MULTI_PERSON_CONF   = 0.50
MULTI_CHECK_MIN_H   = 100
MULTI_CHECK_MIN_ASPECT = 0.0
MERGED_REMOVE_RATIO = 0.5
MERGED_MIN_OBSERVE  = 4

# B' 실시간 스트림용 최대 분석 시간 (초). 30분.
MAX_DURATION_SECONDS = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 모델 로드 (서버 시작 시 1회)
# =====================================================================
_MODELS = {}


def load_models():
    """서버 시작 시 한 번 호출. 모든 모델을 메모리에 올린다."""
    if _MODELS:
        return _MODELS
    print(f"[AI] 모델 로드 중... (device={device})")
    _MODELS['yolo_frontal']  = YOLO(CROWDHUMAN_YOLO_PATH)
    _MODELS['yolo_top']      = YOLO(TOPVIEW_YOLO_PATH)
    _MODELS['yolo_count_f']  = YOLO(CROWDHUMAN_YOLO_PATH)   # 겹침 재검출(정면)
    _MODELS['yolo_count_t']  = YOLO(TOPVIEW_YOLO_PATH)      # 겹침 재검출(드론)
    for k in ('yolo_frontal', 'yolo_top', 'yolo_count_f', 'yolo_count_t'):
        _MODELS[k].to(device)
    _MODELS['fem']  = load_fem_model(weight_path=FEM_WEIGHT_PATH)
    _MODELS['reid'] = load_reid_model(device)
    print("[AI] 모델 로드 완료.")
    return _MODELS


# =====================================================================
# 분석 본체
# =====================================================================
def run_analysis(search_id, video_source, mission_data,
                 query_photo_path=None, is_drone=True):
    """
    한 건의 수색 영상을 분석한다.

    Args:
        search_id       : 탐색 ID (스냅샷 파일명에 사용)
        video_source    : 영상 파일 경로 또는 스트림 URL
        mission_data    : {"gender","up_color","down_color","hair"} (색/성별/머리 조건)
        query_photo_path: 실종자 기준 사진 경로 (있으면 Re-ID 작동, 없으면 None)
        is_drone        : True 면 탑뷰 YOLO, False 면 정면 YOLO

    Returns:
        results: [{"track_id","accuracy","time","img_path"} ...]
                 (img_path 는 로컬 경로 -> app.py 에서 OCI 업로드 후 URL 로 교체)
    """
    M = load_models()
    yolo_model       = M['yolo_top'] if is_drone else M['yolo_frontal']
    yolo_count_model = M['yolo_count_t'] if is_drone else M['yolo_count_f']
    fem_model        = M['fem']
    reid_model       = M['reid']

    target_gender   = mission_data.get("gender", None)
    target_hair     = mission_data.get("hair", None)
    target_up_key   = f"up{mission_data['up_color']}"     if mission_data.get("up_color")   else None
    target_down_key = f"down{mission_data['down_color']}" if mission_data.get("down_color") else None

    # 기준 사진 -> Re-ID 특징
    target_reid_fingerprint = None
    if query_photo_path and os.path.exists(query_photo_path):
        q = cv2.imread(query_photo_path)
        if q is not None:
            target_reid_fingerprint = extract_reid_feature(reid_model, q, device)

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    frame_count = 0
    best_similarity    = {}
    snapshot_registry  = {}
    best_time_registry = {}
    person_features = {}
    gender_sum   = {}
    gender_count = {}
    hair_sum     = {}
    merged_count = {}
    seen_count   = {}

    track_results = yolo_model.track(
        source=video_source, persist=True, tracker=TRACKER_CFG,
        classes=[0], conf=YOLO_CONF_THRESHOLD, iou=0.7, stream=True, verbose=False
    )

    # B' 시작 시각 기록 (최대 시간 한도 체크용)
    start_time = time.time()
    print(f"[분석] 시작 (search_id={search_id}, max_duration={MAX_DURATION_SECONDS}s)")

    # 스트림 끊김/네트워크 에러 등은 try 로 감싸서, 누적된 결과로 마무리 후처리가 도록 한다.
    try:
        for r in track_results:
            # === B': 최대 시간 한도 체크 ===
            elapsed = time.time() - start_time
            if elapsed > MAX_DURATION_SECONDS:
                print(f"[분석] 최대 시간 {MAX_DURATION_SECONDS}s 도달 → 종료 (search_id={search_id}, 처리 프레임 {frame_count})")
                break

            frame = r.orig_img.copy()
            current_time_seconds = frame_count / fps

            if r.boxes is not None and r.boxes.id is not None:
                boxes      = r.boxes.xyxy.cpu().numpy()
                track_ids  = r.boxes.id.int().cpu().tolist()
                yolo_confs = r.boxes.conf.cpu().tolist()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    track_id  = track_ids[i]
                    yolo_conf = yolo_confs[i]

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cropped_person = frame[y1:y2, x1:x2]

                    h, w = cropped_person.shape[:2]
                    if h < 30 or w < 15:
                        continue

                    seen_count[track_id] = seen_count.get(track_id, 0) + 1

                    # 세로로 너무 긴 박스 -> 겹침
                    aspect = h / max(w, 1)
                    if aspect > MAX_ASPECT_RATIO:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 1)
                        continue

                    # 박스 안 사람 수 재검출 (겹침)
                    if CHECK_MULTI_PERSON and aspect >= MULTI_CHECK_MIN_ASPECT and h >= MULTI_CHECK_MIN_H:
                        sub = yolo_count_model.predict(
                            cropped_person, classes=[0], conf=MULTI_PERSON_CONF, verbose=False
                        )
                        n_person = len(sub[0].boxes) if (len(sub) > 0 and sub[0].boxes is not None) else 0
                        if n_person >= 2:
                            merged_count[track_id] = merged_count.get(track_id, 0) + 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 1)
                            continue

                    attrs = predict_attributes(fem_model, cropped_person)

                    # gender / hair 누적
                    _g = attrs.get('gender', None)
                    if _g is not None:
                        gender_sum[track_id]   = gender_sum.get(track_id, 0.0) + float(_g)
                        gender_count[track_id] = gender_count.get(track_id, 0) + 1
                    _hs = attrs.get('hair_short', None)
                    if _hs is not None:
                        acc = hair_sum.setdefault(track_id, [0.0, 0.0, 0.0])
                        acc[0] += float(_hs)
                        acc[1] += float(attrs.get('hair_long', 0.0))
                        acc[2] += float(attrs.get('hair_tied', 0.0))

                    # 색 필터 (메인 게이트)
                    fem_pass = True
                    if target_up_key and attrs.get(target_up_key, 0) < FEM_ATTR_THRESHOLD:
                        fem_pass = False
                    if target_down_key and attrs.get(target_down_key, 0) < FEM_ATTR_THRESHOLD:
                        fem_pass = False
                    if not fem_pass:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        continue

                    # Re-ID 필터 (사진 있을 때만)
                    is_final_target  = True
                    similarity_score = 1.0
                    if target_reid_fingerprint is not None:
                        cur = extract_reid_feature(reid_model, cropped_person, device)
                        similarity_score = 1 - cosine(target_reid_fingerprint, cur)
                        if similarity_score < REID_SIM_THRESHOLD:
                            is_final_target = False

                    if is_final_target:
                        if target_reid_fingerprint is not None:
                            label = f"ID:{track_id} | Re-ID:{similarity_score*100:.1f}%"
                        else:
                            label = f"ID:{track_id} | YOLO:{yolo_conf*100:.0f}%"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        current_score = similarity_score if target_reid_fingerprint is not None else float(yolo_conf)

                        # Re-ID 중복 인물 제거
                        if track_id not in snapshot_registry:
                            dedup_feat = extract_reid_feature(reid_model, cropped_person, device)
                            is_dup = False
                            for other_feat in person_features.values():
                                if (1 - cosine(dedup_feat, other_feat)) >= DEDUP_SIM_THRESHOLD:
                                    is_dup = True
                                    break
                            if is_dup:
                                continue
                            person_features[track_id] = dedup_feat

                        if track_id not in best_similarity or current_score > best_similarity[track_id]:
                            if track_id in snapshot_registry and os.path.exists(snapshot_registry[track_id]):
                                try:
                                    os.remove(snapshot_registry[track_id])
                                except OSError:
                                    pass
                            fname = f"search_{search_id}_target_{track_id}_best.jpg"
                            fpath = os.path.join(SNAPSHOT_DIR, fname)
                            cv2.imwrite(fpath, frame)
                            best_similarity[track_id]    = current_score
                            snapshot_registry[track_id]  = fpath
                            best_time_registry[track_id] = current_time_seconds

            frame_count += 1
    except Exception as e:
        # 스트림 끊김/네트워크/디코딩 에러 등 → 누적 결과로 마무리
        print(f"[분석] 스트림 처리 중단: {e} → 누적 결과로 후처리 진행 (search_id={search_id}, 처리 프레임 {frame_count})")

    # for 루프가 정상 종료(영상 끝/스트림 끊김)된 경우 안내 로그
    elapsed_total = time.time() - start_time
    print(f"[분석] 루프 종료 (search_id={search_id}, 경과 {elapsed_total:.1f}s, 프레임 {frame_count}, 임시 결과 {len(snapshot_registry)}건)")

    # ===== 영상 종료 후 정리 =====
    # 1) 겹침 트랙 제거 (관측의 일정 비율 이상이 2명)
    for tid in list(snapshot_registry.keys()):
        seen = seen_count.get(tid, 0)
        mc   = merged_count.get(tid, 0)
        if seen >= MERGED_MIN_OBSERVE and (mc / seen) >= MERGED_REMOVE_RATIO:
            _drop(tid, snapshot_registry, best_similarity, best_time_registry, person_features)

    # 2) 성별/머리 보조 필터
    if target_gender in ('male', 'female') or target_hair in ('short', 'long', 'tied'):
        for tid in list(snapshot_registry.keys()):
            cnt = gender_count.get(tid, 0)
            if cnt < GENDER_MIN_FRAMES:
                continue
            avg_g = gender_sum[tid] / cnt
            hs, hl, ht = hair_sum.get(tid, [0.0, 0.0, 0.0])
            avg_short = hs / cnt
            avg_long  = (hl + ht) / cnt
            drop = False
            if target_gender == 'female' and avg_g <= (1.0 - GENDER_MARGIN):
                drop = True
            elif target_gender == 'male' and avg_g >= GENDER_MARGIN:
                drop = True
            if not drop and target_hair in ('long', 'tied') and avg_short >= HAIR_MARGIN and avg_short > avg_long:
                drop = True
            if not drop and target_hair == 'short' and avg_long >= HAIR_MARGIN and avg_long > avg_short:
                drop = True
            if not drop and target_hair is None:
                if target_gender == 'female' and avg_g < GENDER_SOFT and avg_short >= HAIR_MARGIN:
                    drop = True
                elif target_gender == 'male' and avg_g >= GENDER_SOFT and avg_long >= HAIR_MARGIN:
                    drop = True
            if drop:
                _drop(tid, snapshot_registry, best_similarity, best_time_registry, person_features)

    # ===== 결과 정리 =====
    results = []
    for tid, path in snapshot_registry.items():
        results.append({
            "track_id": tid,
            "accuracy": round(float(best_similarity[tid]), 4),
            "time":     int(best_time_registry[tid]),
            "img_path": path,
        })
    return results


def _drop(tid, snapshot_registry, best_similarity, best_time_registry, person_features):
    path = snapshot_registry.get(tid)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    snapshot_registry.pop(tid, None)
    best_similarity.pop(tid, None)
    best_time_registry.pop(tid, None)
    person_features.pop(tid, None)
