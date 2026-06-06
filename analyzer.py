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
VIEW_YOLO_PATH       = './models/view/best.pt'      # 시점 분류기(frontal/top)
FEM_WEIGHT_PATH      = './models/fem/best.pt'
TRACKER_CFG          = './custom_tracker.yaml'
SNAPSHOT_DIR         = './results/snapshots'

YOLO_CONF_THRESHOLD  = 0.35
REID_SIM_THRESHOLD   = 0.45
FEM_ATTR_THRESHOLD   = 0.45
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
MAX_DURATION_SECONDS = 120  # LIVE 테스트용 임시

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
    _MODELS['view']          = YOLO(VIEW_YOLO_PATH)        # 시점 분류기
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
            q = _select_target_person(q, mission_data)  # 다인/배경 사진 -> 대상 인물만
            cv2.imwrite(f"/workspace/guseok/work/target_{search_id}_selected.jpg", q)
            target_reid_fingerprint = extract_reid_feature(reid_model, q, device)

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    import glob as _g0
    for _old in _g0.glob(os.path.join(SNAPSHOT_DIR, f"search_{search_id}_target_*")):
        try:
            os.remove(_old)
        except Exception:
            pass

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    frame_count = 0
    best_similarity    = {}
    snapshot_registry  = {}
    best_time_registry = {}
    person_features = {}
    _best_face_q = {}
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
                    if target_down_key:
                        _ds = attrs.get(target_down_key, 0)
                        if target_down_key == "downblue":
                            # 데님 보정: 어두운 청바지는 FEM이 black으로 인식 -> black 점수도 인정
                            _ds = max(_ds, attrs.get("downblack", 0))
                        if _ds < FEM_ATTR_THRESHOLD:
                            fem_pass = False
                    if not fem_pass:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        continue

                    # [얼굴수집] 트랙별 최고 화질 얼굴 갱신
                    if query_photo_path and seen_count[track_id] % 3 == 1 and h >= 100:
                        _fq = _face_quality(cropped_person)
                        if _fq is not None and _fq[0] > _best_face_q.get(track_id, 0.0):
                            _best_face_q[track_id] = _fq[0]
                            cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"search_{search_id}_target_{track_id}_best.jpg.facecrop.jpg"), _fq[1])

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
                            _clean = r.orig_img.copy()
                            cv2.rectangle(_clean, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.imwrite(fpath, _clean)
                            _BEST_BOX[(search_id, track_id)] = (x1, y1)
                            cv2.imwrite(fpath + f".crop{int(current_time_seconds) % 4}.jpg", cropped_person)
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

    # 2) 성별/머리 보조 필터 (기준사진 얼굴 있으면 얼굴 검증이 대체하므로 생략)
    if target_reid_fingerprint is None and (target_gender in ('male', 'female') or target_hair in ('short', 'long', 'tied')) and not _face_target_ok(search_id):
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
                print(f"[보조필터] track {tid} 제외 (성별/머리 조건)")
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
    return _face_rerank(search_id, results, mission_data)


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

def classify_view(video_path, n_frames=8, skip=3):
    """영상 앞부분 프레임 다수결로 시점 반환: 'frontal' 또는 'top'(드론). 못 읽으면 None."""
    import cv2
    M = load_models()
    model = M['view']
    cap = cv2.VideoCapture(str(video_path))
    frames, idx = [], 0
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
          break
        if idx % skip == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    if h > w:
        print(f"[시점] 세로 영상({w}x{h}) -> 폰 촬영 간주, frontal 처리")
        return "frontal"
    votes = {}
    for f in frames:
        r = model.predict(f, imgsz=224, verbose=False)[0]
        name = model.names[int(r.probs.top1)]
        votes[name] = votes.get(name, 0) + 1
    view = max(votes, key=votes.get)

    # 검출기 교차검증: 선택된 모델이 반대 모델보다 사람을 훨씬 못 잡으면 보정 (폰영상 오판 방지)
    try:
        keymap = {"top": "yolo_top", "frontal": "yolo_frontal"}
        if view in keymap:
            other_name = "frontal" if view == "top" else "top"
            sample = frames[::max(1, len(frames)//3)][:3]
            n_pick  = sum(len(M[keymap[view]].predict(f, conf=0.35, imgsz=960, verbose=False)[0].boxes) for f in sample)
            n_other = sum(len(M[keymap[other_name]].predict(f, conf=0.35, imgsz=960, verbose=False)[0].boxes) for f in sample)
            print(f"[시점] 교차검증 수치: {view}={n_pick}명 vs {other_name}={n_other}명")
            if (n_pick == 0 and n_other > 0) or (n_other >= n_pick * 1.3 and n_other - n_pick >= 5):
                print(f"[시점] 교차검증 보정: {view}모델 {n_pick}명 vs {other_name}모델 {n_other}명 -> {other_name}")
                view = other_name
    except Exception as e:
        print(f"[시점] 교차검증 스킵: {e}")
    return view

MAX_RESULTS = 1           # 보고할 최종 결과 수 (기획: 최고 일치 1명)
QE_SIM_THRESHOLD = 0.60   # 쿼리 확장: 확정 트랙과 이 이상 유사하면 점수 갱신
FACE_MIN_PX = 80      # 기준 얼굴 최소 높이(px) - 미만이면 얼굴 검증 신뢰 불가
FACE_SIM_KEEP = 0.55   # 이 이상이면 얼굴 일치로 보고 점수 갱신
FACE_SIM_DROP = 0.40   # 이 미만이면 다른 사람으로 보고 제외

def _face_rerank(search_id, results, mission_data=None):
    """후보 크롭의 얼굴을 기준사진 얼굴과 대조해 재정렬/필터. 얼굴 없으면 Re-ID 점수 유지."""
    if not results:
        return results
    try:
        import glob, torch, cv2
        import numpy as np
        from PIL import Image
        from facenet_pytorch import MTCNN, InceptionResnetV1
        from scipy.spatial.distance import cosine as _cos
        tpaths = sorted(glob.glob(f"/workspace/guseok/work/target_{search_id}.*"))
        if not tpaths:
            results.sort(key=lambda r: -r["accuracy"])
            return _stamp_scores(results[:MAX_RESULTS], search_id)
        if 'face_det' not in _MODELS:
            _MODELS['face_det'] = MTCNN(keep_all=False, device=device)
            _MODELS['face_emb'] = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        if 'face_det_all' not in _MODELS:
            _MODELS['face_det_all'] = MTCNN(keep_all=True, device=device)
        def _femb(bgr):
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            boxes, probs = _MODELS['face_det_all'].detect(rgb)
            if boxes is None:
                return None
            W = bgr.shape[1]
            def _fscore(b, p):
                w, h = b[2] - b[0], b[3] - b[1]
                cx = (b[0] + b[2]) / 2.0
                cent = 1.0 - abs(cx - W / 2.0) / (W / 2.0)
                return float(p) * ((w * h) ** 0.5) * (0.3 + 0.7 * cent)
            i = max(range(len(boxes)), key=lambda k: _fscore(boxes[k], probs[k]))
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            m = int(0.35 * max(x2 - x1, y2 - y1))
   
            sub = bgr[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
            face = _MODELS['face_det'](Image.fromarray(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)))
            if face is None:
                return None
            with torch.no_grad():
                e = _MODELS['face_emb'](face.unsqueeze(0).to(device))[0].cpu().numpy()
            return e / np.linalg.norm(e)
        _selp = f"/workspace/guseok/work/target_{search_id}_selected.jpg"
        if os.path.exists(_selp):
            timg = cv2.imread(_selp)
        else:
            timg = cv2.imread(tpaths[0])
            if timg is not None:
                timg = _select_target_person(timg, mission_data)
        if timg is not None:
            _b, _p = _MODELS['face_det_all'].detect(Image.fromarray(cv2.cvtColor(timg, cv2.COLOR_BGR2RGB)))
            if _b is None or max(bb[3] - bb[1] for bb in _b) < FACE_MIN_PX:
                print("[얼굴] 기준 얼굴 작음/없음 -> 얼굴 검증 생략 (Re-ID 순위 사용)")
                results.sort(key=lambda r: -r["accuracy"])
                return _stamp_scores(results[:MAX_RESULTS], search_id)
        tfe = _femb(timg) if timg is not None else None
        if tfe is None:
            print("[얼굴] 기준사진에서 얼굴 미검출 -> 얼굴 검증 생략")
            results.sort(key=lambda r: -r["accuracy"])
            return _stamp_scores(results[:MAX_RESULTS], search_id)
        kept = []
        for r in results:
            sim = None
            _cpaths = glob.glob(r["img_path"] + ".facecrop.jpg") + sorted(glob.glob(r["img_path"] + ".crop[0-9].jpg"))
            for cpath in _cpaths:
                crop = cv2.imread(cpath)
                fe = _femb(crop) if crop is not None else None
                if fe is not None:
                    _s = 1 - _cos(tfe, fe)
                    sim = _s if sim is None else max(sim, _s)
            if sim is None:
                if r['accuracy'] >= 0.50:
                    print(f"[얼굴] track {r['track_id']}: 얼굴 미검출 -> Re-ID 점수 유지 ({r['accuracy']})")
                    kept.append(r)
                else:
                    print(f"[얼굴] track {r['track_id']}: 얼굴 미검출 + 점수 낮음({r['accuracy']}) -> 제외")
                    continue
            elif sim >= FACE_SIM_KEEP:
                r["accuracy"] = round(float(max(r["accuracy"], sim)), 4)
                r["_face"] = True
                print(f"[얼굴] track {r['track_id']}: 얼굴 일치 {sim:.3f} -> 채택 (점수 갱신)")
                kept.append(r)
            elif sim < FACE_SIM_DROP:
                print(f"[얼굴] track {r['track_id']}: 얼굴 불일치 {sim:.3f} -> 제외")
            else:
                print(f"[얼굴] track {r['track_id']}: 판단 애매 {sim:.3f} -> Re-ID 점수 유지")
                kept.append(r)
        # [쿼리확장] 얼굴 확정 트랙의 영상 내 크롭을 2차 기준으로 나머지 재평가
        try:
            _conf = [r for r in kept if r.get("_face")]
            if _conf:
                _M2 = load_models()
                _base = max(_conf, key=lambda r: r["accuracy"])
                _feats = []
                for cpath in sorted(glob.glob(_base["img_path"] + ".crop[0-9].jpg")):
                    c = cv2.imread(cpath)
                    if c is not None:
                        f = extract_reid_feature(_M2['reid'], c, device)
                        _feats.append(f / (np.linalg.norm(f) + 1e-9))
                if _feats:
                    _q2 = np.mean(np.stack(_feats), axis=0)
                    _q2 = _q2 / (np.linalg.norm(_q2) + 1e-9)
                    for r in kept:
                        if r.get("_face"):
                            continue
                        _s2 = None
                        for cpath in sorted(glob.glob(r["img_path"] + ".crop[0-9].jpg")):
                            c = cv2.imread(cpath)
                            if c is None:
                                continue
                            f = extract_reid_feature(_M2['reid'], c, device)
                            f = f / (np.linalg.norm(f) + 1e-9)
                            _v = float(np.dot(_q2, f))
                            _s2 = _v if _s2 is None else max(_s2, _v)
                        if _s2 is not None and _s2 >= QE_SIM_THRESHOLD and _s2 > r["accuracy"]:
                            print(f"[쿼리확장] track {r['track_id']}: 확정 트랙 유사 {_s2:.3f} -> 점수 갱신")
                            r["accuracy"] = round(_s2, 4)
        except Exception as _qe:
            print(f"[쿼리확장] 생략: {_qe}")
        # 얼굴 확정 트랙이 항상 우선 (쿼리확장=옷 유사가 추월 못 하게)
        kept.sort(key=lambda r: (0 if r.get("_face") else 1, -r["accuracy"]))
        for r in kept:
            r.pop("_face", None)
        if len(kept) > MAX_RESULTS:
            print(f"[얼굴] 후보 {len(kept)}건 -> 상위 {MAX_RESULTS}건만 보고")
            kept = kept[:MAX_RESULTS]
        return _stamp_scores(kept, search_id)
    except Exception as e:
        print(f"[얼굴] 검증 단계 오류 -> 원본 결과 유지: {e}")
        return results


def _face_target_ok(search_id):
    """기준사진에서 얼굴이 검출되면 True -> 보조필터 대신 얼굴 검증 사용."""
    try:
        import glob, cv2
        from PIL import Image
        from facenet_pytorch import MTCNN, InceptionResnetV1
        tpaths = sorted(glob.glob(f"/workspace/guseok/work/target_{search_id}.*"))
        if not tpaths:
            return False
        if 'face_det' not in _MODELS:
            _MODELS['face_det'] = MTCNN(keep_all=False, device=device)
            _MODELS['face_emb'] = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        if 'face_det_all' not in _MODELS:
            _MODELS['face_det_all'] = MTCNN(keep_all=True, device=device)
        img = cv2.imread(tpaths[0])
        if img is None:
            return False
        img = _select_target_person(img)
        boxes, probs = _MODELS['face_det_all'].detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if boxes is None:
            return False
        fh = max(b[3] - b[1] for b in boxes)
        if fh < FACE_MIN_PX:
            print(f"[얼굴] 기준 얼굴 너무 작음({fh:.0f}px < {FACE_MIN_PX}px) -> 얼굴 검증 비활성, 보조필터 사용")
            return False
        print("[얼굴] 기준사진 얼굴 확인 -> 성별/머리 보조필터 생략 (얼굴 검증으로 대체)")
        return True
    except Exception as e:
        print(f"[얼굴] 기준사진 확인 오류 -> 보조필터 유지: {e}")
        return False


def _select_target_person(img, mission_data=None):
    """기준사진에서 대상 인물 크롭 선택.
    1명 -> 그 크롭 / 여러 명 -> FEM 속성을 인상착의와 대조해 최고 득점 / 검출 실패 -> 원본."""
    try:
        M = load_models()
        det = M['yolo_frontal'].predict(img, conf=0.3, imgsz=960, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(int).tolist() if det.boxes is not None else []
        crops = [img[max(0, y1):y2, max(0, x1):x2] for x1, y1, x2, y2 in boxes]
        crops = [c for c in crops if c.size > 0]
        if not crops:
            return img
        if len(crops) == 1:
            print("[기준사진] 인물 1명 -> 크롭 사용")
            return crops[0]
        md = mission_data or {}
        upk = f"up{md['up_color']}" if md.get("up_color") else None
        dnk = f"down{md['down_color']}" if md.get("down_color") else None
        best_c, best_s = None, -1.0
        for c in crops:
            a = predict_attributes(M['fem'], c)
            s = 0.0
            if upk: s += a.get(upk, 0)
            if dnk:
                s += a.get(dnk, 0)  # 선택 단계: 원색 그대로 (변별력 우선, 데님 보정 미적용)
            if md.get("hair") in ("long", "tied"):
                s += max(a.get("hair_long", 0), a.get("hair_tied", 0))
            elif md.get("hair") == "short":
                s += a.get("hair_short", 0)
            s += 1e-7 * (c.shape[0] * c.shape[1])  # 조건 없으면 큰 크롭 선호
            if s > best_s:
                best_s, best_c = s, c
        print(f"[기준사진] 인물 {len(crops)}명 -> 인상착의 최고 일치 크롭 선택 (점수 {best_s:.2f})")
        return best_c
    except Exception as e:
        print(f"[기준사진] 인물 선택 실패 -> 원본 사용: {e}")
        return img

def _face_quality(bgr):
    """크롭에서 중앙 가중 최고 얼굴의 (품질점수, 얼굴영역 이미지) 반환. 없으면 None."""
    try:
        from facenet_pytorch import MTCNN
        from PIL import Image
        if 'face_det_all' not in _MODELS:
            _MODELS['face_det_all'] = MTCNN(keep_all=True, device=device)
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes, probs = _MODELS['face_det_all'].detect(rgb)
        if boxes is None:
            return None
        W = bgr.shape[1]
        bi, bs = -1, -1.0
        for k in range(len(boxes)):
            b, p = boxes[k], probs[k]
            w, h = b[2] - b[0], b[3] - b[1]
            cent = 1.0 - abs((b[0] + b[2]) / 2.0 - W / 2.0) / (W / 2.0)
            s = float(p) * ((w * h) ** 0.5) * (0.3 + 0.7 * cent)
            if s > bs:
                bs, bi = s, k
        x1, y1, x2, y2 = [int(v) for v in boxes[bi]]
        m = int(0.35 * max(x2 - x1, y2 - y1))
        sub = bgr[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
        if sub.size == 0:
            return None
        return bs, sub
    except Exception:
        return None


_BEST_BOX = {}

def _stamp_scores(rs, search_id):
    """최종 점수를 스냅샷에 표기 (저장 순간 점수와 최종 점수 불일치 방지)."""
    for r in rs:
        try:
            bb = _BEST_BOX.get((search_id, r.get("track_id")))
            if not bb:
                continue
            im = cv2.imread(r["img_path"])
            if im is None:
                continue
            cv2.putText(im, f"{r['accuracy']*100:.1f}%", (bb[0], max(30, bb[1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imwrite(r["img_path"], im)
        except Exception:
            pass
    return rs
