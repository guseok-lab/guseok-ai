import os
import json
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google import genai
from google.genai import types
import PIL.Image
import torchreid
from scipy.spatial.distance import cosine
import torchvision.transforms as T
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPRING_SERVER_URL = os.environ.get("SPRING_SERVER_URL", "http://localhost:8080")

# 추론 전처리 (⚠️ train_fem.py 의 IMG_SIZE / Normalize 와 반드시 동일)
inference_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================================================================
# 속성 정의 (33개) - train_fem.py 의 ATTR_KEYS 와 "순서"가 반드시 동일!
#   색 키는 test.py 가 f"up{color}" / f"down{color}" 로 조회하므로
#   언더스코어 없이 upblack / downblack 형태로 둔다.
#   gender: 1에 가까울수록 female, 0에 가까울수록 male (새 모델 기준)
# =====================================================================
ATTR_KEYS = [
    'gender',
    # 상의 색 (10)
    'upblack', 'upwhite', 'upgray', 'upbrown', 'upblue',
    'upgreen', 'upred', 'uppink', 'uppurple', 'upyellow',
    # 하의 색 (10)
    'downblack', 'downwhite', 'downgray', 'downbrown', 'downblue',
    'downgreen', 'downred', 'downpink', 'downpurple', 'downyellow',
    # 소매 / 하의 길이
    'short_sleeve', 'short_lower',
    # 머리 길이 (3)
    'hair_short', 'hair_long', 'hair_tied',
    # 머리 색 (4)
    'haircolor_black', 'haircolor_brown', 'haircolor_white', 'haircolor_yellow',
    # 소지품 (3)
    'has_backpack', 'has_bag', 'has_hat',
]
N_ATTR = len(ATTR_KEYS)   # 33


class MissionParseError(Exception):
    """인상착의를 해석하지 못해 분석을 진행할 수 없을 때 발생."""
    pass


# LLM 시스템 프롬프트용 공통 가이드 (색 팔레트는 모델이 학습한 10색으로 통일)
COLOR_GUIDE = """
[보유 중인 타겟팅 옵션]
1. "gender": "male" 또는 "female"
2. "up_color" (상의) / "down_color" (하의):
   "black", "white", "gray", "brown", "blue", "green", "red", "pink", "purple", "yellow"
3. "hair" (머리 길이/스타일, 알 수 있을 때만): "short"(짧은머리), "long"(긴머리), "tied"(묶은머리/포니테일/번)

[색 변환 가이드 - 위 10색 중 하나로만 매핑]
- 남색/네이비/하늘색/청록(teal) -> "blue"
- 베이지/카키/탄/갈색 -> "brown"
- 연두/민트/올리브 -> "green"
- 아이보리/크림 -> "white"
- 자주/보라 -> "purple"

[머리(hair) 판단 가이드]
- 명령에 머리 정보가 명확히 있을 때만 "hair" 를 채운다 (예: "긴머리 여자" -> "long", "머리 묶은" -> "tied", "짧은머리/단발" -> "short")
- 머리 정보가 전혀 없으면 "hair" 키는 아예 생략한다 (추측 금지)
"""


# =====================================================================
# 규칙 기반 인상착의 파서 (LLM 호출을 최소화 -> 토큰/비용 절약)
#   한국어 색/성별/머리 키워드를 사전으로 직접 매칭한다.
#   대부분의 명령은 이 파서로 처리되어 Gemini 호출이 거의 발생하지 않는다.
# =====================================================================

_COLOR_WORDS = {
    "검정": "black", "검은": "black", "까만": "black", "블랙": "black", "흑": "black",
    "하양": "white", "하얀": "white", "흰": "white", "화이트": "white", "백": "white",
    "아이보리": "white", "크림": "white",
    "회색": "gray", "그레이": "gray", "쥐색": "gray", "회": "gray",
    "갈색": "brown", "브라운": "brown", "베이지": "brown", "카키": "brown", "탄": "brown", "밤색": "brown",
    "남색": "blue", "네이비": "blue", "하늘색": "blue", "하늘": "blue", "청록": "blue",
    "파랑": "blue", "파란": "blue", "블루": "blue", "청": "blue",
    "초록": "green", "초록색": "green", "연두": "green", "민트": "green", "올리브": "green",
    "그린": "green", "녹색": "green",
    "빨강": "red", "빨간": "red", "레드": "red", "적": "red",
    "분홍": "pink", "핑크": "pink", "분홍색": "pink",
    "보라": "purple", "자주": "purple", "퍼플": "purple", "보라색": "purple",
    "노랑": "yellow", "노란": "yellow", "옐로": "yellow", "황": "yellow", "금색": "yellow",
}

_UPPER_WORDS = ["상의", "윗옷", "셔츠", "티셔츠", "티", "후드티", "후드", "맨투맨",
                "재킷", "자켓", "코트", "패딩", "점퍼", "니트", "블라우스", "가디건", "조끼"]
_LOWER_WORDS = ["하의", "청바지", "반바지", "바지", "치마", "스커트", "슬랙스", "레깅스", "트레이닝"]

_MALE_WORDS   = ["남자", "남성", "소년"]
_FEMALE_WORDS = ["여자", "여성", "소녀"]

_HAIR_LONG  = ["긴머리", "장발", "긴머", "롱헤어"]
_HAIR_TIED  = ["묶은", "묶음", "포니테일", "포니", "올림머리", "번", "꽁지"]
_HAIR_SHORT = ["짧은머리", "단발", "숏컷", "짧은머", "스포츠머리", "빡빡"]


def _find_color_near(text, anchors):
    """anchor(상의/하의 단어) 바로 앞 구간에서 가장 가까운 색을 찾는다."""
    best_color, best_pos = None, -1
    for a in anchors:
        idx = text.find(a)
        while idx != -1:
            seg = text[max(0, idx - 12):idx]
            for _sep in (",", ".", "/", ":"):
                _cut = seg.rfind(_sep)
                if _cut != -1:
                    seg = seg[_cut + 1:]
            window = seg + text[idx:idx + len(a)]
            for word, color in sorted(_COLOR_WORDS.items(), key=lambda x: -len(x[0])):
                if word in window and idx > best_pos:
                    best_color, best_pos = color, idx
                    break
            idx = text.find(a, idx + 1)
    return best_color


def parse_appearance_rule(text):
    """규칙 기반 파싱. 반환: {"gender","up_color","down_color","hair"} (없는 키는 생략)."""
    if not text:
        return {}
    t = text.replace(" ", "")
    t = t.replace("차콜", "검정").replace("챠콜", "검정")  # 색 동의어 정규화
    result = {}

    if any(w in t for w in _FEMALE_WORDS):
        result["gender"] = "female"
    elif any(w in t for w in _MALE_WORDS):
        result["gender"] = "male"

    if any(w in t for w in _HAIR_TIED):
        result["hair"] = "tied"
    elif any(w in t for w in _HAIR_LONG):
        result["hair"] = "long"
    elif any(w in t for w in _HAIR_SHORT):
        result["hair"] = "short"
    if "hair" not in result and "단발" in t:
        result["hair"] = "short"

    up_color   = _find_color_near(t, _UPPER_WORDS)
    down_color = _find_color_near(t, _LOWER_WORDS)

    # 청바지=데님: 같은 절에 명시 색 없으면 blue
    if "청바지" in t and not down_color:
        down_color = "blue"

    if "상하의" in t:
        c = _find_color_near(t, ["상하의"]) or _nearest_color(t)
        if c:
            up_color   = up_color or c
            down_color = down_color or c

    if up_color is None and down_color is None:
        c = _nearest_color(t)
        if c:
            up_color = c   # 옷 단어 없이 색만 있으면 상의로 추정

    if up_color:
        result["up_color"] = up_color
    if down_color:
        result["down_color"] = down_color
    return result


def _nearest_color(t):
    for word, color in sorted(_COLOR_WORDS.items(), key=lambda x: -len(x[0])):
        if word in t:
            return color
    return None


def _needs_llm(parsed):
    """규칙 파서 결과가 불충분하면 True (LLM 폴백 필요)."""
    has_color = parsed.get("up_color") or parsed.get("down_color")
    return not (parsed.get("gender") or has_color)


def load_fem_model(weight_path: str):
    """train_fem.py 와 동일한 구조(resnet50 + Dropout 헤드, 출력 33)로 로드."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, N_ATTR),
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_attributes(model, cv2_image):
    """각 속성의 확률값(0~1 float)을 dict 로 반환."""
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = inference_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()

    return {ATTR_KEYS[i]: float(probs[i]) for i in range(N_ATTR)}


def get_mission_command(user_text: str, api_key: str = None, mode: str = "fem"):
    print(f"\n'{user_text}' 명령 해독 중 (LLM API 통신 중)")

    GOOGLE_API_KEY = api_key or os.environ.get("GOOGLE_API_KEY")

    # 인상착의 텍스트 자체가 비어있으면 분석 불가 -> 에러
    if not user_text or not user_text.strip():
        raise MissionParseError("인상착의 정보가 비어 있습니다.")

    # 1) 규칙 기반 파서 우선 (LLM 호출 없이 대부분 처리 -> 토큰 절약)
    parsed = parse_appearance_rule(user_text)
    if not _needs_llm(parsed):
        print(f"[파서] 규칙 기반 해석 성공 (LLM 미사용): {parsed}")
        return parsed

    # 2) 규칙 파서로 부족하면 LLM 폴백
    if not GOOGLE_API_KEY:
        # 파서가 일부라도 뽑았으면 그걸로 진행, 아예 못 뽑았으면 에러
        if parsed:
            print(f"[파서] LLM 키 없음, 규칙 부분결과로 진행: {parsed}")
            return parsed
        raise MissionParseError("인상착의를 해석하지 못했습니다 (규칙 파싱 실패 + LLM 키 없음).")

    print(f"[파서] 규칙 파싱 부족 -> LLM 폴백 (부분결과: {parsed})")

    system_prompt = f"""
        당신은 'GUSEOK(구석)' 시스템에서 사용자의 명령을 분석하여 CCTV 딥러닝 모델에게 전달하는 'AI 통신 장교'입니다.
        사용자의 자연어 수색 명령에서 [성별], [상의 색상], [하의 색상]을 추출하여 반드시 아래의 JSON 포맷으로만 반환하십시오.

        {COLOR_GUIDE}

        [JSON 출력 예시]
        입력: "파란 셔츠에 검은 바지 입은 남자 찾아"
        출력: {{"gender": "male", "up_color": "blue", "down_color": "black"}}
        입력: "긴머리에 빨간 티 입은 여자"
        출력: {{"gender": "female", "up_color": "red", "hair": "long"}}
    """

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        llm_result = json.loads(response.text)
        merged = {**llm_result, **parsed}   # 규칙 파서가 잡은 값 우선 유지
        if _needs_llm(merged):
            # LLM 도 유의미한 조건을 못 뽑음 -> 에러
            raise MissionParseError("인상착의를 해석하지 못했습니다 (LLM 결과도 비어 있음).")
        return merged
    except MissionParseError:
        raise
    except Exception as e:
        # LLM 통신 자체 실패: 파서가 일부라도 뽑았으면 진행, 아니면 에러
        if parsed:
            print(f"[파서] LLM 통신 실패({e}) -> 규칙 부분결과로 진행: {parsed}")
            return parsed
        raise MissionParseError(f"인상착의 해석 실패 (규칙 파싱 실패 + LLM 통신 오류: {e})")


def get_search_detail(search_id):
    """
    GET /api/v1/searches/{searchId} - 탐색 상세 조회
    반환: data dict (gender, height, weight, appearance, targetImageUrl, searchMode, status)
    """
    url = f"{SPRING_SERVER_URL}/api/v1/searches/{search_id}"
    print(f"[백엔드] 탐색 정보 조회: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", {})
        print(f"[백엔드] 수신: {data}")
        return data
    except Exception as e:
        print(f"[백엔드] 탐색 정보 조회 실패: {e}")
        return None


def send_callback_report(search_id, results, status="COMPLETED", error_message=None):
    """
    POST /api/files/callback - AI 분석 완료/실패 콜백
    Swagger 스펙:
      { "searchId": int, "status": "COMPLETED"|"FAILED",
        "results": [ {resultType, status, accuracy, matchedImageUrl, matchedTimeSeconds} ] }
    실패 시 error_message 를 함께 보낸다(백엔드가 사유 표시/로깅에 사용).
    """
    url = f"{SPRING_SERVER_URL}/api/files/callback"
    print(f"\n[백엔드] 분석 {status} 콜백: {url}")

    payload = {
        "searchId": int(search_id),
        "status": status,
        "results": [],
    }
    if error_message:
        payload["errorMessage"] = error_message
    for res in results:
        payload["results"].append({
            "resultType": "VIDEO",
            "status": "FOUND",
            "accuracy": round(float(res.get("accuracy", 0.0)), 4),
            "matchedImageUrl": res.get("matchedImageUrl", ""),
            "matchedTimeSeconds": int(res.get("time", 0)),
        })

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("[백엔드] 콜백 보고 성공.")
        return True
    except Exception as e:
        print(f"[백엔드] 콜백 보고 실패: {e}")
        return False


def download_file(url, save_path):
    """URL(영상/사진)을 로컬에 다운로드. 실패 시 None."""
    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except Exception as e:
        print(f"[다운로드 실패] {url}: {e}")
        return None


def upload_result_image(local_path, search_id):
    """
    결과 스냅샷을 백엔드 presigned URL 방식으로 업로드하고,
    콜백에 넣을 공개 URL(resultImageUrl)을 반환한다.
 
    흐름 (백엔드 협의 결과):
      1) GET /api/files/result-upload-url?searchId=&filename=
         -> { objectKey, uploadUrl(PUT용 10분 유효), resultImageUrl(콜백용) }
      2) uploadUrl 로 이미지 PUT 업로드
      3) resultImageUrl 반환 (app.py 가 matchedImageUrl 에 넣음)
 
    실패 시 None 반환 (호출부에서 빈 문자열 처리).
    """
    if not local_path or not os.path.exists(local_path):
        print(f"[OCI] 업로드할 파일이 없음: {local_path}")
        return None
 
    filename = os.path.basename(local_path)
 
    # 1) presigned URL 발급 요청
    issue_url = f"{SPRING_SERVER_URL}/api/files/result-upload-url"
    try:
        r = requests.get(
            issue_url,
            params={"searchId": int(search_id), "filename": filename},
            timeout=10,
        )
        r.raise_for_status()
        info = r.json()
        # 응답이 {"data": {...}} 로 감싸여 올 수도 있어 양쪽 다 대응
        if "uploadUrl" not in info and isinstance(info.get("data"), dict):
            info = info["data"]
        upload_url       = info["uploadUrl"]
        result_image_url = info["resultImageUrl"]
    except Exception as e:
        print(f"[OCI] presigned URL 발급 실패: {e}")
        return None
 
    # 2) uploadUrl 로 PUT 업로드
    try:
        with open(local_path, "rb") as f:
            put_resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": "image/jpeg"},
                timeout=30,
            )
        put_resp.raise_for_status()
    except Exception as e:
        print(f"[OCI] 이미지 PUT 업로드 실패: {e}")
        return None
 
    print(f"[OCI] 업로드 성공: {filename} -> {result_image_url}")
    # 3) 콜백에 쓸 공개 URL 반환
    return result_image_url


def extract_attributes_from_query_photo(photo_path, api_key=None):
    print(f"\n실종자 사진({photo_path}) 해독 중... (멀티모달 LLM 통신)")

    GOOGLE_API_KEY = api_key or os.environ.get("GOOGLE_API_KEY")

    # 기본값: 검정 상의 / 검정 하의 / 여성
    DEFAULT = {"up_color": "black", "down_color": "black", "gender": "female"}

    if not GOOGLE_API_KEY or not os.path.exists(photo_path):
        print("API 키 또는 사진 파일이 없어 기본값으로 진행합니다.")
        return DEFAULT

    img = PIL.Image.open(photo_path)

    system_prompt = f"""
        당신은 실종자 수색 시스템의 '영상 분석 장교'입니다.
        제시된 사진 속 실종자의 [성별], [상의 색상], [하의 색상]을 분석하여 반드시 아래의 JSON 포맷으로만 반환하십시오.

        {COLOR_GUIDE}

        [JSON 출력 예시]
        {{ "gender": "male", "up_color": "blue", "down_color": "black", "hair": "short" }}
    """

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img, system_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"사진 해독 실패 ({e}). 기본값으로 진행합니다.")
        return DEFAULT


# =====================================================================
# Re-ID (OSNet)
# =====================================================================
reid_transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_reid_model(device):
    print("[Re-ID] OSNet 가중치 로드 중...")
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    # Re-ID 전용 학습 가중치(MSMT17+Duke+CUHK)로 교체 — 파일 없으면 imagenet 유지
    _w = "/workspace/guseok/models/osnet_reid.pth"
    if os.path.exists(_w):
        _ck = torch.load(_w, map_location="cpu", weights_only=False)  # torchreid 공식 zoo 파일
        _sd = _ck.get("state_dict", _ck) if isinstance(_ck, dict) else _ck
        _sd = {(k[7:] if k.startswith("module.") else k): v for k, v in _sd.items()}
        _msd = model.state_dict()
        _sd = {k: v for k, v in _sd.items() if k in _msd and _msd[k].shape == v.shape}
        model.load_state_dict(_sd, strict=False)
        print(f"[Re-ID] Re-ID 학습 가중치 적용: {len(_sd)}개 레이어")
    model.eval().to(device)
    print("OSNet 로드 완료.")
    return model


def extract_reid_feature(model, cv2_image, device):
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = reid_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)

    return features.cpu().numpy().flatten()
