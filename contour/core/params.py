"""Every adjustable value in one table: label, group, range, unit and what kind of processing it drives.

kinds
  body   skeleton driven width changes (segments)
  shape  vertical stretches and whole-body ratios
  face   face landmark driven shape changes
  head   head / hair silhouette driven changes
  skin   skin retouch (smoothing, tone, grain, wrinkles)
  color  local colour work (lips, teeth, eye whites, knees)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ParamDef:
    key: str
    label: str
    group: str
    lo: float
    hi: float
    default: float = 0.0
    unit: str = "%"          # % (signed) | pct (unsigned %) | deg
    kind: str = "body"
    tip: str = ""


GROUPS: List[Tuple[str, str, str]] = [
    # key, title, hint shown at the right of the header
    ("skin", "피부", ""),
    ("body", "몸", "− 가늘게  /  + 넓게"),
    ("shape", "키 · 비율", "+ 길게 / 넓게"),
    ("face", "얼굴형", "− 작게 / 짧게"),
    ("features", "이목구비", "− 작게  /  + 크게"),
    ("head", "머리", "+ 크게 / 높게"),
    ("color", "메이크업 · 색", ""),
]

PARAMS: List[ParamDef] = [
    # ---- body
    ParamDef("waist", "허리", "body", -50, 50, tip="허리 둘레. 배경은 몸 가까이에서만 움직입니다"),
    ParamDef("upper_arm", "위팔", "body", -50, 50, tip="어깨 아래부터 팔꿈치까지. 어깨선은 그대로 둡니다"),
    ParamDef("forearm", "아래팔", "body", -50, 50, tip="팔꿈치부터 손목까지. 손은 그대로 둡니다"),
    ParamDef("shoulders", "어깨", "body", -50, 50, tip="어깨 폭"),
    ParamDef("thighs", "허벅지", "body", -50, 50),
    ParamDef("calves", "종아리", "body", -50, 50),
    ParamDef("legs", "다리 전체", "body", -50, 50, tip="허벅지부터 발목까지 한 번에 (허벅지·종아리 값에 더해집니다)"),
    ParamDef("knee", "무릎", "body", -50, 30, tip="− 쪽으로 튀어나온 무릎 윤곽을 줄입니다"),
    ParamDef("neck", "목", "body", -50, 50, tip="목 두께. 턱과 얼굴은 움직이지 않습니다"),
    ParamDef("shoulder_line", "직각 어깨", "body", -30, 40, tip="+ 쪽이 처진 어깨 끝을 올려 어깨선을 반듯하게"),
    ParamDef("pelvis", "골반", "body", -30, 30, tip="+ 쪽이 골반(허리 아래 엉덩이 옆선)을 넓게"),
    ParamDef("hip_up", "힙업", "body", -20, 40, tip="+ 쪽이 엉덩이 아래 선을 위로 (뒷모습·옆모습에서 잘 보입니다)"),
    ParamDef("arm_under", "팔뚝살", "body", -40, 20, tip="− 쪽이 위팔 아래쪽(처지는 쪽) 살만 줄입니다. 위쪽 윤곽은 그대로"),
    ParamDef("belly", "뱃살", "body", -40, 20, tip="− 쪽이 배 앞쪽 윤곽을 넣습니다. 옆모습일수록 크게, 정면에서는 거의 없음"),
    ParamDef("body_sculpt", "몸 점 편집", "body", 0, 150, default=100, unit="pct",
             tip="몸 윤곽 점을 직접 옮겨 만든 모양의 강도 (편집 버튼으로 점 이동)"),
    # ---- length / ratio
    ParamDef("height", "키", "shape", -5, 15, kind="shape",
             tip="목 아래부터 발목까지 세로로 늘립니다. 배경은 가로줄이 휘지 않게 띠 전체가 고르게 늘어납니다"),
    ParamDef("leg_length", "다리 길이", "shape", -10, 15, kind="shape", tip="골반부터 발목까지만 늘립니다"),
    ParamDef("body_width", "몸 가로 비율", "shape", -20, 20, kind="shape", tip="목 아래 몸 전체의 가로 폭 (머리는 그대로)"),
    # ---- face shape
    ParamDef("face", "얼굴 크기", "face", -30, 30, kind="face",
             tip="얼굴만 (머리카락과 배경은 거의 그대로). 머리카락까지는 '머리 크기'"),
    ParamDef("face_length", "얼굴 길이", "face", -20, 20, kind="face", tip="턱 끝 위치로 얼굴 세로 길이 조절"),
    ParamDef("lower_face", "하관", "face", -25, 25, kind="face", tip="코 밑부터 턱 끝까지의 길이"),
    ParamDef("jaw", "턱선", "face", -40, 40, kind="face", tip="광대 아래 턱선을 둥글게 안쪽(−) 또는 바깥쪽(+)으로"),
    ParamDef("jaw_fix", "턱선 교정", "face", 0, 100, unit="pct", kind="face",
             tip="각진 턱과 좌우 비대칭을 매끈한 곡선으로 다듬습니다"),
    ParamDef("cheekbone", "광대", "face", -40, 40, kind="face", tip="− 쪽이 옆으로 나온 광대를 안으로 (턱선은 그대로)"),
    ParamDef("chin", "턱 끝 길이", "face", -30, 30, kind="face", tip="+ 쪽이 턱 끝만 아래로 (얼굴 다른 곳은 그대로)"),
    ParamDef("chin_width", "턱 끝 폭", "face", -40, 40, kind="face", tip="− 쪽이 턱 끝을 좁고 갸름하게 (V라인)"),
    # ---- features
    ParamDef("eyes", "눈 크기", "features", -30, 40, kind="face", tip="눈썹·코는 그대로"),
    ParamDef("eye_height", "눈 세로", "features", -30, 40, kind="face", tip="눈을 세로로만 키우거나 줄입니다"),
    ParamDef("eye_dist", "눈 사이", "features", -30, 30, kind="face", tip="두 눈 사이 거리. + 쪽이 넓게"),
    ParamDef("eye_tilt", "눈꼬리", "features", -30, 30, kind="face", tip="+ 쪽이 눈꼬리를 올리고, − 쪽이 내립니다"),
    ParamDef("nose", "코 크기", "features", -30, 30, kind="face"),
    ParamDef("nose_bridge", "콧대 폭", "features", -40, 40, kind="face", tip="− 쪽이 콧대를 가늘게"),
    ParamDef("mouth", "입 크기", "features", -30, 30, kind="face"),
    ParamDef("mouth_corner", "입꼬리", "features", -30, 40, kind="face", tip="+ 쪽이 입꼬리를 올립니다"),
    ParamDef("philtrum", "인중", "features", -40, 30, kind="face", tip="− 쪽이 인중을 짧게 (입이 코 쪽으로, 턱 끝은 그대로)"),
    ParamDef("sculpt", "세부 성형", "features", 0, 150, default=100, unit="pct", kind="face",
             tip="얼굴 점을 직접 옮겨 만든 모양의 강도 (편집 버튼으로 점 이동)"),
    # ---- head
    ParamDef("head", "머리 크기", "head", -25, 25, kind="head", tip="머리카락까지 포함한 머리 전체"),
    ParamDef("crown", "정수리", "head", -30, 30, kind="head", tip="+ 쪽이 정수리를 위로"),
    ParamDef("hair_volume", "헤어 볼륨", "head", -30, 40, kind="head", tip="얼굴은 그대로 두고 머리카락 부피만"),
    # ---- skin
    ParamDef("skin", "피부결", "skin", 0, 100, unit="pct", kind="skin",
             tip="피부결을 매끈하게. 눈·눈썹·입술·콧구멍·머리카락은 건드리지 않습니다"),
    ParamDef("nasolabial", "팔자주름", "skin", 0, 100, unit="pct", kind="skin"),
    ParamDef("mouth_lines", "입가주름", "skin", 0, 100, unit="pct", kind="skin"),
    ParamDef("blemish", "잡티", "skin", 0, 100, unit="pct", kind="skin",
             tip="점·트러블·뾰루지처럼 주변보다 어둡거나 붉은 작은 점을 주변 피부로 메웁니다. 크기는 '잡티 크기'로"),
    ParamDef("dark_circle", "다크서클", "skin", 0, 100, unit="pct", kind="skin",
             tip="눈 밑 어둡고 푸른 그늘을 볼 피부 밝기·색으로 끌어올립니다"),
    ParamDef("shine", "번들거림", "skin", 0, 100, unit="pct", kind="skin",
             tip="이마·코·볼의 하얗게 번들거리는 유분 광만 눌러 줍니다"),
    ParamDef("redness", "홍조", "skin", 0, 100, unit="pct", kind="skin",
             tip="볼·코 주변의 붉은 기를 얼굴 피부톤으로 가라앉힙니다 (입술은 그대로)"),
    ParamDef("neck_lines", "목주름", "skin", 0, 100, unit="pct", kind="skin",
             tip="목의 가로 주름 골만 주변 피부로 채웁니다 (목 그늘과 결은 그대로)"),
    ParamDef("neck_bumps", "목 닭살", "skin", 0, 100, unit="pct", kind="skin",
             tip="목의 오돌토돌한 닭살 결을 부드럽게 눌러 줍니다. 최대로 올려도 결은 남고, 목걸이·옷깃·목 그늘은 그대로"),
    ParamDef("skin_bright", "피부 밝기", "skin", -30, 30, kind="skin"),
    ParamDef("skin_warm", "피부 톤", "skin", -30, 30, kind="skin", tip="− 차갑게 / + 따뜻하게"),
    ParamDef("skin_sat", "피부 채도", "skin", -40, 40, kind="skin"),
    ParamDef("grain", "그레인", "skin", 0, 100, unit="pct", kind="skin", tip="피부 영역에 필름 그레인 (매끈해진 피부에 질감)"),
    ParamDef("grain_color", "색상 그레인", "skin", 0, 100, unit="pct", kind="skin", tip="0이면 모노크롬 그레인"),
    # ---- colour
    ParamDef("lip_hue", "입술 색조", "color", -40, 40, unit="deg", kind="color", tip="− 쪽이 핑크/보라, + 쪽이 코랄/오렌지"),
    ParamDef("lip_sat", "입술 채도", "color", -60, 60, kind="color"),
    ParamDef("lip_bright", "입술 밝기", "color", -40, 40, kind="color"),
    ParamDef("blush", "볼터치", "color", 0, 100, unit="pct", kind="color", tip="광대 앞쪽 볼에 블러셔"),
    ParamDef("blush_hue", "볼터치 색", "color", -40, 40, unit="deg", kind="color", tip="− 쪽이 핑크/로즈, + 쪽이 코랄/피치"),
    ParamDef("shading", "쉐딩", "color", 0, 100, unit="pct", kind="color", tip="콧대 옆, 광대 아래, 턱선 안쪽을 어둡게 (윤곽)"),
    ParamDef("highlight", "하이라이트", "color", 0, 100, unit="pct", kind="color", tip="콧대, 광대 위, 이마 가운데, 턱 끝을 밝게"),
    ParamDef("teeth", "치아 미백", "color", 0, 100, unit="pct", kind="color"),
    ParamDef("eye_white", "눈 흰자", "color", 0, 100, unit="pct", kind="color", tip="실핏줄과 누런 기를 걷어 깨끗한 흰색으로"),
    ParamDef("knee_dark", "무릎 어두움", "color", 0, 100, unit="pct", kind="color", tip="무릎의 어둡고 붉은 부분을 주변 피부톤에 맞춥니다"),
]

PARAM: Dict[str, ParamDef] = {p.key: p for p in PARAMS}
ALL_KEYS: List[str] = [p.key for p in PARAMS]
BODY_KEYS = [p.key for p in PARAMS if p.kind == "body"]
SHAPE_KEYS = [p.key for p in PARAMS if p.kind == "shape"]
FACE_KEYS = [p.key for p in PARAMS if p.kind == "face"]
HEAD_KEYS = [p.key for p in PARAMS if p.kind == "head"]
GEOMETRY_KEYS = BODY_KEYS + SHAPE_KEYS + FACE_KEYS + HEAD_KEYS     # everything that moves pixels
RETOUCH_KEYS = [p.key for p in PARAMS if p.kind in ("skin", "color")]
# body parts that have a measured width (range multiplier)
WIDTH_KEYS = ["waist", "upper_arm", "forearm", "thighs", "calves", "knee", "neck", "pelvis"]

STRENGTH_KEY = "__strength__"


def default(key: str) -> float:
    return PARAM[key].default if key in PARAM else 0.0


def is_neutral(key: str, v: float) -> bool:
    return abs(v - default(key)) < 1e-6


# --------------------------------------------------------------------------- skin: basic adjustment (a preset of the detail sliders)
# 표준 is the engine's reference: skin 70 takes the texture down as much as the hand retouch the engine was fitted
# to (cut 069, skin.FS_REF); 약하게 is about two thirds of it, 강하게 stops at 90 (at 100 the pores are nearly gone).
# The wrinkles and spots each have their own 보호 (0) / 제거 (the level's value) switch.
SKIN_LEVELS: List[Tuple[str, str]] = [("light", "약하게"), ("std", "표준"), ("strong", "강하게")]
SKIN_LEVEL_VALUES: Dict[str, Dict[str, float]] = {
    "light": {"skin": 45, "shine": 0, "dark_circle": 15, "redness": 0, "neck_lines": 15, "neck_bumps": 0},
    "std": {"skin": 70, "shine": 15, "dark_circle": 30, "redness": 15, "neck_lines": 30, "neck_bumps": 20},
    "strong": {"skin": 90, "shine": 30, "dark_circle": 50, "redness": 30, "neck_lines": 50, "neck_bumps": 40},
}
# switch -> (detail slider, label, value at light / std / strong when 제거)
SKIN_REMOVE: List[Tuple[str, str, str]] = [("mouth", "mouth_lines", "입가주름"), ("naso", "nasolabial", "팔자주름"),
                                           ("acne", "blemish", "여드름")]
SKIN_REMOVE_VALUES: Dict[str, float] = {"light": 25, "std": 40, "strong": 60}
SKIN_BASIC_KEYS: List[str] = list(SKIN_LEVEL_VALUES["std"]) + [k for _, k, _ in SKIN_REMOVE]


def skin_basic_values(level: str, remove) -> Dict[str, float]:
    """The detail slider values of a basic adjustment: level (light | std | strong) and the switches set to 제거."""
    level = level if level in SKIN_LEVEL_VALUES else "std"
    out = {k: float(v) for k, v in SKIN_LEVEL_VALUES[level].items()}
    rm = set(remove or ())
    for sw, key, _ in SKIN_REMOVE:
        out[key] = float(SKIN_REMOVE_VALUES[level]) if sw in rm else 0.0
    return out
