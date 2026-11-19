"""AI review material for an analysed project (instruction 2026-11-19 §6).

    python -m contour.review <project folder> [--out DIR] [--cuts ID,ID] [--no-images] [--max-images N]

The menu item 'AI 검수 자료 만들기' (project menu) runs the same thing. It reads what the analysis already saved
(project.json, each cut's tracks.npz, the matte / owner / clothes / head-part stores, face layer, matte record, the
QC results) and writes, under <project>/ai_review/<time>/ (or --out):
- review.json: per cut its id, name, timecodes, frame range, people, analysis state, saved QC, and findings, each
  with where (frames, timecodes, person), what it rests on, and what to do about it in DIAP;
- report.md: the same for a person to read, linking the pictures;
- img/: for the worst findings a picture of the frame (or, without the original, of the saved matte) with the
  tracked skeletons and who owns which pixel drawn on it.

Nothing is analysed again (no model is run), nothing in the project is changed, and nothing is sent anywhere; the
originals are not copied. A cut whose original is gone or whose stores are missing is reviewed as far as what is
saved allows, and says what could not be checked. Findings say where they come from: 'auto' (measured here from
the saved results), 'qc' (the saved render-before QC) or 'user' (what a person decided in DIAP: a status, a
checked or locked range, a joint correction, a QC issue set aside). An automatic finding inside a range the user
marked as right says so."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

FORMAT = "diap-ai-review"
VERSION = 1

LOW_MIN = 3                 # frames in a row under the confidence threshold before it is a finding
SPIKE_TORSO = 0.06          # a joint off the line of its neighbours by this share of the person's height (1 frame)
SPIKE_LIMB = 0.12
ID_JUMP = 0.5               # the person's box centre moving this many box heights in one frame (no picture cut)
MATTE_LOST = 0.03           # the person's owner pixels cover less than this share of their box ...
MATTE_MIN_BOX = 0.004       # ... for a box at least this share of the frame
MAX_IMG_CUT = 4
MAX_IMG = 150
IMG_W = 960

SEV_ORDER = {"error": 0, "warn": 1, "info": 2}
SEV_KO = {"error": "오류", "warn": "경고", "info": "참고"}
SRC_KO = {"auto": "자동 측정", "qc": "저장된 QC", "user": "사람 판정"}

KIND_KO = {
    "not_analysed": "분석 안 됨",
    "skipped_prescan": "빠른 검사로 건너뜀",
    "old_analysis": "이전 버전 분석",
    "tracks_missing": "추적 파일 없음",
    "tracks_mismatch": "추적 데이터와 원본 불일치",
    "persons_mismatch": "인물 목록 불일치",
    "lowconf": "추적 신뢰도 낮음",
    "missing_joints": "관절 누락",
    "jitter": "관절 튐",
    "id_jump": "인물 번호 바뀜 의심",
    "matte_missing": "인물 매트 없음",
    "matte_broken": "인물 매트 파일 손상",
    "matte_rvm": "SAM 매트 아님 (RVM)",
    "matte_lost": "매트에서 인물이 빠짐",
    "matte_unknown_owner": "없는 인물이 소유한 매트 픽셀",
    "matte_pending": "편집 후 매트 갱신 대기",
    "cloth_stale": "의상 마스크가 매트보다 오래됨",
    "cloth_missing": "의상 마스크 없음",
    "face_missing": "얼굴·목 마스크 없음",
    "parts_missing": "머리 부위 마스크 없음",
    "source_missing": "원본 연결 끊김",
    "lowconf_retouch": "추적 불확실 구간에 보정 적용",
}


# ---------------------------------------------------------------------- small helpers
def _ranges(mask: np.ndarray, gap: int = 2, min_len: int = 1, offset: int = 0) -> List[List[int]]:
    """[a, b) runs of True, joined across gaps up to `gap` frames, kept from `min_len` frames."""
    idx = np.nonzero(np.asarray(mask, bool))[0]
    out: List[List[int]] = []
    for t in idx:
        t = int(t) + offset
        if out and t <= out[-1][1] + gap:
            out[-1][1] = t + 1
        else:
            out.append([t, t + 1])
    return [r for r in out if r[1] - r[0] >= min_len]


def _overlap(r, rs) -> bool:
    return any(a <= r[1] - 1 and r[0] <= b for a, b in rs)


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(s)).strip("_")[:40] or "cut"


class _Ctx:
    def __init__(self, prj, out: Path, images: bool, max_images: int):
        self.prj, self.out, self.images, self.max_images = prj, out, images, max_images
        self.n_images = 0
        self.root = Path(prj.path)


def _tc(cut, t: int) -> str:
    try:
        return cut.info.tc_at(int(t))
    except Exception:
        return ""


def _rec_tc(prj, cut, t: int) -> str:
    """The timeline (record) timecode of cut frame t, for a cut that comes from an edited timeline."""
    tl, pt = cut.tl or {}, prj.timeline or {}
    if not tl or "rec_in" not in tl:
        return ""
    try:
        from .core.media import frames_to_tc
        return frames_to_tc(int(pt.get("seq_start", 0)) + int(tl["rec_in"]) + int(t), float(pt.get("fps") or
                                                                                          cut.info.fps),
                            bool(pt.get("drop", False)))
    except Exception:
        return ""


def _where(prj, cut, a: int, b: int) -> dict:
    """Frames [a, b) of the cut with their timecodes (the last frame's for the end)."""
    d = dict(frame_in=int(a), frame_out=int(b), tc_in=_tc(cut, a), tc_out=_tc(cut, max(a, b - 1)))
    r0, r1 = _rec_tc(prj, cut, a), _rec_tc(prj, cut, max(a, b - 1))
    if r0:
        d.update(rec_tc_in=r0, rec_tc_out=r1)
    return d


# ---------------------------------------------------------------------- what is saved for a cut
def _store_file(cache: Path, prefix: str) -> Optional[dict]:
    """A matte-layout store (<prefix>_<w>x<h>.u8) in the cut's cache, without creating anything."""
    if not cache.is_dir():
        return None
    # the newest one when the proxy size changed and an older store is left beside it
    for f in sorted(cache.glob(f"{prefix}_*x*.u8"), key=lambda f: -f.stat().st_mtime):
        m = re.match(rf"{prefix}_(\d+)x(\d+)\.u8$", f.name)
        if not m:
            continue
        w, h = int(m.group(1)), int(m.group(2))
        size = f.stat().st_size
        return dict(path=f, w=w, h=h, frames=size // max(1, w * h), whole=size % max(1, w * h) == 0)
    return None


def _open(info: Optional[dict]):
    if not info or not info["whole"] or info["frames"] <= 0:
        return None
    try:
        return np.memmap(info["path"], dtype=np.uint8, mode="r", shape=(info["frames"], info["h"], info["w"]))
    except Exception:
        return None


def _json(f: Path) -> dict:
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _caches(cut_dir: Path) -> dict:
    cache = cut_dir / "cache"
    out = dict(tracks=(cut_dir / "tracks.npz").exists())
    for k in ("matte", "seglab", "cloth", "parts"):
        out[k] = _store_file(cache, k)
    out["facelayer"] = (cache / "facelayer.npz").exists()
    out["seg_meta"] = _json(cache / "seg_meta.json") if (cache / "seg_meta.json").exists() else None
    cl = out["cloth"]
    out["cloth_state"] = _json(Path(cl["path"]).with_suffix(".json")) if cl else None
    out["alias"] = {int(k): int(v) for k, v in _json(cache / "seglab_alias.json").items()} \
        if (cache / "seglab_alias.json").exists() else {}
    return out


# ---------------------------------------------------------------------- findings
def _f(kind, severity, source, where=None, person=None, evidence="", suggest="", **extra) -> dict:
    d = dict(kind=kind, title=KIND_KO.get(kind, kind), severity=severity, source=source)
    if person is not None:
        d["person"] = int(person)
    if where:
        d.update(where)
    d["evidence"] = evidence
    d["suggest"] = suggest
    d.update(extra)
    return d


def _track_findings(prj, cut, st, wr) -> List[dict]:
    from .track import skeleton as S
    out: List[dict] = []
    a0, b0 = wr
    thr = float(getattr(cut.tracking, "conf_threshold", 0.5) or 0.5)
    torso = [S.L_SH, S.R_SH, S.L_HIP, S.R_HIP]
    limbs = [S.L_EL, S.R_EL, S.L_WR, S.R_WR, S.L_KN, S.R_KN, S.L_AN, S.R_AN]
    jump = np.asarray(getattr(st, "jump", np.zeros(st.n, bool)), bool)
    names = getattr(S, "JOINT_NAMES_KO", [str(j) for j in range(S.K)])
    for p in cut.persons:
        if not st.has(p.id):
            continue
        i = st.index(p.id)
        pres = np.asarray(st.present[i, a0:b0], bool)
        if not pres.any():
            continue
        conf = np.nan_to_num(st.conf[i, a0:b0], nan=0.0)
        fin = st.final[i, a0:b0, :, :2].astype(np.float64)
        sc = st.scale[i, a0:b0].astype(np.float64)
        sc = np.where(np.isfinite(sc), sc, np.nanmedian(sc) if np.isfinite(sc).any() else 400.0)
        # confidence under the threshold
        low = pres & (conf < thr)
        for a, b in _ranges(low, gap=2, min_len=LOW_MIN, offset=a0):
            seg = conf[a - a0:b - a0][pres[a - a0:b - a0]]
            worst = a + int(np.argmin(np.where(pres[a - a0:b - a0], conf[a - a0:b - a0], 9)))
            out.append(_f("lowconf", "warn", "auto", _where(prj, cut, a, b), p.id,
                          f"추적 신뢰도 평균 {float(seg.mean()):.2f} (기준 {thr:.2f}), {b - a}프레임",
                          "추적 탭 빨간 구간에서 관절을 확인하고 틀린 관절을 끌어 바로잡기. 많이 틀렸으면 다시 분석",
                          worst_frame=worst))
        # torso joints without any estimate while the person is there
        miss = pres[:, None] & ~np.isfinite(fin[:, torso, 0])
        mf = miss.any(1)
        for a, b in _ranges(mf, gap=2, min_len=2, offset=a0):
            js = sorted({names[torso[j]] for j in range(len(torso)) if miss[a - a0:b - a0, j].any()})
            out.append(_f("missing_joints", "warn", "auto", _where(prj, cut, a, b), p.id,
                          f"{', '.join(js)} 위치 추정 없음, {b - a}프레임",
                          "가려진 관절이면 우클릭으로 '안 보이는 관절'로 지정, 보이는데 빠졌으면 관절을 끌어 위치 지정",
                          worst_frame=a))
        # a joint off the line of the frames before and after (a one-frame spike)
        if len(pres) >= 3:
            mid = fin[1:-1] - 0.5 * (fin[:-2] + fin[2:])
            dev = np.hypot(mid[..., 0], mid[..., 1]) / sc[1:-1, None]
            ok3 = pres[:-2] & pres[1:-1] & pres[2:] & ~jump[a0:b0 - 2] & ~jump[a0 + 1:b0 - 1]
            # a spike: the frames before and after agree with each other, not a step (someone moving on fast, a
            # box that changed person) whose two sides differ
            span = fin[2:] - fin[:-2]
            dev = np.where(np.hypot(span[..., 0], span[..., 1]) / sc[1:-1, None] < dev, dev, np.nan)
            lim = np.full(S.K, np.inf)
            lim[torso] = SPIKE_TORSO
            lim[limbs] = SPIKE_LIMB
            bad = ok3[:, None] & np.isfinite(dev) & (dev > lim[None, :])
            bf = bad.any(1)
            for a, b in _ranges(bf, gap=1, min_len=1, offset=a0 + 1):
                sl = slice(a - a0 - 1, b - a0 - 1)
                k = int(np.nanargmax(np.where(bad[sl], dev[sl], -1).max(0)))
                worst = a + int(np.nanargmax(np.where(bad[sl], dev[sl], -1).max(1)))
                out.append(_f("jitter", "warn" if b - a >= 2 else "info", "auto", _where(prj, cut, a, b), p.id,
                              f"{names[k]} 등 {int(bad[sl].any(0).sum())}개 관절이 앞뒤 프레임 선에서 키의 "
                              f"{float(np.nanmax(np.where(bad[sl], dev[sl], np.nan))) * 100:.0f}%까지 벗어남",
                              "튀는 프레임의 관절을 끌어 바로잡기(키프레임). 넓게 떨리면 추적 탭에서 부드럽게 조정",
                              worst_frame=worst))
        # the box jumping (another person followed for a while)
        bx = st.boxes[i, a0:b0].astype(np.float64)
        cx, cy = 0.5 * (bx[:, 0] + bx[:, 2]), 0.5 * (bx[:, 1] + bx[:, 3])
        bh = np.maximum(1.0, bx[:, 3] - bx[:, 1])
        mv = np.hypot(np.diff(cx), np.diff(cy)) / np.minimum(bh[1:], bh[:-1])
        both = pres[1:] & pres[:-1] & np.isfinite(mv) & ~jump[a0:b0 - 1]
        hit = both & (mv > ID_JUMP)
        for a, b in _ranges(hit, gap=0, min_len=1, offset=a0 + 1):
            out.append(_f("id_jump", "warn", "auto", _where(prj, cut, a, b), p.id,
                          f"인물 상자 중심이 한 프레임에 상자 높이의 {float(np.nanmax(mv[a - a0 - 1:b - a0 - 1])):.1f}배 "
                          "이동 (화면 전환 없음)",
                          "추적 탭 > 인물 정리 > '번호 바로잡기…'로 맞바꾸거나 새 인물로 떼어 내기", worst_frame=a))
    return out


def _matte_findings(prj, cut, st, caches, wr) -> List[dict]:
    out: List[dict] = []
    a0, b0 = wr
    mt, sl = caches["matte"], caches["seglab"]
    if mt is None:
        out.append(_f("matte_missing", "warn", "auto", _where(prj, cut, a0, b0),
                      evidence="매트 파일(matte_*.u8)이 없음",
                      suggest="추적 탭 고급 > '인물 매트 강제 갱신' 또는 다시 분석"))
        return out
    for nm, info in (("matte", mt), ("seglab", sl)):
        if info is not None and (not info["whole"] or info["frames"] != st.n):
            out.append(_f("matte_broken", "error", "auto", _where(prj, cut, a0, b0),
                          evidence=f"{info['path'].name}: {info['frames']}프레임 분량 (추적 {st.n}프레임)",
                          suggest="추적 탭 고급 > '인물 매트 강제 갱신', 안 되면 다시 분석"))
            return out
    if sl is None:
        out.append(_f("matte_rvm", "info", "auto", _where(prj, cut, a0, b0),
                      evidence="인물별 소유 정보(seglab)가 없는 RVM 매트",
                      suggest="겹친 인물 보정이 번지면 추적 탭 고급 > '인물 매트 강제 갱신'"))
        return out
    lab = _open(sl)
    if lab is None:
        return out
    alias = caches["alias"]
    W, H = float(cut.info.width or 1), float(cut.info.height or 1)
    kx, ky = sl["w"] / W, sl["h"] / H
    frame_area = float(sl["w"] * sl["h"])
    known = {int(pid) for pid in st.pids} | set(alias)
    n = b0 - a0
    step = 1 if n <= 1500 else int(np.ceil(n / 1500))
    ts = list(range(a0, b0, step))
    P = len(st.pids)
    share = np.full((P, len(ts)), np.nan)
    unknown: Dict[int, int] = {}
    for k, t in enumerate(ts):
        cnt = np.bincount(np.asarray(lab[t]).ravel(), minlength=256)
        for v in np.nonzero(cnt[1:])[0]:
            if int(v) not in known:
                unknown[int(v)] = unknown.get(int(v), 0) + 1
        for p in range(P):
            if not st.present[p, t]:
                continue
            pid = int(st.pids[p])
            c = cnt[pid + 1] + sum(cnt[v + 1] for v, to in alias.items() if to == pid and v + 1 < 256)
            b = st.boxes[p, t]
            if not np.isfinite(b).all():
                continue
            area = max(1.0, (b[2] - b[0]) * kx * (b[3] - b[1]) * ky)
            if area / frame_area >= MATTE_MIN_BOX:
                share[p, k] = c / area
    for p in range(P):
        pid = int(st.pids[p])
        q = cut.person(pid)
        if q is None or q.still or q.hold:
            continue                    # a picture on the set, or a track the person finder never confirmed
        lost = np.isfinite(share[p]) & (share[p] < MATTE_LOST)
        for a, b in _ranges(lost, gap=1, min_len=max(1, 3 // step)):
            fa, fb = ts[a], min(b0, ts[b - 1] + step)
            out.append(_f("matte_lost", "warn", "auto", _where(prj, cut, fa, fb), pid,
                          f"인물 상자 안에서 이 인물 소유 픽셀이 {float(np.nanmin(share[p, a:b])) * 100:.1f}% 이하 "
                          f"(기준 {MATTE_LOST * 100:.0f}%)",
                          "렌더에서 보정이 빠지거나 옆 사람에게 번질 수 있음: 추적 확인 후 '인물 매트 강제 갱신'",
                          worst_frame=fa))
    if unknown:
        ids = ", ".join(str(v) for v in sorted(unknown))
        out.append(_f("matte_unknown_owner", "info", "auto", _where(prj, cut, a0, b0),
                      evidence=f"추적에 없는 인물 번호 {ids}가 매트 픽셀을 가짐 ({sum(unknown.values())}프레임 표본)",
                      suggest="지운 인물의 픽셀이면 보정에는 영향 없음. 겹친 인물 경계가 이상하면 '인물 매트 강제 갱신'"))
    return out


def _cache_findings(prj, cut, caches, wr) -> List[dict]:
    out: List[dict] = []
    a0, b0 = wr
    marks = (cut.status or {}).get("matte_dirty") or []
    for a, b in marks:
        out.append(_f("matte_pending", "warn", "auto", _where(prj, cut, max(a0, a), min(b0, b)),
                      evidence="관절·인물 편집 뒤 이 구간의 매트를 아직 다시 만들지 않음 (저장된 표시)",
                      suggest="컷을 열어 두면 자동으로 갱신됨. 바로 하려면 추적 탭 고급 > '인물 매트 강제 갱신'"))
    targets = [p for p in cut.persons if p.role == "target"]
    pp = [cut.params_for(p.id) for p in targets]
    wants_cloth = any(getattr(q, "cloth", "include") != "include" for q in pp)
    wants_face = any(q.retouch_active() for q in pp)
    cl = caches["cloth"]
    if wants_cloth and cl is None:
        out.append(_f("cloth_missing", "warn", "auto", _where(prj, cut, a0, b0),
                      evidence="몸매 보정이 옷을 따로 다루는데(옷 제외 설정) 의상 마스크가 없음",
                      suggest="몸매 탭 > '의상 마스크 다시 계산'"))
    if cl is not None and caches["seg_meta"]:
        made = float((caches["cloth_state"] or {}).get("made") or 0)
        later = [r[:2] for r in caches["seg_meta"].get("refreshed") or [] if len(r) > 2 and r[2] > made]
        if later:
            out.append(_f("cloth_stale", "warn" if wants_cloth else "info", "auto", _where(prj, cut, a0, b0),
                          evidence=f"의상 마스크를 만든 뒤 인물 매트가 {len(later)}곳 다시 만들어짐 "
                                   f"(첫 구간 {later[0][0]}~{later[0][1] - 1})",
                          suggest="몸매 탭 > '의상 마스크 다시 계산'"))
    if wants_face and not caches["facelayer"]:
        out.append(_f("face_missing", "info", "auto", _where(prj, cut, a0, b0),
                      evidence="피부 보정이 켜져 있는데 저장된 얼굴·목 마스크(facelayer)가 없음 (열 때 만들어짐)",
                      suggest="피부 탭 > '얼굴·목 마스크 다시 계산'"))
    hair = any(q._active(k) for q in pp for k in ("hair_volume", "head") if k in q.amounts)
    if hair and caches["parts"] is None:
        out.append(_f("parts_missing", "info", "auto", _where(prj, cut, a0, b0),
                      evidence="머리 크기·헤어 볼륨을 쓰는데 머리 부위 마스크가 없음",
                      suggest="몸매 탭 머리 그룹 > '머리 부위 인식'"))
    return out


def _qc(prj, cut) -> List[dict]:
    out = []
    try:
        from .qc.checks import KIND_TEXT
    except Exception:
        KIND_TEXT = {}
    for q in cut.qc:
        d = dict(kind="qc_" + q.kind, title=KIND_TEXT.get(q.kind, q.kind), severity=q.severity,
                 source="user" if q.status in ("ignored", "fixed") else "qc", status=q.status, value=q.value)
        if q.person >= 0:
            d["person"] = int(q.person)
        d.update(_where(prj, cut, q.start, q.end + 1))
        d["evidence"] = q.message
        d["suggest"] = {"ignored": "사용자가 무시로 표시함", "fixed": "사용자가 고친 것으로 표시함"}.get(
            q.status, "렌더 전에 QC 탭에서 확인 (보정값을 낮추거나 관절 확인)")
        out.append(d)
    return out


def _user(prj, cut) -> List[dict]:
    """What a person decided in DIAP for this cut."""
    out = []
    st = cut.status or {}
    if st.get("user"):
        out.append(dict(kind="status", source="user", text=f"컷 상태를 '{st['user']}'(으)로 직접 지정"))
    if st.get("reviewed"):
        out.append(dict(kind="reviewed", source="user", text="검수 완료로 표시"))
    if st.get("work"):
        out.append(dict(kind="work", source="user", text={"done": "작업 완료로 표시", "skip": "작업 불필요로 지정"}.get(
            st["work"], str(st["work"]))))
    for pid, rs in (cut.confirmed or {}).items():
        for a, b in rs:
            out.append(dict(kind="confirmed", source="user", person=int(pid), text="추적이 정확하다고 확정",
                            **_where(prj, cut, a, b + 1)))
    for pid, rs in (cut.locked or {}).items():
        for a, b in rs:
            out.append(dict(kind="locked", source="user", person=int(pid), text="다시 계산해도 바뀌지 않게 잠금",
                            **_where(prj, cut, a, b + 1)))
    for pid, jd in (cut.joint_keys or {}).items():
        fr = sorted({int(f) for d in jd.values() for f in d})
        if fr:
            out.append(dict(kind="joint_edits", source="user", person=int(pid),
                            text=f"관절 위치 직접 수정 {sum(len(d) for d in jd.values())}곳 ({len(fr)}프레임)",
                            frames=fr[:200]))
    for pid, jd in (cut.hidden or {}).items():
        n = sum(len(rs) for rs in jd.values())
        if n:
            out.append(dict(kind="hidden", source="user", person=int(pid),
                            text=f"안 보이는 관절 지정 {n}구간"))
    return out


# ---------------------------------------------------------------------- pictures
def _hex(c: str):
    c = (c or "#ffffff").lstrip("#")
    try:
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except Exception:
        r, g, b = 255, 255, 255
    return (b, g, r)


def _picture(ctx: _Ctx, cut, st, caches, frames, t: int, tag: str) -> Optional[str]:
    """The frame (or the saved matte when the original cannot be read) with skeletons and owners drawn on it."""
    import cv2
    from .track import skeleton as S
    if not ctx.images or ctx.n_images >= ctx.max_images:
        return None
    W, H = int(cut.info.width or 1920), int(cut.info.height or 1080)
    w = IMG_W
    h = max(2, int(round(H * w / W)))
    img = None
    if frames is not None:
        try:
            rgb = np.asarray(frames.frame(int(t)))
            img = cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    base = "원본" if img is not None else "저장된 매트"
    mt = _open(caches["matte"])
    if img is None:
        img = np.full((h, w, 3), 24, np.uint8)
        if mt is not None and t < mt.shape[0]:
            a = cv2.resize(np.asarray(mt[t]), (w, h), interpolation=cv2.INTER_LINEAR)
            img[:] = np.clip(24 + a[..., None].astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    lab = _open(caches["seglab"])
    if lab is not None and t < lab.shape[0]:
        L = cv2.resize(np.asarray(lab[t]), (w, h), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(img)
        for p in cut.persons:
            tint[L == (p.id + 1) % 256] = _hex(p.color)
        for v, to in caches["alias"].items():
            pr = cut.person(to)
            if pr is not None:
                tint[L == (v + 1) % 256] = _hex(pr.color)
        m = tint.any(2)
        img[m] = (0.62 * img[m] + 0.38 * tint[m]).astype(np.uint8)
    k = w / W
    for p in cut.persons:
        if not st.has(p.id):
            continue
        i = st.index(p.id)
        if not st.present[i, t]:
            continue
        col = _hex(p.color)
        kp = st.final[i, t, :, :2] * k
        for a_, b_ in S.BONES:
            if np.isfinite(kp[[a_, b_]]).all():
                cv2.line(img, tuple(int(v) for v in kp[a_]), tuple(int(v) for v in kp[b_]), col, 2, cv2.LINE_AA)
        for j in range(S.K):
            if np.isfinite(kp[j]).all():
                cv2.circle(img, tuple(int(v) for v in kp[j]), 3, col, -1, cv2.LINE_AA)
        b = st.boxes[i, t]
        if np.isfinite(b).all():
            x0, y0 = int(b[0] * k), int(b[1] * k)
            cv2.rectangle(img, (x0, y0), (int(b[2] * k), int(b[3] * k)), col, 1)
            cv2.putText(img, f"P{p.id}", (x0 + 3, max(14, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2,
                        cv2.LINE_AA)
    cap = f"{re.sub(r'[^ -~]', '', cut.name)[:28]}  f{t}  {_tc(cut, t)}  {tag}"
    cv2.rectangle(img, (0, h - 26), (w, h), (20, 20, 20), -1)
    cv2.putText(img, cap, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    d = ctx.out / "img"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{_slug(cut.name)}_{cut.id}_{tag}_{int(t):06d}.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    f.write_bytes(buf.tobytes())
    ctx.n_images += 1
    return _rel(f, ctx.out) + ("" if base == "원본" else "#matte")


# ---------------------------------------------------------------------- one cut
def review_cut(ctx: _Ctx, cut, number: str = "") -> dict:
    from .core.frames import source_present
    from .track.store import TrackStore
    prj = ctx.prj
    wr = tuple(int(v) for v in cut.work_range)
    cut_dir = ctx.root / "cuts" / cut.id               # read where it is: nothing created for a cut without one
    caches = _caches(cut_dir)
    src = str(cut.source or cut.info.path)
    found = source_present(cut.info)
    st_ = cut.status or {}
    from .track import status as STT
    d = dict(id=cut.id, name=cut.name, number=number, source=dict(path=src, found=bool(found)),
             fps=round(float(cut.info.fps), 3), size=[int(cut.info.width), int(cut.info.height)],
             frames=int(cut.info.frames), work=dict(**_where(prj, cut, *wr)),
             timeline=dict(index=cut.tl.get("index"), kind=cut.tl.get("kind"), name=cut.tl.get("name", ""))
             if cut.tl else None,
             analysis=dict(analyzed=bool(cut.analyzed), version=int(cut.analysis_version),
                           state=STT.effective(cut), auto=st_.get("auto"), reason=STT.reason(cut) or st_.get("reason", ""),
                           user=st_.get("user"), reviewed=bool(st_.get("reviewed")), work=st_.get("work"),
                           prescan=st_.get("prescan_info") or (dict(verdict="none") if st_.get("prescan") else None),
                           models=dict(cut.models_used or {}), colour=cut.cs_analysed or cut.input_cs or ""),
             persons=[], caches={}, findings=[], qc=_qc(prj, cut), user=_user(prj, cut), checked=[], unchecked=[])
    cs = dict(tracks=caches["tracks"])
    for k in ("matte", "seglab", "cloth", "parts"):
        v = caches[k]
        cs[k] = None if v is None else dict(file=_rel(Path(v["path"]), ctx.root), size=[v["w"], v["h"]],
                                            frames=v["frames"], whole=v["whole"])
    cs["facelayer"] = caches["facelayer"]
    cs["matte_model"] = (caches["seg_meta"] or {}).get("model") or ("" if caches["seglab"] else
                                                                     ("RVM" if caches["matte"] else ""))
    cs["matte_refreshed"] = len((caches["seg_meta"] or {}).get("refreshed") or [])
    cs["cloth_model"] = (caches["cloth_state"] or {}).get("model", "") if caches["cloth_state"] else ""
    d["caches"] = cs
    F = d["findings"]
    if not found:
        F.append(_f("source_missing", "warn", "auto", _where(prj, cut, *wr),
                    evidence=f"원본을 찾을 수 없음: {src}",
                    suggest="원본 위치를 다시 연결해야 렌더 가능. 이 검수는 저장된 추적·매트로만 확인함"))
        d["unchecked"].append("원본 화면 (그림은 저장된 매트 위에 그림)")
    if not cut.analyzed:
        if st_.get("prescan") or (st_.get("auto") == "none" and "빠른 검사" in str(st_.get("reason", ""))):
            F.append(_f("skipped_prescan", "info", "auto", _where(prj, cut, *wr),
                        evidence=str(st_.get("reason") or "빠른 검사에서 인물 없음"),
                        suggest="인물이 있는 컷이면 컷 리스트에서 '선택 컷 분석'"))
        elif not (cut.tl and cut.tl.get("kind") in ("gap", "transition")):
            F.append(_f("not_analysed", "info", "auto", _where(prj, cut, *wr), evidence="분석 기록 없음",
                        suggest="보정할 컷이면 '분석 시작'"))
        d["unchecked"].append("추적·매트 (분석 안 됨)")
        d["persons"] = [dict(id=p.id, name=p.name, role=p.role) for p in cut.persons]
        return _finish(ctx, d, cut, None, caches, wr)
    st = TrackStore.load(cut_dir / "tracks.npz") if caches["tracks"] else None
    if st is None:
        F.append(_f("tracks_missing", "error", "auto", _where(prj, cut, *wr),
                    evidence="분석한 컷인데 cuts/<id>/tracks.npz가 없거나 읽을 수 없음",
                    suggest="다시 분석 (보정값·키프레임은 프로젝트에 남아 있음)"))
        d["unchecked"].append("추적 (파일 없음)")
        d["persons"] = [dict(id=p.id, name=p.name, role=p.role) for p in cut.persons]
        return _finish(ctx, d, cut, None, caches, wr)
    d["checked"].append("추적 (tracks.npz)")
    if st.n != int(cut.info.frames):
        F.append(_f("tracks_mismatch", "error", "auto", _where(prj, cut, *wr),
                    evidence=f"추적 {st.n}프레임, 원본 {cut.info.frames}프레임",
                    suggest="원본이 바뀐 것으로 보임: 다시 분석"))
        d["persons"] = [dict(id=p.id, name=p.name, role=p.role) for p in cut.persons]
        return _finish(ctx, d, cut, None, caches, wr)
    wr = (max(0, wr[0]), min(st.n, wr[1]))
    in_store = {int(x) for x in st.pids}
    lost = [p.id for p in cut.persons if p.id not in in_store]
    extra = sorted(in_store - {p.id for p in cut.persons} - set(int(x) for x in cut.removed))
    if lost or extra:
        F.append(_f("persons_mismatch", "warn", "auto", _where(prj, cut, *wr),
                    evidence=(f"프로젝트에만 있는 인물 {lost}" if lost else "") +
                             (" · " if lost and extra else "") + (f"추적에만 있는 인물 {extra}" if extra else ""),
                    suggest="인물 목록이 추적과 어긋남: 다시 분석하거나 인물 합치기·삭제로 정리"))
    if not (st.has_vis and st.has_face):
        F.append(_f("old_analysis", "info", "auto", _where(prj, cut, *wr),
                    evidence="가려진 관절 판정 또는 얼굴 점이 없는 이전 버전 분석",
                    suggest="얼굴·피부 보정이나 가림 판정을 쓰려면 다시 분석"))
    for p in cut.persons:
        pe = dict(id=p.id, name=p.name, role=p.role, member=p.member or "")
        if st.has(p.id):
            i = st.index(p.id)
            pr = np.asarray(st.present[i, wr[0]:wr[1]], bool)
            fr = np.nonzero(pr)[0]
            pe.update(frames=[int(fr[0] + wr[0]), int(fr[-1] + wr[0]) + 1] if len(fr) else None,
                      present=int(pr.sum()),
                      conf=round(float(np.nanmean(np.where(pr, st.conf[i, wr[0]:wr[1]], np.nan))), 3)
                      if pr.any() else None)
        pe["retouch"] = bool(cut.params_for(p.id).any_active()) if p.role == "target" else False
        d["persons"].append(pe)
    try:
        F.extend(_track_findings(prj, cut, st, wr))
    except Exception as e:
        d["unchecked"].append(f"추적 측정 실패: {e}")
    try:
        F.extend(_matte_findings(prj, cut, st, caches, wr))
        d["checked"].append("인물 매트·소유" if caches["seglab"] else "인물 매트")
    except Exception as e:
        d["unchecked"].append(f"매트 측정 실패: {e}")
    try:
        F.extend(_cache_findings(prj, cut, caches, wr))
        d["checked"].append("얼굴·의상·머리 마스크 유무")
    except Exception as e:
        d["unchecked"].append(f"마스크 확인 실패: {e}")
    # retouch on frames where the tracking is unsure: what the render would show
    tgt = {p.id for p in cut.persons if p.role == "target" and cut.params_for(p.id).any_active()}
    for f in [x for x in F if x["kind"] == "lowconf" and x.get("person") in tgt]:
        f["render_risk"] = True
    if cut.qc:
        d["checked"].append("저장된 QC")
    else:
        d["unchecked"].append("QC (실행한 기록 없음)")
    return _finish(ctx, d, cut, st, caches, wr)


def _finish(ctx: _Ctx, d: dict, cut, st, caches, wr) -> dict:
    """Ranges the user marked right noted on the automatic findings, order, pictures."""
    conf = {}
    for pid, rs in (cut.confirmed or {}).items():
        conf[int(pid)] = [[a, b] for a, b in rs]
    for f in d["findings"]:
        if f.get("person") in conf and "frame_in" in f and _overlap([f["frame_in"], f["frame_out"]],
                                                                      conf[f["person"]]):
            f["inside_user_confirmed"] = True
            f["note"] = "사용자가 이 구간의 추적을 정확하다고 확정함: 측정 기준이 과민했을 수 있음"
    d["findings"].sort(key=lambda f: (SEV_ORDER.get(f["severity"], 3), f.get("frame_in", 0)))
    for i, f in enumerate(d["findings"]):
        f["id"] = f"{cut.id}-{i + 1:03d}"
    if st is not None and ctx.images:
        frames = None
        cache = ctx.root / "cuts" / cut.id / "cache"
        link = cache / "proxy_shared.json"
        dead_link = link.exists() and not Path(str(_json(link).get("path", ""))).exists()
        if cache.is_dir() and not dead_link:
            # the original (or its proxy): read only. A dead link to a bake proxy would be cleaned up on the way,
            # so then the pictures are drawn on the saved matte instead
            try:
                from .core.frames import frames_for
                frames = frames_for(ctx.prj, cut)
            except Exception:
                frames = None
        try:
            done = 0
            for f in d["findings"]:
                if done >= MAX_IMG_CUT:
                    break
                t = f.get("worst_frame")
                if t is None or f["severity"] == "info" or not (0 <= int(t) < st.n):
                    continue
                pic = _picture(ctx, cut, st, caches, frames, int(t), f["kind"])
                if pic:
                    f["image"], _, how = pic.partition("#")
                    if how:
                        f["image_base"] = "saved_matte"
                    done += 1
        finally:
            if frames is not None:
                try:
                    frames.close()
                except Exception:
                    pass
    n = {k: sum(1 for f in d["findings"] if f["severity"] == k) for k in SEV_ORDER}
    d["counts"] = dict(**n, qc_open=sum(1 for q in d["qc"] if q.get("status") == "open"))
    return d


# ---------------------------------------------------------------------- the whole project
def build(project_path: str, out_dir: Optional[str] = None, cut_ids: Optional[Sequence[str]] = None,
          images: bool = True, max_images: int = MAX_IMG,
          progress: Optional[Callable[[float, str], None]] = None,
          cancel: Optional[Callable[[], bool]] = None) -> Path:
    """Writes the review; returns its folder."""
    from .core.project import Project
    from .core.settings import APP_VERSION
    prj = Project.load(project_path)
    root = Path(prj.path)
    out = Path(out_dir) if out_dir else root / "ai_review" / time.strftime("%Y%m%d_%H%M%S")
    k = 2
    base = out
    while out.exists() and any(out.iterdir()):
        out = base.with_name(f"{base.name}_{k}")
        k += 1
    out.mkdir(parents=True, exist_ok=True)
    ctx = _Ctx(prj, out, images, max_images)
    cuts = list(prj.timeline_cuts()) if prj.timeline else []
    cuts += [c for c in prj.cuts if c not in cuts]
    if cut_ids:
        want = set(cut_ids)
        cuts = [c for c in cuts if c.id in want or c.name in want]
    t0 = time.time()
    rev = dict(format=FORMAT, version=VERSION, app=APP_VERSION, made=time.strftime("%Y-%m-%dT%H:%M:%S"),
               models_run=False, project_changed=False,
               project=dict(name=prj.name, folder=str(root), cuts=len(prj.cuts),
                            timeline=dict(name=prj.timeline.get("name", ""), fps=prj.timeline.get("fps"),
                                          start_tc=prj.timeline.get("start_tc", "")) if prj.timeline else None),
               conventions=dict(frames="컷 안의 프레임 번호 (0부터). frame_out은 포함하지 않음",
                                tc="tc_*: 원본 타임코드, rec_tc_*: 편집 타임라인 타임코드 (tc_out은 마지막 프레임)",
                                source="auto: 저장된 결과로 이 도구가 측정, qc: 저장된 렌더 전 QC, user: 사람이 DIAP에서 "
                                       "판정·수정한 내용",
                                images="img/ 아래, 이 폴더 기준 상대 경로. image_base=saved_matte는 원본 대신 저장된 매트 위에 그림"),
               cuts=[])
    for i, c in enumerate(cuts):
        if cancel and cancel():
            break
        if progress:
            progress(i / max(1, len(cuts)), f"검수 자료 {i + 1}/{len(cuts)}: {c.name}")
        num = f"{c.tl['index'] + 1:03d}" if c.tl and "index" in c.tl else f"{prj.cuts.index(c) + 1:03d}"
        try:
            rev["cuts"].append(review_cut(ctx, c, num))
        except Exception as e:
            rev["cuts"].append(dict(id=c.id, name=c.name, number=num, error=f"검수 실패: {e}", findings=[], qc=[],
                                    user=[], counts={}))
    rev["seconds"] = round(time.time() - t0, 1)
    rev["summary"] = _summary(rev)
    (out / "review.json").write_text(json.dumps(rev, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(report(rev), encoding="utf-8")
    if progress:
        progress(1.0, "검수 자료 완료")
    return out


def _summary(rev: dict) -> dict:
    cs = rev["cuts"]
    by_kind: Dict[str, int] = {}
    for c in cs:
        for f in c.get("findings", []):
            by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    return dict(cuts=len(cs), analysed=sum(1 for c in cs if (c.get("analysis") or {}).get("analyzed")),
                source_missing=sum(1 for c in cs if not (c.get("source") or {}).get("found", True)),
                errors=sum((c.get("counts") or {}).get("error", 0) for c in cs),
                warnings=sum((c.get("counts") or {}).get("warn", 0) for c in cs),
                qc_open=sum((c.get("counts") or {}).get("qc_open", 0) for c in cs), by_kind=by_kind)


def report(rev: dict) -> str:
    s = rev["summary"]
    L = [f"# AI 검수 자료: {rev['project']['name']}", "",
         f"- 만든 시각 {rev['made']} · DIAP {rev['app']} · {rev['seconds']}초",
         f"- 컷 {s['cuts']}개 (분석 {s['analysed']}) · 오류 {s['errors']} · 경고 {s['warnings']} · 열린 QC {s['qc_open']}"
         + (f" · 원본 끊김 {s['source_missing']}컷" if s["source_missing"] else ""),
         "- 모델을 다시 돌리지 않았고 프로젝트 파일은 바꾸지 않았습니다. 저장된 추적·매트·QC만 읽었습니다.",
         "- 출처: **자동 측정**은 이 도구가 저장된 결과로 잰 의심, **저장된 QC**는 DIAP의 렌더 전 검사, "
         "**사람 판정**은 사용자가 DIAP에서 지정·수정한 내용입니다.",
         "- 프레임은 컷 안의 번호(0부터), 구간 끝은 포함하지 않습니다. TC는 원본 타임코드, 편집 TC는 타임라인 기준입니다.",
         "", "## 컷 요약", "", "| 번호 | 컷 | 원본 TC | 상태 | 오류 | 경고 | 열린 QC |", "|---|---|---|---|---|---|---|"]
    for c in rev["cuts"]:
        a = c.get("analysis") or {}
        w = c.get("work") or {}
        n = c.get("counts") or {}
        from .track.status import LABEL as _SL
        L.append(f"| {c.get('number', '')} | {c['name']} | {w.get('tc_in', '')}–{w.get('tc_out', '')} | "
                 f"{_SL.get(a.get('state', ''), a.get('state', '')) if not c.get('error') else '검수 실패'} | {n.get('error', 0)} | "
                 f"{n.get('warn', 0)} | {n.get('qc_open', 0)} |")
    for c in rev["cuts"]:
        L += ["", f"## {c.get('number', '')} {c['name']} (`{c['id']}`)", ""]
        if c.get("error"):
            L.append(f"- {c['error']}")
            continue
        a, w = c["analysis"], c["work"]
        from .track.status import LABEL as _SL
        L.append(f"- 작업 구간 {w['frame_in']}~{w['frame_out'] - 1} · TC {w['tc_in']}–{w['tc_out']}"
                 + (f" · 편집 TC {w['rec_tc_in']}–{w['rec_tc_out']}" if w.get("rec_tc_in") else ""))
        L.append(f"- 분석 {'됨' if a['analyzed'] else '안 됨'} · 상태 {_SL.get(a['state'], a['state'])}"
                 + (f" ({a['reason']})" if a.get("reason") else "")
                 + f" · 원본 {'있음' if c['source']['found'] else '없음'}"
                 + (f" · 매트 {c['caches'].get('matte_model') or '-'}" if a["analyzed"] else ""))
        if c["persons"]:
            L.append("- 인물: " + ", ".join(
                f"P{p['id']} {p['name']}({'보정' if p['role'] == 'target' else '제외'}"
                + (f", {p['frames'][0]}~{p['frames'][1] - 1}" if p.get("frames") else "") + ")" for p in c["persons"]))
        if c.get("unchecked"):
            L.append("- 확인하지 못한 항목: " + "; ".join(c["unchecked"]))
        if c["findings"]:
            L += ["", "| ID | 구분 | 내용 | 인물 | 구간 | TC | 근거 | 제안 |", "|---|---|---|---|---|---|---|---|"]
            for f in c["findings"]:
                rng = "" if "frame_in" not in f else (str(f["frame_in"]) if f["frame_out"] - f["frame_in"] <= 1 else
                                                      f"{f['frame_in']}~{f['frame_out'] - 1}")
                img = f" [그림]({f['image']})" if f.get("image") else ""
                note = f" ({f['note']})" if f.get("note") else ""
                risk = " · 렌더 위험" if f.get("render_risk") else ""
                L.append(f"| {f['id']} | {SEV_KO.get(f['severity'], f['severity'])}·{SRC_KO[f['source']]} | "
                         f"{f['title']}{risk} | {('P' + str(f['person'])) if 'person' in f else ''} | {rng} | "
                         f"{f.get('tc_in', '')} | {f['evidence']}{note}{img} | {f['suggest']} |")
        if c["qc"]:
            L += ["", "저장된 QC:", ""]
            for q in c["qc"]:
                L.append(f"- [{SRC_KO[q['source']]}·{q['status']}] {q['title']} "
                         f"{q['frame_in']}~{q['frame_out'] - 1} ({q['tc_in']}){(' P' + str(q['person'])) if 'person' in q else ''}"
                         f": {q['evidence']}")
        if c["user"]:
            L += ["", "사람이 확인한 내용:", ""]
            for u in c["user"]:
                r = f" {u['frame_in']}~{u['frame_out'] - 1}" if "frame_in" in u else ""
                L.append(f"- {('P' + str(u['person']) + ' ') if 'person' in u else ''}{u['text']}{r}")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m contour.review",
                                 description="분석된 DIAP 프로젝트의 AI 검수 자료 (모델 재실행 없음, 프로젝트 수정 없음)")
    ap.add_argument("project", help="프로젝트 폴더 (.dpprj) 또는 그 안의 project.json")
    ap.add_argument("--out", help="출력 폴더 (기본: <프로젝트>/ai_review/<시각>)")
    ap.add_argument("--cuts", help="이 컷만 (id 또는 이름, 쉼표로 구분)")
    ap.add_argument("--no-images", action="store_true", help="그림 없이 보고서와 JSON만")
    ap.add_argument("--max-images", type=int, default=MAX_IMG)
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    out = build(a.project, a.out, [x.strip() for x in a.cuts.split(",")] if a.cuts else None,
                images=not a.no_images, max_images=a.max_images,
                progress=lambda f, m: print(f"{f * 100:5.1f}% {m}", flush=True))
    print(f"OUT {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
