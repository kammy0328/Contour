"""Analysis orchestration for one cut and the edit-time re-solve logic."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import skeleton as S
from . import solver as SV
from .adapt import bias_offsets, learn_bias, template_measurements
from .assoc import appearance, assign, fill_gaps, iou, track_people
from .exposure import analysis_view
from .people import detect_people, load_identity_models
from .camera import estimate_all
from .store import TrackStore
from ..core.project import PERSON_COLORS, Cut, Person, Project
from ..core.proxy import MatteStore, Proxy
from ..warp import body as B
from .modelpkg import cut_models as _cut_models

_LABEL_LOCK = threading.RLock()        # CutTracker.labels: what it lets go of from its small cache, one at a time

Progress = Optional[Callable[[float, str], None]]


class Cancelled(Exception):
    pass


RECOVER = True        # carry a person through missed detections while tracking (people standing clear only)
FILL_GAPS = True      # after linking, fill the remaining gaps by following the pose
# people whose face is out of focus through the whole shot (track/focus.py; 2611 IWALY c057): no close-up matte, no face flow
# and face layer, protected. Environment IWALY_DEFOCUS_SKIP=0 switches it off
DEFOCUS_SKIP = __import__("os").environ.get("IWALY_DEFOCUS_SKIP", "1") not in ("0", "", "off")


class _Recover:
    """Pose estimation on a box where a person should be (the ensemble, as in pass 1)."""

    def __init__(self, proxy, poses, a: int):
        self.proxy, self.poses, self.a = proxy, poses, a

    def local(self, t: int, boxes: np.ndarray):
        return self(t + self.a, boxes)

    def __call__(self, t: int, boxes: np.ndarray):
        rgb = analysis_view(np.asarray(self.proxy.frame(t)))
        H, W = rgb.shape[:2]
        per_model = {k: pe(rgb, boxes) for k, pe in self.poses.items()}
        out = []
        for i, bx in enumerate(boxes):
            area = max(1.0, float((bx[2] - bx[0]) * (bx[3] - bx[1])))
            inside = max(0.0, min(W, bx[2]) - max(0, bx[0])) * max(0.0, min(H, bx[3]) - max(0, bx[1])) / area
            if inside < 0.6:
                out.append(None)
                continue
            avg = np.stack([np.column_stack([kp[i, :17], sc[i, :17]]) for kp, sc in per_model.values()]).mean(0)
            out.append(dict(kp=avg, app=appearance(rgb, avg, bx),
                            raw={k: (kp[i].copy(), sc[i].copy()) for k, (kp, sc) in per_model.items()}))
        return out


def _identity(tr: dict, dets) -> tuple:
    """Body re-ID mean over frames where the person stood clear of others, and up to 6 of the clearest faces."""
    from .assoc import _isolated
    feats, faces, crowded = [], [], []
    for f, i in tr.items():
        d = dets[f][i]
        r = d.get("reid")
        if r is not None and r.any():
            (feats if _isolated(dets[f])[i] else crowded).append(r)
        fc = d.get("face")
        if fc is not None:
            faces.append((fc["score"] * fc["w"], fc["feat"]))
    reid = []
    feats = feats or crowded          # never clear of others: the crowded frames are all there is
    if feats:
        m = np.mean(feats, 0)
        reid = [round(float(v), 4) for v in m / (np.linalg.norm(m) + 1e-9)]
    faces.sort(key=lambda x: -x[0])
    picked = []
    for _, fe in faces:
        # keep them different from each other (other angles, other light)
        if all(float(fe @ q) < 0.85 for q in picked):
            picked.append(fe)
        if len(picked) >= 6:
            break
    return reid, [[round(float(v), 4) for v in fe] for fe in picked]


def _masked_pose(rgb: np.ndarray, rec: dict, poses: Dict[str, object], dbg: Optional[dict] = None,
                 upright: bool = False) -> Optional[dict]:
    """Pose of the person a finder instance holds, with everybody else hidden (grey): a pose model given a box over
    two people takes the more visible one; here it can see only this one. -> model -> (kp (K,2) rgb px, sc (K,)),
    or None when fewer than 5 sure joints fall on the person, or (upright) when the head is below the feet (a mirror
    image on a glossy floor: only asked for people who are not tracked yet; a dancer bending back is not one)."""
    from . import segment as SG
    (x0, y0, x1, y1), m = SG.record_mask(rec)
    H, W = rgb.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    m = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    r = max(2, int(0.02 * (cy1 - cy0)))
    m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    img = np.full_like(rgb, 114)
    sub, src = img[cy0:cy1, cx0:cx1], rgb[cy0:cy1, cx0:cx1]
    sub[m] = src[m]
    box = np.array([[cx0, cy0, cx1, cy1]], np.float32)
    out = {}
    for k, pe in poses.items():
        kp, sc = pe(img, box)
        out[k] = (np.asarray(kp[0], np.float32), np.asarray(sc[0], np.float32))
    if not out:
        return None
    kp, sc = out.get("pose_rtmw_x", next(iter(out.values())))
    good = 0
    for j in range(S.K):
        if sc[j] >= 0.3:
            x, y = int(kp[j, 0]) - cx0, int(kp[j, 1]) - cy0
            if 0 <= x < m.shape[1] and 0 <= y < m.shape[0] and m[y, x]:
                good += 1
    head = [kp[j, 1] for j in (S.NOSE, S.L_EYE, S.R_EYE) if sc[j] >= 0.3]
    feet = [kp[j, 1] for j in (S.L_AN, S.R_AN) if sc[j] >= 0.3]
    upside = bool(head and feet and np.mean(head) > np.mean(feet))
    if dbg is not None:
        dbg.update(good=good, upside=upside, box=[cx0, cy0, cx1, cy1], img=img, kp=kp, sc=sc)
    if good < 5 or (upright and upside):
        return None
    return out


def _kp_dist(k0: Optional[np.ndarray], k1: Optional[np.ndarray]) -> Optional[float]:
    """Median distance of the sure joints two skeletons share (>= 4), else None."""
    if k0 is None or k1 is None:
        return None
    ok = np.isfinite(k0[:, :2]).all(1) & np.isfinite(k1[:, :2]).all(1) & (np.nan_to_num(k0[:, 2]) >= 0.3) \
        & (np.nan_to_num(k1[:, 2]) >= 0.3)
    if ok.sum() < 4:
        return None
    return float(np.median(np.hypot(*(k0[ok, :2] - k1[ok, :2]).T)))


def _save_identity_samples(project, cut, tracklets, dets, pids) -> None:
    """For linking people across cuts and for identity self-learning: per person up to 12 body re-ID and
    12 face features from different moments, and which people were on screen at the same time (never the
    same person). cuts/<id>/identity.npz"""
    from .assoc import _isolated
    P = len(tracklets)
    R, Fc, owner_r, owner_f = [], [], [], []
    spans = []
    for p, tr in enumerate(tracklets):
        fr = sorted(tr)
        spans.append(set(fr))
        rs, fs = [], []
        for f in fr:
            d = dets[f][tr[f]]
            r = d.get("reid")
            if r is not None and r.any() and _isolated(dets[f])[tr[f]]:
                rs.append(np.asarray(r, np.float32))
            fc = d.get("face")
            if fc is not None and fc.get("feat") is not None:
                fs.append((float(fc["score"] * fc["w"]), np.asarray(fc["feat"], np.float32)))
        if len(rs) > 12:
            rs = [rs[i] for i in np.linspace(0, len(rs) - 1, 12).round().astype(int)]
        fs.sort(key=lambda x: -x[0])
        fs = [x[1] for x in fs[:24]]
        if len(fs) > 12:
            fs = [fs[i] for i in np.linspace(0, len(fs) - 1, 12).round().astype(int)]
        R += rs
        owner_r += [pids[p]] * len(rs)
        Fc += fs
        owner_f += [pids[p]] * len(fs)
    co = []
    for i in range(P):
        for j in range(i + 1, P):
            if len(spans[i] & spans[j]) >= 3:
                co.append((pids[i], pids[j]))
    out = project.cut_dir(cut) / "identity.npz"
    np.savez_compressed(out, reid=np.stack(R).astype(np.float32) if R else np.zeros((0, 0), np.float32),
                        reid_pid=np.array(owner_r, np.int32),
                        face=np.stack(Fc).astype(np.float32) if Fc else np.zeros((0, 0), np.float32),
                        face_pid=np.array(owner_f, np.int32), cooccur=np.array(co, np.int32).reshape(-1, 2),
                        frames=np.array([len(s_) for s_ in spans], np.int32), pids=np.array(pids, np.int32))


class CutTracker:
    def __init__(self, project: Project, cut: Cut, proxy: Proxy):
        self.project = project
        self.cut = cut
        self.proxy = proxy
        self.path = project.cut_dir(cut) / "tracks.npz"
        mw, mh = max(2, proxy.w // 2), max(2, proxy.h // 2)
        self.matte = MatteStore(project.cache_dir(cut), proxy.n, mw, mh)
        # SAM 2 person labels (person id + 1 per matte pixel), written with the matte
        self.seglab = MatteStore(project.cache_dir(cut), proxy.n, mw, mh, prefix="seglab")
        # clothing classes per pixel (class + 1), filled on demand for 'keep the clothes'
        self.cloth = MatteStore(project.cache_dir(cut), proxy.n, mw, mh, prefix="cloth")
        # head parts per pixel (1 = hair, 2 = face), filled on demand for precise head edits
        self.parts = MatteStore(project.cache_dir(cut), proxy.n, mw, mh, prefix="parts")
        self._alias_path = project.cache_dir(cut) / "seglab_alias.json"
        self._stage_extra = None             # files of a whole new matte being written (matte_rev.begin)
        self._stage_meta = None
        # a matte revision that was being put in use when the app ended: the cut is put back on its last complete
        # one (matte_rev: the whole matte before the tracks are read, a refresh of some frames after)
        self.recovered = ""
        try:
            from . import matte_rev as MRV
            self.recovered = MRV.recover_files(project.cache_dir(cut), self.path, self.seg_meta)
        except Exception:
            import traceback
            traceback.print_exc()
        self.store: Optional[TrackStore] = TrackStore.load(self.path)
        if self.store is not None and self.store.n != proxy.n:
            self.store = None
        self._label_cache: Dict[int, np.ndarray] = {}
        try:
            from . import matte_rev as MRV
            self.recovered = MRV.recover(self) or self.recovered
        except Exception:
            import traceback
            traceback.print_exc()
        self.widths_stale = False
        st = self.store
        if st is not None and st.widths.shape[2] != B.N_SEG:
            # body segments changed between versions: keep what still matches, measure the rest later
            old_w = st.widths
            st.widths = np.full((len(st.pids), st.n, B.N_SEG, 2), np.nan, np.float32)
            k = min(old_w.shape[2], B.N_SEG)
            if old_w.shape[2] < B.N_SEG:
                # segments are only ever appended: the old ones keep their place
                st.widths[:, :, :k] = old_w[:, :, :k]
            self.widths_stale = True
        if st is not None and len(st.pids) and not st.has_contour:
            # projects analysed before body outline points existed: measure them in the background
            self.widths_stale = True

    # ------------------------------------------------------------------ basics
    @property
    def ready(self) -> bool:
        return self.store is not None and len(self.store.pids) > 0

    def save(self) -> None:
        if self.store is not None:
            self.store.save(self.path)

    def frame_source(self) -> SV.FrameSource:
        return SV.FrameSource(self.proxy.gray, self.proxy.sx, self.proxy.sy)

    def keys_by_index(self) -> Dict[int, Dict[int, Dict[int, List[float]]]]:
        st = self.store
        out = {}
        for pid, jd in self.cut.joint_keys.items():
            if st is not None and st.has(pid):
                out[st.index(pid)] = jd
        return out

    def invalidate_labels(self, frames=None):
        if frames is None:
            self._label_cache.clear()
        else:
            for f in list(self._label_cache):
                if f in frames:
                    del self._label_cache[f]

    def _seg_alias(self) -> Dict[int, int]:
        import json
        try:
            return {int(k): int(v) for k, v in json.loads(self._alias_path.read_text(encoding="utf-8")).items()}
        except Exception:
            return {}

    def has_seg(self) -> bool:
        return self.seglab.exists()

    def labels(self, t: int) -> Optional[np.ndarray]:
        """Person label map at matte resolution (index into store.pids, -1 = nobody).

        With a SAM 2 matte every person pixel has its owner from segmentation; anything the matte holds that
        SAM did not claim (untracked people, a blurred hand) falls back to the nearest tracked skeleton."""
        if self.store is None:
            return None
        lab = self._label_cache.get(t)
        if lab is None:
            seg = self.seglab.get(t) if self.seglab.exists() else None
            lab = self.labels_from(t, seg, lambda: self.matte.get(t))
            with _LABEL_LOCK:                       # (the people of a frame ask side by side: retouch.engine.PREP_THREADS)
                while len(self._label_cache) > 48:
                    try:
                        self._label_cache.pop(next(iter(self._label_cache)))
                    except (StopIteration, KeyError, RuntimeError):
                        break
                self._label_cache[t] = lab
        return lab

    def labels_from(self, t: int, seg, alpha) -> np.ndarray:
        """labels(t) for a given owner frame and alpha (alpha: the frame, or a function that gives it): the owners
        in use, or those of a matte revision that is not in use yet (matte_rev.stage). Nothing is kept."""
        if seg is None or not np.asarray(seg).any():
            kps = [self.store.kp(pid, t) for pid in self.store.pids]
            return B.label_map(kps, (self.matte.h, self.matte.w),
                               (self.matte.w / self.cut.info.width, self.matte.h / self.cut.info.height))
        lut = np.full(256, -1, np.int16)
        alias = self._seg_alias()
        for idx, pid in enumerate(self.store.pids):
            if 0 <= pid < 255:
                lut[pid + 1] = idx
        for a_, b_ in alias.items():
            if 0 <= a_ < 255 and b_ in self.store.pids:
                lut[a_ + 1] = self.store.pids.index(b_)
        lab = lut[seg]
        al = alpha() if callable(alpha) else alpha
        if al is not None:
            # 255: a person nobody tracks (an extra): stays nobody's
            miss = (lab < 0) & (np.asarray(al) >= 128) & (np.asarray(seg) != 255)
            if miss.any():
                kps = [self.store.kp(pid, t) for pid in self.store.pids]
                sk = B.label_map(kps, (self.matte.h, self.matte.w),
                                 (self.matte.w / self.cut.info.width, self.matte.h / self.cut.info.height))
                lab = np.where(miss, sk, lab).astype(np.int16)
        return lab

    # ------------------------------------------------------------------ full analysis
    def analyze(self, progress: Progress = None, cancel: Optional[Callable[[], bool]] = None,
                accuracy: str = "high") -> TrackStore:
        """The whole analysis as one revision (instruction 2026-11-20 §1-2): its matte, owners and face layer are
        written beside the ones in use and put in use together with the new tracks at the very end. Cancelled or
        failed, the files in use were never touched, and the tracker and the cut's people are what they were."""
        import copy as _copy
        from . import matte_rev as MRV
        cut = self.cut
        keep = dict(store=self.store, persons=_copy.deepcopy(cut.persons), joint_keys=dict(cut.joint_keys),
                    analyzed=cut.analyzed, analysis_version=cut.analysis_version, cs_analysed=cut.cs_analysed,
                    models_used=dict(cut.models_used or {}), model_versions=dict(cut.model_versions or {}))
        a, b = cut.work_range
        MRV.begin(self, copy=not (a <= 0 and b >= self.proxy.n))
        try:
            # the model packages in use now are the ones this analysis runs with to its end (instruction 2026-11-20
            # §4-2), whatever is installed meanwhile; they are written on the cut with its result
            from . import modelpkg as MPK
            with MPK.pinned() as pins:
                st = self._analyze(progress, cancel, accuracy)
                cut.model_versions = {k: v for k, v in pins.items() if k in (cut.models_used or {})}
                return st
        except BaseException:
            if MRV.staging(self):
                MRV.drop(self)
            self.store = keep["store"]
            cut.persons, cut.joint_keys = keep["persons"], keep["joint_keys"]
            cut.analyzed, cut.analysis_version = keep["analyzed"], keep["analysis_version"]
            cut.cs_analysed, cut.models_used = keep["cs_analysed"], keep["models_used"]
            cut.model_versions = keep["model_versions"]
            self._label_cache.clear()
            self._geo_cache = {}
            self._head_raw = {}
            self._skin_models = {}
            raise

    def _analyze(self, progress: Progress = None, cancel: Optional[Callable[[], bool]] = None,
                 accuracy: str = "high") -> TrackStore:
        from .infer import Detector, FaceDetector, Matter, PoseEstimator
        from .models import POSE_MODEL_KEYS

        def step(lo, hi):
            def f(x, msg=""):
                if progress:
                    progress(lo + (hi - lo) * float(np.clip(x, 0, 1)), msg)
                if cancel and cancel():
                    raise Cancelled()
            return f

        cut, proxy = self.cut, self.proxy
        ts = cut.tracking
        a, b = cut.work_range
        N = proxy.n
        sx, sy = proxy.sx, proxy.sy
        W, H = cut.info.width, cut.info.height
        p_load = step(0.0, 0.04)
        p_load(0, "모델 불러오는 중")
        # a lower score and a looser overlap rule than the defaults: dancers overlap and crouch; doubles of one
        # person are removed after pose estimation (same skeleton)
        det = Detector("det_yolox_x" if accuracy == "max" else "det_yolox_m", score_thr=0.3, nms_thr=0.6)
        reid, face_id = load_identity_models(bool(getattr(ts, "reid", True)))
        pose_keys = [k for k in ts.models if k in POSE_MODEL_KEYS] or ["pose_rtmw_x"]
        if "pose_rtmw_x" not in pose_keys:
            pose_keys.append("pose_rtmw_x")      # face landmarks for face reshaping and skin masks
        if accuracy == "fast":
            pose_keys = ["pose_rtmw_x"]
        if accuracy == "max" and "pose_dwpose_l" not in pose_keys:
            pose_keys.append("pose_dwpose_l")    # a fourth opinion from a differently trained model
        flip = accuracy == "max" or bool(getattr(ts, "flip_tta", False))
        poses = {}
        for i, k in enumerate(pose_keys):
            poses[k] = PoseEstimator(k, flip=flip)
            p_load((i + 1) / (len(pose_keys) + 1), f"모델 불러오는 중 ({k})")
        face = FaceDetector() if ts.face else None
        # identities followed by SAM 2's video memory (with a GPU); else the frame-to-frame matcher
        vt = None
        try:
            from . import sam2video as VT
            vt = VT.make_tracker(getattr(ts, "video_track", "off"), float(cut.info.fps or 30.0), p_load)
        except Cancelled:
            raise
        except Exception:
            import traceback
            traceback.print_exc()
            vt = None

        # ---------------- pass 1: detection + pose on every frame
        p1 = step(0.04, 0.40)
        dets: List[List[dict]] = [[] for _ in range(N)]
        raw1: List[List[dict]] = [[] for _ in range(N)]
        t0 = time.time()
        # (1.27) the frames decoded a few ahead in a thread of their own (track/ahead.py): the detector and the pose models
        # no longer stand while the next frame is read from the original (0.2 s of a 4K log frame's 0.6 s here)
        from .ahead import FramesAhead
        with FramesAhead(proxy, a, b) as reader:
            for t in range(a, b):
                rgb, _ = reader.get(t)
                people, raw = detect_people(rgb, det, poses, reid, face_id, face, face_always=(t - a) % 4 == 0)
                for d, r in zip(people, raw):
                    d["i"] = len(dets[t])
                    dets[t].append(d)
                    raw1[t].append(r)
                if vt is not None:
                    try:
                        vt.step(t, rgb, dets[t])
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        vt = None
                el = time.time() - t0
                n_done = t - a + 1
                p1(n_done / max(1, b - a), f"인물·포즈 검출{' + 비디오 추적' if vt is not None else ''} "
                                           f"{n_done}/{b - a} ({n_done / max(el, 1e-3):.1f} fps)")

        # ---------------- association
        p_as = step(0.40, 0.42)
        p_as(0, "인물 구분 중")
        recover = _Recover(proxy, poses, a)
        tracklets = None
        if vt is not None:
            from .assoc import link_tracklets
            vtr, free = vt.results()
            if RECOVER and vtr:
                p_as(0.2, "비디오 추적이 이어 준 프레임의 포즈 찾는 중")
                from . import sam2video as VT
                VT.recover_free(vtr, free, dets, recover)
            tracklets = link_tracklets(vtr, dets) or None
            self.video_tracked = tracklets is not None
        if tracklets is None:
            self.video_tracked = False
            tracklets = track_people(dets[a:b], recover=recover.local if RECOVER else None,
                                     members=getattr(self.cut.tracking, "members", 0) or None)
            tracklets = [{f + a: i for f, i in tr.items()} for tr in tracklets]
        tracklets = [tr for tr in tracklets if len(tr) >= max(3, int(0.03 * (b - a)))] or tracklets
        if not tracklets:
            raise RuntimeError("인물을 찾지 못했습니다. 작업 구간과 영상을 확인하세요.")
        if FILL_GAPS:
            p_as(0.5, "검출이 빠진 프레임 잇는 중")
            fill_gaps(tracklets, dets, recover)
        # poses found without a detection carry their per-model results; keep raw1 aligned with dets
        for t in range(a, b):
            while len(raw1[t]) < len(dets[t]):
                raw1[t].append(dets[t][len(raw1[t])].pop("raw", {}))
        # prominence: mean box height x presence
        def prominence(tr):
            hs = [dets[f][i]["box"][3] - dets[f][i]["box"][1] for f, i in tr.items()]
            return float(np.mean(hs)) * len(tr)
        tracklets.sort(key=prominence, reverse=True)
        P = len(tracklets)
        old_store = self.store
        pids = self._map_person_ids(tracklets, dets, old_store)
        keep_i = [i for i, pid in enumerate(pids) if pid != -2]
        if keep_i and len(keep_i) < len(pids):
            tracklets = [tracklets[i] for i in keep_i]
            pids = [pids[i] for i in keep_i]
            P = len(tracklets)
        st = TrackStore(N, pids)
        if old_store is not None:
            st.removed = dict(old_store.removed)
        st.homog = old_store.homog if old_store is not None else st.homog
        for p, tr in enumerate(tracklets):
            for f, i in tr.items():
                st.present[p, f] = True
                st.boxes[p, f] = dets[f][i]["box"] / [sx, sy, sx, sy]
            # fill short gaps
            fr = sorted(tr)
            for f0, f1 in zip(fr, fr[1:]):
                if 1 < f1 - f0 <= 15:
                    for f in range(f0 + 1, f1):
                        w_ = (f - f0) / (f1 - f0)
                        st.present[p, f] = True
                        st.boxes[p, f] = (1 - w_) * st.boxes[p, f0] + w_ * st.boxes[p, f1]
        for k in poses:
            st.raw[k] = np.full((P, N, S.K, 3), np.nan, np.float32)
        if "pose_rtmw_x" in poses:
            st.raw["face_rtmw"] = np.full((P, N, S.K, 3), np.nan, np.float32)
        for p, tr in enumerate(tracklets):
            for f, i in tr.items():
                for k, (kp, sc) in raw1[f][i].items():
                    st.raw[k][p, f, :, :2] = kp[:17] / [sx, sy]
                    st.raw[k][p, f, :, 2] = sc[:17]
                    if k == "pose_rtmw_x":
                        self._face_from_rtmw(st, p, f, kp / [sx, sy], sc)
                        self._hand42(st, p, f, kp / [sx, sy], sc)
        del raw1

        # ---------------- faces out of focus (track/focus.py): people the shot is not about. They get no close-up matte, no
        # face flow and no face layer, and are protected (not retouched) when this analysis is the first to see them
        st.defocus = set()
        self._focus_info = {}
        if DEFOCUS_SKIP:
            try:
                from . import focus as FC
                sc_ = FC.scores(st, proxy.gray, sx, sy, (a, b))
                out_, info_ = FC.defocused(sc_)
                st.defocus = {int(st.pids[p]) for p in out_}
                self._focus_info = {int(st.pids[p]): i_ for p, i_ in info_.items()}
                if st.defocus:
                    print(f"초점 밖 인물 {sorted(st.defocus)}: 보정 안 함, 분석 일부 생략", flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                st.defocus = set()

        # ---------------- matte (needed to tell visible joints from guessed ones)
        p_m = step(0.42, 0.48)
        self._label_cache.clear()
        collect: dict = {}

        def repose(t, p, rgb, rec):
            res = _masked_pose(rgb, rec, poses)
            if res is None:
                return False
            self._write_pose(st, p, t, res, rec["box"])
            return True

        self._run_matte(a, b, p_m, st=st, accuracy=accuracy, cancel=cancel, repose=repose, collect=collect)
        # ---------------- what the person finder says about the tracking: identities that jumped to someone else,
        # people nobody follows
        self._added_pids = set()
        if collect.get("src"):
            p_fx = step(0.48, 0.50)
            try:
                p_fx(0.0, "인물 정체 확인")
                self._fix_swaps(st, tracklets, collect, a, b)
                p_fx(0.3, "놓친 인물 찾는 중")
                added = self._add_found_people(st, tracklets, dets, pids, collect, poses, a, b, cancel)
                if added:
                    P = len(st.pids)
                from . import segment as SG
                self._seg_stats = SG.stats_from(collect.get("src"))
            except Cancelled:
                raise
            except Exception:
                import traceback
                traceback.print_exc()

        # ---------------- blur + faces
        p_bf = step(0.50, 0.55)
        if face is not None:
            st.raw["face_yunet"] = np.full((P, N, S.K, 3), np.nan, np.float32)
        lap_raw = np.full((P, N), np.nan, np.float32)
        for t in range(a, b):
            if not st.present[:, t].any():
                continue
            g = proxy.gray(t)
            rgb = None
            for p in np.nonzero(st.present[:, t])[0]:
                bx = st.boxes[p, t] * [sx, sy, sx, sy]
                x1, y1, x2, y2 = [int(v) for v in np.round(bx)]
                x1, y1 = max(0, x1), max(0, y1)
                crop = g[y1:max(y1 + 2, y2), x1:max(x1 + 2, x2)]
                if crop.size > 64:
                    s_ = 200.0 / max(1, crop.shape[0])
                    crop = cv2.resize(crop, (max(4, int(crop.shape[1] * s_)), 200), interpolation=cv2.INTER_AREA)
                    lap_raw[p, t] = cv2.Laplacian(crop, cv2.CV_32F).var()
                if face is not None:
                    if rgb is None:
                        rgb = analysis_view(np.asarray(proxy.frame(t)))
                    self._face_yunet(st, face, rgb, p, t)
            p_bf((t - a + 1) / max(1, b - a), "얼굴·모션블러 분석")
        for p in range(P):
            series = lap_raw[p]
            ref = SV._smooth_series(series, 15)
            st.blur[p] = np.where(np.isfinite(series), np.clip(series / np.maximum(ref, 1e-3), 0, 2), 1.0)

        # ---------------- camera motion (background only; people masked by their boxes)
        p_c = step(0.55, 0.58)

        dil = max(3, int(0.012 * max(proxy.w, proxy.h))) | 1

        def box_mask(t):
            # the people matte is much tighter than the boxes (open arms make a box as wide as the frame)
            al = self.matte.get(t)
            if al is not None:
                m = cv2.resize(np.asarray(al), (proxy.w, proxy.h), interpolation=cv2.INTER_LINEAR)
                m = np.where(m > 30, 255, 0).astype(np.uint8)
                return cv2.dilate(m, np.ones((dil, dil), np.uint8))
            m = np.zeros((proxy.h, proxy.w), np.uint8)
            for p in range(P):
                if st.present[p, t] and np.isfinite(st.boxes[p, t]).all():
                    x1, y1, x2, y2 = st.boxes[p, t] * [sx, sy, sx, sy]
                    cv2.rectangle(m, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), 255, -1)
            return m

        Hs, ok, jumps = estimate_all(N, proxy.gray, box_mask, sx, sy, progress=p_c, cancel=cancel)
        if self.cut.edit_points:
            # the editor's own cuts (editing XML) are the truth: inside a plate made of jump cuts they are exactly
            # where tracking must restart, and nowhere else (a flash is then never taken for a cut)
            jumps = np.zeros(N, bool)
            for e in self.cut.edit_points:
                if 1 <= e < N:
                    jumps[e - 1] = True
        st.homog, st.homog_ok, st.jump = Hs, ok, jumps
        st.invalidate_cache()

        # ---------------- fuse + first solve
        p_s1 = step(0.58, 0.66)
        fuse_kw = dict(frame_wh=(W, H), alpha_at=self.matte.get,
                       alpha_scale=(self.matte.w / W, self.matte.h / H),
                       owner_at=(self.seglab.get if self.seglab.exists() else None))
        # people turning around: one left/right labelling across models and through the shot
        from .orient import fix_left_right
        fix_left_right(st, (a, b))
        SV.fuse_measurements(st, ts.conf_threshold, **fuse_kw)
        fs = SV.FrameSource(proxy.gray, sx, sy) if ts.point_tracker else None
        refx = self._refiner_extra(st) if ts.refiner else None
        trk, fconf, flags = SV.solve(st, fs, range(P), (a, b), extra=[refx] if refx else None,
                                     smoothing=ts.smoothing, conf_threshold=ts.conf_threshold, progress=p_s1,
                                     vis=st.vis_meas)
        st.auto[:, a:b] = trk
        st.flags[:, a:b] = flags

        # ---------------- pass 2: re-estimate with keypoint-derived boxes
        if ts.second_pass and accuracy != "fast":
            p2 = step(0.66, 0.84)
            for k in poses:
                st.raw[k + "@2"] = np.full((P, N, S.K, 3), np.nan, np.float32)
            for t in range(a, b):
                pres = np.nonzero(st.present[:, t])[0]
                if not len(pres):
                    continue
                rgb = analysis_view(np.asarray(proxy.frame(t)))
                boxes = []
                owners = []
                for p in pres:
                    kp = st.auto[p, t].copy()
                    kp[:, 0] *= sx
                    kp[:, 1] *= sy
                    bb = S.bbox_from_kp(kp, (proxy.w, proxy.h))
                    if bb is not None:
                        boxes.append(bb)
                        owners.append(p)
                if not boxes:
                    continue
                boxes = np.array(boxes, np.float32)
                for k, pe in poses.items():
                    kp, sc = pe(rgb, boxes)
                    for i, p in enumerate(owners):
                        st.raw[k + "@2"][p, t, :, :2] = kp[i, :17] / [sx, sy]
                        st.raw[k + "@2"][p, t, :, 2] = sc[i, :17]
                        if k == "pose_rtmw_x":
                            self._face68(st, p, t, kp[i] / [sx, sy], sc[i], second=True)
                            self._hand42(st, p, t, kp[i] / [sx, sy], sc[i], second=True)
                p2((t - a + 1) / max(1, b - a), f"2차 정밀 추정 {t - a + 1}/{b - a}")
            p_s2 = step(0.84, 0.87)
            fix_left_right(st, (a, b))
            SV.fuse_measurements(st, ts.conf_threshold, **fuse_kw)
            refx = self._refiner_extra(st) if ts.refiner else None
            trk, fconf, flags = SV.solve(st, fs, range(P), (a, b), extra=[refx] if refx else None,
                                         smoothing=ts.smoothing, conf_threshold=ts.conf_threshold, progress=p_s2,
                                         vis=st.vis_meas)
            st.auto[:, a:b] = trk
            st.flags[:, a:b] = flags
        st.auto_conf[:, a:b] = SV.frame_conf_from_track(st.auto[:, a:b], st.present[:, a:b], st.vis_meas[:, a:b],
                                                        st.flags[:, a:b])
        st.final = st.auto.copy()
        st.conf = st.auto_conf.copy()
        st.vis = st.vis_meas.copy()
        # ---------------- face and hand detail: dedicated models on tight crops
        if accuracy != "fast" and getattr(ts, "detail", True):
            self._detail_pass(st, (a, b), step(0.87, 0.93), cancel)
        # ---------------- face landmarks and hands through the shot
        p_f = step(0.93, 0.96)
        from .face import solve_face, solve_hands
        solve_face(st, fs, (a, b), ts.smoothing, p_f)
        if fs is not None:
            try:
                from .faceflow import refine as _ff
                _ff(st, fs, (a, b), skip={i for i, q in enumerate(st.pids) if int(q) in getattr(st, "defocus", ())})
            except Exception:
                import traceback
                traceback.print_exc()
        try:
            from .face import occlusion_fade
            occlusion_fade(st, (a, b), 40.0 / max(self.proxy.sx, 1e-6))
        except Exception:
            import traceback
            traceback.print_exc()
        solve_hands(st, fs, (a, b), max(0.15, ts.smoothing * 0.7), None)
        self._head_raw = {}
        self.store = st
        self._label_cache.clear()
        self._geo_cache = {}
        self._skin_models = {}

        # ---------------- persons in the project
        self._update_persons(tracklets, dets, pids)
        for q in self.cut.persons:
            if q.id in getattr(self, "_added_pids", set()) and q.id in getattr(self, "_new_pids", set()):
                # found by the finder, not by the detector: kept out of the warp, not retouched until the user says
                q.name = f"{q.name} (추가)"
                q.role, q.hold = "protect", "added"
        p_w = step(0.96, 1.0)
        # keys from a previous analysis still apply (same person ids)
        if any(self.cut.joint_keys.get(pid) for pid in pids):
            self.resolve_all(progress=None)
        self.update_widths()
        # posters, photos and screens on the set: a figure that never moves is a picture, not someone to
        # retouch (its mask would sit on the print, and it would count as a face in the cut's status). Only
        # for people new in this analysis: a role the user chose stays.
        try:
            from .status import still_pictures
            new = getattr(self, "_new_pids", set())
            pics = still_pictures(st) & new if new else set()
            for q in self.cut.persons:
                if q.id in pics:
                    q.role, q.still = "protect", True
                    q.name = f"{q.name} (사진)"
        except Exception:
            import traceback
            traceback.print_exc()
        # a track SAM 3.1's person finder never confirms (a shadow on a glossy wall, a reflection, a coat on a
        # chair): not retouched unless the user says so. Only for people new in this analysis.
        try:
            from . import segment as SG
            new = getattr(self, "_new_pids", set())
            for q in self.cut.persons:
                if q.id in new and q.id in SG.unconfirmed(getattr(self, "_seg_stats", None)) and not getattr(q, "still", False):
                    q.role, q.hold = "protect", "unconfirmed"
                    q.name = f"{q.name} (확인)"
        except Exception:
            import traceback
            traceback.print_exc()
        # people out of focus (track/focus.py): protected, only for people new in this analysis (a role the user chose stays)
        try:
            new = getattr(self, "_new_pids", set())
            for q in self.cut.persons:
                if q.id in getattr(st, "defocus", ()) and q.id in new and not getattr(q, "still", False) and not getattr(q, "hold", ""):
                    q.role, q.hold = "protect", "defocus"
                    q.name = f"{q.name} (초점밖)"
        except Exception:
            import traceback
            traceback.print_exc()
        # face layer (parsing + steady skin colour) for the retouch masks (not for the people out of focus)
        try:
            from . import facelayer as FL
            soft_ = getattr(st, "defocus", set())
            only_ = {i for i, q in enumerate(st.pids) if int(q) not in soft_} if soft_ else None
            FL.build(self, lambda f, m: p_w(0.3 + 0.65 * f, m), cancel, persons=only_)
        except Exception:
            import traceback
            traceback.print_exc()
        # originals for later training of the masks and the tracking (Cloud folder, this computer only)
        try:
            from . import cloudcollect as CC
            CC.collect(self)
        except Exception:
            import traceback
            traceback.print_exc()
        p_w(1.0, "완료")
        self.cut.analyzed = True
        self.cut.analysis_version += 1
        self.cut.cs_analysed = self.cut.input_cs or ""
        try:
            from . import models as _M
            self.cut.models_used = _M.fingerprints()
        except Exception:
            pass
        from . import matte_rev as MRV
        if MRV.staging(self):
            MRV.commit(self, (a, b))        # matte, owners, face layer and these tracks: in use from here
        else:
            self.save()
        return st

    # ------------------------------------------------------------------ helpers for analysis
    @staticmethod
    def _face68(st: TrackStore, p: int, f: int, kp133: np.ndarray, sc133: np.ndarray, second: bool = False) -> None:
        """Keep all 68 RTMW face landmarks; the second (tighter crop) pass is averaged in by score."""
        face = kp133[S.RTMW_FACE]
        fsc = sc133[S.RTMW_FACE]
        new = np.column_stack([face, fsc]).astype(np.float32)
        old = st.face_raw[p, f]
        if second and np.isfinite(old[:, :2]).all():
            w1 = np.maximum(old[:, 2:3], 1e-3)
            w2 = np.maximum(new[:, 2:3], 1e-3)
            mix = old.copy()
            mix[:, :2] = (old[:, :2] * w1 + new[:, :2] * w2) / (w1 + w2)
            mix[:, 2] = np.maximum(old[:, 2], new[:, 2])
            st.face_raw[p, f] = mix
        else:
            st.face_raw[p, f] = new

    @staticmethod
    def _hand42(st: TrackStore, p: int, f: int, kp133: np.ndarray, sc133: np.ndarray, second: bool = False) -> None:
        """RTMW hand points (person's left hand 91..111, right 112..132)."""
        new = np.column_stack([kp133[91:133], sc133[91:133]]).astype(np.float32)
        old = st.hand_raw[p, f]
        if second and np.isfinite(old[:, :2]).all():
            w1 = np.maximum(old[:, 2:3], 1e-3)
            w2 = np.maximum(new[:, 2:3], 1e-3)
            mix = old.copy()
            mix[:, :2] = (old[:, :2] * w1 + new[:, :2] * w2) / (w1 + w2)
            mix[:, 2] = np.maximum(old[:, 2], new[:, 2])
            st.hand_raw[p, f] = mix
        else:
            st.hand_raw[p, f] = new

    def _detail_pass(self, st: TrackStore, rng: Tuple[int, int], progress, cancel=None) -> None:
        """106-point face model and 21-point hand model on crops around the tracked face and hands."""
        from .infer import PoseEstimator
        from .face import MAP106_68, bad_regions
        proxy = self.proxy
        sx, sy = proxy.sx, proxy.sy
        a, b = rng
        try:
            fm = PoseEstimator("face_rtm106", padding=1.25)
        except Exception:
            fm = None
        try:
            hm = PoseEstimator("hand_rtm21", padding=1.25)
        except Exception:
            hm = None
        if fm is None and hm is None:
            return
        try:
            from .infer import FaceDetector
            fdet = FaceDetector(score_thr=0.6) if fm is not None else None
        except Exception:
            fdet = None
        for t in range(a, b):
            pres = np.nonzero(st.present[:, t])[0]
            if not len(pres):
                continue
            fboxes, fown, frots, fvar, hboxes, hown = [], [], [], [], [], []
            claims = {}          # detected face -> [(share of points on it, person)]
            rgb = None
            yfaces = None
            for p in pres:
                lm = st.face_raw[p, t]
                if fm is not None and np.isfinite(lm[:, :2]).all():
                    q = lm[:, :2] * [sx, sy]
                    x0, y0 = q.min(0)
                    x1, y1 = q.max(0)
                    # the 68 points stop at the brows: crops with and without the forehead
                    h_ = y1 - y0
                    tight = [x0, y0, x1, y1]
                    y0 -= 0.25 * h_
                    # how far the head is tilted (eye line)
                    el, er = q[36:42].mean(0), q[42:48].mean(0)
                    rot = float(np.arctan2(er[1] - el[1], er[0] - el[0]))
                    if abs(rot) > np.radians(100):
                        rot = 0.0                   # eyes swapped (head turned away), not an upside-down head
                    # where the face detector sees this person's face: in close-ups, with arms up in front, the
                    # whole-body model scatters its face points; a face box is then the reliable crop, and the
                    # scattered points are dropped so they do not pull the dedicated model's landmarks around
                    if fdet is not None and max(x1 - x0, y1 - y0) >= 40:
                        if rgb is None:
                            rgb = analysis_view(np.asarray(proxy.frame(t)))
                            yfaces = fdet.detect(rgb, None, target=960)
                            yfaces = yfaces[yfaces[:, 14] >= 0.7] if len(yfaces) else yfaces
                        fb, fi, fn = self._face_for(q, yfaces, True)
                        if fb is not None:
                            ctr = np.array([fb[0] + fb[2] / 2, fb[1] + fb[3] / 2])
                            off = float(np.hypot(*(np.median(q[27:68], 0) - ctr))) / max(float(fb[2]), 1.0)
                            claims.setdefault(fi, []).append((round(fn, 1), off, p))
                            fx, fy, fw_, fh_ = fb[:4]
                            if len(fb) >= 14 and np.isfinite(fb[4:8]).all():
                                # the face detector's eyes (its right eye first: image left on a frontal face)
                                rot = float(np.arctan2(fb[7] - fb[5], fb[6] - fb[4]))
                            # the face box covers brows to chin: with the forehead and a little on the sides, or as is
                            tight = [fx, fy, fx + fw_, fy + fh_]
                            x0, x1 = fx - 0.12 * fw_, fx + fw_ * 1.12
                            y0, y1 = fy - 0.30 * fh_, fy + fh_ * 1.08
                            inside = ((q[:, 0] >= fx - 0.15 * fw_) & (q[:, 0] <= fx + 1.15 * fw_) &
                                      (q[:, 1] >= fy - 0.3 * fh_) & (q[:, 1] <= fy + 1.2 * fh_)).mean()
                            spread = max(np.ptp(q[:, 0]) / max(fw_, 1.0), np.ptp(q[:, 1]) / max(fh_, 1.0))
                            if inside < 0.85 or spread > 1.8:
                                st.face_raw[p, t] = np.nan
                    if min(x1 - x0, y1 - y0) >= 14:
                        # the 106-point model is touchy about the crop: on some frames one framing gives a clean face
                        # and another a tangle of points (every point on some feature, not its own). Three framings
                        # (with forehead, tight, tight and turned upright); the one that makes a face is kept
                        fvar.append(len(fboxes))
                        fboxes.append([x0, y0, x1, y1])
                        frots.append(0.0)
                        fown.append(p)
                        fboxes.append(tight)
                        frots.append(0.0)
                        fown.append(p)
                        fboxes.append(tight)
                        frots.append(rot if abs(rot) > np.radians(8) else 0.0)
                        fown.append(p)
                if hm is not None:
                    for h in range(2):
                        box = self._hand_box(st, p, t, h)
                        if box is not None:
                            hboxes.append(box)
                            hown.append((p, h))
            # one face, one person: when the face points of two tracks land on the same detected face (a second
            # track on the hair or the shoulder of the person in a close-up, a background figure behind), the
            # track with most of its face points on it keeps it; the other has no face in this frame
            # the person segmentation says whose face it is: the person whose pixels hold the face's centre
            seg_owner = {}
            bare = [p for p in pres if not np.isfinite(st.face_raw[p, t, :, :2]).all()]
            if yfaces is None and fdet is not None and (self.seglab.exists() or bare):
                if rgb is None:
                    rgb = analysis_view(np.asarray(proxy.frame(t)))
                yfaces = fdet.detect(rgb, None, target=960)
                yfaces = yfaces[yfaces[:, 14] >= 0.7] if len(yfaces) else yfaces
            if yfaces is not None and len(yfaces) and self.seglab.exists():
                lab = self.labels(t)
                if lab is not None:
                    kx, ky = lab.shape[1] / proxy.w, lab.shape[0] / proxy.h
                    for fi_, f in enumerate(yfaces):
                        cx, cy = (f[0] + f[2] / 2) * kx, (f[1] + f[3] / 2) * ky
                        if 0 <= int(cy) < lab.shape[0] and 0 <= int(cx) < lab.shape[1]:
                            o = int(lab[int(cy), int(cx)])
                            if o >= 0 and st.present[o, t]:
                                seg_owner[fi_] = o
            granted = set()          # face crops given from a detected face (kept when the track lost another)
            # a frame the person detector missed (the track is bridged across it) has no whole-body face points and
            # often no segmentation either, while the face is in plain view: the detected face goes to the bridged
            # track whose head is on it (a whip pan or someone passing in front used to leave such a frame bare)
            if fm is not None and yfaces is not None and len(yfaces):
                owned = set(seg_owner.values()) | {o for o in fown}
                for fi_, f in enumerate(yfaces):
                    if fi_ in seg_owner or fi_ in claims or float(max(f[2], f[3])) < 40:
                        continue
                    ctr = np.array([f[0] + f[2] / 2, f[1] + f[3] / 2])
                    best, bd = None, 0.7 * float(max(f[2], f[3]))
                    for p in pres:
                        if p in owned or np.isfinite(st.face_raw[p, t, :, :2]).all():
                            continue
                        hd = st.final[p, t, :3]
                        if not np.isfinite(hd[:, :2]).all():
                            hd = st.auto[p, t, :3]
                        ok = np.isfinite(hd[:, :2]).all(1)
                        if ok.sum() < 2:
                            continue
                        d = float(np.linalg.norm(hd[ok, :2].mean(0) * [sx, sy] - ctr))
                        if d < bd:
                            best, bd = p, d
                    if best is not None:
                        seg_owner[fi_] = best
                        owned.add(best)
            lose = set()
            for fi, cl in claims.items():
                if len(cl) > 1:
                    cl.sort(key=lambda c: (c[2] != seg_owner.get(fi), -c[0], c[1]))  # owner, most points, centred
                    lose.update(c[2] for c in cl[1:])
            # a face whose owner has no face points of its own here (dropped as scattered, or the whole-body model
            # put them on the hair) is still that person's: it gets the 106-point crops from the face box, and a
            # track that claimed it from outside the owner's pixels (a second track on the same person) loses it
            if fm is not None:
                # a track that just lost a face it shared with another (its points spilled onto a neighbour's
                # face) may still be given its own face below
                has = set(fown) - lose
                for fi_, o in seg_owner.items():
                    if o in has:
                        continue
                    # a track whose face points lie squarely on this face keeps it when it is at least as sure of
                    # its head as the label map's owner (a track that slid onto the head in front, a mask that
                    # took in the face in front of its person)
                    strong = [c_ for c_ in claims.get(fi_, []) if c_[0] >= 0.8 and c_[1] < 0.3]
                    if strong:
                        hc = lambda q_: float(np.mean(np.nan_to_num(st.final[q_, t, :3, 2])))
                        if max(hc(c_[2]) for c_ in strong) >= hc(o) - 0.05:
                            continue
                    f = yfaces[fi_]
                    fx, fy, fw_, fh_ = [float(v) for v in f[:4]]
                    if max(fw_, fh_) < 40:
                        continue
                    for c in claims.get(fi_, []):
                        lose.add(c[2])
                    rot = float(np.arctan2(f[7] - f[5], f[6] - f[4])) if np.isfinite(f[4:8]).all() else 0.0
                    tight = [fx, fy, fx + fw_, fy + fh_]
                    fvar.append(len(fboxes))
                    granted.update(range(len(fboxes), len(fboxes) + 3))
                    fboxes += [[fx - 0.12 * fw_, fy - 0.30 * fh_, fx + 1.12 * fw_, fy + 1.08 * fh_], tight, tight]
                    frots += [0.0, 0.0, rot if abs(rot) > np.radians(8) else 0.0]
                    fown += [o, o, o]
                    has.add(o)
            if lose:
                for pp in lose:
                    st.face_raw[pp, t] = np.nan
                keep_i = [i for i in range(len(fboxes)) if fown[i] not in lose or i in granted]
                remap = {i: k for k, i in enumerate(keep_i)}
                fvar = [remap[i] for i in fvar if i in remap]
                fboxes = [fboxes[i] for i in keep_i]
                frots = [frots[i] for i in keep_i]
                fown = [fown[i] for i in keep_i]
            if not (fboxes or hboxes):
                continue
            if rgb is None:
                rgb = analysis_view(np.asarray(proxy.frame(t)))
            if fboxes:
                kp, sc = fm(rgb, np.array(fboxes, np.float32), frots)
                for i0 in fvar:
                    p = fown[i0]
                    best, bkey = i0, None
                    for i in range(i0, i0 + 3):
                        lm = np.concatenate([kp[i, MAP106_68], sc[i, MAP106_68, None]], 1)
                        key = (len(bad_regions(lm)), -float(np.median(sc[i])))
                        if bkey is None or key < bkey:
                            best, bkey = i, key
                    st.face106_raw[p, t, :, :2] = kp[best, MAP106_68] / [sx, sy]
                    st.face106_raw[p, t, :, 2] = sc[best, MAP106_68]
            if hboxes:
                kp, sc = hm(rgb, np.array(hboxes, np.float32))
                for i, (p, h) in enumerate(hown):
                    sl = slice(h * 21, (h + 1) * 21)
                    st.hand21_raw[p, t, sl, :2] = kp[i] / [sx, sy]
                    st.hand21_raw[p, t, sl, 2] = sc[i]
            progress((t - a + 1) / max(1, b - a), f"얼굴·손 정밀 추정 {t - a + 1}/{b - a}")

    @staticmethod
    def _face_for(q: np.ndarray, faces, with_index: bool = False):
        """The detected face (x, y, w, h, 5 landmarks, score) that this person's face points are on: the one
        holding most of them, and at least a third. with_index: (face, its index, share of points)."""
        if faces is None or not len(faces):
            return (None, -1, 0.0) if with_index else None
        best, bi, bn = None, -1, 0.0
        for i, f in enumerate(faces):
            x, y, w, h = [float(v) for v in f[:4]]
            n = float(((q[:, 0] >= x - 0.1 * w) & (q[:, 0] <= x + 1.1 * w) &
                       (q[:, 1] >= y - 0.2 * h) & (q[:, 1] <= y + 1.15 * h)).mean())
            if n > bn:
                best, bi, bn = np.asarray(f[:15], np.float32), i, n
        if bn < 0.33:
            best, bi = None, -1
        return (best, bi, bn) if with_index else best

    def _hand_box(self, st: TrackStore, p: int, t: int, h: int):
        """Square box (proxy px) around one hand: from RTMW hand points, else from the forearm."""
        sx, sy = self.proxy.sx, self.proxy.sy
        sl = slice(h * 21, (h + 1) * 21)
        pts = st.hand_raw[p, t, sl]
        wr, el = (S.L_WR, S.L_EL) if h == 0 else (S.R_WR, S.R_EL)
        if st.vis_meas[p, t, wr] == 0:
            return None
        ok = np.isfinite(pts[:, :2]).all(1)
        good = ok & (pts[:, 2] > 0.3 * np.nanmax(np.r_[pts[ok, 2], 1e-6])) if ok.any() else ok
        w = st.auto[p, t, wr, :2]
        e = st.auto[p, t, el, :2]
        if not (np.isfinite(w).all() and np.isfinite(e).all()):
            return None
        fl = float(np.hypot(*(w - e)))
        if good.sum() >= 8:
            q = pts[good, :2]
            c = (q.min(0) + q.max(0)) / 2
            size = max(float((q.max(0) - q.min(0)).max()), 0.55 * fl)
        else:
            c = w + 0.35 * (w - e)
            size = 0.75 * fl
        size *= 1.15
        if size * sx < 12:
            return None
        cx, cy = c[0] * sx, c[1] * sy
        hs = size * sx / 2
        return [cx - hs, cy - hs, cx + hs, cy + hs]

    def _face_from_rtmw(self, st: TrackStore, p: int, f: int, kp133: np.ndarray, sc133: np.ndarray) -> None:
        self._face68(st, p, f, kp133, sc133)
        face = kp133[S.RTMW_FACE]
        fs = sc133[S.RTMW_FACE]
        if np.median(fs) < 0.3 * np.median(sc133[:17] + 1e-6):
            return
        nose = face[30]
        eye_a = face[36:42].mean(0)     # subject's right eye
        eye_b = face[42:48].mean(0)     # subject's left eye
        out = st.raw["face_rtmw"][p, f]
        s = float(np.median(fs))
        out[S.NOSE] = [nose[0], nose[1], s]
        out[S.R_EYE] = [eye_a[0], eye_a[1], s]
        out[S.L_EYE] = [eye_b[0], eye_b[1], s]

    def _face_yunet(self, st: TrackStore, face, rgb: np.ndarray, p: int, t: int) -> None:
        sx, sy = self.proxy.sx, self.proxy.sy
        ref = st.raw.get("pose_rtmw_x", st.raw.get(next(iter(k for k in st.raw if k.startswith("pose_")))))[p, t]
        head = [S.NOSE, S.L_EYE, S.R_EYE, S.L_EAR, S.R_EAR]
        hp = ref[head]
        ok = np.isfinite(hp[:, :2]).all(1) & (hp[:, 2] > 0.2 * np.nanmax(ref[:, 2]) if np.isfinite(ref[:, 2]).any() else False)
        if ok.sum() < 2:
            return
        c = hp[ok, :2].mean(0) * [sx, sy]
        scale = S.person_scale(ref) if np.isfinite(S.person_scale(ref)) else 0
        size = max(24.0, 0.22 * scale * sx)
        box = np.array([c[0] - size, c[1] - size, c[0] + size, c[1] + size])
        faces = face.detect(rgb, box)
        if not len(faces):
            return
        centers = faces[:, 0:2] + faces[:, 2:4] / 2
        k = int(np.argmin(np.linalg.norm(centers - c, axis=1)))
        fc = faces[k]
        # the search box reaches past this head: a face a whole face width away is someone else's (a head seen
        # from behind, next to a face turned towards it)
        if np.linalg.norm(centers[k] - c) > 0.8 * max(float(fc[2]), float(fc[3])):
            return
        lm = fc[4:14].reshape(5, 2) / [sx, sy]
        s = float(fc[14])
        out = st.raw["face_yunet"][p, t]
        out[S.NOSE] = [lm[2, 0], lm[2, 1], s]
        # eye identity by proximity to the pose estimate
        e1, e2 = lm[0], lm[1]
        le, re = ref[S.L_EYE, :2], ref[S.R_EYE, :2]
        if np.isfinite(le).all() and np.isfinite(re).all():
            if np.linalg.norm(e1 - le) + np.linalg.norm(e2 - re) > np.linalg.norm(e1 - re) + np.linalg.norm(e2 - le):
                e1, e2 = e2, e1
            out[S.L_EYE] = [e1[0], e1[1], s]
            out[S.R_EYE] = [e2[0], e2[1], s]

    def _map_person_ids(self, tracklets, dets, old: Optional[TrackStore]) -> List[int]:
        """Keep person ids (and with them names, roles, keys) stable across re-analysis."""
        sx, sy = self.proxy.sx, self.proxy.sy
        existing = [p.id for p in self.cut.persons]
        next_id = max(existing + list(self.cut.removed) + (list(old.removed) if old is not None else []) + [-1]) + 1
        if old is None or (not old.pids and not old.removed):
            return list(range(next_id, next_id + len(tracklets)))
        cost = np.ones((len(tracklets), len(old.pids)))
        for i, tr in enumerate(tracklets):
            for j, pid in enumerate(old.pids):
                ious = []
                for f, di in tr.items():
                    if old.present[j, f] and np.isfinite(old.boxes[j, f]).all():
                        ious.append(iou(dets[f][di]["box"] / [sx, sy, sx, sy], old.boxes[j, f]))
                if len(ious) >= 3:
                    cost[i, j] = 1 - float(np.mean(ious))
        out = [-1] * len(tracklets)
        for i, j in assign(cost, 0.6):
            out[i] = old.pids[j]
        # tracklets that are a person the user deleted stay deleted (-2)
        for i, tr in enumerate(tracklets):
            if out[i] >= 0:
                continue
            for pid, boxes in old.removed.items():
                ious = [iou(dets[f][di]["box"] / [sx, sy, sx, sy], boxes[f]) for f, di in tr.items()
                        if f < len(boxes) and np.isfinite(boxes[f]).all()]
                if len(ious) >= 3 and float(np.mean(ious)) > 0.4:
                    out[i] = -2
                    break
        for i in range(len(out)):
            if out[i] == -1:
                out[i] = next_id
                next_id += 1
        return out

    def _update_persons(self, tracklets, dets, pids: List[int]) -> None:
        old = {p.id: p for p in self.cut.persons}
        persons = []
        self._new_pids = set()
        for p, (tr, pid) in enumerate(zip(tracklets, pids)):
            apps = [dets[f][i]["app"] for f, i in tr.items() if dets[f][i]["app"].any()]
            prof = np.mean(apps, 0).tolist() if apps else []
            reid, faces = _identity(tr, dets)
            if pid in old:
                op = old[pid]
                op.profile = prof
                op.reid, op.faces = reid, faces
                op.first, op.last = min(tr), max(tr)
                persons.append(op)
            else:
                role = "target"                       # everyone is retouched unless the user protects them
                self._new_pids.add(int(pid))
                persons.append(Person(id=pid, name=f"인물 {pid + 1}", color=PERSON_COLORS[pid % len(PERSON_COLORS)],
                                      role=role, profile=prof, first=min(tr), last=max(tr), reid=reid, faces=faces))
        self.cut.persons = persons
        try:
            _save_identity_samples(self.project, self.cut, tracklets, dets, pids)
        except Exception:
            import traceback
            traceback.print_exc()
        keep = set(pids)
        self.cut.joint_keys = {k: v for k, v in self.cut.joint_keys.items() if k in keep}

    # ------------------------------------------------------------------ fixes from the person finder (analysis)
    def _write_pose(self, st: TrackStore, p: int, t: int, res: dict, box_proxy) -> None:
        """A pose estimated again (on the person's own instance) replaces what the pose models gave this track."""
        sx, sy = self.proxy.sx, self.proxy.sy
        for k, (kp, sc) in res.items():
            if k in st.raw and st.raw[k].shape[2] == S.K:
                st.raw[k][p, t, :, :2] = kp[:S.K] / [sx, sy]
                st.raw[k][p, t, :, 2] = sc[:S.K]
            if k == "pose_rtmw_x":
                if "face_rtmw" in st.raw:
                    st.raw["face_rtmw"][p, t] = np.nan
                self._face_from_rtmw(st, p, t, kp / [sx, sy], sc)
                self._hand42(st, p, t, kp / [sx, sy], sc)
        st.boxes[p, t] = np.asarray(box_proxy, np.float32) / [sx, sy, sx, sy]

    def _fix_swaps(self, st: TrackStore, tracklets, collect: dict, a: int, b: int) -> list:
        """A track that went over to someone else between two frames (the tracker's identity switch): the finder's
        masks say the person a track had is now under another track, and the skeletons say the same (this track's
        joints jump, the other's continue where this one was). From that frame on the two exchange everything.
        Both signs are needed: a person stepping in front of another moves the masks only, a fast move the joints
        only."""
        from . import segment as SG
        if not self.seglab.exists():
            return []
        mw, mh = self.seglab.w, self.seglab.h
        min_area = 0.002 * mw * mh
        sx, sy = self.proxy.sx, self.proxy.sy
        swaps = []
        prev = None
        for t in range(a, b):
            g = self.seglab.get(t)
            cur = None if g is None else np.array(g, np.uint8)
            if prev is not None and cur is not None:
                pres0 = [p for p in range(len(st.pids)) if st.present[p, t - 1]]
                pres1 = [p for p in range(len(st.pids)) if st.present[p, t]]
                for pa in [p for p in pres0 if p in pres1]:
                    va = int(st.pids[pa]) + 1
                    if va >= 255:
                        continue
                    A0 = prev == va
                    na = int(A0.sum())
                    if na < min_area:
                        continue
                    cnt = np.bincount(cur[A0], minlength=256).astype(np.float64) / na
                    cands = [(cnt[int(st.pids[q]) + 1], q) for q in pres1 if q != pa and int(st.pids[q]) + 1 < 255]
                    if not cands:
                        continue
                    cax, px_ = max(cands)
                    if not (cax >= 0.4 and cax >= 2.0 * cnt[va] + 0.1):
                        continue
                    kA0 = SG._kp_for(st, pa, t - 1, False)
                    kX1 = SG._kp_for(st, px_, t, False)
                    kA1 = SG._kp_for(st, pa, t, False)
                    bx = st.boxes[pa, t - 1]
                    h = float(bx[3] - bx[1]) if np.isfinite(bx).all() else 0.0
                    # the skeleton may take a frame or two to follow the box and face that went over: this frame
                    # or one of the next two
                    jump = False
                    for k in range(3):
                        if t + k >= b or not (st.present[pa, t + k] and st.present[px_, t + k]):
                            break
                        dAA = _kp_dist(kA0, SG._kp_for(st, pa, t + k, False))
                        dAX = _kp_dist(kA0, SG._kp_for(st, px_, t + k, False))
                        if dAA is not None and dAX is not None and h > 0 and dAX < 0.6 * dAA and dAA > 0.15 * h * (1 + 0.5 * k):
                            jump = True
                            break
                    if not jump:
                        continue
                    vx = int(st.pids[px_]) + 1
                    if st.present[px_, t - 1]:
                        X0 = prev == vx
                        nx = int(X0.sum())
                        if nx >= min_area:
                            cx = np.bincount(cur[X0], minlength=256)
                            if cx[va] < cx[vx]:
                                continue
                        kX0 = SG._kp_for(st, px_, t - 1, False)
                        dXX, dXA = _kp_dist(kX0, kX1), _kp_dist(kX0, kA1)
                        if dXX is not None and dXA is not None and not dXA < dXX:
                            continue
                    pid_a, pid_x = int(st.pids[pa]), int(st.pids[px_])
                    self._swap_local(st, tracklets, collect, pa, px_, t, b)
                    lut = np.arange(256, dtype=np.uint8)
                    lut[va], lut[vx] = vx, va
                    cur = lut[cur]
                    swaps.append((pid_a, pid_x, t))
                    import logging
                    logging.getLogger("contour").info("identity switch fixed: %d <-> %d from frame %d", pid_a, pid_x, t)
                    break
            prev = cur
        self._swaps_fixed = swaps
        return swaps

    def _swap_local(self, st: TrackStore, tracklets, collect: dict, pa: int, pb: int, t0: int, t1: int) -> None:
        """Exchange two tracks over t0..t1-1 during analysis: the store, the labels, the tracklets, the sources."""
        va, vb = int(st.pids[pa]) + 1, int(st.pids[pb]) + 1
        st.swap_persons(int(st.pids[pa]), int(st.pids[pb]), t0, t1)
        lut = np.arange(256, dtype=np.uint8)
        lut[va], lut[vb] = vb, va
        mm = self.seglab.open_rw()
        try:
            for t in range(t0, t1):
                mm[t] = lut[np.asarray(mm[t])]
        finally:
            mm.flush()
            del mm
            self.seglab.close()
        ta, tb = tracklets[pa], tracklets[pb]
        ma = {f: i for f, i in ta.items() if t0 <= f < t1}
        mb = {f: i for f, i in tb.items() if t0 <= f < t1}
        for f in ma:
            del ta[f]
        for f in mb:
            del tb[f]
        ta.update(mb)
        tb.update(ma)
        ida, idb = int(st.pids[pa]), int(st.pids[pb])
        for t, fr in (collect.get("src") or {}).items():
            if t0 <= t < t1:
                sa, sb = fr.pop(ida, None), fr.pop(idb, None)
                if sa is not None:
                    fr[idb] = sa
                if sb is not None:
                    fr[ida] = sb
        self._label_cache.clear()

    def _label_new(self, pid: int, poses_t: dict, collect: dict) -> None:
        """A person added from the finder's instances takes the pixels marked nobody's (255) inside their instance."""
        from . import segment as SG
        if not self.seglab.exists() or not 0 <= pid < 254:
            return
        mw, mh = self.seglab.w, self.seglab.h
        kx, ky = mw / self.proxy.w, mh / self.proxy.h
        mm = self.seglab.open_rw()
        try:
            for t, (res, rec) in poses_t.items():
                (x0, y0, x1, y1), m = SG.record_mask(rec)
                bx0, by0 = int(round(x0 * kx)), int(round(y0 * ky))
                bx1, by1 = max(bx0 + 1, int(round(x1 * kx))), max(by0 + 1, int(round(y1 * ky)))
                mk = cv2.resize(m.astype(np.uint8), (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_NEAREST).astype(bool)
                cx0, cy0, cx1, cy1 = max(0, bx0), max(0, by0), min(mw, bx1), min(mh, by1)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                mk = mk[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
                sub = np.array(mm[t, cy0:cy1, cx0:cx1])
                sub[(sub == SG.UNTRACKED) & mk] = pid + 1
                mm[t, cy0:cy1, cx0:cx1] = sub
                collect.setdefault("src", {}).setdefault(t, {})[int(pid)] = "find"
        finally:
            mm.flush()
            del mm
            self.seglab.close()
        self._label_cache.clear()

    DUP_DIST = 0.15      # median joint distance, share of the box height, of a skeleton drawn again on a tracked person
    DUP_SHARE = 0.3      # share of the frames that makes the finder's instance part of that person

    def _duplicate_of(self, st: TrackStore, poses_t: dict):
        """(pid, frames) of the tracked person whose skeleton the masked pose of a finder chain repeats, or None."""
        sx, sy = self.proxy.sx, self.proxy.sy
        # the skeletons as the pose models gave them: final is only made from them after this step
        refs = [st.raw[k] for k in ("pose_rtmw_x", "pose_rtmpose_x") if k in st.raw and st.raw[k].shape[2] >= S.K]
        hits = {}
        for t, (res, rec) in poses_t.items():
            kp_, sc_ = res.get("pose_rtmw_x", next(iter(res.values())))
            k = np.asarray(kp_, np.float32)[:S.K, :2] / [sx, sy]
            sc = np.asarray(sc_, np.float32)[:S.K]
            bh = max(float(rec["box"][3] - rec["box"][1]) / sy, 1.0)
            for q in range(len(st.pids)):
                if not st.present[q, t]:
                    continue
                for kq in [r[q, t, :S.K] for r in refs] + [st.final[q, t, :S.K]]:
                    ok = np.isfinite(kq[:, :2]).all(1) & np.isfinite(k).all(1)
                    ok &= sc >= float(np.median(sc))          # the chain's surer half of the joints
                    if ok.sum() < 5:
                        continue
                    d = float(np.median(np.linalg.norm(k[ok] - kq[ok, :2], axis=1)))
                    if d < self.DUP_DIST * bh:
                        hits.setdefault(int(st.pids[q]), []).append(t)
                        break
        if not hits:
            return None
        q_pid, fr = max(hits.items(), key=lambda kv: len(kv[1]))
        if len(fr) < self.DUP_SHARE * len(poses_t):
            return None
        return q_pid, sorted(fr)

    def _add_found_people(self, st: TrackStore, tracklets, dets, pids, collect: dict, poses, a: int, b: int,
                          cancel=None) -> list:
        """People the finder sees on many frames whom tracking never followed (a dancer behind another: the pose
        model, given her box, took the dancer in front, and the double was dropped). Their instances are linked
        from frame to frame, their pose estimated with everybody else hidden, and they become tracked people."""
        from .assoc import appearance
        from .people import MAX_PEOPLE
        ex = collect.get("extras") or {}
        if not ex:
            return []
        chains = []
        for t in sorted(ex):
            recs = ex[t]
            used = set()
            pairs = []
            for ci, c in enumerate(chains):
                if t - c["last"] > 3:
                    continue
                for ri, r in enumerate(recs):
                    v = iou(c["box"], r["box"])
                    if v >= 0.3:
                        pairs.append((v, ci, ri))
            pairs.sort(reverse=True)
            done_c = set()
            for v, ci, ri in pairs:
                if ci in done_c or ri in used:
                    continue
                c = chains[ci]
                c["recs"][t] = recs[ri]
                c["last"], c["box"] = t, recs[ri]["box"]
                done_c.add(ci)
                used.add(ri)
            for ri, r in enumerate(recs):
                if ri not in used:
                    chains.append(dict(recs={t: r}, last=t, box=r["box"]))
        need = max(6, int(0.15 * (b - a)))
        chains = [c for c in chains if len(c["recs"]) >= need
                  and len(c["recs"]) >= 0.5 * (max(c["recs"]) - min(c["recs"]) + 1)]
        chains.sort(key=lambda c: -len(c["recs"]))
        sx, sy = self.proxy.sx, self.proxy.sy
        added = []
        frames = {}
        for c in chains:
            if len(st.pids) >= MAX_PEOPLE + 4:
                break
            poses_t = {}
            for t, rec in sorted(c["recs"].items()):
                if cancel and cancel():
                    raise Cancelled()
                if t not in frames:
                    frames[t] = analysis_view(np.asarray(self.proxy.frame(t)))
                res = _masked_pose(frames[t], rec, poses, upright=True)
                if res is not None:
                    poses_t[t] = (res, rec)
            if len(poses_t) < need:
                continue
            # someone already tracked: the finder split a person in two (the head of a dancer behind two others
            # apart from her body, ILLIT 213ZQ6YQ) and the pose model, shown only that part, drew her whole skeleton
            # again where her track already has it. Not a new person: the part's pixels nobody had are hers
            dup = self._duplicate_of(st, poses_t)
            if dup is not None:
                q_pid, q_frames = dup
                self._label_new(q_pid, {t: (None, c["recs"][t]) for t in q_frames}, collect)
                import logging
                logging.getLogger("contour").info("finder instance is part of person %d (%d frames): not added",
                                                  q_pid, len(q_frames))
                continue
            # what is a dancer to add and what is not: a sure pose (a blurred shoulder in the foreground gives a
            # pose of sorts too), not a blob filling the frame or cut by two of its edges (out-of-focus foreground),
            # and something that moves (a figure on a poster does not)
            W_, H_ = float(self.proxy.w), float(self.proxy.h)
            qs, big, mot = [], 0, []
            for t, (res, rec) in poses_t.items():
                kp_, sc_ = res.get("pose_rtmw_x", next(iter(res.values())))
                qs.append(float(np.median(sc_[:S.K])))
                x0_, y0_, x1_, y1_ = [float(v) for v in rec["box"]]
                edges = (x0_ <= 2) + (y0_ <= 2) + (x1_ >= W_ - 2) + (y1_ >= H_ - 2)
                if (x1_ - x0_) * (y1_ - y0_) > 0.35 * W_ * H_ or edges >= 2:
                    big += 1
            bxs = np.array([rec["box"] for t, (res, rec) in sorted(poses_t.items())], np.float32)
            hh = float(np.median(bxs[:, 3] - bxs[:, 1]))
            span = float(np.max(np.ptp(bxs, axis=0))) / max(hh, 1.0)
            # pose scores are not 0..1 (they depend on the model): against the tracked people of this shot
            ref = st.raw.get("pose_rtmw_x")
            if ref is not None:
                rr = ref[:len(st.pids), a:b, :, 2]
                rr = rr[np.isfinite(rr)]
                ref_q = float(np.median(rr)) if rr.size else 0.0
            else:
                ref_q = 0.0
            if np.median(qs) < 0.6 * ref_q or big > 0.5 * len(poses_t) or span < 0.03:
                import logging
                logging.getLogger("contour").info("finder instance not added (pose %.2f of %.2f, foreground %d/%d, motion %.3f)",
                                                  float(np.median(qs)), ref_q, big, len(poses_t), span)
                continue
            existing = [q.id for q in self.cut.persons] + list(self.cut.removed) + list(st.removed) + list(st.pids)
            pid = max(existing + [-1]) + 1
            st.add_person(pid)
            p = len(st.pids) - 1
            tr = {}
            for t, (res, rec) in poses_t.items():
                self._write_pose(st, p, t, res, rec["box"])
                st.present[p, t] = True
                kp, sc = res.get("pose_rtmw_x", next(iter(res.values())))
                avg = np.column_stack([kp[:S.K], sc[:S.K]]).astype(np.float32)
                d = dict(box=np.asarray(rec["box"], np.float32), kp=avg, score=float(rec["score"]),
                         app=appearance(frames[t], avg, rec["box"]), reid=None, face=None)
                d["i"] = len(dets[t])
                dets[t].append(d)
                tr[t] = d["i"]
            fr = sorted(poses_t)
            for f0, f1 in zip(fr, fr[1:]):
                if 1 < f1 - f0 <= 4:
                    for f in range(f0 + 1, f1):
                        w_ = (f - f0) / (f1 - f0)
                        st.present[p, f] = True
                        st.boxes[p, f] = (1 - w_) * st.boxes[p, f0] + w_ * st.boxes[p, f1]
            tracklets.append(tr)
            pids.append(pid)
            added.append(pid)
            # every frame the finder had this person on (also where the masked pose was unsure)
            self._label_new(pid, {t: (None, rec) for t, rec in c["recs"].items()}, collect)
            import logging
            logging.getLogger("contour").info("person the finder saw added: %d (%d frames)", pid, len(poses_t))
        self._added_pids = set(added)
        st.invalidate_cache()
        return added

    def _refiner_extra(self, st: TrackStore):
        try:
            from ..learn.refiner import active_runtime, refine_measurements
        except Exception:
            return None
        rt = active_runtime()
        if rt is None:
            return None
        try:
            return refine_measurements(st, self.proxy, rt)
        except Exception:
            return None

    def _run_matte(self, a: int, b: int, progress, st=None, accuracy: str = "high", use_final: bool = False,
                   cancel=None, repose=None, collect: Optional[dict] = None) -> None:
        from .infer import Matter
        try:
            matter = Matter()
        except Exception:
            matter = None
        if st is not None and accuracy != "fast" and getattr(self.cut.tracking, "seg", "sam2") == "sam2":
            from . import segment as SG
            try:
                if not self.matte.exists():
                    self.matte.create()
                    self.matte.close()
                used: dict = {}
                self._seg_stats = SG.run(st, self.proxy, self.matte, self.seglab, a, b, SG.model_for(accuracy),
                                         use_final, progress, cancel, matter, repose=repose, collect=collect,
                                         models=used)
                self._label_cache.clear()
                if self._stage_extra is not None:
                    # written beside the matte in use: its record and the marks follow when it is put in use
                    self._stage_meta = dict(model=used.get("used", ""))
                    return
                self.write_seg_meta(model=used.get("used", ""), ranges=[[int(a), int(b)]], full=True)
                self._matte_marks_done(a, b)
                return
            except Cancelled:
                raise                       # cancel pressed: stop here (it was taken for a failure and the whole RVM matte followed)
            except Exception as e:          # no model / no memory: the RVM matte still works
                import logging
                logging.getLogger("contour").warning("SAM matte failed, using RVM: %s", e)
                self.seglab.remove()
                if matter is None and self._stage_extra is not None:
                    # neither model can make it: a matte written half way is never put in use (1.22.2 left it
                    # in the file in use); the one before stays
                    raise RuntimeError(f"인물 매트를 만들지 못했습니다 (SAM 실패, RVM 모델 없음): {e}") from e
        if matter is None:
            return
        self.seglab.remove()
        mm = self.matte.create()
        mw, mh = self.matte.w, self.matte.h
        matter.reset()
        # warm the recurrent state on the first frames
        for t in range(a, min(b, a + 4)):
            matter(cv2.resize(analysis_view(np.asarray(self.proxy.frame(t))), (mw, mh), interpolation=cv2.INTER_AREA))
        matter.reset()
        for t in range(a, b):
            small = cv2.resize(analysis_view(np.asarray(self.proxy.frame(t))), (mw, mh), interpolation=cv2.INTER_AREA)
            al = matter(small)
            mm[t] = np.clip(al * 255 + 0.5, 0, 255).astype(np.uint8)
            progress((t - a + 1) / max(1, b - a), "인물 매트 생성")
        mm.flush()
        del mm
        self.matte.close()

    def _person_mask_proxy(self, t: int) -> Optional[np.ndarray]:
        m = self.matte.get(t)
        if m is not None:
            return cv2.resize(np.asarray(m), (self.proxy.w, self.proxy.h), interpolation=cv2.INTER_LINEAR)
        if self.store is None:
            return None
        mask = np.zeros((self.proxy.h, self.proxy.w), np.uint8)
        for p in range(len(self.store.pids)):
            if self.store.present[p, t] and np.isfinite(self.store.boxes[p, t]).all():
                x1, y1, x2, y2 = self.store.boxes[p, t] * [self.proxy.sx, self.proxy.sy, self.proxy.sx, self.proxy.sy]
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        return mask

    # ------------------------------------------------------------------ re-solve with corrections
    def _extras_for(self, persons: List[int], frame_range: Tuple[int, int], progress: Progress = None):
        st = self.store
        cut = self.cut
        extras = []
        offset = np.zeros((len(st.pids), st.n, S.K, 2), np.float32)
        if cut.tracking.refiner:
            rx = self._refiner_extra(st)
            if rx:
                extras.append(rx)
        if cut.tracking.adapt:
            keys = self.keys_by_index()
            txy = np.full((len(st.pids), st.n, S.K, 2), np.nan, np.float32)
            tvar = np.full((len(st.pids), st.n, S.K), np.inf, np.float32)
            any_t = False
            for p in persons:
                kd = keys.get(p) or {}
                conf_refs = self._confirmed_refs(p)
                if not kd and not conf_refs:
                    continue
                bias, wgt = learn_bias(st, p, kd)
                st.adapt_bias[p] = bias * wgt[:, None]
                offset[p] = bias_offsets(st, p, bias, wgt)
                xy, var = template_measurements(st, p, kd, self.proxy.gray, self.proxy.sx, self.proxy.sy,
                                                st.final[p, :, :, :2], extra_refs=conf_refs, progress=progress,
                                                frame_range=frame_range)
                txy[p], tvar[p] = xy, var
                any_t = True
            if any_t:
                extras.append((txy, tvar))
        return extras, offset

    def _confirmed_refs(self, p: int) -> Dict[int, Dict[int, List[float]]]:
        st = self.store
        pid = st.pids[p]
        refs: Dict[int, Dict[int, List[float]]] = {}
        for a, b in self.cut.confirmed.get(pid, []):
            for f in range(a, b + 1, 12):
                if 0 <= f < st.n and st.present[p, f]:
                    for j in S.CORE_JOINTS:
                        v = st.final[p, f, j]
                        if np.isfinite(v[:2]).all():
                            refs.setdefault(j, {})[f] = v[:2].tolist()
        return refs

    def resolve_all(self, progress: Progress = None) -> None:
        st = self.store
        if st is None:
            return
        a, b = self.cut.work_range
        P = list(range(len(st.pids)))
        extras, offset = self._extras_for(P, (a, b), progress)
        fs = self.frame_source() if self.cut.tracking.point_tracker else None
        vis = self.effective_vis()
        trk, fconf, flags = SV.solve(st, fs, P, (a, b), user_keys=self.keys_by_index(), extra=extras or None,
                                     smoothing=self.cut.tracking.smoothing,
                                     conf_threshold=self.cut.tracking.conf_threshold, progress=progress,
                                     offset=offset, vis=vis)
        self._commit(P, (a, b), trk, fconf, flags, blend=0, vis=vis)
        self.update_widths()

    def resolve_local(self, pid: int, frame: int, margin: int = 40, frame_end: Optional[int] = None) -> Tuple[int, int]:
        """Re-solve one person around an edit (frame .. frame_end). Returns the frame range that changed."""
        st = self.store
        if st is None or not st.has(pid):
            return (frame, frame + 1)
        p = st.index(pid)
        a0, b0 = self.cut.work_range
        last = frame if frame_end is None else max(frame, frame_end)
        keys = self.cut.joint_keys.get(pid, {})
        kframes = sorted({f for fr in keys.values() for f in fr})
        prev_k = max([f for f in kframes if f < frame], default=frame)
        next_k = min([f for f in kframes if f > last], default=last)
        a = max(a0, prev_k - margin)
        b = min(b0, next_k + margin + 1)
        extras, offset = self._extras_for([p], (a, b))
        fs = self.frame_source() if self.cut.tracking.point_tracker else None
        vis = self.effective_vis()
        trk, fconf, flags = SV.solve(st, fs, [p], (a, b), user_keys=self.keys_by_index(), extra=extras or None,
                                     smoothing=self.cut.tracking.smoothing,
                                     conf_threshold=self.cut.tracking.conf_threshold, offset=offset, vis=vis)
        self._commit([p], (a, b), trk, fconf, flags, blend=min(12, margin // 2), keep_edges=(a > a0, b < b0),
                     vis=vis)
        self.update_widths([p], (a, b))
        self.invalidate_labels(range(a, b))
        return a, b

    def effective_vis(self) -> np.ndarray:
        """Model visibility with the user's decisions applied (keyframes show, hidden ranges hide)."""
        from .visibility import apply_overrides
        st = self.store
        base = st.vis_meas if st.has_vis else np.ones_like(st.vis_meas)
        hidden = {}
        for pid, jd in self.cut.hidden.items():
            if st.has(pid):
                hidden[st.index(pid)] = jd
        return apply_overrides(base, st, self.keys_by_index(), hidden)

    # ------------------------------------------------------------------ face / head / hands for the warp
    def face_geom(self, p: int, t: int):
        from ..warp.face import FaceGeom
        st = self.store
        if st is None or not st.has_face:
            return None
        return FaceGeom.from_landmarks(st.face[p, t])

    def _virtual_face(self, p: int, t: int) -> Optional[np.ndarray]:
        """(68,2) where the face would be on frame t, carried over from the nearest frame where it was seen
        (by the neck and the head centre): keeps head size / hair edits going while she shows her back."""
        st = self.store
        refs = np.nonzero((st.face_vis[p] >= 0.55) & np.isfinite(st.face[p, :, :, :2]).all((1, 2)))[0]
        if not len(refs):
            return None
        r = int(refs[np.argmin(np.abs(refs - t))])

        def anchor(f):
            kp = st.final[p, f]
            n_, h_ = B._pt(kp, "neck"), B._pt(kp, "head")
            if n_ is None or h_ is None or np.hypot(*(h_ - n_)) < 2:
                return None
            return n_, h_
        a0, a1 = anchor(r), anchor(t)
        if a0 is None or a1 is None:
            return None
        v0, v1 = a0[1] - a0[0], a1[1] - a1[0]
        sc = float(np.hypot(*v1) / max(np.hypot(*v0), 1e-6))
        if not 0.4 < sc < 2.5:
            return None
        ang = float(np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0]))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * sc
        return a1[0][None] + (st.face[p, r, :, :2].astype(np.float64) - a0[0][None]) @ R.T

    def head_face(self, p: int, t: int):
        """Face geometry for head edits: the tracked face, blended into a carried-over one where the face is
        not seen (turned away), so head size, crown and hair volume hold through a turn."""
        from ..warp.face import FaceGeom
        st = self.store
        if st is None or not st.has_face:
            return None
        real = st.face[p, t]
        real_ok = bool(np.isfinite(real[:, :2]).all())
        w = float(np.clip((st.face_vis[p, t] - 0.1) / 0.4, 0, 1)) if real_ok else 0.0
        if w >= 0.999:
            return FaceGeom.from_landmarks(real)
        virt = self._virtual_face(p, t)
        if virt is None:
            return FaceGeom.from_landmarks(real) if real_ok else None
        lm = virt if w <= 0 else w * real[:, :2].astype(np.float64) + (1 - w) * virt
        return FaceGeom.from_landmarks(lm)

    def head_geom(self, p: int, t: int, face=None, use_parts: bool = True):
        """Head outline (with hair) from the matte, steadied over +-3 frames in face units. With head parts
        found (hair + face), the outline is measured on those, so a hand or an arm beside the head is not hair."""
        from ..warp import face as FW
        from . import parts as PT
        st = self.store
        use_parts = use_parts and PT.ready_for(self, st.pids[p]) if st is not None else False
        if st is None:
            return None
        face = face if face is not None else self.face_geom(p, t)
        if face is None:
            return None
        cache = getattr(self, "_head_raw", None)
        if cache is None:
            cache = self._head_raw = {}
        rows = []
        for f in range(max(0, t - 3), min(st.n, t + 4)):
            key = (p, f, use_parts)
            if key not in cache:
                r = None
                fg = face if f == t else self.face_geom(p, f)
                al = PT.head_alpha(self, p, f) if use_parts else None
                if al is not None:
                    lab = None
                else:
                    al = self.matte.get(f)
                    lab = self.labels(f) if al is not None else None
                if fg is not None and al is not None and st.present[p, f]:
                    r = FW.head_outline(np.asarray(al), lab, p, self.matte.w / self.cut.info.width,
                                        self.matte.h / self.cut.info.height, fg)
                cache[key] = r
                if len(cache) > 4000:
                    cache.pop(next(iter(cache)))
            if cache[key] is not None:
                rows.append(cache[key])
        if not rows:
            return None
        radii = np.nanmedian(np.vstack(rows), axis=0)
        return FW.head_from_radii(face, radii)

    def hand_polys(self, p: int, t: int):
        """Hand outlines (source px) from the tracked hand points, used to keep hands out of arm warps."""
        st = self.store
        if st is None or getattr(st, "hand", None) is None or not st.has_hands:
            return []
        out = []
        for h in range(2):
            pts = st.hand[p, t, h * 21:(h + 1) * 21, :2]
            ok = np.isfinite(pts).all(1) & (st.hand[p, t, h * 21:(h + 1) * 21, 2] > 0.25)
            if ok.sum() < 6:
                continue
            hull = cv2.convexHull(pts[ok].astype(np.float32)).reshape(-1, 2).astype(np.float64)
            c = hull.mean(0)
            hull = c + (hull - c) * 1.25
            out.append(hull)
        return out

    def warp_geometry(self, p: int):
        """(geometry (N,K,3), segment weights (N,N_SEG), shoulder weight (N,)) for the warp, cached."""
        from .visibility import segment_weights
        cache = getattr(self, "_geo_cache", None)
        if cache is None:
            cache = self._geo_cache = {}
        if p not in cache:
            st = self.store
            cache[p] = segment_weights(st.final[p], st.vis[p], st.present[p])
        return cache[p]

    def _commit(self, persons, rng, trk, fconf, flags, blend: int = 0, keep_edges=(False, False),
                vis: Optional[np.ndarray] = None) -> None:
        from .store import VIS_HIDDEN, VIS_VISIBLE
        st = self.store
        a, b = rng
        T = b - a
        w = np.ones(T, np.float32)
        if blend > 0:
            ramp = np.linspace(0, 1, blend + 2)[1:-1]
            if keep_edges[0]:
                w[:len(ramp)] = ramp[:T]
            if keep_edges[1]:
                w[-len(ramp):] = np.minimum(w[-len(ramp):], ramp[::-1][-T:])
        for qi, p in enumerate(persons):
            pid = st.pids[p]
            old = st.final[p, a:b].copy()
            new = trk[qi]
            ww = w[:, None, None]
            mix = np.where(np.isfinite(old) & np.isfinite(new), old * (1 - ww) + new * ww,
                           np.where(np.isfinite(new), new, old))
            new_vis = vis[p, a:b].copy() if vis is not None else st.vis[p, a:b].copy()
            if vis is not None:
                mix[new_vis == VIS_HIDDEN] = np.nan
            # locked ranges keep their previous result
            for la, lb in self.cut.locked.get(pid, []):
                lo, hi = max(a, la), min(b, lb + 1)
                if lo < hi:
                    mix[lo - a:hi - a] = old[lo - a:hi - a]
                    new_vis[lo - a:hi - a] = st.vis[p, lo:hi]
            # keyframes are exact
            for j, fr in (self.cut.joint_keys.get(pid) or {}).items():
                for f, xy in fr.items():
                    if a <= f < b:
                        mix[f - a, j, :2] = xy
                        mix[f - a, j, 2] = 1.0
                        new_vis[f - a, j] = VIS_VISIBLE
            st.final[p, a:b] = mix
            st.vis[p, a:b] = new_vis
            st.flags[p, a:b] = flags[qi]
            if getattr(self, "_geo_cache", None):
                self._geo_cache.pop(p, None)
        st.conf[:, a:b] = SV.frame_conf_from_track(st.final[:, a:b], st.present[:, a:b], st.vis[:, a:b],
                                                   st.flags[:, a:b])

    # ------------------------------------------------------------------ widths
    # ------------------------------------------------------------------ person matte: what made it, when
    @property
    def _seg_meta_path(self) -> Path:
        return self.project.cache_dir(self.cut) / "seg_meta.json"

    def seg_meta(self) -> dict:
        """{'model': 'SAM 3.1' | 'SAM 2.1' | '', 'made': time of the whole matte, 'refreshed': [[a, b, time], ...]}"""
        import json
        try:
            return json.loads(self._seg_meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _matte_marks_done(self, a: int, b: int) -> None:
        """The matte of frames a..b was just made from the current tracking: the frames an edit had marked for a
        refresh there (ui/matte_refresh.py) are up to date."""
        st = dict(self.cut.status or {})
        if not st.get("matte_dirty"):
            return
        from .matte_refresh import subtract
        left = subtract(st["matte_dirty"], [[int(a), int(b)]])
        if left:
            st["matte_dirty"] = left
        else:
            st.pop("matte_dirty", None)
        self.cut.status = st

    def write_seg_meta(self, model: str = "", ranges=None, full: bool = False,
                       revision: Optional[int] = None) -> None:
        """revision: the matte revision this makes the one in use (matte_rev: written last, after everything that
        belongs to the revision). full with no model: the matte is not a SAM one (no owners)."""
        import json
        import time as _t
        m = {} if full else self.seg_meta()
        keep_rev = int(self.seg_meta().get("revision", 0) or 0) if full else int(m.get("revision", 0) or 0)
        if model:
            m["model"] = model
        m["revision"] = int(revision) if revision is not None else keep_rev
        now = _t.time()
        if full:
            m["made"] = now
            m["refreshed"] = []
            if ranges:
                m["range"] = [int(ranges[0][0]), int(ranges[0][1])]
        else:
            m.setdefault("refreshed", []).extend([[int(a), int(b), now] for a, b in (ranges or [])])
            m["refreshed"] = m["refreshed"][-50:]
        try:
            tmp = self._seg_meta_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(m), encoding="utf-8")
            tmp.replace(self._seg_meta_path)
        except OSError:
            if revision is not None:
                raise                       # the revision is not in use without it: the caller takes it back

    def update_widths(self, persons: Optional[List[int]] = None, rng: Optional[Tuple[int, int]] = None) -> None:
        out = self.measure_widths(persons, rng)
        if out is None:
            return
        st = self.store
        for p in out["persons"]:
            st.widths[p], st.contour_raw[p], st.contour[p] = out["widths"][p], out["contour_raw"][p], out["contour"][p]
        st.has_contour = True

    def measure_widths(self, persons: Optional[List[int]] = None, rng: Optional[Tuple[int, int]] = None,
                       staged: Optional[dict] = None, base: Optional[dict] = None) -> Optional[dict]:
        """The body widths and outline points of the people over rng (default: the work range), measured and
        smoothed, without changing the tracks: {'persons': [...], 'widths': {p: array}, 'contour_raw': {p: ...},
        'contour': {p: ...}}. update_widths() puts them in.

        staged: {'matte': {t: alpha}, 'labels': {t: owner frame}} of a matte revision that is not in use yet:
        those frames are measured on it, every other frame on the matte in use (matte_rev.stage). base: the result
        of an earlier call to go on from (several ranges of one revision)."""
        st = self.store
        if st is None:
            return None
        a, b = rng or self.cut.work_range
        persons = list(range(len(st.pids))) if persons is None else persons
        mw, mh = self.matte.w, self.matte.h
        scale_xy = (mw / self.cut.info.width, mh / self.cut.info.height)
        A, Bb = self.cut.work_range
        sm = (staged or {}).get("matte") or {}
        sl = (staged or {}).get("labels") or {}
        lab_cache: Dict[int, np.ndarray] = {}

        def alpha_at(t):
            return sm[t] if t in sm else self.matte.get(t)

        def labels_at(t, alpha):
            if alpha is None:
                return None
            if t not in sm and t not in sl:
                return self.labels(t)
            lab = lab_cache.get(t)
            if lab is None:
                seg = sl[t] if t in sl else (self.seglab.get(t) if self.seglab.exists() else None)
                lab = lab_cache[t] = self.labels_from(t, seg, alpha)
            return lab

        out = dict(persons=list(persons), widths={}, contour_raw={}, contour={})
        if base is not None:
            for k in ("widths", "contour_raw", "contour"):
                out[k].update(base[k])
            out["persons"] = sorted(set(base["persons"]) | set(persons))
        for p in persons:
            prev = out["widths"][p] if p in out["widths"] else st.widths[p]
            raw = np.full((st.n, B.N_SEG, 2), np.nan, np.float32)
            fb = np.full((st.n, B.N_SEG), np.nan, np.float32)
            # measure over the whole work range so smoothing has context
            for t in range(A, Bb):
                kp = st.final[p, t]
                if not st.present[p, t] or not np.isfinite(kp[:, :2]).any():
                    continue
                ends = B.segment_endpoints(kp)
                fb[t] = B.fallback_widths(kp, ends)
                if a <= t < b or not np.isfinite(prev[t]).any():
                    alpha = alpha_at(t)
                    lab = labels_at(t, alpha)
                    raw[t] = B.measure_widths(kp, None if alpha is None else np.asarray(alpha), lab, p, scale_xy)
                else:
                    raw[t] = np.nan
            # reuse previous raw-equivalent values outside the edit window (approximate by the stored widths)
            outside = np.ones(st.n, bool)
            outside[a:b] = False
            raw = np.where(outside[:, None, None] & np.isfinite(prev), prev, raw)
            # a width never measured in this cut takes 30; one measured in some frames takes its median elsewhere (the
            # median only of those: one never measured made numpy warn "All-NaN slice" into the log at every analysis)
            fin = np.isfinite(fb)
            has = fin.any(axis=0)
            med = np.full((1,) + fb.shape[1:], np.nan, fb.dtype)
            if has.any():
                med[0][has] = np.nanmedian(fb[:, has], axis=0)
            fbf = np.where(fin, fb, med if has.any() else 30.0)
            fbf = np.where(np.isfinite(fbf), fbf, 30.0)
            out["widths"][p] = B.smooth_widths(raw, fbf, st.present[p])
            # body contour points (edge distances along the limbs and torso)
            craw = (out["contour_raw"][p] if p in out["contour_raw"] else st.contour_raw[p]).copy()
            for t in range(max(a, A), min(b, Bb)):
                kp = st.final[p, t]
                if not st.present[p, t] or not np.isfinite(kp[:, :2]).any():
                    craw[t] = np.nan
                    continue
                alpha = alpha_at(t)
                lab = labels_at(t, alpha)
                gray = self.proxy.gray(t)
                craw[t] = B.measure_contour(kp, None if alpha is None else np.asarray(alpha), lab, p, scale_xy,
                                            gray, (self.proxy.sx, self.proxy.sy))
            cfb = fbf[:, [B.SEG_INDEX[nm] for nm, _ in B.CONTOUR_SEGS]]
            out["contour_raw"][p] = craw
            out["contour"][p] = B.smooth_contour(craw, cfb, st.present[p])
        return out

    # ------------------------------------------------------------------ hidden joints (user decisions)
    def set_hidden(self, pid: int, joints: List[int], a: int, b: int, hide: bool) -> None:
        """Mark joints as not visible on frames a..b (inclusive), or give that range back to the automatic
        decision. Hiding also drops keyframes of those joints inside the range."""
        cut = self.cut
        for j in joints:
            ranges = [list(r) for r in cut.hidden.get(pid, {}).get(j, [])]
            out = []
            for r0, r1 in ranges:
                if r1 < a or r0 > b:
                    out.append([r0, r1])
                else:
                    if r0 < a:
                        out.append([r0, a - 1])
                    if r1 > b:
                        out.append([b + 1, r1])
            if hide:
                out.append([a, b])
                kf = cut.joint_keys.get(pid, {}).get(j, {})
                for f in [f for f in kf if a <= f <= b]:
                    del kf[f]
            out.sort()
            merged = []
            for r in out:
                if merged and r[0] <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], r[1])
                else:
                    merged.append(r)
            if merged:
                cut.hidden.setdefault(pid, {})[j] = merged
            else:
                cut.hidden.get(pid, {}).pop(j, None)
        if pid in cut.hidden and not cut.hidden[pid]:
            cut.hidden.pop(pid)

    def hidden_at(self, pid: int, j: int, t: int) -> bool:
        return any(r0 <= t <= r1 for r0, r1 in self.cut.hidden.get(pid, {}).get(j, []))

    def ghost(self, pid: int, t: int) -> Dict[int, np.ndarray]:
        """Best guesses of the joints that count as not visible on frame t (for drawing and reviving)."""
        from .store import VIS_VISIBLE
        st = self.store
        if st is None or not st.has(pid):
            return {}
        p = st.index(pid)
        if not st.present[p, t]:
            return {}
        out = {}
        for j in range(S.K):
            if st.vis[p, t, j] == VIS_VISIBLE and np.isfinite(st.final[p, t, j, :2]).all():
                continue
            if np.isfinite(st.final[p, t, j, :2]).all():
                continue          # outside the frame: drawn with the skeleton
            for src in (st.meas, st.auto):
                v = src[p, t, j, :2]
                if np.isfinite(v).all():
                    out[j] = v.copy()
                    break
        return out

    # ------------------------------------------------------------------ persons editing
    def delete_person(self, pid: int) -> None:
        """Remove a wrongly found person (or object). Re-analysis will not bring it back."""
        st = self.store
        cut = self.cut
        if st is not None and st.has(pid):
            p = st.index(pid)
            try:
                from . import matte_rev as MRV
                MRV.forget_person(self, p)       # also out of what the matte was made from (no refresh for it)
            except Exception:
                import traceback
                traceback.print_exc()
            st.removed[pid] = st.boxes[p].copy()
            st.keep_persons([q for q in st.pids if q != pid])
        cut.persons = [p for p in cut.persons if p.id != pid]
        for d in (cut.params, cut.joint_keys, cut.hidden, cut.locked, cut.confirmed):
            d.pop(pid, None)
        if pid not in cut.removed:
            cut.removed.append(pid)
        self._label_cache.clear()
        self._geo_cache = {}
        self._head_raw = {}
        self._skin_models = {}
        self.save()

    @_cut_models
    def rebuild_matte(self, progress: Progress = None, cancel: Optional[Callable[[], bool]] = None,
                      accuracy: str = "high") -> None:
        """Make the person matte again with SAM 2, prompted by the current (corrected) tracking, then
        re-measure body widths and outline points."""
        st = self.store
        if st is None:
            return
        import copy as _copy
        from . import matte_rev as MRV
        a, b = self.cut.work_range
        acc = "max" if accuracy == "max" else "high"
        p_m = (lambda x, m="": progress(0.9 * x, m)) if progress else None
        # one revision (instruction 2026-11-20 §1-2): the new matte and owners beside the ones in use, the widths
        # and outlines measured on them, all put in use together; cancelled or failed, what was in use stays
        keep = (st.widths.copy(), st.contour_raw.copy(), st.contour.copy(), st.has_contour, st.matte_rev)
        MRV.begin(self, copy=not (a <= 0 and b >= self.proxy.n))
        try:
            self._run_matte(a, b, p_m, st=st, accuracy=acc, use_final=True, cancel=cancel)
            self._label_cache.clear()
            self._head_raw = {}
            self._skin_models = {}
            if progress:
                progress(0.92, "몸 폭·윤곽 점 다시 측정")
            self.update_widths()
            MRV.commit(self, (a, b))
        except BaseException:
            if MRV.staging(self):
                MRV.drop(self)
            st.widths, st.contour_raw, st.contour, st.has_contour, st.matte_rev = keep
            self._label_cache.clear()
            self._geo_cache = {}
            self._head_raw = {}
            self._skin_models = {}
            raise
        if progress:
            progress(1.0, "완료")

    def merge_persons(self, keep_pid: int, drop_pid: int) -> None:
        if self.seglab.exists():
            import json
            al = self._seg_alias()
            al[int(drop_pid)] = int(keep_pid)
            for k, v in list(al.items()):
                if v == drop_pid:
                    al[k] = int(keep_pid)
            try:
                self._alias_path.write_text(json.dumps(al), encoding="utf-8")
            except OSError:
                pass
        self.store.merge_persons(keep_pid, drop_pid)
        try:
            # the two parts may have been labelled left/right differently (a turn split them): make it one
            from .orient import fix_left_right
            st = self.store
            W, H = self.cut.info.width, self.cut.info.height
            fix_left_right(st, self.cut.work_range)
            SV.fuse_measurements(st, self.cut.tracking.conf_threshold, frame_wh=(W, H), alpha_at=self.matte.get,
                                 alpha_scale=(self.matte.w / W, self.matte.h / H))
        except Exception:
            pass
        kk = self.cut.joint_keys.pop(drop_pid, {})
        for j, fr in kk.items():
            self.cut.joint_keys.setdefault(keep_pid, {}).setdefault(j, {}).update(fr)
        hh = self.cut.hidden.pop(drop_pid, {})
        for j, rs in hh.items():
            self.cut.hidden.setdefault(keep_pid, {}).setdefault(j, []).extend(rs)
        self.cut.persons = [p for p in self.cut.persons if p.id != drop_pid]
        self.cut.params.pop(drop_pid, None)
        self._label_cache.clear()
        self._geo_cache = {}
        self._head_raw = {}
        self._skin_models = {}
        self.resolve_all()

    # ------------------------------------------------------------------ identity fixes (group dance)
    def swap_persons(self, pid_a: int, pid_b: int, t0: int, t1: Optional[int] = None) -> None:
        """From frame t0 (to t1, exclusive; default the end of the work range) the two people swap identities:
        tracking, segmentation, joint edits and hidden-joint marks all move with them."""
        t1 = int(self.cut.work_range[1] if t1 is None else t1)
        self.store.swap_persons(pid_a, pid_b, t0, t1)
        self._move_identity({pid_a: pid_b, pid_b: pid_a}, t0, t1)

    def split_person(self, pid: int, t0: int, t1: Optional[int] = None) -> int:
        """Frames t0..t1-1 of this person become a new person (someone else was followed there)."""
        t1 = int(self.cut.work_range[1] if t1 is None else t1)
        cut, st = self.cut, self.store
        new = max([p.id for p in cut.persons] + list(cut.removed) + list(st.removed) + list(st.pids) + [-1]) + 1
        st.add_person(new)
        st.swap_persons(pid, new, t0, t1)
        self._move_identity({pid: new}, t0, t1)
        old = cut.person(pid)
        cut.persons.append(Person(id=new, name=f"인물 {new + 1}", color=PERSON_COLORS[new % len(PERSON_COLORS)],
                                  role="protect", profile=list(old.profile) if old else [], first=int(t0),
                                  last=int(t1) - 1))
        return new

    def _move_identity(self, m: Dict[int, int], t0: int, t1: int) -> None:
        cut, st = self.cut, self.store
        # segmentation owners (person id + 1 per pixel)
        if self.seglab.exists():
            alias = self._seg_alias()
            lut = np.arange(256, dtype=np.uint8)
            for v in range(1, 256):
                pid = alias.get(v - 1, v - 1)
                if pid in m and 0 <= m[pid] < 255:
                    lut[v] = m[pid] + 1
            mm = self.seglab.open_rw()
            try:
                for t in range(t0, t1):
                    mm[t] = lut[np.asarray(mm[t])]
            finally:
                mm.flush()
                del mm
                self.seglab.close()
        # joint edits and hidden-joint marks inside the range
        moved_k = {pid: {} for pid in m}
        moved_h = {pid: {} for pid in m}
        for pid in m:
            for j, fr in cut.joint_keys.get(pid, {}).items():
                for f in [f for f in fr if t0 <= int(f) < t1]:
                    moved_k[pid].setdefault(j, {})[f] = fr.pop(f)
            for j, rs in cut.hidden.get(pid, {}).items():
                keep, take = [], []
                for r0, r1 in rs:
                    if r1 < t0 or r0 >= t1:
                        keep.append([r0, r1])
                        continue
                    if r0 < t0:
                        keep.append([r0, t0 - 1])
                    if r1 >= t1:
                        keep.append([t1, r1])
                    take.append([max(r0, t0), min(r1, t1 - 1)])
                rs[:] = keep
                if take:
                    moved_h[pid][j] = take
        for src, dst in m.items():
            for j, fr in moved_k[src].items():
                cut.joint_keys.setdefault(dst, {}).setdefault(j, {}).update(fr)
            for j, rs in moved_h[src].items():
                cut.hidden.setdefault(dst, {}).setdefault(j, []).extend(rs)
        for pid in list(cut.joint_keys):
            cut.joint_keys[pid] = {j: fr for j, fr in cut.joint_keys[pid].items() if fr}
        for p in cut.persons:
            if st.has(p.id):
                fr = np.nonzero(st.present[st.index(p.id)])[0]
                if len(fr):
                    p.first, p.last = int(fr[0]), int(fr[-1])
        # each part keeps its own left/right labelling: both were already consistent with the facing rule
        self._label_cache.clear()
        self._geo_cache = {}
        self._head_raw = {}
        self._skin_models = {}
        self.resolve_all()
        self.update_widths()

    def active_area(self) -> Optional[Tuple[float, float, float, float]]:
        """The picture area inside letterbox or pillarbox bars (source px), or None when the picture fills the
        frame. Bars are rows / columns that stay black on every sampled frame."""
        if getattr(self, "_active", "?") != "?":
            return self._active
        self._active = None
        px = self.proxy
        try:
            if not px.has_frames():
                return None
            a, b = self.cut.work_range
            ts = np.unique(np.linspace(a, max(a, b - 1), min(12, max(1, b - a))).astype(int))
            mx = None
            for t in ts:
                g = cv2.cvtColor(np.asarray(px.frame(int(t))), cv2.COLOR_RGB2GRAY)
                mx = g if mx is None else np.maximum(mx, g)
        except Exception:
            return None
        h, w = mx.shape
        rows = mx.max(1) <= 8
        cols = mx.max(0) <= 8

        def run(flags):
            n = 0
            for f in flags:
                if not f:
                    break
                n += 1
            return n

        top, bot = run(rows), run(rows[::-1])
        left, right = run(cols), run(cols[::-1])
        top = top if top >= 0.01 * h else 0
        bot = bot if bot >= 0.01 * h else 0
        left = left if left >= 0.01 * w else 0
        right = right if right >= 0.01 * w else 0
        if top + bot >= 0.9 * h or left + right >= 0.9 * w or not (top or bot or left or right):
            return None
        sx, sy = px.sx, px.sy
        self._active = (left / sx, top / sy, (w - right) / sx, (h - bot) / sy)
        return self._active

    def auto_ok_rate(self, thr: float) -> float:
        st = self.store
        if st is None:
            return -1.0
        pres = st.present
        if not pres.any():
            return -1.0
        return float((st.auto_conf[pres] >= thr).mean())
