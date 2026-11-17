"""Image viewer: result / original / side-by-side / wipe, skeleton editing, protection drawing, overlays."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF,
                           QTransform)
from PySide6.QtWidgets import QMenu, QWidget

from ..track import skeleton as S
from ..warp.protect import line_at, shape_at
from .theme import C, font, qc

HEAT_STOPS = [(0.0, (232, 176, 74, 0)), (0.06, (232, 176, 74, 40)), (0.35, (232, 176, 74, 130)),
              (0.7, (240, 112, 154, 175)), (1.0, (255, 228, 240, 215))]


def _heat_lut() -> np.ndarray:
    lut = np.zeros((256, 4), np.uint8)
    xs = [s[0] for s in HEAT_STOPS]
    for c in range(4):
        lut[:, c] = np.interp(np.linspace(0, 1, 256), xs, [s[1][c] for s in HEAT_STOPS]).astype(np.uint8)
    return lut


HEAT_LUT = _heat_lut()
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def np_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    h, w = a.shape[:2]
    if a.ndim == 2:
        img = QImage(a.data, w, h, a.strides[0], QImage.Format_Grayscale8)
    elif a.shape[2] == 4:
        img = QImage(a.data, w, h, a.strides[0], QImage.Format_RGBA8888)
    else:
        img = QImage(a.data, w, h, a.strides[0], QImage.Format_RGB888)
    img._keep = a      # keep the buffer alive
    return img


class Viewer(QWidget):
    jointDragged = Signal(int, int, float, float)       # pid, joint, x, y (source px) while dragging
    jointCommitted = Signal(int, int, float, float)
    jointKeyDeleted = Signal(int, int)
    personClicked = Signal(int)
    lineAdded = Signal(list)                             # [x1,y1,x2,y2] source px
    lineEdited = Signal(str, list)                       # id, [x1,y1,x2,y2]
    shapeAdded = Signal(str, list)                       # kind, [cx,cy,rx,ry]
    shapeEdited = Signal(str, list)
    interacted = Signal()
    contextAction = Signal(str, int, int)                # action, pid, joint
    personAction = Signal(str, int)                      # rename | role | delete, pid (right click on a person)
    editStarted = Signal()                               # a drag that changes the project begins (undo point)
    sculptEdited = Signal(int, object, bool)             # dense face point, offset (face units) or None, committed
    bodyEdited = Signal(object, bool)                    # {body point key: edge change or None}, committed

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.t = 0
        self.mode = "result"           # result | orig | side | wipe
        self.ab_flip = False
        self.wipe = 0.5
        self.show_skel = True
        self.show_heat = False
        self.show_matte = False
        self.show_protect = True
        self.show_skin = False
        self.hover_pick = True                           # highlight the person under the mouse (edit tool)
        self.selection: Optional[Tuple[int, int]] = None   # timeline range (for joint menus)
        self._hover_ghost: Optional[Tuple[int, int]] = None
        self._hover_sculpt: Optional[int] = None
        self._hover_body: Optional[Tuple[int, int, int]] = None
        self.playing = False
        self._play_raw = False              # showing the original because the result is not ready yet
        self.tool = "edit"             # edit | line | ellipse | rect
        self.sel_pid: Optional[int] = None
        self.zoom = 1.0
        self.fit = True
        self.pan = QPointF(0, 0)
        self._hover: Optional[Tuple[int, int]] = None
        self._hover_pid: Optional[int] = None         # the tracked person under the mouse (edit tool)
        self._drag = None
        self._panning = None
        self._draw = None
        self._wipe_drag = False
        self._img_cache = {}
        self._last_res = None            # (frame, QImage) last result shown, used while a new one renders
        from .workers import PreviewWorker
        self.preview = PreviewWorker(self._compute_preview, self)
        self.preview.ready.connect(self._preview_ready)
        # the skin area overlay is worked out on its own thread too (seconds per frame on a 4K cut)
        self.skin_worker = PreviewWorker(self._compute_skin, self)
        self.skin_worker.ready.connect(self._skin_ready)
        self._skin_last = None           # (frame, QImage) last skin overlay shown
        # a small turning arc while a result is being worked out
        from PySide6.QtCore import QTimer as _QT
        self._spin_a = 0.0
        self._spin_t = _QT(self)
        self._spin_t.setInterval(40)
        self._spin_t.timeout.connect(self._spin_tick)
        self.message = ""
        self.setMouseTracking(True)
        self._hover_timer = QTimer(self)                  # hover state follows the real pointer (see _pointer_inside)
        self._hover_timer.setInterval(200)
        self._hover_timer.timeout.connect(self._hover_check)
        self._hover_timer.start()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 200)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    # ------------------------------------------------------------------ geometry
    def _img_size(self) -> Tuple[int, int]:
        if self.s.proxy is None:
            return 1920, 1080
        return self.s.proxy.w, self.s.proxy.h

    def _panes(self) -> List[QRectF]:
        """Widget rects where images are drawn (1 or 2)."""
        iw, ih = self._img_size()
        W, H = self.width(), self.height()
        n = 2 if self.mode == "side" else 1
        pane_w = (W - 24 - (12 if n == 2 else 0)) / n
        pane_h = H - 24
        if self.fit:
            z = min(pane_w / iw, pane_h / ih)
        else:
            z = self.zoom
        self.zoom = z
        rects = []
        for i in range(n):
            cx = 12 + pane_w * (i + 0.5) + (12 * i if n == 2 else 0)
            cy = 12 + pane_h / 2
            r = QRectF(cx - iw * z / 2 + self.pan.x(), cy - ih * z / 2 + self.pan.y(), iw * z, ih * z)
            rects.append(r)
        return rects

    def _cells(self) -> List[QRectF]:
        """The viewer areas the panes live in (images are clipped to them: a zoomed image must not run over the
        neighbouring pane)."""
        W, H = self.width(), self.height()
        n = 2 if self.mode == "side" else 1
        pane_w = (W - 24 - (12 if n == 2 else 0)) / n
        return [QRectF(12 + (pane_w + 12) * i, 12, pane_w, H - 24) for i in range(n)]

    def _wipe_x(self) -> float:
        """Wipe divider: fixed on the screen (zooming or panning moves the picture under it, not the line)."""
        c = self._cells()[0]
        return c.left() + c.width() * self.wipe

    def _limit_pan(self):
        """The picture can be pushed around but never out of sight."""
        panes, cells = self._panes(), self._cells()
        r, c = panes[0], cells[0]
        keep = 60.0
        dx = dy = 0.0
        if r.right() < c.left() + keep:
            dx = c.left() + keep - r.right()
        elif r.left() > c.right() - keep:
            dx = c.right() - keep - r.left()
        if r.bottom() < c.top() + keep:
            dy = c.top() + keep - r.bottom()
        elif r.top() > c.bottom() - keep:
            dy = c.bottom() - keep - r.top()
        if dx or dy:
            self.pan = QPointF(self.pan.x() + dx, self.pan.y() + dy)

    def _edit_rect(self) -> QRectF:
        ps = self._panes()
        return ps[-1]

    def img_to_w(self, x: float, y: float, rect: Optional[QRectF] = None) -> QPointF:
        r = rect or self._edit_rect()
        iw, _ = self._img_size()
        z = r.width() / iw
        return QPointF(r.left() + x * z, r.top() + y * z)

    def w_to_img(self, p: QPointF, rect: Optional[QRectF] = None) -> Tuple[float, float]:
        r = rect or self._edit_rect()
        iw, _ = self._img_size()
        z = r.width() / iw
        return (p.x() - r.left()) / z, (p.y() - r.top()) / z

    def src_to_w(self, x: float, y: float, rect=None) -> QPointF:
        px = self.s.proxy
        return self.img_to_w(x * px.sx, y * px.sy, rect)

    def w_to_src(self, p: QPointF, rect=None) -> Tuple[float, float]:
        px = self.s.proxy
        x, y = self.w_to_img(p, rect)
        return x / px.sx, y / px.sy

    # ------------------------------------------------------------------ data
    def set_frame(self, t: int):
        self.t = t
        self.update()

    def _store_img(self, key, img):
        if len(self._img_cache) > 8:
            self._img_cache.pop(next(iter(self._img_cache)))
        self._img_cache[key] = img

    def _qimg(self, which: str) -> Optional[QImage]:
        cid = self.s.cut.id if self.s.cut is not None else ""
        if which == "orig":
            key = ("orig", self.t, cid)
            if key in self._img_cache:
                return self._img_cache[key]
            a = self.s.frame(self.t)
            if a is None:
                return None
            img = np_to_qimage(a)
            self._store_img(key, img)
            return img
        if self.s.cut is None:
            return None
        rkey = self.s.result_key(self.t)
        key = ("result", rkey)
        self._play_raw = False
        self._res_pending = False
        if key in self._img_cache:
            return self._img_cache[key]
        if not self.s.interactive:
            pa = self.s.play_get(self.t)
            if pa is not None:
                img = np_to_qimage(pa)
                self._store_img(key, img)
                self._last_res = (self.t, img)
                return img
        a = self.s.cached_result(self.t)
        if a is None and self.playing:
            # playback never waits: frames not computed yet show the original, marked
            self._play_raw = True
            self.preview.request((self.t, rkey))
            return self._qimg("orig")
        if a is None and self._last_res is not None and self._last_res[0] == self.t:
            # same frame, new settings: keep showing the last result while the new one renders
            self.preview.request((self.t, rkey))
            self._res_pending = True
            return self._last_res[1]
        if a is None:
            # never computed on the UI thread: a frame can take seconds (and the playback cache may hold the
            # computation lock), which froze the window after it had been left alone. The original shows, marked,
            # until the preview thread has the result.
            self.preview.request((self.t, rkey))
            self._res_pending = True
            return self._qimg("orig")
        img = np_to_qimage(a)
        self._store_img(key, img)
        self._last_res = (self.t, img)
        return img

    def _compute_preview(self, key):
        t, rkey = key
        if self.s.cut is None:
            return None
        cid = self.s.cut.id
        rk = self.s.result_key(t)
        self.s.preview_wait += 1             # the playback cache steps aside after the frame it is on
        try:
            a = self.s.result(t)
        finally:
            self.s.preview_wait -= 1
        return (t, rk, a, cid) if a is not None else None

    def _preview_ready(self, key, payload):
        if payload is None:
            return
        t, rkey, a, cid = payload
        if self.s.cut is None or cid != self.s.cut.id:
            return                      # computed for a cut that is no longer open
        img = np_to_qimage(a)
        self._store_img(("result", rkey), img)
        if t == self.t:
            self._last_res = (t, img)
            self.update()

    def _heat_img(self) -> Optional[QImage]:
        key = ("heat", self.t, self.s.vers["geo"], self.s.vers["trk"])
        if key in self._img_cache:
            return self._img_cache[key]
        fr = self.s.field(self.t)
        if fr is None or fr.empty():
            return None
        from ..warp.field import upsample
        D = upsample(fr)
        mag = np.hypot(D[..., 0], D[..., 1])
        # 0..max over a fixed 1.5% of frame height scale so colours mean the same thing across frames
        ref = 0.015 * fr.out_h
        idx = np.clip(mag / ref * 255 / 1.6, 0, 255).astype(np.uint8)
        rgba = HEAT_LUT[idx]
        img = np_to_qimage(rgba)
        self._img_cache[key] = img
        return img

    def _hover_img(self, pid: int):
        """The hovered person's area (tinted) and its box in matte pixels."""
        key = ("hover", self.t, pid, self.s.vers["trk"], self.s.cut.id if self.s.cut else "")
        if key in self._img_cache:
            return self._img_cache[key]
        tr = self.s.tracker
        if tr is None or tr.store is None or not tr.store.has(pid):
            return None
        lab = tr.labels(self.t)
        if lab is None:
            return None
        m = lab == tr.store.index(pid)
        import cv2
        # the skeleton fallback of the label map also covers background near the body: keep only what the person
        # matte holds, and drop specks
        al = tr.matte.get(self.t) if tr.matte is not None else None
        if al is not None and al.shape == m.shape:
            m = m & (np.asarray(al) >= 128)
            if m.any():
                n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
                if n > 2:
                    keep = np.zeros(n, bool)
                    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(30, 0.02 * st[1:, cv2.CC_STAT_AREA].max())
                    m = keep[cc]
        if not m.any():
            return None
        h, w = lab.shape
        p_ = self.s.cut.person(pid)
        col = QColor(p_.color if p_ is not None else "#FFFFFF")
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[m] = (col.red(), col.green(), col.blue(), 70)
        edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        rgba[edge] = (col.red(), col.green(), col.blue(), 235)
        ys, xs = np.nonzero(m)
        out = (np_to_qimage(rgba), (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), w, h))
        if len(self._img_cache) > 64:
            self._img_cache.clear()
        self._img_cache[key] = out
        return out

    def _paint_hover(self, p: QPainter, r: QRectF, pid: int):
        hi = self._hover_img(pid)
        if hi is None:
            return
        img, (x0, y0, x1, y1, w, h) = hi
        p.drawImage(r, img)
        q = self.s.cut.person(pid)
        if q is None:
            return
        sx, sy = r.width() / w, r.height() / h
        name = q.name + ("  ·  보호" if q.role != "target" else "")
        p.setFont(font(11, 600))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(name) + 16
        cx = r.left() + (x0 + x1) / 2 * sx
        ty = max(r.top() + 4, r.top() + y0 * sy - 26)
        box = QRectF(cx - tw / 2, ty, tw, 21)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 185))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QColor(q.color))
        p.drawText(box, Qt.AlignCenter, name)

    def _person_menu(self, pid: int, gpos):
        q = self.s.cut.person(pid) if self.s.cut else None
        if q is None:
            return
        m = QMenu(self)
        head = m.addAction(q.name)
        head.setEnabled(False)
        m.addSeparator()
        a_sel = m.addAction("이 인물 선택")
        a_ren = m.addAction("이름 변경…")
        a_role = m.addAction("보호로 바꾸기" if q.role == "target" else "보정 대상으로 바꾸기")
        m.addSeparator()
        a_del = m.addAction("이 인물 삭제…")
        act = m.exec(gpos)
        if act is None:
            return
        if act == a_sel:
            self.sel_pid = pid
            self.personClicked.emit(pid)
            self.update()
            return
        self.personAction.emit({a_ren: "rename", a_role: "role", a_del: "delete"}.get(act, ""), pid)

    def _matte_img(self) -> Optional[QImage]:
        key = ("matte", self.t, self.s.vers["trk"], self.s.vers["geo"], self.s.cut.id if self.s.cut else "",
               self.sel_pid)
        if key in self._img_cache:
            return self._img_cache[key]
        tr = self.s.tracker
        if tr is None or tr.store is None:
            return None
        al = tr.matte.get(self.t)
        lab = tr.labels(self.t)
        if al is None or lab is None:
            return None
        h, w = lab.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        a = np.asarray(al)
        for p in self.s.cut.persons:
            if not tr.store.has(p.id):
                continue
            i = tr.store.index(p.id)
            m = lab == i
            col = QColor(p.color)
            rgba[m, 0], rgba[m, 1], rgba[m, 2] = col.red(), col.green(), col.blue()
            rgba[m, 3] = (a[m].astype(np.float32) * (0.45 if p.role == "target" else 0.3)).astype(np.uint8)
        other = (lab < 0) & (a > 30)
        rgba[other] = (200, 200, 200, 60)
        # clothes that stay as they are (body edits held back there): striped orange
        try:
            from ..track import clothing as CL
            for p in self.s.cut.targets():
                mode = getattr(self.s.cut.params_for(p.id), "cloth", "include")
                if mode == "include" or not tr.store.has(p.id) or not CL.ready_for(tr, p.id):
                    continue
                km = CL.keep_mask(tr, self.t, tr.store.index(p.id), mode)
                if km is None:
                    continue
                yy, xx = np.mgrid[0:h, 0:w]
                stripe = ((xx + yy) // 4) % 2 == 0
                rgba[km] = (255, 150, 40, 150)
                rgba[km & stripe] = (255, 190, 90, 190)
        except Exception:
            pass
        img = np_to_qimage(rgba)
        self._img_cache[key] = img
        return img

    def _skin_key(self):
        return ("skin", self.t, self.s.vers["ret"], self.s.vers["trk"], self.s.vers["geo"],
                self.s.cut.id if self.s.cut else "", self.sel_pid)

    def _skin_img(self) -> Optional[QImage]:
        """The skin area overlay of this frame, from the background thread. Until it is ready: the last one of the
        same frame (settings changed), else nothing; during playback only frames already worked out show it."""
        key = self._skin_key()
        if key in self._img_cache:
            return self._img_cache[key]
        self.skin_worker.request((key, self.t, self.sel_pid))
        if self._skin_last is not None and self._skin_last[0] == self.t and not self.playing:
            self._res_pending = True
            return self._skin_last[1]
        return None

    def _compute_skin(self, req):
        key, t, sel_pid = req
        if self.s.cut is None or key != self._skin_key():
            return None                          # settings or frame moved on: a newer request follows
        rf = None
        if sel_pid is not None and self.s.tracker is not None:
            try:
                from ..retouch.engine import person_masks
                rf = person_masks(self.s.cut, self.s.tracker, t, sel_pid)
            except Exception:
                rf = None
        if rf is None or rf.empty():
            rf = self.s.retouch(t)
        if rf is None:
            return (key, t, None)
        m = rf.union()
        if m is None:
            return (key, t, None)
        # show the area where it ends up: the overlay follows face size and other shape changes
        fr = self.s.peek_field(t)
        if fr is not None and not fr.empty():
            from ..warp.field import warp_image
            m = warp_image(m.astype(np.float32), fr, cv2.INTER_LINEAR)
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 79, 139, 255
        rgba[..., 3] = np.clip(m * 170, 0, 255).astype(np.uint8)
        return (key, t, rgba)

    def _skin_ready(self, req, payload):
        if payload is None:
            return
        key, t, rgba = payload
        img = np_to_qimage(rgba) if rgba is not None else QImage()
        self._store_img(key, img)
        if t == self.t:
            self._skin_last = (t, img)
            self.update()

    def _spin_tick(self):
        self._spin_a = (self._spin_a + 14.0) % 360.0
        self.update()

    # ------------------------------------------------------------------ paint
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), qc("viewer"))
        if self.s.proxy is None:
            p.setPen(qc("text3"))
            p.setFont(font(13, 500))
            p.drawText(self.rect(), Qt.AlignCenter, self.message or "왼쪽에서 컷을 추가하거나 선택하세요")
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 1.6)
        p.setRenderHint(QPainter.Antialiasing, True)
        panes = self._panes()
        mode = self.mode
        if mode in ("result", "orig") and self.ab_flip:
            mode = "orig" if mode == "result" else "result"
        cells = self._cells()
        if mode == "side":
            for r, cell, which, lab in zip(panes, cells, ("orig", "result"), ("원본", "결과")):
                img = self._qimg(which)
                p.save()
                p.setClipRect(cell)
                if img is not None:
                    p.drawImage(r, img)
                p.restore()
                self._tag(p, cell, lab)
            r = panes[1]
        elif mode == "wipe":
            r = panes[0]
            cell = cells[0]
            a, b = self._qimg("orig"), self._qimg("result")
            if b is not None:
                p.drawImage(r, b)
            x = self._wipe_x()
            if a is not None:
                p.save()
                p.setClipRect(QRectF(cell.left(), cell.top(), x - cell.left(), cell.height()))
                p.drawImage(r, a)
                p.restore()
                p.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
                p.drawLine(QPointF(x, cell.top()), QPointF(x, cell.bottom()))
                p.setBrush(qc("bg3"))
                p.setPen(QPen(qc("line2"), 1))
                p.drawEllipse(QPointF(x, cell.center().y()), 9, 9)
            self._tag(p, cell, "원본", left=True)
            self._tag(p, cell, "결과")
        else:
            r = panes[0]
            img = self._qimg("orig" if mode == "orig" else "result")
            if img is not None:
                p.drawImage(r, img)
            if mode == "orig":
                self._tag(p, cells[0], "원본")
        # overlays on the edit pane (in wipe mode only on the result side, in side mode inside its pane)
        clipped = mode in ("wipe", "side")
        if mode == "wipe":
            p.save()
            xw = self._wipe_x()
            p.setClipRect(QRectF(xw, cells[0].top(), cells[0].right() - xw, cells[0].height()))
        elif mode == "side":
            p.save()
            p.setClipRect(cells[1])
        if self.show_matte:
            mi = self._matte_img()
            if mi is not None:
                p.drawImage(r, mi)
        if self.show_skin:
            si = self._skin_img()
            if si is not None and si.isNull():
                si = None
            if si is not None:
                p.drawImage(r, si)
        if clipped:
            p.restore()
        if mode == "side":
            p.save()
            p.setClipRect(cells[1])
        if self._hover_pid is not None and self.tool == "edit" and self._drag is None and not self.playing \
                and self._pointer_inside():
            self._paint_hover(p, r, self._hover_pid)
        if self.show_protect:
            self._paint_protect(p, r)
        if self.show_skel and self.tool not in ("sculpt", "bodypts"):
            self._paint_skeleton(p, r)
        if self.tool == "sculpt":
            self._paint_sculpt(p, r)
        if self.tool == "bodypts":
            self._paint_body(p, r)
        if self._draw is not None:
            self._paint_draft(p)
        if mode == "side":
            p.restore()
        if self.message:
            self._tag(p, QRectF(self.rect()).adjusted(12, 12, -12, -12), self.message, left=True, bottom=True)
        # the "not computed yet" tag only after several original frames in a row, and it stays a moment:
        # single frames flipping between result and original made it blink
        if self.playing:
            self._raw_run = (getattr(self, "_raw_run", 0) + 1) if self._play_raw else 0
            if self._raw_run >= 6:
                self._raw_shown_until = self._paint_n + 12
        self._paint_n = getattr(self, "_paint_n", 0) + 1
        if self.playing and getattr(self, "_raw_shown_until", -1) > self._paint_n:
            self._tag(p, cells[-1], "미리 계산 전 · 원본")
        pend = not self.playing and getattr(self, "_res_pending", False) and mode in ("result", "side", "wipe")
        if pend:
            self._busy_tag(p, cells[-1], "보정 계산 중")
            if not self._spin_t.isActive():
                self._spin_t.start()
        elif self._spin_t.isActive():
            self._spin_t.stop()

    def _busy_tag(self, p: QPainter, r: QRectF, text: str):
        """Bottom right: a turning arc and the text (a result is being worked out)."""
        p.save()
        p.setFont(font(11, 600))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 14 + 22
        x, y = r.right() - w - 8, r.bottom() - 30
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(QRectF(x, y, w, 22), 5, 5)
        pen = QPen(qc("accent"), 2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(x + 8, y + 5, 12, 12), int(-self._spin_a * 16), int(-250 * 16))
        p.setPen(QColor(235, 235, 235))
        p.drawText(QRectF(x + 24, y, w - 28, 22), Qt.AlignVCenter | Qt.AlignLeft, text)
        p.restore()

    def _tag(self, p: QPainter, r: QRectF, text: str, left: bool = False, bottom: bool = False):
        p.save()
        p.setFont(font(11, 600))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 14
        x = r.left() + 8 if left else r.right() - w - 8
        y = r.bottom() - 30 if bottom else r.top() + 8
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(QRectF(x, y, w, 22), 5, 5)
        p.setPen(QColor(235, 235, 235))
        p.drawText(QRectF(x, y, w, 22), Qt.AlignCenter, text)
        p.restore()

    def _paint_skeleton(self, p: QPainter, r: QRectF):
        tr = self.s.tracker
        if tr is None or tr.store is None:
            return
        st = tr.store
        cut = self.s.cut
        z = r.width() / self._img_size()[0]
        for person in cut.persons:
            if person.role == "ignore" or not st.has(person.id):
                continue
            kp = st.kp(person.id, self.t)
            if kp is None:
                continue
            sel = person.id == self.sel_pid
            col = QColor(person.color)
            alpha = 235 if sel else (150 if person.role == "target" else 90)
            pi = st.index(person.id)
            flags = st.flags[pi, self.t]
            vis = st.vis[pi, self.t]
            outside = vis == 2
            keys = cut.joint_keys.get(person.id, {})
            pts = [self.src_to_w(x, y, r) if np.isfinite(x) and np.isfinite(y) else None for x, y in kp[:, :2]]
            pen = QPen(QColor(col.red(), col.green(), col.blue(), int(alpha * 0.8)), 2.2 if sel else 1.6)
            pen.setCapStyle(Qt.RoundCap)
            dpen = QPen(QColor(col.red(), col.green(), col.blue(), int(alpha * 0.45)), 1.4, Qt.DashLine)
            for a, b in S.BONES:
                if pts[a] is not None and pts[b] is not None:
                    p.setPen(dpen if (outside[a] or outside[b]) else pen)
                    p.drawLine(pts[a], pts[b])
            rad = 4.6 if sel else 3.4
            if getattr(st, "has_hands", False):
                # hands: thin bones only (21 points per hand)
                hp = st.hand[pi, self.t]
                hpen = QPen(QColor(col.red(), col.green(), col.blue(), int(alpha * 0.55)), 1.1)
                p.setPen(hpen)
                for h in range(2):
                    q = hp[h * 21:(h + 1) * 21]
                    if not np.isfinite(q[:, :2]).any():
                        continue
                    W_ = [self.src_to_w(x, y, r) if np.isfinite(x) and np.isfinite(y) else None for x, y in q[:, :2]]
                    for a_, b_ in HAND_BONES:
                        if W_[a_] is not None and W_[b_] is not None:
                            p.drawLine(W_[a_], W_[b_])
            if sel:
                self._paint_ghosts(p, r, person, col)
            for j, q in enumerate(pts):
                if q is None:
                    continue
                if outside[j] and not (self.t in keys.get(j, {})):
                    # beyond the frame edge: only gives the limb its direction
                    p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), int(alpha * 0.6)), 1.2))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(q, rad - 0.6, rad - 0.6)
                    continue
                is_key = self.t in keys.get(j, {})
                hover = self._hover == (person.id, j)
                dragging = self._drag is not None and self._drag[0] == person.id and self._drag[1] == j
                low = bool(flags[j] & S.FLAG_LOW)
                rr = rad + (2 if hover or dragging else 0)
                if low and not is_key:
                    p.setPen(QPen(qc("error"), 1.6))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(q, rr + 3.2, rr + 3.2)
                if is_key:
                    poly = QPolygonF([q + QPointF(0, -rr - 1.5), q + QPointF(rr + 1.5, 0), q + QPointF(0, rr + 1.5),
                                      q + QPointF(-rr - 1.5, 0)])
                    p.setPen(QPen(QColor(20, 20, 20, 200), 1.2))
                    p.setBrush(QColor(255, 255, 255, alpha))
                    p.drawPolygon(poly)
                else:
                    p.setPen(QPen(QColor(15, 15, 15, 180), 1.0))
                    p.setBrush(QColor(col.red(), col.green(), col.blue(), alpha))
                    p.drawEllipse(q, rr, rr)
            if sel:
                # name tag near the head
                head = next((pts[j] for j in (S.NOSE, S.L_EAR, S.R_EAR) if pts[j] is not None), None)
                if head is not None:
                    p.save()
                    p.setFont(font(11, 600))
                    fm = p.fontMetrics()
                    w = fm.horizontalAdvance(person.name) + 12
                    box = QRectF(head.x() - w / 2, head.y() - 46 * max(0.6, min(1.4, z * 2)), w, 20)
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(col.red(), col.green(), col.blue(), 220))
                    p.drawRoundedRect(box, 5, 5)
                    p.setPen(QColor(18, 18, 18))
                    p.drawText(box, Qt.AlignCenter, person.name)
                    p.restore()
        if self._hover is not None and self._drag is None:
            pid, j = self._hover
            kp = st.kp(pid, self.t)
            if kp is not None and np.isfinite(kp[j, :2]).all():
                q = self.src_to_w(kp[j, 0], kp[j, 1], r)
                name = S.JOINT_NAMES_KO[j]
                if st.has(pid) and st.vis[st.index(pid), self.t, j] == 2:
                    name += " · 화면 밖"
                self._tag(p, QRectF(q.x() + 10, q.y() - 34, 0, 0), name, left=True)
        elif self._hover_ghost is not None and self._drag is None:
            pid, j = self._hover_ghost
            g = tr.ghost(pid, self.t).get(j)
            if g is not None:
                q = self.src_to_w(g[0], g[1], r)
                self._tag(p, QRectF(q.x() + 10, q.y() - 34, 0, 0),
                          f"{S.JOINT_NAMES_KO[j]} · 안 보임 (드래그하면 위치 지정)", left=True)

    def _paint_ghosts(self, p: QPainter, r: QRectF, person, col: QColor):
        """Joints counted as not visible: their best guess as a faint dashed ring (selected person only)."""
        g = self.s.tracker.ghost(person.id, self.t)
        if not g:
            return
        for j, v in g.items():
            if self._drag is not None and self._drag[:2] == (person.id, j):
                continue
            q = self.src_to_w(v[0], v[1], r)
            hover = self._hover_ghost == (person.id, j)
            a = 210 if hover else 120
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), a), 1.3, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            rr = 6.5 if hover else 5.0
            p.drawEllipse(q, rr, rr)
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), a), 1.3))
            p.drawLine(q + QPointF(-rr * 0.6, rr * 0.6), q + QPointF(rr * 0.6, -rr * 0.6))

    # ------------------------------------------------------------------ face sculpt
    def _sculpt_points(self):
        """(face geom, tracked dense points, displayed points) for the selected person, or None."""
        tr = self.s.tracker
        cut = self.s.cut
        if tr is None or tr.store is None or cut is None or self.sel_pid is None or not tr.store.has(self.sel_pid):
            return None
        fg = tr.face_geom(tr.store.index(self.sel_pid), self.t)
        if fg is None:
            return None
        dense = fg.dense()
        pp = cut.params_for(self.sel_pid)
        shown = dense.copy()
        for i, uv in pp.sculpt.items():
            if 0 <= i < len(dense):
                shown[i] = dense[i] + fg.from_units_vec(np.array(uv[:2]))
        return fg, dense, shown, pp

    def _paint_sculpt(self, p: QPainter, r: QRectF):
        sp = self._sculpt_points()
        if sp is None:
            self._tag(p, QRectF(self.rect()).adjusted(12, 12, -12, -12), "이 프레임에서 얼굴을 찾지 못했습니다", left=True)
            return
        from ..warp import face as FW
        fg, dense, shown, pp = sp
        W = [self.src_to_w(x, y, r) for x, y in shown]
        # outline
        p.setPen(QPen(QColor(255, 255, 255, 110), 1.0))
        cont = list(range(FW.N_JAW)) + list(range(FW.D_FORE.start + 1, FW.D_FORE.stop - 1))
        poly = QPolygonF([W[i] for i in cont])
        p.drawPolygon(poly)
        for sl in (FW.D_BROW, FW.D_NOSE, FW.D_EYE, FW.D_MOUTH):
            idx = list(range(sl.start, sl.stop))
            p.drawPolyline(QPolygonF([W[i] for i in idx]))
        for i, q in enumerate(W):
            moved = i in pp.sculpt
            hover = self._hover_sculpt == i
            if moved:
                p.setPen(QPen(qc("accent"), 1.0, Qt.DashLine))
                p.drawLine(self.src_to_w(dense[i, 0], dense[i, 1], r), q)
                p.setPen(QPen(QColor(15, 15, 15, 200), 1.0))
                p.setBrush(qc("accent"))
                p.drawEllipse(q, 4.2 if hover else 3.4, 4.2 if hover else 3.4)
            else:
                p.setPen(QPen(QColor(15, 15, 15, 160), 1.0))
                p.setBrush(QColor(255, 255, 255, 235 if hover else 170))
                p.drawEllipse(q, 3.6 if hover else 2.4, 3.6 if hover else 2.4)

    def _hit_sculpt(self, pos: QPointF) -> Optional[int]:
        sp = self._sculpt_points()
        if sp is None:
            return None
        r = self._edit_rect()
        best, bi = 9.0, None
        for i, (x, y) in enumerate(sp[2]):
            q = self.src_to_w(x, y, r)
            d = float(np.hypot(q.x() - pos.x(), q.y() - pos.y()))
            if d < best:
                best, bi = d, i
        return bi

    # ------------------------------------------------------------------ body points
    def _body_points(self):
        """Body outline points of the selected person on this frame, or None:
        dict(shown (NC,K,2,2) source px where the edge is now, centre/normal per point, d0, pp, scale)."""
        from ..warp import body as B
        from ..warp.field import forward_points
        tr = self.s.tracker
        cut = self.s.cut
        if tr is None or tr.store is None or cut is None or self.sel_pid is None or not tr.store.has(self.sel_pid):
            return None
        st = tr.store
        if not getattr(st, "has_contour", False):
            return None
        pi = st.index(self.sel_pid)
        if not st.present[pi, self.t]:
            return None
        kp = tr.warp_geometry(pi)[0][self.t]
        dist = st.contour[pi, self.t]
        if not np.isfinite(dist).any():
            return None
        ends = B.segment_endpoints(kp)
        cen = np.full((B.NC, B.CONTOUR_K, 2), np.nan)
        nrm = np.full((B.NC, 2), np.nan)
        for ci, (nm, _) in enumerate(B.CONTOUR_SEGS):
            e = ends[B.SEG_INDEX[nm]]
            if not np.isfinite(e).all():
                continue
            a = np.array([e[0], e[1]])
            v = np.array([e[2], e[3]]) - a
            L = float(np.hypot(*v))
            if L < 2:
                continue
            nrm[ci] = np.array([-v[1], v[0]]) / L
            cen[ci] = a[None] + v[None] * B.contour_ts(ci)[:, None]
        orig = B.contour_points(kp, dist)
        pp = cut.params_for(self.sel_pid)
        flips = B.side_flips(kp) if getattr(pp, "body_sides", "person") == "person" else np.zeros(B.NC, bool)
        px = self.s.proxy
        flat = orig.reshape(-1, 2) * np.array([px.sx, px.sy])
        mode = self.mode
        if mode in ("result", "orig") and self.ab_flip:
            mode = "orig" if mode == "result" else "result"
        fr = self.s.peek_field(self.t) if mode != "orig" else None
        pp = cut.params_for(self.sel_pid)
        scale = max(0.05, pp.amount("body_sculpt", self.t) / 100.0 * pp.strength.value(self.t) / 100.0)
        if mode == "orig":
            shown = orig.copy()
        elif fr is not None:
            shown = forward_points(fr, flat) / np.array([px.sx, px.sy])
            shown = shown.reshape(orig.shape)
        else:
            shown = orig.copy()
            for key, dv in pp.body_sculpt.items():
                try:
                    nm, k, side = key.split(":")
                    ci, k, side = B.CONTOUR_INDEX[nm], int(k), int(side)
                except (ValueError, KeyError):
                    continue
                side = side ^ int(flips[ci])
                sg = 1.0 if side == 0 else -1.0
                shown[ci, k, side] = cen[ci, k] + sg * nrm[ci] * dist[ci, k, side] * (1 + dv * scale)
        # points that end up outside the picture cannot be grabbed
        iw, ih = cut.info.width, cut.info.height
        out = (shown[..., 0] < 0) | (shown[..., 0] > iw - 1) | (shown[..., 1] < 0) | (shown[..., 1] > ih - 1)
        shown[out] = np.nan
        return dict(orig=orig, shown=shown, cen=cen, nrm=nrm, dist=dist, pp=pp, scale=scale, flips=flips)

    def _paint_body(self, p: QPainter, r: QRectF):
        from ..warp import body as B
        bp = self._body_points()
        if bp is None:
            self._tag(p, QRectF(self.rect()).adjusted(12, 12, -12, -12), "이 프레임에서 몸 윤곽 점을 찾지 못했습니다", left=True)
            return
        shown, pp = bp["shown"], bp["pp"]
        for ci in range(B.NC):
            for side in (0, 1):
                seq = [self.src_to_w(*shown[ci, k, side], r) if np.isfinite(shown[ci, k, side]).all() else None
                       for k in range(B.CONTOUR_K)]
                p.setPen(QPen(QColor(255, 255, 255, 90), 1.0))
                for a, b in zip(seq, seq[1:]):
                    if a is not None and b is not None:
                        p.drawLine(a, b)
                for k, q in enumerate(seq):
                    if q is None:
                        continue
                    key = B.sculpt_key(ci, k, side ^ int(bp["flips"][ci]))
                    moved = abs(pp.body_sculpt.get(key, 0.0)) > 1e-4
                    hover = self._hover_body == (ci, k, side)
                    rr = 4.4 if hover else (3.6 if moved else 2.8)
                    p.setPen(QPen(QColor(15, 15, 15, 180), 1.0))
                    p.setBrush(qc("accent") if moved else QColor(120, 220, 255, 240 if hover else 200))
                    p.drawEllipse(q, rr, rr)

    def _hit_body(self, pos: QPointF):
        from ..warp import body as B
        bp = self._body_points()
        if bp is None:
            return None
        r = self._edit_rect()
        best, bi = 10.0, None
        sh = bp["shown"]
        for ci in range(B.NC):
            for k in range(B.CONTOUR_K):
                for side in (0, 1):
                    x, y = sh[ci, k, side]
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    q = self.src_to_w(x, y, r)
                    d = float(np.hypot(q.x() - pos.x(), q.y() - pos.y()))
                    if d < best:
                        best, bi = d, (ci, k, side)
        return bi

    def _start_body_drag(self, hit, pos, single: bool):
        from ..warp import body as B
        bp = self._body_points()
        if bp is None:
            return
        ci, k, side = hit
        sg = 1.0 if side == 0 else -1.0
        m = bp["nrm"][ci] * sg
        c = bp["cen"][ci, k]
        P = bp["shown"][ci, k, side]
        dd0 = float((P - c) @ m)
        x, y = self.w_to_src(pos)
        m0 = float((np.array([x, y]) - c) @ m)
        pside = side ^ int(bp["flips"][ci])
        start = {kk: bp["pp"].body_sculpt.get(B.sculpt_key(ci, kk, pside), 0.0) for kk in range(B.CONTOUR_K)}
        self._drag = ("body", (ci, k, pside), c, m, dd0, m0, start, bp["scale"], single, bp["pp"].feather_factor())

    def _drag_body(self, pos, committed: bool):
        from ..warp import body as B
        _, (ci, k, side), c, m, dd0, m0, start, scale, single, fth = self._drag
        if abs(dd0) < 1.0:
            return
        x, y = self.w_to_src(pos)
        dd = dd0 + (float((np.array([x, y]) - c) @ m) - m0)
        ratio = float(np.clip(dd / dd0, 0.45, 1.8))
        new_k = ((1.0 + start[k] * scale) * ratio - 1.0) / scale
        new_k = float(np.clip(new_k, -0.5, 0.8))
        delta = new_k - start[k]
        sig = 0.6 * fth
        out = {}
        for kk, v0 in start.items():
            if single and kk != k:
                continue
            w = 1.0 if kk == k else float(np.exp(-0.5 * ((kk - k) / sig) ** 2))
            if w < 0.03:
                continue
            nv = float(np.clip(v0 + delta * w, -0.5, 0.8))
            out[B.sculpt_key(ci, kk, side)] = nv if abs(nv) > 1e-4 else None
        self.bodyEdited.emit(out, committed)

    def _paint_protect(self, p: QPainter, r: QRectF):
        cut = self.s.cut
        st = self.s.tracker.store if self.s.tracker else None
        pen = QPen(QColor(120, 220, 255, 220), 1.6, Qt.DashLine)
        for ln in cut.protect.lines:
            v = line_at(ln, self.t, st)
            if v is None:
                continue
            a, b = self.src_to_w(v[0], v[1], r), self.src_to_w(v[2], v[3], r)
            p.setPen(pen if ln.enabled else QPen(QColor(120, 120, 120, 160), 1.2, Qt.DashLine))
            p.drawLine(a, b)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(120, 220, 255, 230) if self.t in ln.keys else QColor(120, 220, 255, 120))
            for q in (a, b):
                p.drawRect(QRectF(q.x() - 4, q.y() - 4, 8, 8))
        for sh in cut.protect.shapes:
            v = shape_at(sh, self.t, st)
            if v is None:
                continue
            c = self.src_to_w(v[0], v[1], r)
            px = self.s.proxy
            rx = v[2] * px.sx * r.width() / px.w
            ry = v[3] * px.sy * r.width() / px.w
            p.setPen(pen if sh.enabled else QPen(QColor(120, 120, 120, 160), 1.2, Qt.DashLine))
            p.setBrush(QColor(120, 220, 255, 26))
            if sh.kind == "rect":
                p.drawRect(QRectF(c.x() - rx, c.y() - ry, 2 * rx, 2 * ry))
            else:
                p.drawEllipse(c, rx, ry)

    def _paint_draft(self, p: QPainter):
        kind, a, b = self._draw
        p.setPen(QPen(QColor(120, 220, 255, 240), 1.6, Qt.DashLine))
        p.setBrush(QColor(120, 220, 255, 30))
        if kind == "line":
            p.drawLine(a, b)
        elif kind == "rect":
            p.drawRect(QRectF(a, b).normalized())
        else:
            p.drawEllipse(QRectF(a, b).normalized())

    # ------------------------------------------------------------------ hit testing
    def _hit_joint(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        if not self.show_skel or self.s.tracker is None or self.s.proxy is None:
            return None
        r = self._edit_rect()
        x, y = self.w_to_src(pos, r)
        px = self.s.proxy
        z = r.width() / px.w
        rad = 11 / z / px.sx
        hit = self.s.joint_hit(self.t, x, y, rad)
        if hit and self.sel_pid is not None and hit[0] != self.sel_pid:
            # prefer the selected person when both are close
            st = self.s.tracker.store
            kp = st.kp(self.sel_pid, self.t)
            if kp is not None:
                d = np.hypot(kp[:, 0] - x, kp[:, 1] - y)
                d = np.where(np.isfinite(d), d, np.inf)
                j = int(np.argmin(d))
                if d[j] < rad:
                    return self.sel_pid, j
        return hit

    def _hit_ghost(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        if not self.show_skel or self.s.tracker is None or self.s.proxy is None or self.sel_pid is None:
            return None
        r = self._edit_rect()
        x, y = self.w_to_src(pos, r)
        px = self.s.proxy
        z = r.width() / px.w
        j = self.s.ghost_hit(self.sel_pid, self.t, x, y, 10 / z / px.sx)
        return (self.sel_pid, j) if j is not None else None

    def _hit_protect(self, pos: QPointF):
        """('line', id, end_index) / ('shape', id, 'move'|'size')"""
        if not self.show_protect or self.s.cut is None:
            return None
        r = self._edit_rect()
        st = self.s.tracker.store if self.s.tracker else None
        for ln in self.s.cut.protect.lines:
            v = line_at(ln, self.t, st)
            if v is None:
                continue
            for k, (x, y) in enumerate(((v[0], v[1]), (v[2], v[3]))):
                q = self.src_to_w(x, y, r)
                if abs(q.x() - pos.x()) < 8 and abs(q.y() - pos.y()) < 8:
                    return ("line", ln.id, k, v)
        for sh in self.s.cut.protect.shapes:
            v = shape_at(sh, self.t, st)
            if v is None:
                continue
            c = self.src_to_w(v[0], v[1], r)
            if abs(c.x() - pos.x()) < 10 and abs(c.y() - pos.y()) < 10:
                return ("shape", sh.id, "move", v)
        return None

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e):
        self.interacted.emit()
        self.setFocus()
        pos = e.position()
        if self.s.proxy is None:
            return
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.ControlModifier
                                               and self.tool == "edit"):
            self._panning = (pos, QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self.tool == "sculpt":
            i = self._hit_sculpt(pos)
            if i is not None:
                if e.button() == Qt.RightButton:
                    sp = self._sculpt_points()
                    if sp is not None and i in sp[3].sculpt:
                        self.editStarted.emit()
                        self.sculptEdited.emit(i, None, True)
                    return
                if e.button() == Qt.LeftButton:
                    self.editStarted.emit()
                    self._drag = ("sculpt", i)
                    self.setCursor(Qt.BlankCursor)
                    return
        if self.tool == "bodypts":
            hb = self._hit_body(pos)
            if hb is not None:
                from ..warp import body as B
                if e.button() == Qt.RightButton:
                    bp_ = self._body_points()
                    key = B.sculpt_key(hb[0], hb[1], hb[2] ^ int(bp_["flips"][hb[0]]) if bp_ else hb[2])
                    pp = self.s.cut.params_for(self.sel_pid)
                    if key in pp.body_sculpt:
                        self.editStarted.emit()
                        self.bodyEdited.emit({key: None}, True)
                    return
                if e.button() == Qt.LeftButton:
                    self.editStarted.emit()
                    self._start_body_drag(hb, pos, bool(e.modifiers() & Qt.ShiftModifier))
                    if self._drag is not None:
                        self.setCursor(Qt.BlankCursor)
                    return
            if e.button() == Qt.RightButton:
                return
        if e.button() == Qt.RightButton:
            hit = self._hit_joint(pos)
            if hit:
                self._joint_menu(hit, e.globalPosition().toPoint())
                return
            gh = self._hit_ghost(pos)
            if gh:
                self._ghost_menu(gh, e.globalPosition().toPoint())
                return
            if self.tool == "edit":
                pid = self.s.person_at(self.t, *self.w_to_src(pos))
                if pid is not None:
                    self._person_menu(pid, e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton:
            return
        if self.mode == "wipe":
            x = self._wipe_x()
            if abs(pos.x() - x) < 10:
                self._wipe_drag = True
                return
        if self.tool in ("edit", "sculpt", "bodypts"):
            ph = self._hit_protect(pos)
            if ph is not None:
                self.editStarted.emit()
                self._drag = ("protect",) + ph
                return
            hit = self._hit_joint(pos)
            if hit:
                pid, j = hit
                if e.modifiers() & Qt.AltModifier:
                    self.jointKeyDeleted.emit(pid, j)
                    return
                self._drag = (pid, j)
                if pid != self.sel_pid:
                    self.sel_pid = pid
                    self.personClicked.emit(pid)
                self.setCursor(Qt.BlankCursor)
                return
            gh = self._hit_ghost(pos)
            if gh:
                self._drag = gh
                self._hover_ghost = None
                self.setCursor(Qt.BlankCursor)
                return
            pid = self.s.person_at(self.t, *self.w_to_src(pos))
            if pid is not None:
                self.sel_pid = pid
                self.personClicked.emit(pid)
                self.update()
        else:
            self._draw = (self.tool, pos, pos)
            self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._panning is not None:
            start, pan0 = self._panning
            self.fit = False
            self.pan = pan0 + (pos - start)
            self._limit_pan()
            self.update()
            return
        if self._wipe_drag:
            c = self._cells()[0]
            self.wipe = float(np.clip((pos.x() - c.left()) / max(1, c.width()), 0, 1))
            self.update()
            return
        if self._draw is not None:
            k, a, _ = self._draw
            self._draw = (k, a, pos)
            self.update()
            return
        if self._drag is not None:
            if self._drag[0] == "protect":
                self._drag_protect(pos)
                return
            if self._drag[0] == "sculpt":
                self._drag_sculpt(pos, False)
                return
            if self._drag[0] == "body":
                self._drag_body(pos, False)
                return
            pid, j = self._drag
            x, y = self.w_to_src(pos)
            self.jointDragged.emit(pid, j, x, y)
            return
        if self.tool == "bodypts":
            hb = self._hit_body(pos)
            if hb != self._hover_body:
                self._hover_body = hb
                self.update()
            self.setCursor(Qt.PointingHandCursor if hb is not None else Qt.ArrowCursor)
            return
        if self.tool == "sculpt":
            hs = self._hit_sculpt(pos)
            if hs != self._hover_sculpt:
                self._hover_sculpt = hs
                self.update()
            self.setCursor(Qt.PointingHandCursor if hs is not None else Qt.ArrowCursor)
            return
        if self.tool == "edit":
            hv = self._hit_joint(pos)
            gh = self._hit_ghost(pos) if hv is None else None
            ph = self._hit_protect(pos) if hv is None and gh is None else None
            if hv != self._hover or gh != self._hover_ghost:
                self._hover = hv
                self._hover_ghost = gh
                self.update()
            hp = None
            if self.hover_pick and hv is None and gh is None and ph is None and self.s.cut is not None and not self.playing:
                hp = self.s.person_at(self.t, *self.w_to_src(pos))
            if hp != self._hover_pid:
                self._hover_pid = hp
                self.update()
            if hv is not None or ph is not None or gh is not None or hp is not None:
                self.setCursor(Qt.PointingHandCursor)
            elif self.mode == "wipe" and abs(pos.x() - self._wipe_x()) < 10:
                self.setCursor(Qt.SplitHCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def _pointer_inside(self) -> bool:
        """The real pointer, not the last mouse event: a leave event is lost when the pointer leaves fast, the window
        goes inactive or another window opens, and the hover highlight then stayed on."""
        try:
            from PySide6.QtGui import QCursor
            return self.isVisible() and self.rect().contains(self.mapFromGlobal(QCursor.pos())) \
                and (self.window().isActiveWindow() or self.underMouse())
        except Exception:
            return True

    def clear_hover(self) -> None:
        if (self._hover_pid is not None or self._hover is not None or self._hover_ghost is not None
                or self._hover_sculpt is not None or self._hover_body is not None):
            self._hover_pid = self._hover = self._hover_ghost = self._hover_sculpt = self._hover_body = None
            self.update()

    def _hover_check(self) -> None:
        if not self._pointer_inside():
            self.clear_hover()

    def leaveEvent(self, e):
        self.clear_hover()
        super().leaveEvent(e)

    def hideEvent(self, e):
        self.clear_hover()
        super().hideEvent(e)

    def focusOutEvent(self, e):
        self.clear_hover()
        super().focusOutEvent(e)

    def _drag_sculpt(self, pos, committed: bool):
        sp = self._sculpt_points()
        if sp is None:
            return
        fg, dense, shown, pp = sp
        i = self._drag[1]
        x, y = self.w_to_src(pos)
        uv = fg.to_units(np.array([[x, y]]))[0] - fg.to_units(dense[i:i + 1])[0]
        self.sculptEdited.emit(i, (float(uv[0]), float(uv[1])), committed)

    def _drag_protect(self, pos):
        _, kind, oid, part, v = self._drag
        x, y = self.w_to_src(pos)
        if kind == "line":
            nv = list(v)
            if part == 0:
                nv[0], nv[1] = x, y
            else:
                nv[2], nv[3] = x, y
            self._drag = ("protect", kind, oid, part, np.array(nv))
            self.lineEdited.emit(oid, [float(a) for a in nv])
        else:
            nv = [x, y, v[2], v[3]]
            self._drag = ("protect", kind, oid, part, np.array(nv))
            self.shapeEdited.emit(oid, [float(a) for a in nv])

    def mouseReleaseEvent(self, e):
        pos = e.position()
        if self._panning is not None:
            self._panning = None
            self.setCursor(Qt.ArrowCursor)
            return
        if self._wipe_drag:
            self._wipe_drag = False
            return
        if self._draw is not None:
            kind, a, b = self._draw
            self._draw = None
            if (b - a).manhattanLength() > 6:
                xa, ya = self.w_to_src(a)
                xb, yb = self.w_to_src(b)
                if kind == "line":
                    self.lineAdded.emit([xa, ya, xb, yb])
                else:
                    self.shapeAdded.emit(kind, [(xa + xb) / 2, (ya + yb) / 2, abs(xb - xa) / 2, abs(yb - ya) / 2])
            self.update()
            return
        if self._drag is not None:
            d = self._drag
            if d[0] == "sculpt":
                self._drag_sculpt(pos, True)
            if d[0] == "body":
                self._drag_body(pos, True)
            self._drag = None
            self.setCursor(Qt.ArrowCursor)
            if d[0] in ("protect", "sculpt", "body"):
                if d[0] == "protect":
                    self.s.interactive = False
                    self.s.touch("geo")
                    self.update()
                return
            pid, j = d
            x, y = self.w_to_src(pos)
            self.jointCommitted.emit(pid, j, x, y)

    def _joint_menu(self, hit, gpos):
        pid, j = hit
        m = QMenu(self)
        keys = self.s.cut.joint_keys.get(pid, {}).get(j, {})
        name = S.JOINT_NAMES_KO[j]
        a_del = m.addAction("이 프레임의 키 삭제")
        a_del.setEnabled(self.t in keys)
        a_clear = m.addAction(f"{name} 키 전체 삭제")
        a_clear.setEnabled(bool(keys))
        m.addSeparator()
        hm = m.addMenu("안 보이는 관절로 지정")
        acts = {hm.addAction(f"{name} · 이 프레임"): "hide_frame"}
        sel = self.selection
        a_sel_one = hm.addAction(f"{name} · 선택 구간 {sel[0]}~{sel[1]}" if sel else f"{name} · 선택 구간 (타임라인 Shift+드래그)")
        a_sel_one.setEnabled(sel is not None)
        acts[a_sel_one] = "hide_sel"
        desc = S.DESCENDANTS.get(j, [])
        if desc:
            chain = "·".join(S.JOINT_NAMES_KO[x] for x in [j] + desc)
            hm.addSeparator()
            acts[hm.addAction(f"{chain} · 이 프레임")] = "hide_frame_chain"
            a_chain = hm.addAction(f"{chain} · 선택 구간" + (f" {sel[0]}~{sel[1]}" if sel else ""))
            a_chain.setEnabled(sel is not None)
            acts[a_chain] = "hide_sel_chain"
        tr = self.s.tracker
        if tr is not None and tr.hidden_at(pid, j, self.t):
            acts[m.addAction("숨김 지정 해제 (자동 판단으로)")] = "unhide"
        m.addSeparator()
        a_sel = m.addAction("이 인물 선택")
        act = m.exec(gpos)
        if act == a_del:
            self.jointKeyDeleted.emit(pid, j)
        elif act == a_clear:
            self.contextAction.emit("clear_joint", pid, j)
        elif act in acts:
            self.contextAction.emit(acts[act], pid, j)
        elif act == a_sel:
            self.sel_pid = pid
            self.personClicked.emit(pid)

    def _ghost_menu(self, hit, gpos):
        pid, j = hit
        m = QMenu(self)
        name = S.JOINT_NAMES_KO[j]
        a_rev = m.addAction(f"{name}: 추정 위치를 보이는 관절로 확정")
        tr = self.s.tracker
        a_un = m.addAction("숨김 지정 해제 (자동 판단으로)")
        a_un.setEnabled(tr is not None and tr.hidden_at(pid, j, self.t))
        m.addSeparator()
        info = m.addAction("가려지거나 화면 밖이라 보정에 쓰지 않는 관절입니다")
        info.setEnabled(False)
        act = m.exec(gpos)
        if act == a_rev:
            self.contextAction.emit("revive", pid, j)
        elif act == a_un:
            self.contextAction.emit("unhide", pid, j)

    def wheelEvent(self, e):
        if self.s.proxy is None:
            return
        pos = e.position()
        panes = self._panes()
        # the pane is the one whose viewer area (cell) the cursor is in: a zoomed-in picture's own rect runs into the
        # neighbouring cell, so testing the picture rects picked the left pane while the cursor was over the right
        # one (비교: the zoom did not centre on the pointer)
        cells = self._cells()
        k = next((i for i, cc in enumerate(cells) if cc.contains(pos)), None)
        if k is None:
            k = int(np.argmin([abs(cc.center().x() - pos.x()) for cc in cells]))
        r = panes[k]
        ix, iy = self.w_to_img(pos, r)
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.fit = False
        old = self.zoom
        self.zoom = float(np.clip(old * f, 0.05, 16))
        # keep the pixel under the cursor fixed (both panes share zoom and pan)
        r2 = self._panes()[k]
        q = self.img_to_w(ix, iy, r2)
        self.pan += pos - q
        c0 = self._cells()[0]
        iw, ih = self._img_size()
        if self.zoom <= min(c0.width() / iw, c0.height() / ih) * 1.001:
            self.fit_view()                 # zoomed out to the whole picture: back to the fitted, centred view
            return
        self._limit_pan()
        self.update()

    def fit_view(self):
        self.fit = True
        self.pan = QPointF(0, 0)
        self.update()

    def zoom_100(self):
        self.fit = False
        self.zoom = 1.0 / max(1e-6, self.s.proxy.sx) if self.s.proxy else 1.0
        self.zoom = min(self.zoom, 1.0) if self.s.proxy else 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def resizeEvent(self, e):
        self.update()
