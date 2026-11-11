"""Lucide icons (ISC licence), recoloured on the fly."""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .theme import ASSETS, C

ICON_DIR = ASSETS / "icons"
_CACHE = Path(tempfile.gettempdir()) / "contour_icons"


def _svg(name: str, color: str, stroke: float = 2.0) -> bytes:
    p = ICON_DIR / f"{name}.svg"
    if not p.exists():
        p = ICON_DIR / "square.svg"
    s = p.read_text(encoding="utf-8")
    s = s.replace("currentColor", color).replace('stroke-width="2"', f'stroke-width="{stroke}"')
    return s.encode("utf-8")


def icon_file(name: str, color: str) -> str:
    """Path to a recoloured SVG (for stylesheets)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    out = _CACHE / f"{name}_{color.strip('#')}.svg"
    if not out.exists():
        out.write_bytes(_svg(name, color))
    return out.as_posix()


@lru_cache(maxsize=512)
def pixmap(name: str, size: int = 16, color: str = "", stroke: float = 1.75, dpr: float = 2.0) -> QPixmap:
    color = color or C["text"]
    r = QSvgRenderer(QByteArray(_svg(name, color, stroke)))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, size * dpr, size * dpr))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, size: int = 16, color: str = "", active: str = "") -> QIcon:
    ic = QIcon()
    ic.addPixmap(pixmap(name, size, color or C["text2"]), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(name, size, active or C["text"]), QIcon.Normal, QIcon.On)
    ic.addPixmap(pixmap(name, size, active or C["text"]), QIcon.Active, QIcon.Off)
    ic.addPixmap(pixmap(name, size, C["text3"]), QIcon.Disabled, QIcon.Off)
    return ic
