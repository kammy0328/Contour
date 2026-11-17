"""App-wide settings and data locations.

Data (models, learning datasets, benchmarks, stats) lives outside the app
folder so a newer app version can be dropped in next to the old one and keep
everything that has been learned.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "DIAP"
APP_VERSION = "1.20.0"


def default_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents"
    else:
        base = Path.home()
    # 1.9 renamed the app to DIAP; data learned under the old name stays where it is
    old = base / "Contour_Data"
    new = base / "DIAP_Data"
    if old.exists() and not new.exists():
        return old
    return new


@dataclass
class AppSettings:
    data_dir: str = ""
    cache_dir: str = ""                   # proxies (made again from the footage whenever missing); "" = data folder\cache
    ffmpeg_dir: str = ""                  # (unused since 1.14: ffmpeg ships with DIAP) kept so old files load
    ocio_config: str = ""                 # (unused since 1.14: always the built-in ACES Studio config)
    display: str = ""
    view: str = ""
    proxy_long_side: int = 1920
    gpu: str = "auto"                     # auto | cuda | dml | cpu
    accuracy: str = "high"                # high | balanced | fast
    collect_learning_data: bool = True
    self_learning: bool = True        # learn from every analysed cut on its own (no corrections needed)
    last_project: str = ""
    recent_projects: list = field(default_factory=list)
    known_projects: list = field(default_factory=list)   # every project the project manager lists (no limit)
    pm_sort: str = "modified"             # project manager order: modified | name | created (last chosen)
    play_loop: bool = False               # transport: repeat the cut (off: playback goes on into the next cut)
    open_last_on_start: bool = False      # skip the project manager at start and open the last project
    output_dir: str = ""
    qc_thresholds: dict = field(default_factory=dict)
    cloth_model_ok: bool = False          # the user has read the clothing model's licence note
    proxy_mode: str = "auto"              # auto: a proxy in the background when the original is slow / always / off
    frame_cache_mb: int = 1536            # memory for frames decoded from originals (all cuts together)
    hover_pick: bool = True               # edit tool: highlight the person under the mouse
    ask_proxy: bool = True                # ask for the proxy resolution when footage comes in
    autosave: bool = True
    autosave_minutes: int = 5
    autosave_keep: int = 10
    skip_version: str = ""                # an update the user chose to skip
    auto_update_check: bool = True
    analysis_spf: float = 0.0             # seconds per frame of a whole-cut analysis on this PC (learned; 0 = unknown)

    # ------------------------------------------------------------------ paths
    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir) if self.data_dir else default_data_dir()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        """This computer's cache of proxies: a fast local disk, not the project's (often network) folder."""
        p = Path(self.cache_dir) if self.cache_dir else self.data_path / "cache"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            p = self.data_path / "cache"
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_path(self) -> Path:
        p = self.data_path / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def learning_path(self) -> Path:
        p = self.data_path / "learning"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def benchmark_path(self) -> Path:
        p = self.data_path / "benchmark"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def stats_path(self) -> Path:
        p = self.data_path / "stats"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ io
    @staticmethod
    def file() -> Path:
        d = default_data_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d / "settings.json"

    @classmethod
    def load(cls) -> "AppSettings":
        f = cls.file()
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        f = self.file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(f)

    def add_recent(self, path: str) -> None:
        path = str(path)
        if path in self.recent_projects:
            self.recent_projects.remove(path)
        self.recent_projects.insert(0, path)
        del self.recent_projects[10:]
        if path in self.known_projects:
            self.known_projects.remove(path)
        self.known_projects.insert(0, path)
        self.last_project = path


_settings: AppSettings | None = None


def settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings.load()
    return _settings


QC_DEFAULTS = {
    "max_warp_pct": 6.0,        # % of person height
    "fold_det": 0.25,           # Jacobian determinant floor
    "jitter_px": 0.8,           # temporal second-difference of the field (render px)
    "line_bend_px": 1.5,        # deviation of straight background lines (render px)
    "leak_px": 0.5,             # displacement on protected people (render px)
    "low_conf": 0.45,           # tracking confidence under which an applied warp is flagged
}


def qc_threshold(name: str) -> float:
    s = settings()
    return float(s.qc_thresholds.get(name, QC_DEFAULTS[name]))
