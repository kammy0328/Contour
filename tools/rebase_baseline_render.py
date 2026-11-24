"""Renders the frozen baseline projects (tests/data/baseline_1222, baseline_1222_body) with the code of now and puts what
it renders and shows into their golden.npz, where the retouch was changed on purpose (1.25.0: the user chose the
retouch of the 2611 IWALY work as the app's own, 2026-11-24). The projects themselves stay exactly as 1.22.2 saved them
(their opening is still checked against them), and so do the fields of golden.npz that say how 1.22.2 saved them.

    python tools/rebase_baseline_render.py 1.25.0

Only for a deliberate change of the picture: the point of the fixture is that it does not move."""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))


def main(version: str) -> int:
    work = Path(tempfile.mkdtemp(prefix="diap_rebase_"))
    os.environ["HOME"] = os.environ["USERPROFILE"] = str(work)
    os.environ["CONTOUR_NO_UPDATE"] = "1"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    import baseline_util as BU
    from contour.core.project import Project
    from contour.core.settings import APP_VERSION
    if APP_VERSION != version:
        print(f"the code is {APP_VERSION}, not {version}", file=sys.stderr)
        return 2
    for name, body in (("baseline_1222", False), ("baseline_1222_body", True)):
        fix = HERE / "tests" / "data" / name
        with np.load(fix / "golden.npz", allow_pickle=False) as d:
            g = {k: d[k] for k in d.files}
        pdir, src = BU.copy_fixture(fix, work / name)
        prj = Project.load(str(pdir))
        cut = prj.cuts[0]
        with BU.NoModels():
            planes = BU.render_planes(prj, cut)
            prev = BU.preview_frames(prj, cut, [int(t) for t in g["frames"]])
            # (preview_face stays 1.22.2's, as preview_1222: the fixture's own check that its body regions did something)
        d_ = np.abs(planes.astype(np.int32) - g["planes"].astype(np.int32))
        print(f"{name}: planes changed max {int(d_.max())} mean {float(d_.mean()):.4f}")
        g.setdefault("planes_1222", g["planes"])          # what 1.22.2 itself rendered and showed stays beside them
        g.setdefault("preview_1222", g["preview"])
        g["planes"], g["preview"] = planes, prev
        # never "the same libraries": the render is taken on whatever machine runs this, and its threads may round one value differently
        g["env"] = np.array(f"rebased {version} ({BU.env_key()})")
        g["render_version"] = np.array(version)
        np.savez_compressed(fix / "golden.npz", **g)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
