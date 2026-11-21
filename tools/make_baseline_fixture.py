"""Makes tests/data/baseline_1222: a small analysed project exactly as DIAP 1.22.2 saves it, and what 1.22.2 shows
and renders from it. tests/test_baseline_1222.py opens and renders the frozen project with the code of now and
compares (instruction 2026-11-20 §6-1: the output and the opening of existing projects are held first).

Run with the 1.22.2 code on the path, from the repository of now (for tests/baseline_util.py):

    git worktree add ../main_1222 6a9f96e
    PYTHONPATH=../main_1222 python tools/make_baseline_fixture.py tests/data/baseline_1222

    PYTHONPATH=../main_1222 python tools/make_baseline_fixture.py tests/data/baseline_1222_body --body

--body: the second fixture (instruction 2026-11-20 §2: "기존 프로젝트의 명시적인 몸 영역 선택은 마이그레이션 후에도 같은
결과를 유지"): the same person with bare shoulders and the skin retouch's body regions chosen by hand; besides the
planes and the preview, the preview 1.22.2 shows with the face and neck alone (what the body regions add is the
difference of the two).

Only run again when the fixture itself has to change; the point of it is that it does not."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))


def main(out: str, body: bool = False) -> int:
    out = Path(out).resolve()
    work = Path(tempfile.mkdtemp(prefix="diap_baseline_"))
    home = work / "home"
    home.mkdir()
    os.environ["HOME"] = os.environ["USERPROFILE"] = str(home)
    os.environ["CONTOUR_NO_UPDATE"] = "1"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    import contour
    from contour.core.settings import APP_VERSION
    import baseline_util as BU
    print("code:", os.path.dirname(contour.__file__), "version", APP_VERSION)
    if APP_VERSION != "1.22.2":
        print("NOT 1.22.2: the fixture must be made with the 1.22.2 code (PYTHONPATH)", file=sys.stderr)
        return 2
    extra = {}
    if body:
        src = BU.write_source(work / "src.mkv", bare=True)
        prj, cut = BU.make_project(work, src, regions=BU.BODY_REGIONS)
        assert cut.params_for(0).skin_scope == BU.BODY_REGIONS
    else:
        src = BU.write_source(work / "src.mkv")
        prj, cut = BU.make_project(work, src)
    with BU.NoModels():
        planes = BU.render_planes(prj, cut)
        prev = BU.preview_frames(prj, cut)
        if body:
            cut.params_for(0).skin_scope = "face,neck"
            extra["preview_face"] = BU.preview_frames(prj, cut)
            cut.params_for(0).skin_scope = BU.BODY_REGIONS
    from contour.core.project import Project
    loaded = Project.load(prj.path).cuts[0].to_dict()
    text = json.dumps(loaded, ensure_ascii=False, sort_keys=True)
    text = text.replace(json.dumps(str(src))[1:-1], json.dumps(BU.PLACEHOLDER)[1:-1])
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copyfile(src, out / "src.mkv")
    pdir = Path(prj.path)
    shutil.rmtree(pdir / "out", ignore_errors=True)
    shutil.copytree(pdir, out / "project")
    for f in (out / "project").glob("project*.json"):
        t = f.read_text(encoding="utf-8")
        a, b = json.dumps(str(src))[1:-1], json.dumps(BU.PLACEHOLDER)[1:-1]
        assert a in t or f.name != "project.json", f.name      # (the backup is the project before the cut was added)
        f.write_text(t.replace(a, b), encoding="utf-8")
    np.savez_compressed(out / "golden.npz", planes=planes, preview=prev, cut_json=np.array(text),
                        version=np.array(APP_VERSION), frames=np.array([0, 3, 5]), env=np.array(BU.env_key()), **extra)
    src_rgb = BU.frames_rgb(body)
    if body:
        d = np.abs(prev.astype(int) - extra["preview_face"].astype(int)).max(-1)
        print("body regions change", int((d > 0).sum()), "preview pixels, max", int(d.max()))
    print("planes", planes.shape, planes.dtype, "preview", prev.shape,
          "preview changed vs source: mean |d| =", float(np.abs(prev.astype(float) - src_rgb[[0, 3, 5]]).mean()))
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out).as_posix():50s} {f.stat().st_size:8d}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    body_ = "--body" in sys.argv[1:]
    sys.exit(main(args[0] if args else str(HERE / "tests" / "data" / ("baseline_1222_body" if body_ else "baseline_1222")), body_))
