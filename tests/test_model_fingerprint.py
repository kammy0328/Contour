"""The models an analysis ran with are recorded on the cut (a fingerprint of their files), and a changed file changes it."""


def test_fingerprint_follows_the_file(tmp_path):
    from contour.track import models as M
    from contour.core.project import Cut
    p = M.model_path("face_yunet")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x08" + b"a" * 4000)
    a = M.fingerprint("face_yunet")
    assert len(a) == 16
    assert M.fingerprint("face_yunet") == a                     # cached, same
    import os, time
    p.write_bytes(b"\x08" + b"b" * 4000)
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert M.fingerprint("face_yunet") != a
    assert M.fingerprint("det_yolox_x") == ""                 # not downloaded
    fps = M.fingerprints()
    assert set(fps) == {"face_yunet"}
    c = Cut.from_dict(dict(Cut(id="x", name="x", source="", info=__import__("contour.core.media", fromlist=["MediaInfo"]).MediaInfo(path="")).to_dict(), models_used=fps))
    assert c.models_used == fps
