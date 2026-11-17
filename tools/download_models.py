"""Download every model once (about 1.1 GB) so the first analysis starts immediately."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour.track import models as M  # noqa: E402


def reset_acl():
    """Windows: models downloaded by 1.9.0's elevated installer kept an Administrators-only ACL. The installer runs
    this elevated again: give every file in the models folder the folder's inherited permissions back."""
    if sys.platform != "win32":
        return
    import subprocess
    root = M.model_path("det_yolox_m").parent.parent
    if root.exists():
        try:
            subprocess.run(["icacls", str(root), "/reset", "/T", "/C", "/Q"], capture_output=True, timeout=300)
        except Exception:
            pass


def main():
    reset_acl()
    keys = ["det_yolox_m", "pose_rtmpose_x", "pose_rtmw_x", "pose_rtmpose_m", "face_yunet", "matte_rvm",
            "face_rtm106", "hand_rtm21", "seg_sam31", "seg_sam31_find", "seg_sam2_s", "face_sface", "face_parse"]
    if "--max" in sys.argv:
        keys += ["det_yolox_x", "pose_dwpose_l", "seg_sam2_b"]
    failed = []
    for k in keys:
        spec = M.MODELS[k]
        if M.is_downloaded(k):
            print(f"  [있음] {spec['label']}")
            continue
        last = [-1]

        def prog(f, msg):
            pct = int(f * 100)
            if pct != last[0] and pct % 5 == 0:
                last[0] = pct
                print(f"\r  {msg}   ", end="", flush=True)
        try:
            M.ensure_model(k, prog)
            print(f"\r  [완료] {spec['label']}                                  ")
        except Exception as e:
            failed.append(k)
            print(f"\r  [실패] {spec['label']}: {e}")
            print("         인터넷 연결을 확인하고 install.bat을 다시 실행하세요. (처음 분석할 때도 자동으로 받습니다)")
    print(f"  모델 위치: {M.model_path('det_yolox_m').parent}")
    if failed:
        print(f"  받지 못한 모델 {len(failed)}개: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
