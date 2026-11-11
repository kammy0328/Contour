"""Download every model once (about 1.1 GB) so the first analysis starts immediately."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour.track import models as M  # noqa: E402


def main():
    keys = ["det_yolox_m", "pose_rtmpose_x", "pose_rtmw_x", "pose_rtmpose_m", "face_yunet", "matte_rvm",
            "face_rtm106", "hand_rtm21", "seg_sam2_s", "face_sface", "face_parse"]
    if "--max" in sys.argv:
        keys += ["det_yolox_x", "pose_dwpose_l", "seg_sam2_b"]
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
            print(f"\r  [실패] {spec['label']}: {e}")
            print("         인터넷 연결을 확인하고 install.bat을 다시 실행하세요. (처음 분석할 때도 자동으로 받습니다)")
    print(f"  모델 위치: {M.model_path('det_yolox_m').parent}")


if __name__ == "__main__":
    main()
