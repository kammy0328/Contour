"""Download every model once (about 1.1 GB) so the first analysis starts immediately.

    python tools/download_models.py [--max] [--registry ADDRESS|FOLDER|ZIP]

With a model registry (the argument, or 설정 > 모델 관리's address) and a publishing key in this version, the models
the registry offers are installed as packages first (a signed list, each file checked: track/modelpkg.py); the
models it does not offer are downloaded the way they always were. Only model files come in; nothing is sent."""
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
    reg = ""
    if "--registry" in sys.argv:
        reg = sys.argv[sys.argv.index("--registry") + 1]
    else:
        try:
            from contour.core.settings import settings
            reg = getattr(settings(), "model_registry", "") or ""
        except Exception:
            reg = ""
    if reg:
        from contour.track import modelpkg as MP
        if not MP.TRUSTED:
            print("  [패키지] 이 버전에는 모델 배포 키가 없어 패키지 설치를 건너뜁니다")
        else:
            try:
                with MP.registry_at(reg) as r:
                    offered = {o["key"] for o in MP.offered(r) if not o["why"]}
                res = MP.bring_in(reg, [k for k in keys if k in offered])
                for k, v in res["installed"] + res["kept"]:
                    print(f"  [패키지] {M.MODELS.get(k, {}).get('label', k)} {v}")
                for k, t in res["failed"].items():
                    print(f"  [패키지 실패] {k}: {t}")
            except MP.PackageError as e:
                print(f"  [패키지] 목록을 쓰지 못했습니다 ({e}). 이전 방식으로 받습니다")
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
