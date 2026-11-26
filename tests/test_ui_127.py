"""1.27, what the person at the computer sees of it: the TensorRT RTX switch in the settings, the device named in the top
bar, a render's pace told so that a slow frame reads as one, the note while people are linked."""
import pytest


def test_render_pace_reads_as_seconds_a_frame_when_a_frame_takes_that_long():
    from contour.warp.render import pace_text
    assert pace_text(12, 40, 112.8) == "12/40 · 프레임당 9.4초 · 남은 시간 4분 23초"
    assert pace_text(30, 240, 10.0) == "30/240 · 3.0 fps · 남은 시간 1분 10초"
    assert pace_text(3, 2000, 30.0) == "3/2000 · 프레임당 10.0초 · 남은 시간 5시간 32분"      # hours when it is that long
    assert pace_text(40, 40, 100.0).endswith("남은 시간 0분 0초")
    assert pace_text(0, 40, 0.0) == "0/40" and pace_text(0, 40, 5.0) == "0/40"               # nothing to tell a pace from


def test_the_top_bar_names_cuda_and_tensorrt_rtx_when_both_run(monkeypatch):
    from contour.track import backend as BK
    BK.forget()
    assert BK.summary_label("CUDA") == "CUDA"
    BK.note("det.onnx", asked=["cuda", "cpu"], session=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert BK.summary_label("x") == "CUDA"
    BK.note("image-encoder-gpu.onnx", asked=["rtx", "cuda", "cpu"], session=[BK.RTX_EP, "CPUExecutionProvider"])
    monkeypatch.setitem(BK.BACKEND, BK.RTX_EP, "rtx")
    assert BK.summary_label("x") == "CUDA + TensorRT RTX"
    BK.note("seg.onnx", asked=["cuda", "cpu"], session=["CPUExecutionProvider"])
    assert BK.summary_label("x") == "CUDA + TensorRT RTX (일부 CPU)"
    BK.forget()


def test_the_settings_line_tells_what_is_known_without_loading_anything(monkeypatch):
    import importlib.util
    from contour.track import backend as BK
    monkeypatch.setattr(BK, "system", lambda: "win32")
    monkeypatch.setattr(BK, "_rtx", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    assert "구성요소 없음" in BK.rtx_brief()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    assert BK.rtx_brief().startswith("설치됨")
    monkeypatch.setattr(BK, "_rtx", dict(devices=["gpu"], why="", version="0.5.0", name=BK.RTX_EP, state="ok"))
    assert BK.rtx_brief() == "사용 가능 (TensorRT RTX 0.5.0)"
    monkeypatch.setattr(BK, "_rtx", dict(devices=[], why="이 GPU는 TensorRT RTX 대상이 아님 (RTX 30 시리즈 이상)", version="0.5.0",
                                         name=BK.RTX_EP, state="no_device"))
    assert "대상이 아님" in BK.rtx_brief()
    monkeypatch.setattr(BK, "system", lambda: "darwin")
    assert BK.rtx_brief() == ""


def test_the_settings_switch_is_off_until_it_is_switched_on_and_lets_go_of_the_sessions_kept(qapp, monkeypatch):
    from contour.core.settings import settings
    from contour.track import models as TM
    from contour.ui.dialogs import SettingsDialog
    assert settings().rtx == "off"                          # an analysis is CUDA's until the user says otherwise
    d = SettingsDialog()
    assert not d.rtx.isChecked() and d.rtx.isEnabled()
    # only beside CUDA: another device, and the switch cannot be used
    d.gpu.setCurrentIndex(d.gpu.findData("cpu"))
    assert not d.rtx.isEnabled()
    d.gpu.setCurrentIndex(d.gpu.findData("auto"))
    assert d.rtx.isEnabled()
    TM._kept["x"] = object()
    d.apply()                                               # nothing changed: nothing let go
    assert settings().rtx == "off" and "x" in TM._kept
    d.rtx.setChecked(True)
    d.apply()
    assert settings().rtx == "auto" and not TM._kept        # the sessions kept were opened by the choice before
    d2 = SettingsDialog()
    assert d2.rtx.isChecked()
    TM._kept["x"] = object()
    d2.rtx.setChecked(False)
    d2.apply()
    assert settings().rtx == "off" and not TM._kept


def test_the_note_in_the_top_bar_is_cleared_only_when_it_is_still_the_one_shown(qapp):
    from contour.ui import widgets as Wd
    s = Wd.SaveIndicator()
    s.start("인물 연결 중")
    assert s.text.text() == "인물 연결 중" and not s.spin.isHidden()
    s.clear("저장 중")                                       # another note: stays
    assert s.text.text() == "인물 연결 중"
    s.clear("인물 연결 중")
    assert s.text.text() == "" and s.spin.isHidden() and s.check.isHidden()
    s.done("저장됨 12:00")
    s.clear("인물 연결 중")
    assert s.text.text() == "저장됨 12:00"


def test_the_render_page_tells_the_speed_settings_while_a_cut_renders(qapp, monkeypatch):
    from contour.core import perf
    from contour.ui import pages as PG
    monkeypatch.setattr(perf, "summary", lambda: "RAM 31 GB, 코어 18: 사람 4명 동시, 다음 3프레임 미리")
    line = PG.RenderQueuePage._perf_line(None)
    assert line == "속도 설정 · RAM 31 GB, 코어 18: 사람 4명 동시, 다음 3프레임 미리"
