import os, sys, threading, subprocess
from types import SimpleNamespace as NS
from unittest.mock import Mock
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def local_settings(tmp_path):
    from contour.core import settings as S
    S._settings = S.AppSettings(data_dir=str(tmp_path/'data'), cache_dir=str(tmp_path/'cache'), gpu='cpu',auto_update_check=False)
    yield
    S._settings = None

def test_manager_refuses_rename_delete_if_close_failed(monkeypatch):
    from contour.ui.projectmgr import ProjectsPage
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k: QMessageBox.Yes)
    page=NS(current_path=lambda:'/tmp/current.dpprj', main=NS(close_project=Mock(return_value=False)))
    allowed=ProjectsPage._release_if_open(page,'/tmp/current.dpprj','이름 변경')
    assert not allowed, 'project manager allowed rename/delete although close_project returned False'

def test_save_refuses_if_pending_edits_did_not_drain(tmp_path):
    from contour.ui.main_window import MainWindow
    session=NS(project=NS(path=str(tmp_path)),save=Mock())
    w=NS(session=session,activity=NS(flush=Mock()),edit_worker=NS(drain=Mock(return_value=False)),
         saved_lbl=NS(start=Mock(),done=Mock()),_clear_recovery=Mock(),_collect_learning=Mock(),show_error=Mock())
    ok=MainWindow.save(w)
    assert ok is False and session.save.call_count == 0, f'save returned {ok}, writes after timeout: {session.save.call_count}'

def test_encoded_full_range_keeps_appearance(tmp_path):
    from contour.core import media as M
    from contour.core.project import Project,OutputSettings
    from contour.warp.render import render_cut
    W,H=64,48
    pl=[np.full((H,W),128,np.uint16),np.full((H,W),512,np.uint16),np.full((H,W),512,np.uint16)]
    src=tmp_path/'source.mkv'
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','yuv444p10le','-s','64x48','-r','24','-i','-',
        '-frames:v','3','-c:v','ffv1','-level','3','-color_range','pc','-colorspace','bt709',str(src)],input=b''.join(p.tobytes() for p in pl)*3,check=True)
    info=M.probe(str(src)); assert info.color_range == 'pc'
    prj=Project.create(str(tmp_path),'range'); cut=prj.add_cut(info)
    tracker=NS(store=None,matte=NS(w=W//2,h=H//2))
    rep=render_cut(prj,cut,tracker,OutputSettings(format='prores_4444',out_dir=str(tmp_path/'out'),include_audio=False))
    out=M.probe(rep.out_path)
    before=M.read_frame_rgb(info,0).astype(float)*255
    after=M.read_frame_rgb(out,0).astype(float)*255
    print('full range before/after:',before[0,0],after[0,0],'tags:',info.color_range,out.color_range)
    assert np.max(np.abs(before-after)) <= 2, f'source RGB {before[0,0]}, output RGB {after[0,0]}; ranges {info.color_range} -> {out.color_range}'

def test_final_result_cache_isolated_by_epoch(monkeypatch):
    from contour.ui.session import EditSession
    class PX:
        def __init__(self,v): self.v=v
        def frame(self,t): return np.full((4,4,3),self.v,np.uint8)
        def close(self): pass
    s=EditSession(); cut=NS(id='A',mode='warp'); s.cut=cut; s.proxy=PX(17)
    entered=threading.Event(); release=threading.Event()
    original=s.retouched
    def delayed(t):
        img=original(t)
        entered.set()
        assert release.wait(5)
        return img
    monkeypatch.setattr(s,'retouched',delayed)
    thread=threading.Thread(target=lambda:s.result(0))
    thread.start(); assert entered.wait(5)
    s.open_cut(NS(id='B',mode='warp'))
    s.open_cut(cut); s.proxy=PX(203)
    release.set(); thread.join(5)
    monkeypatch.setattr(s,'retouched',original)
    got=s.result(0)
    assert int(got[0,0,0]) == 203, f'old final result came back under new epoch: {got[0,0,0]} instead of 203'

def test_incomplete_proxy_not_usable_for_render_queue(tmp_path):
    from contour.core import media as M
    from contour.core.proxy import Proxy
    from contour.core.colour import manager, ViewSettings, PASSTHROUGH
    src=tmp_path/'source.mkv'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=24','-frames:v','3','-c:v','ffv1',str(src)],check=True)
    info=M.probe(str(src)); info.frames=6
    px=Proxy(tmp_path/'proxy',info,64)
    with pytest.raises(RuntimeError):px.build(manager(),ViewSettings(input_cs=PASSTHROUGH))
    assert not px.has_frames(), f'has_frames True despite complete={px.meta()["complete"]}, n_decoded={px.meta().get("n_decoded")}'

def test_implicit_cpu_fallback_is_reported(monkeypatch,tmp_path):
    from contour.track import models as M
    import onnxruntime as ort
    monkeypatch.setattr(M,'providers',lambda:['CUDAExecutionProvider','CPUExecutionProvider'])
    monkeypatch.setattr(ort,'InferenceSession',lambda *a,**k:NS(get_providers=lambda:['CPUExecutionProvider']))
    M.take_cpu_fallbacks()
    session=M._session_raw(tmp_path/'test.onnx')
    assert session.get_providers()==['CPUExecutionProvider']
    assert M.take_cpu_fallbacks(), 'actual provider CPU, requested CUDA, but no warning recorded'


def test_batch_before_does_not_drop_unsaved_cut():
    from contour.ui.main_window import MainWindow
    cut=NS(id='A')
    s=NS(project=NS(cut=lambda cid:cut),cut=cut,open_cut=Mock())
    w=NS(session=s,save=Mock(return_value=False),viewer=NS(message='',update=Mock()))
    allowed=MainWindow._batch_before(w,'A')
    assert not allowed and not s.open_cut.called, f'batch allowed={allowed}, cut closed={s.open_cut.called} after save=False'


def test_skinbench_mask_is_the_persons_own():
    """skinbench scores one person with that person's masks only, not the union of every target in the frame."""
    from contour.learn.skinbench import person_mask
    from contour.retouch import engine as RE
    a = np.zeros((4, 4), np.float32); a[0, 0] = 1
    b = np.zeros((4, 4), np.float32); b[3, 3] = 1
    mk = lambda pid, m: RE.Job(p=pid, pid=pid, v=defaultdict(float), keep=0.5, scope="face", grain_size=1, face=None,
                               face_w=1, scale=1, masks={"skin": m})
    rf = RE.RetouchFrame(t=0, jobs=[mk(1, a), mk(2, b)], proxy_wh=(4, 4), src_wh=(4, 4))
    m1 = person_mask(rf, 1)
    assert m1[0, 0] == 1 and m1[3, 3] == 0, "person 2's skin must not be in person 1's score"
    assert person_mask(rf, 9) is None
