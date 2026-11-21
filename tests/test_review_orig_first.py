"""Review of c1c48b7 (orig-first): five edge cases, each failing before its fix."""
import os, sys, time, threading, subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import replace
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

@pytest.fixture(autouse=True)
def settings(tmp_path):
    from contour.core import settings as S
    from contour.core import frames as F
    S._settings=S.AppSettings(data_dir=str(tmp_path/'data'),cache_dir=str(tmp_path/'cache'),gpu='cpu',proxy_mode='off',ask_proxy=False,auto_update_check=False)
    F.pool().clear()
    yield S._settings
    F.pool().clear()
    S._settings=None

@pytest.fixture
def project(tmp_path):
    from contour.core import media as M
    from contour.core.project import Project
    src=tmp_path/'src.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=24','-frames:v','100','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(src)],check=True)
    p=Project.create(str(tmp_path),'test'); p.add_cut(M.probe(str(src)),'test'); p.save()
    return p

def test_sequence_start_is_part_of_cache_key(tmp_path):
    from contour.core import frames as F, media as M
    from contour.core.colour import manager,ViewSettings,PASSTHROUGH
    for k in range(10):
        cv2.imwrite(str(tmp_path/f'f.{1000+k:04d}.png'),np.full((48,64,3),30 if k<5 else 220,np.uint8))
    info=M.probe(str(tmp_path/'f.1000.png'))
    a=replace(info,frames=5,seq_start=1000)
    b=replace(info,frames=5,seq_start=1005,path=str(tmp_path/'f.1005.png'))
    vs=ViewSettings(input_cs=PASSTHROUGH)
    fa=F.SourceFrames(tmp_path/'a',a,(64,48),manager(),vs)
    fb=F.SourceFrames(tmp_path/'b',b,(64,48),manager(),vs)
    try:
        fa.frame(0)
        got=fb.frame(0).copy()
        F.pool().clear()
        expected=fb.frame(0).copy()
        assert np.array_equal(got,expected), f'seq start 1005 returned {got[0,0].tolist()}, correct {expected[0,0].tolist()}, owner equality={fa.owner==fb.owner}'
    finally:
        fa.close(); fb.close()

def test_next_cut_cancel_closes_decoders(project,monkeypatch):
    from contour.ui.session import EditSession
    from contour.ui.workers import NextCutPrep
    from contour.core import frames as F
    entered=threading.Event(); release=threading.Event()
    def paused_result(s,t,final=False):
        entered.set()
        assert release.wait(5)
        return s.frame(t)
    monkeypatch.setattr(EditSession,'result',paused_result)
    prep=NextCutPrep(); prep.start(project,project.cuts[0],head=1)
    assert entered.wait(5)
    sf=prep.session.proxy
    threads=[t for t in threading.enumerate() if t.name=='diap-nextcut']
    procs=[c.reader._proc for c in sf._cursors]
    assert procs and any(p.poll() is None for p in procs)
    try:
        prep.cancel(); release.set()
        for t in threads:t.join(3)
        assert all(not t.is_alive() for t in threads)
        assert all(p.poll() is not None for p in procs), 'next-cut thread exited but its ffmpeg decoder is still running'
    finally:
        release.set(); sf.close()

def test_original_decode_does_not_block_ui():
    from contour.ui.viewer import Viewer
    app=QApplication.instance() or QApplication([])
    began=[]; observed=[]
    def slow_frame(t):
        time.sleep(.25)
        return np.zeros((48,64,3),np.uint8)
    viewer=NS(s=NS(cut=NS(id='A'),frame=slow_frame),t=0,_img_cache={},_store_img=Mock())
    # The clock starts when the event loop reaches this test's own timers: what earlier tests left for the loop
    # (windows not deleted yet, their queued events) is handled first and is not the original's decoding. Measured
    # from before app.exec(), the heartbeat came 0.19 to 0.22 s late in a run of every test (Windows and Linux)
    # and in time alone: j2800, j2805.
    def first():
        began.append(time.perf_counter()); Viewer._qimg(viewer,'orig')
    QTimer.singleShot(0,first)
    QTimer.singleShot(10,lambda:observed.append(time.perf_counter()-began[0]))
    QTimer.singleShot(350,app.quit)
    app.exec()
    assert observed and observed[0]<.15, f'UI heartbeat delayed {observed[0]:.3f}s by original frame decoding'

def test_batch_rejects_offline_wrong_view(project,tmp_path,monkeypatch):
    from contour.core import frames as F
    from contour.core.colour import manager
    from contour.warp.render import _vs_for
    from contour.track.analyze import CutTracker
    from contour import batch
    c=project.cuts[0]
    px=F.proxy_for(project,c); assert px.build(manager(),_vs_for(c)); px.close()
    Path(c.info.path).rename(tmp_path/'moved.mp4')
    c.input_cs='Linear Rec.709 (sRGB)'; project.save()
    calls=[]
    class ReachedAnalysis(Exception):pass
    def entered(*a,**k):calls.append(1); raise ReachedAnalysis()
    monkeypatch.setattr(CutTracker,'analyze',entered)
    try:batch.analyze(project.path,c.id,prescan=False)
    except ReachedAnalysis:pass
    assert not calls, 'batch analysed old proxy in the wrong view despite original being missing'

def test_optional_proxy_disk_failure_falls_back_in_batch(project,settings,monkeypatch):
    from contour.core import proxy as P
    from contour.track.analyze import CutTracker
    from contour import batch
    settings.proxy_mode='always'
    def no_space(*a,**k):raise P.DiskSpaceError('simulated disk full')
    monkeypatch.setattr(P.Proxy,'build',no_space)
    class ReachedAnalysis(Exception):pass
    def entered(*a,**k):raise ReachedAnalysis()
    monkeypatch.setattr(CutTracker,'analyze',entered)
    with pytest.raises(ReachedAnalysis):batch.analyze(project.path,project.cuts[0].id,prescan=False)
