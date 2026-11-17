import os,sys,time,threading,subprocess
from pathlib import Path
from types import SimpleNamespace as NS
"""Review c84dc81: an original decoded for an earlier opening of the same cut is not shown; cancelling the next-cut preparation does not wait for the decoder on the UI thread."""
import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

@pytest.fixture(autouse=True)
def setup(tmp_path):
    from contour.core import settings as S, frames as F
    S._settings=S.AppSettings(data_dir=str(tmp_path/'data'),cache_dir=str(tmp_path/'cache'),gpu='cpu',proxy_mode='off',ask_proxy=False,auto_update_check=False)
    F.pool().clear()
    yield
    F.pool().clear(); S._settings=None

@pytest.fixture
def project(tmp_path):
    from contour.core import media as M
    from contour.core.project import Project
    src=tmp_path/'src.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=24','-frames:v','30','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(src)],check=True)
    p=Project.create(str(tmp_path),'test'); p.add_cut(M.probe(str(src)),'test'); p.save()
    return p

def test_original_loader_rejects_previous_epoch(project):
    from contour.ui.viewer import Viewer
    from contour.ui.session import EditSession
    app=QApplication.instance() or QApplication([])
    gate=threading.Event(); entered=threading.Event()
    class PX:
        kind='source'
        def __init__(self,value,wait=False):self.value=value;self.wait=wait
        def frame(self,t):
            if self.wait:
                entered.set(); assert gate.wait(5)
            return np.full((48,64,3),self.value,np.uint8)
        def peek(self,t):return None
        def close(self):pass
    s=EditSession(); c=project.cuts[0]
    s.cut=c; s.proxy=PX(17,True)
    v=Viewer(s);v.t=0
    try:
        v._qimg('orig'); assert entered.wait(3)
        old_epoch=s.epoch
        # Same cut reopened after a colour/LUT change; memory/Qt image caches cleared by the real path.
        s.open_cut(c); s.proxy=PX(203)
        assert old_epoch!=s.epoch
        v._img_cache.clear();v._last_orig=None
        gate.set()
        deadline=time.monotonic()+3
        while ('orig',0,c.id) not in v._img_cache and time.monotonic()<deadline:
            app.processEvents();time.sleep(.005)
        image=v._qimg('orig')
        assert image is None or image.pixelColor(0,0).red()!=17, 'old epoch RGB 17 inserted into new epoch; correct new view is RGB 203'
    finally:
        gate.set();v.stop_workers();s.open_cut(None);v.close()

def test_cancel_next_cut_does_not_wait_for_source_decode_on_ui(project,monkeypatch):
    from contour.core.frames import SourceFrames
    from contour.ui.workers import NextCutPrep
    app=QApplication.instance() or QApplication([])
    entered=threading.Event();release=threading.Event()
    original=SourceFrames._decode_to
    def delayed(self,i):
        entered.set(); assert release.wait(5)
        return original(self,i)
    monkeypatch.setattr(SourceFrames,'_decode_to',delayed)
    prep=NextCutPrep();prep.start(project,project.cuts[0],head=1)
    assert entered.wait(3)
    own=prep._own
    threads=[t for t in threading.enumerate() if t.name=='diap-nextcut']
    elapsed=[];start=time.monotonic()
    unblock=threading.Timer(.25,release.set);unblock.start()
    QTimer.singleShot(0,prep.cancel)
    QTimer.singleShot(10,lambda:elapsed.append(time.monotonic()-start))
    QTimer.singleShot(450,app.quit)
    try:
        app.exec()
        for t in threads:t.join(3)
        assert elapsed and elapsed[0]<.15, f'cancel held UI for {elapsed[0]:.3f}s waiting on SourceFrames.close reader lock'
    finally:
        release.set();unblock.join()
        for t in threads:t.join(3)
        if own['sh'] is not None and own['sh'].proxy is not None:own['sh'].proxy.close()
