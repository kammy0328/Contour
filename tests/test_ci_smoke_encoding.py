"""tools/ci_smoke.py says its steps in Korean in part; on CI the console is a pipe on a Windows code page (cp1252) and printing
them stopped the installed app's short run (1.24.0 release build, 2026-11-21). The script sets its own streams to UTF-8."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_smoke_script_prints_korean_on_a_narrow_code_page():
    code = ("import sys, runpy; sys.argv = ['ci_smoke.py', '--help']; "
            "sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='cp1252', errors='strict'); "
            "sys.stderr = __import__('io').TextIOWrapper(sys.stderr.buffer, encoding='cp1252', errors='strict'); "
            "g = runpy.run_path(r'%s', run_name='smoke_probe'); print('한글 출력'); sys.stdout.flush()" % (ROOT / "tools" / "ci_smoke.py"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]
    assert "한글 출력".encode("utf-8") in r.stdout
