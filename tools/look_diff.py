"""The engine's settings that differ between two versions of the code (for retouch/looks.py: what a version that changes the
defaults has to put under the outgoing look's name).

    git worktree add --detach /tmp/old v1.26.2
    python tools/look_diff.py /tmp/old .

Every module-level UPPER_CASE setting of the retouch modules that is a plain value (number, text, tuple, list, dict, None) is
compared; NEW: only in the second (give it the value that leaves it out), GONE: only in the first, CHG: changed."""
import json
import os
import subprocess
import sys

MODS = ("engine", "skin", "detect", "bodyskin", "bodyspots", "bodytone")
DUMP = r'''
import sys, json, os
sys.path.insert(0, sys.argv[1])
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import importlib
out = {}
for mn in %r:
    m = importlib.import_module("contour.retouch." + mn)
    for k, v in vars(m).items():
        if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None), tuple, list, dict)):
            try:
                json.dumps(v)
                out[mn + "." + k] = v
            except TypeError:
                pass
print(json.dumps(out))
''' % (MODS,)


def dump(root: str) -> dict:
    r = subprocess.run([sys.executable, "-c", DUMP, os.path.abspath(root)], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(root)))
    if r.returncode:
        raise SystemExit(r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main(a: str, b: str) -> int:
    o, n = dump(a), dump(b)
    for k in sorted(set(o) | set(n)):
        if k not in o:
            print("NEW ", k, "=", json.dumps(n[k]))
        elif k not in n:
            print("GONE", k, "=", json.dumps(o[k]))
        elif o[k] != n[k]:
            print("CHG ", k, ":", json.dumps(o[k]), "->", json.dumps(n[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
