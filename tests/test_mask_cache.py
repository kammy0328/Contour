"""1.27: the face masks kept between a frame's retouch steps (engine._cached) are held to a number of bytes, not only of
masks: with a body effect a mask is the whole 4K frame (33 MB, nearly all zeros round one face) and 48 of them were up to
1.6 GB held between frames (IWALY c004, 6 people, j2905). A mask let go is made again, the same."""
import numpy as np

from contour.retouch import engine as E


def _lm(seed):
    rng = np.random.default_rng(seed)
    return (rng.uniform(100, 900, (68, 3))).astype(np.float32)


def test_masks_are_kept_up_to_a_number_of_bytes_and_made_again_the_same(monkeypatch):
    E._MCACHE.clear()
    monkeypatch.setattr(E, "MCACHE_BYTES", 3 * 1000 * 1000 * 4 + 100)       # room for three masks of 1000 x 1000
    made = []

    def make(i):
        def f():
            made.append(i)
            return np.full((1000, 1000), float(i), np.float32)
        return f
    lms = [_lm(i) for i in range(5)]
    for i in range(5):
        m = E._cached("t", lms[i], (1000, 1000), 1.0, 300.0, make(i))
        assert float(m[0, 0]) == i
    assert len(E._MCACHE) == 3 and made == [0, 1, 2, 3, 4]                   # the two least recently used were let go
    assert float(E._cached("t", lms[4], (1000, 1000), 1.0, 300.0, make(4))[0, 0]) == 4 and made[-1] == 4 and len(made) == 5   # kept: not made again
    assert float(E._cached("t", lms[0], (1000, 1000), 1.0, 300.0, make(0))[0, 0]) == 0 and len(made) == 6                    # let go: made again, the same
    # one asked for stays the newest: 2 goes before 3 after 3 is asked again
    E._cached("t", lms[3], (1000, 1000), 1.0, 300.0, make(3))
    E._cached("t", _lm(9), (1000, 1000), 1.0, 300.0, make(9))
    keys = [np.frombuffer(k[1], np.float32)[0] for k in E._MCACHE]
    assert float(lms[3][0, 0]) in keys and float(lms[4][0, 0]) not in keys
    # small masks (a person's own piece of the picture): by their number, as before
    E._MCACHE.clear()
    monkeypatch.setattr(E, "MCACHE_BYTES", 192 * 2 ** 20)
    for i in range(60):
        E._cached("s", _lm(100 + i), (64, 64), 1.0, 50.0, lambda: np.zeros((64, 64), np.float32))
    assert len(E._MCACHE) == E.MCACHE_N
    # a mask bigger than the whole allowance is still given (and kept alone)
    E._MCACHE.clear()
    monkeypatch.setattr(E, "MCACHE_BYTES", 10)
    big = E._cached("b", _lm(1), (500, 500), 1.0, 100.0, lambda: np.ones((500, 500), np.float32))
    assert big.shape == (500, 500) and len(E._MCACHE) == 1
    # tuples and None are counted and kept as they are
    assert E._mc_bytes((np.zeros(10, np.float32), None, [np.zeros(5, np.uint8)])) == 45 and E._mc_bytes(None) == 0
    E._MCACHE.clear()
