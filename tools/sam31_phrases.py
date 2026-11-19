"""Encodes phrases for SAM 3.1's finder once, so the app ships their encodings and never downloads the 700 MB text
encoder (as data/sam31_text_person.npz for "person").

  python tools/sam31_phrases.py <text-encoder.onnx> <bpe_simple_vocab_16e6.txt.gz> <out.npz> phrase [phrase ...]

Tokenizer: the makers' byte-pair encoding (CLIP's simple tokenizer, MIT), cleaned as smile-forge/sam3.1-onnx's
README says: lower case, whitespace collapsed, between the start (49406) and end (49407) tokens, cut to 32, padded
with 0. Plain ASCII phrases only (no ftfy / HTML clean-up needed for them). The text encoder and the vocabulary are
smile-forge/sam3.1-onnx's (SAM License)."""
from __future__ import annotations

import gzip
import re
import sys
from functools import lru_cache

import numpy as np


@lru_cache()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class Tokenizer:
    def __init__(self, bpe_path: str):
        self.byte_encoder = _bytes_to_unicode()
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for m in merges:
            vocab.append("".join(m))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\s\w]+""", re.I)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str):
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for tok in self.pat.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[b] for b in self.bpe(tok).split(" "))
        return ids

    def tokens(self, text: str, n: int = 32) -> np.ndarray:
        ids = [49406] + self.encode(text)[: n - 2] + [49407]
        out = np.zeros((1, n), np.int64)
        out[0, :len(ids)] = ids
        return out


def encode(text_encoder: str, bpe: str, phrases) -> dict:
    import onnxruntime as ort
    s = ort.InferenceSession(text_encoder, providers=["CPUExecutionProvider"])
    tk = Tokenizer(bpe)
    name = s.get_inputs()[0].name
    out = {}
    for ph in phrases:
        t = tk.tokens(ph)
        text, pad = s.run(None, {name: t.astype(np.int64)})
        out[ph] = (text.astype(np.float32), pad)
    return out


if __name__ == "__main__":
    te, bpe, dst, *phrases = sys.argv[1:]
    enc = encode(te, bpe, phrases)
    arrays = {}
    for ph, (text, pad) in enc.items():
        k = ph.replace(" ", "_")
        arrays[f"{k}__text"] = text
        arrays[f"{k}__text_pad"] = pad
    np.savez_compressed(dst, phrases=np.array(phrases), **arrays)
    print("saved", dst, list(enc))
