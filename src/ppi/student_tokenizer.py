import torch

AA_BASE = list("ACDEFGHIKLMNPQRSTVWY")
AA_EXTRA = ["X", "B", "Z", "U", "O"]


def build_vocab():
    vocab = {"<pad>": 0}
    idx = 1
    for c in AA_BASE + AA_EXTRA:
        vocab[c] = idx
        idx += 1
    return vocab


VOCAB = build_vocab()


def to_indices(seq: str, max_len: int):
    s = seq.strip().upper()[:max_len]
    ids = []
    for ch in s:
        ids.append(VOCAB.get(ch, VOCAB["X"]))
    return ids


def pad_batch(seqs_a, seqs_b, max_len: int):
    a_ids = [to_indices(s, max_len) for s in seqs_a]
    b_ids = [to_indices(s, max_len) for s in seqs_b]
    la = max(len(x) for x in a_ids) if a_ids else 1
    lb = max(len(x) for x in b_ids) if b_ids else 1
    la = min(la, max_len)
    lb = min(lb, max_len)
    pa = torch.zeros((len(a_ids), la), dtype=torch.long)
    pb = torch.zeros((len(b_ids), lb), dtype=torch.long)
    for i, ids in enumerate(a_ids):
        n = min(len(ids), la)
        pa[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
    for i, ids in enumerate(b_ids):
        n = min(len(ids), lb)
        pb[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
    return pa, pb

