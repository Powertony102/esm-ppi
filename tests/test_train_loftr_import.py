import torch
import importlib


def test_import_and_normalize_feats():
    m = importlib.import_module("scripts.train_loftr")
    fa = torch.randn(2, 10, 64)
    fb = torch.randn(2, 12, 64)
    fa_n, fb_n = m.normalize_feats(fa, fb)
    assert fa_n.shape == (2, 10, 64)
    assert fb_n.shape == (2, 12, 64)


def test_contacts_outer_map():
    m = importlib.import_module("scripts.train_loftr")
    ca = torch.randn(2, 10, 10).abs()
    cb = torch.randn(2, 12, 12).abs()
    ma = torch.ones(2, 10, dtype=torch.bool)
    mb = torch.ones(2, 12, dtype=torch.bool)
    tmap = m.contacts_outer_map(ca, cb, ma, mb)
    assert tmap.shape == (2, 10, 12)
