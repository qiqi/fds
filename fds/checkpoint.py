import pickle
from collections import namedtuple

Checkpoint = namedtuple('Checkpoint', 'u0 V v lss G_lss g_lss J G_dil g_dil')

def save_checkpoint(checkpoint_file, cp):
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(cp, f)

def load_checkpoint(checkpoint_file):
    return pickle.load(open(checkpoint_file, 'rb'))

def verify_checkpoint(checkpoint):
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint
    return lss.m_segments() == len(G_lss) \
                            == len(g_lss) \
                            == len(J_hist) \
                            == len(G_dil) \
                            == len(g_dil)
