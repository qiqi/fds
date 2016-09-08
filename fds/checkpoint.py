import os
import dill as pickle
from collections import namedtuple

Checkpoint = namedtuple('Checkpoint', 'u0 V v lss G_lss g_lss J G_dil g_dil')

def save_checkpoint(checkpoint_path, cp):
    '''
    save a checkpoint file under the path checkpoint_path,
    naming convention is mXX_segmentYYY, where XX and YY are given by cp.lss
    '''
    filename = 'm{0}_segment{1}'.format(cp.lss.m_modes(), cp.lss.K_segments())
    with open(os.path.join(checkpoint_path, filename), 'wb') as f:
        pickle.dump(cp, f)

def load_checkpoint(checkpoint_file):
    return pickle.load(open(checkpoint_file, 'rb'))

def verify_checkpoint(checkpoint):
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint
    return lss.K_segments() == len(G_lss) \
                            == len(g_lss) \
                            == len(J_hist) \
                            == len(G_dil) \
                            == len(g_dil)

def load_last_checkpoint(checkpoint_path, m):
    '''
    load checkpoint in path checkpoint_path, with file name mXX_segmentYYY,
    where XX matches the given m, and YY is the largest
    '''
    def m_modes(filename):
        try:
            m, _ = filename.split('_segment')
            assert m.startswith('m')
            return int(m[1:])
        except:
            return None

    def segments(filename):
        try:
            _, segments = filename.split('_segment')
            return int(segments)
        except:
            return None

    files = filter(lambda f : m_modes(f) == m and segments(f) is not None,
                   os.listdir(checkpoint_path))
    files = sorted(files, key=segments)
    if len(files):
        return load_checkpoint(os.path.join(checkpoint_path, files[-1]))
