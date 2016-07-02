# 2016 CTR SP Qiqi Wang
import sys
import struct
import ctypes
from numpy import *

__all__ = ['NO_CHANGE', 'save_les', 'load_les']

NO_CHANGE = None

UGP_IO_MAGIC_NUMBER = 123581321
UGP_IO_EOF = 51
UGP_IO_NO_D1 = 32
UGP_IO_NO_D2 = 33
UGP_IO_FAZONE_NO_D2 = 114

def read_header(fp):
    '''
    Header structure according to core/CTIDefs.hpp
    returns name (str)
            iid (integer, record type according to core/CTIDefs.hpp
            skip (integer, distance in bytes to next record)
            idata (integer array of size 16, meaning depends on record type)
            rdata (double array of size 16, meaning depends on record type)
    '''
    UGP_IO_HEADER_NAME_LEN = 52
    name = ctypes.create_string_buffer(fp.read(UGP_IO_HEADER_NAME_LEN)).value
    iid, skip, zero = struct.unpack('iii', fp.read(12))
    assert zero == 0
    idata = frombuffer(fp.read(16*4), 'i')
    rdata = frombuffer(fp.read(16*8), 'd')
    return name, iid, skip, idata, rdata

def load_les(fname, verbose=True):
    '''
    Load CTI restart file
    Returns a dict, whose keys are record names,
                    and whose values are numpy arrays containing data
    Data types loaded include
        UGP_IO_NO_D1 = 32
        UGP_IO_NO_D2 = 33
        UGP_IO_FAZONE_NO_D2 = 114
    Other record types are ignored.
    '''
    fp = open(fname, 'rb')
    magic_number, io_version = struct.unpack('ii', fp.read(8))
    assert magic_number == UGP_IO_MAGIC_NUMBER
    if verbose:
        print('loading {0}, io_version={1}'.format(fname, io_version))
    data, iid, offset = {}, 0, 8
    while iid != UGP_IO_EOF:
        fp.seek(offset)
        name, iid, skip, idata, rdata = read_header(fp)
        offset += skip
        if iid == UGP_IO_NO_D1:
            size = idata[0]
            if verbose: print(name, size)
            data[name] = frombuffer(fp.read(size*8), 'd')
        elif iid == UGP_IO_NO_D2:
            size, dim = idata[:2]
            if verbose: print(name, size, dim)
            assert dim == 3
            data[name] = frombuffer(fp.read(size*3*8), 'd').reshape([size, 3])
        elif iid == UGP_IO_FAZONE_NO_D2:
            size, dim = idata[0], 3
            if verbose: print(name, size)
            data[name] = frombuffer(fp.read(size*3*8), 'd').reshape([size, 3])
    return data

def save_data_field(fp, size, data, name):
    '''
    Checks the size of data[name], saves it to fp, then delete the data entry.
    if data[name] is NO_CHANGE, the do not change this record in file.
    Raises AssertionError if things are not right
    '''
    assert name in data
    if data[name] is not NO_CHANGE:
        assert size == data[name].size
        fp.write(ascontiguousarray(data[name], dtype='d').tobytes())
    del data[name]

def save_les(fname, data, verbose=True):
    '''
    Saves data to an existing CTI restart file that already contains the
        data records consistent to the content of data
    data is a dict, whose keys are record names,
                    and whose values are numpy arrays containing data
    Data types saved include
        UGP_IO_NO_D1 = 32
        UGP_IO_NO_D2 = 33
        UGP_IO_FAZONE_NO_D2 = 114
    Other record types are ignored.
    '''
    fp = open(fname, 'r+b')
    magic_number, io_version = struct.unpack('ii', fp.read(8))
    assert magic_number == UGP_IO_MAGIC_NUMBER
    if verbose:
        print('saving to {0}, io_version={1}'.format(fname, io_version))
    iid, offset = 0, 8
    unsaved = dict(data)
    while iid != UGP_IO_EOF:
        fp.seek(offset)
        name, iid, skip, idata, rdata = read_header(fp)
        offset += skip
        if iid == UGP_IO_NO_D1:
            size = idata[0]
            if verbose: print(name, size)
            save_data_field(fp, size, unsaved, name)
        elif iid == UGP_IO_NO_D2:
            size, dim = idata[:2]
            if verbose: print(name, size, dim)
            assert dim == 3
            save_data_field(fp, size*3, unsaved, name)
        elif iid == UGP_IO_FAZONE_NO_D2:
            size, dim = idata[0], 3
            if verbose: print(name, size)
            save_data_field(fp, size*3, unsaved, name)
    if len(unsaved):
        sys.stderr.write('Warning: cannot find names in les file:\n')
        sys.stderr.write(unsaved.keys())
        sys.stderr.write('\n')

if __name__ == '__main__':
    'smoke test'
    data = load_les('result.les')
    save_les('result.les', data)
