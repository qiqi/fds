# 2016 CTR SP Qiqi Wang
import sys
import struct
import ctypes
from numpy import *

__all__ = ['NO_CHANGE', 'save_les', 'load_les']

NO_CHANGE = None

UGP_IO_MAGIC_NUMBER = 123581321
UGP_IO_EOF = 51
UGP_IO_I0 = 11
UGP_IO_D0 = 12
UGP_IO_NO_D1 = 32
UGP_IO_NO_D2 = 33
UGP_IO_FAZONE_NO_D2 = 114

UGP_IO_HEADER_NAME_LEN = 52

def read_header(fp):
    '''
    Header structure according to core/CTIDefs.hpp
    returns name (str)
            iid (integer, record type according to core/CTIDefs.hpp
            skip (integer, distance in bytes to next record)
            idata (integer array of size 16, meaning depends on record type)
            rdata (double array of size 16, meaning depends on record type)
    '''
    name = ctypes.create_string_buffer(fp.read(UGP_IO_HEADER_NAME_LEN)).value
    iid, skip, zero = struct.unpack('iii', fp.read(12))
    if zero != 0:
        raise IOError('Byte 4-7 in CTIHeader "skip" is expected to be 0')
    idata = frombuffer(fp.read(16*4), 'i')
    rdata = frombuffer(fp.read(16*8), 'd')
    return name.decode(), iid, skip, idata, rdata

def same_skip_in_header(fp, skip):
    loc = fp.tell()
    _, _, old_skip, _, _ = read_header(fp)
    fp.seek(loc)
    return old_skip == skip

def write_header(fp, name, iid, skip, idata, rdata):
    '''
    Header structure according to core/CTIDefs.hpp
    returns name (str)
            iid (integer, record type according to core/CTIDefs.hpp
            skip (integer, distance in bytes to next record)
            idata (integer array of size 16, meaning depends on record type)
            rdata (double array of size 16, meaning depends on record type)
    '''
    if not same_skip_in_header(fp, skip):
        raise ValueError('Cannot overwrite CTIHeader with a different "skip"')
    name = fromstring(name, dtype='b')
    fp.write(name.tobytes())
    fp.write(zeros(UGP_IO_HEADER_NAME_LEN - name.size, dtype='b').tobytes())
    fp.write(struct.pack('iii', iid, skip, 0))
    idata = ascontiguousarray(idata, dtype='i')
    rdata = ascontiguousarray(rdata, dtype='d')
    if not idata.shape == rdata.shape == (16,):
        raise ValueError('CTIHeader idata and rdata both must be shape (16,)')
    fp.write(idata.tobytes())
    fp.write(rdata.tobytes())

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
    if magic_number != UGP_IO_MAGIC_NUMBER:
        raise IOError('Magic number error in read_les')
    if verbose:
        print('loading {0}, io_version={1}'.format(fname, io_version))
    data, iid, offset = {}, 0, 8
    while iid != UGP_IO_EOF:
        fp.seek(offset)
        name, iid, skip, idata, rdata = read_header(fp)
        offset += skip
        if iid == UGP_IO_I0:
            data[name] = idata[0]
            if verbose: print(name, data[name])
        elif iid == UGP_IO_D0:
            data[name] = rdata[0]
            if verbose: print(name, data[name])
        elif iid == UGP_IO_NO_D1:
            size = idata[0]
            if verbose: print(name, size)
            data[name] = frombuffer(fp.read(size*8), 'd')
        elif iid == UGP_IO_NO_D2:
            size, dim = idata[:2]
            if verbose: print(name, size, dim)
            if dim != 3:
                raise IOError('dim!=3 in UGP_IO_NO_D2, load_les')
            data[name] = frombuffer(fp.read(size*3*8), 'd').reshape([size, 3])
        elif iid == UGP_IO_FAZONE_NO_D2:
            size, dim = idata[0], 3
            if verbose: print(name, size)
            data[name] = frombuffer(fp.read(size*3*8), 'd').reshape([size, 3])
        else:
            if verbose: print('skipping ', name, iid)
    sys.stdout.flush()
    return data

def save_data_field(fp, size, data, name):
    '''
    Checks the size of data[name], saves it to fp, then delete the data entry.
    if data[name] is NO_CHANGE, the do not change this record in file.
    Raises ValueError if things are not right
    '''
    if name not in data:
        raise ValueError(
                'Saving field {0} not in data with fields {1}'.format(
                    name, data.keys()))
    if data[name] is not NO_CHANGE:
        if size != data[name].size:
            raise ValueError(
                'Saving field {0} size mismatch {1}!={2}'.format(
                    name, size, data[name].size))
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
    if magic_number != UGP_IO_MAGIC_NUMBER:
        raise IOError('Magic number error in save_les')
    if verbose:
        print('saving to {0}, io_version={1}'.format(fname, io_version))
    iid, offset = 0, 8
    unsaved = dict(data)
    while iid != UGP_IO_EOF:
        fp.seek(offset)
        name, iid, skip, idata, rdata = read_header(fp)
        offset += skip
        if name not in unsaved:
            continue
        elif iid == UGP_IO_I0:
            if verbose: print(name, data[name])
            fp.seek(offset - skip)
            idata = idata.copy()
            idata[0] = unsaved[name]
            del unsaved[name]
            write_header(fp, name, iid, skip, idata, rdata)
        elif iid == UGP_IO_D0:
            if verbose: print(name, data[name])
            data[name] = rdata[0]
            fp.seek(offset - skip)
            rdata = rdata.copy()
            rdata[0] = unsaved[name]
            del unsaved[name]
            write_header(fp, name, iid, skip, idata, rdata)
        elif iid == UGP_IO_NO_D1:
            size = idata[0]
            if verbose: print(name, size)
            save_data_field(fp, size, unsaved, name)
        elif iid == UGP_IO_NO_D2:
            size, dim = idata[:2]
            if verbose: print(name, size, dim)
            if dim != 3:
                raise IOError('dim!=3 in UGP_IO_NO_D2, save_les')
            save_data_field(fp, size*3, unsaved, name)
        elif iid == UGP_IO_FAZONE_NO_D2:
            size, dim = idata[0], 3
            if verbose: print(name, size)
            save_data_field(fp, size*3, unsaved, name)
    sys.stdout.flush()
    if len(unsaved):
        sys.stderr.write('Warning: cannot find names in les file:\n')
        for key in unsaved:
            sys.stderr.write('\t' + key + '\n')

# if __name__ == '__main__':
#     'smoke test'
#     data = load_les('result.les')
#     data['STEP'] = 0
#     save_les('result.les', data)
