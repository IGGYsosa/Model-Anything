"""Minimal binary PLY reading/writing for the splat pipeline (numpy only).

read_vertices() reads the first element of a binary_little_endian PLY whose
properties are all scalars: the app's LiDAR mesh (mesh/room-N.ply: x y z nx ny nz
coverage red green blue inroom) and 3DGS splat files both qualify.
"""
import numpy as np

TYPES = {'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1', 'short': '<i2', 'ushort': '<u2',
         'int16': '<i2', 'uint16': '<u2', 'int': '<i4', 'uint': '<u4', 'int32': '<i4', 'uint32': '<u4',
         'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8'}


def read_header(f):
    if f.readline().strip() != b'ply':
        raise ValueError('not a PLY file')
    fmt, elements = None, []
    while True:
        line = f.readline()
        if not line:
            raise ValueError('PLY header has no end_header')
        words = line.decode('ascii', 'replace').split()
        if not words or words[0] == 'comment':
            continue
        if words[0] == 'format':
            fmt = words[1]
        elif words[0] == 'element':
            elements.append({'name': words[1], 'count': int(words[2]), 'props': []})
        elif words[0] == 'property':
            elements[-1]['props'].append(words[1:])
        elif words[0] == 'end_header':
            return fmt, elements


def read_vertices(path):
    """Structured array of the first element (normally `vertex`)."""
    with open(path, 'rb') as f:
        fmt, elements = read_header(f)
        if fmt != 'binary_little_endian':
            raise ValueError(f'unsupported PLY format {fmt}')
        first = elements[0]
        if any(p[0] == 'list' for p in first['props']):
            raise ValueError(f'element {first["name"]} has list properties')
        dtype = np.dtype([(p[1], TYPES[p[0]]) for p in first['props']])
        data = np.frombuffer(f.read(dtype.itemsize * first['count']), dtype=dtype)
        if len(data) != first['count']:
            raise ValueError('PLY file is truncated')
        return data


def write_points(path, xyz, rgb):
    """Binary point cloud (x y z red green blue), readable by most splat trainers."""
    n = len(xyz)
    arr = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())
