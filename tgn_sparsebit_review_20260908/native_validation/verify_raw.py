"""Independent NumPy replay of captured native inputs and outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
records = []
for path in sorted(args.directory.glob('case_*.bin'), key=lambda p: int(p.stem.split('_')[1])):
    data = path.read_bytes()
    magic, tiles, k, words, dim, mode = struct.unpack_from('<6I', data)
    assert magic == 0x4c41594f
    offset = 24

    def take(dtype, shape):
        global offset
        count = int(np.prod(shape))
        value = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += value.nbytes
        return value

    bits = take('<u4', (tiles, 48, words))
    weights = take('<f4', (tiles, 16, 32))
    x = take('<f2', (tiles, 32, dim))
    q = take('u1', (tiles, 16, 32))
    y = take('<f4', (tiles, 16, dim))
    assert offset == len(data)
    reference_q = np.any(bits[:, :16, None, :] & bits[:, None, 16:, :], axis=-1)
    c = reference_q.astype(np.float64) * weights.astype(np.float64)
    reference_y = c @ x.astype(np.float64)
    qbad = int(np.count_nonzero(q != reference_q))
    error = np.abs(y.astype(np.float64) - reference_y)
    bad = int(np.count_nonzero(~np.isfinite(y) | (error > 1e-5)))
    # Padding beyond logical K must be zero; changing K must not silently
    # introduce contributions from padding bits.
    padding_bad = 0
    for bit in range(k, words * 32):
        padding_bad += int(np.count_nonzero((bits[..., bit // 32] >> (bit % 32)) & 1))
    records.append(dict(file=path.name, tiles=tiles, k=k, dim=dim, mode=mode,
                        q_entries=q.size, y_entries=y.size, q_failures=qbad,
                        y_failures=bad, padding_failures=padding_bad,
                        max_abs_error=float(np.max(error)),
                        sha256=hashlib.sha256(data).hexdigest()))
result = dict(scope='Independent CPU replay of native Q and Y; no timing or TNCN claim',
              records=records, files=len(records),
              q_entries=sum(r['q_entries'] for r in records),
              y_entries=sum(r['y_entries'] for r in records),
              all_pass=bool(records) and all(not (r['q_failures'] or r['y_failures'] or r['padding_failures']) for r in records),
              max_abs_error=max((r['max_abs_error'] for r in records), default=None))
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))
raise SystemExit(0 if result['all_pass'] else 1)
