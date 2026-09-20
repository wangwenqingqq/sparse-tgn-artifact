"""CPU-only fragment ownership screen; no CUDA or performance claims.

Reference ABI: NVIDIA PTX ISA, mma.m16n8k128 b1 (Figure 99),
and sparse mma.m16n8k32 f16/bf16 (Figures 120 and 122).
Only one 16x32 forward coefficient tile is modeled. The matrix producer,
input gathers, precision policy, backward ownership, and native timings
remain independent obligations. Uses standard Python only.
"""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import struct


ROOT = Path(__file__).resolve().parent


def column(panel, slot, permuted):
    if not permuted:
        return 8 * panel + slot
    # Each of the four lanes now receives all four candidates for its
    # own Sparse MMA quartet across two successive BMMA output panels.
    return 16 * (panel // 2) + 4 * (slot // 2) + 2 * (panel % 2) + slot % 2


def producer_map(permuted):
    result = {}
    for panel in range(4):
        for lane in range(32):
            for register in range(4):
                row = lane // 4 + 8 * (register // 2)
                slot = 2 * (lane % 4) + register % 2
                col = column(panel, slot, permuted)
                assert (row, col) not in result
                result[row, col] = (lane, panel, register)
    assert len(result) == 512
    return result


def consumer_lane(row, col):
    return 4 * (row % 8) + (col % 16) // 4


def consumer_register(row, col):
    # One packed half2 register for each quartet of four logical entries.
    return 2 * (col // 16) + row // 8


def metadata_owner(row, col):
    # sparsity selector i=0 in PTX Figure 122.
    return 4 * (row % 8) + col // 16, 16 * (row // 8) + col % 16


def pack_tile(matrix, permuted):
    mapping = producer_map(permuted)
    produced = [[[None] * 4 for _ in range(4)] for _ in range(32)]
    for (row, col), (lane, panel, reg) in mapping.items():
        produced[lane][panel][reg] = matrix[row][col]
    packed = [[[[0, 0] for _ in range(4)] for _ in range(32)] for _ in range(2)]
    metadata = [[0] * 32 for _ in range(2)]
    value_remote = 0
    metadata_remote = 0
    for row in range(16):
        for start in range(0, 32, 4):
            nz = [j for j in range(4) if matrix[row][start + j] != 0]
            for plane in range(2):
                positions = nz[2 * plane:2 * plane + 2]
                positions += [j for j in range(4) if j not in positions][:2 - len(positions)]
                positions.sort()
                lane = consumer_lane(row, start)
                reg = consumer_register(row, start)
                for half, j in enumerate(positions):
                    src, panel, outreg = mapping[row, start + j]
                    value = produced[src][panel][outreg] if j in nz[2 * plane:2 * plane + 2] else 0
                    if value:
                        value_remote += src != lane
                    # Values in our synthetic fixtures are half-exact integers.
                    assert struct.unpack('e', struct.pack('e', value))[0] == value
                    packed[plane][lane][reg][half] = value
                owner, shift = metadata_owner(row, start)
                nibble = positions[0] | (positions[1] << 2)
                metadata[plane][owner] |= nibble << shift
                metadata_remote += owner != lane
    decoded = [[0] * 32 for _ in range(16)]
    for plane in range(2):
        for lane in range(32):
            g, t = divmod(lane, 4)
            for reg in range(4):
                row = g + (reg % 2) * 8
                start = 4 * t + (reg // 2) * 16
                # Independent read of the metadata ABI, from logical row/col.
                owner = 4 * g + reg // 2
                shift = 4 * t + 16 * (reg % 2)
                nibble = (metadata[plane][owner] >> shift) & 15
                positions = (nibble & 3, nibble >> 2)
                assert positions[0] < positions[1]
                for half, j in enumerate(positions):
                    decoded[row][start + j] += packed[plane][lane][reg][half]
    assert decoded == matrix
    return decoded, value_remote, metadata_remote


def main():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    natural = producer_map(False)
    permuted = producer_map(True)
    rows = []
    for row in range(16):
        for col in range(32):
            rows.append(dict(row=row, col=col, consumer=consumer_lane(row, col),
                             natural=natural[row, col][0], permuted=permuted[row, col][0]))
    with (out / 'owners.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert sum(r['natural'] != r['consumer'] for r in rows) == 384
    assert sum(r['permuted'] != r['consumer'] for r in rows) == 0
    # Exhaust all local support masks at every quartet; linear composition
    # extends this local pack/decode check to arbitrary 16x32 support patterns.
    support_cases = 0
    for row in range(16):
        for start in range(0, 32, 4):
            for mask in range(16):
                matrix = [[0] * 32 for _ in range(16)]
                for j in range(4):
                    matrix[row][start + j] = ((-1) ** j) * (j + 1) if mask & (1 << j) else 0
                _, remote, _ = pack_tile(matrix, True)
                assert remote == 0
                support_cases += 1
    # Compare two panel orders on actual integer binary dot products with
    # partial K chunks. Both orders must give exactly the same dense result.
    rng = random.Random(20260908)
    dot_cases = 0
    for k in [0, 1, 31, 32, 127, 128, 129, 255, 256, 257]:
        left = [[rng.randrange(2) for _ in range(k)] for _ in range(16)]
        right = [[rng.randrange(2) for _ in range(k)] for _ in range(32)]
        reference = [[sum(a * b for a, b in zip(x, y)) for y in right] for x in left]
        for ordering in [False, True]:
            result = [[0] * 32 for _ in range(16)]
            for panel in range(4):
                for row in range(16):
                    for slot in range(8):
                        col = column(panel, slot, ordering)
                        # Model K128 accumulation, not hardware execution.
                        result[row][col] = sum(
                            sum(left[row][z] & right[col][z] for z in range(begin, min(begin + 128, k)))
                            for begin in range(0, k, 128))
            assert result == reference
            dot_cases += 1
    # Signed integer reconstruction and forward/transpose application.
    random_cases = 0
    for _ in range(64):
        matrix = [[rng.randint(-16, 16) if rng.random() < 0.45 else 0 for _ in range(32)] for _ in range(16)]
        decoded, moved, meta_moved = pack_tile(matrix, True)
        assert moved == 0 and meta_moved == 192  # 96 per plane, fixed padded encoding.
        x, grad = [rng.randint(-8, 8) for _ in range(32)], [rng.randint(-8, 8) for _ in range(16)]
        forward = [sum(matrix[r][c] * x[c] for c in range(32)) for r in range(16)]
        backward = [sum(matrix[r][c] * grad[r] for r in range(16)) for c in range(32)]
        assert forward == [sum(decoded[r][c] * x[c] for c in range(32)) for r in range(16)]
        assert backward == [sum(decoded[r][c] * grad[r] for r in range(16)) for c in range(32)]
        random_cases += 1
    # Dense 2-of-4 patterns: the natural layout's local maximum is 128/256.
    min_remote = max_remote = 0
    for row in range(16):
        for start in range(0, 32, 4):
            counts = [sum(natural[row, start + j][0] != consumer_lane(row, start) for j in pair)
                      for pair in itertools.combinations(range(4), 2)]
            min_remote += min(counts)
            max_remote += max(counts)
    assert (min_remote, max_remote) == (128, 256)
    # A separate geometry check for sparse TF32 m16n8k16 (1:2).
    # Its candidate pair already matches each BMMA lane's output pair.
    tf32_coordinates = []
    for row in range(16):
        for col in range(16):
            tf32_coordinates.append(
                natural[row, col][0] == 4 * (row % 8) + (col % 8) // 2)
    assert all(tf32_coordinates)
    result = dict(
        scope='CPU coordinate/encoding screen only; no native kernel, training, or timing',
        all_pass=True,
        panel_order=[[column(p, j, True) for j in range(8)] for p in range(4)],
        coordinate_slots=512, natural_remote_slots=384, permuted_remote_slots=0,
        natural_exactly_2_of_4_remote_range=[min_remote, max_remote],
        exhaustive_local_support_cases=support_cases,
        binary_dot_panel_cases=dot_cases,
        signed_tile_forward_transpose_cases=random_cases,
        metadata_remote_nibbles_per_plane=96,
        metadata_total_nibbles_per_plane=128,
        tf32_m16n8k16_candidate_slots=256,
        tf32_m16n8k16_natural_remote_slots=0,
        caveats=[
            'Coordinate slots and nibbles are not warp instruction counts or HBM traffic.',
            'Coefficient values are assumed to be produced in the BMMA accumulator owners.',
            'BMMA gives Q witnesses/counts, not full weighted TNCN coefficients.',
            'Input packing, weighted S delivery, metadata gathers and live registers must be paid.',
            'Two planes are exact in this forward row format but do not guarantee transpose 2:4.',
            'Transpose arithmetic check is algebraic, not a backward native layout.',
            'Full packing modeled only for FP16/BF16 sparse m16n8k32; TF32 check is coordinates only.',
            'A separate implementation may use the identical column schedule.',
        ],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / 'screen.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
