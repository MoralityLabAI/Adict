"""BF-01: blind Fourier-basis discovery on frozen anti-collapse encoders (CPU only).

The discovery step sees complex phase features and opaque index pairs (x, T x)
for the generators T(1,0), T(0,1) and their composite. It receives no operand
identities, no labels and no target frequencies. It estimates each feature's
multiplier under each generator, rounds the angle to a p-th root of unity and
reports the frequency pair modulo p. A group-aware decoder then solves for the
exponents that produce the shear character and fits one output phase.

Run from addict_probe_lab/addict_probe_lab:
    python -m research.accessible_causal_information.blind_fourier --out /tmp/bf01
Encoders are re-derived deterministically from the anti-collapse manifest
(no checkpoints were saved by AC-01) and checked against its committed summary.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from .anti_collapse import (components, shift_indices, spectrum_diagnostic,
                            train_only_character_basis_decode)
from .pilot import Encoder, ridge_scores, states

HERE = Path(__file__).resolve().parent
TARGET = (2, 1)  # shear 2a+b


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def complex_features(z: torch.Tensor) -> torch.Tensor:
    """Pair coordinates (0,1) and (2,3) into complex features [N,2]."""
    z = z.double()
    return torch.stack((torch.complex(z[:, 0], z[:, 1]),
                        torch.complex(z[:, 2], z[:, 3])), 1)


def shift_pairs(p: int, pool: torch.Tensor | None, n_shift: int | None) -> dict:
    """Opaque index pairs (i, j) with j = T i. Operand values never leave here.

    pool restricts both endpoints to the given state indices; n_shift keeps the
    first n pairs per generator in index order.
    """
    ia, ib = shift_indices(p)
    n = p * p
    base = torch.arange(n)
    out = {'a': (base, ia), 'b': (base, ib), 'ab': (base, ib[ia])}
    if pool is not None:
        inpool = torch.zeros(n, dtype=torch.bool)
        inpool[pool] = True
        for key, (i, j) in list(out.items()):
            keep = inpool[i] & inpool[j]
            out[key] = (i[keep], j[keep])
    if n_shift is not None:
        out = {k: (i[:n_shift], j[:n_shift]) for k, (i, j) in out.items()}
    return out


def discover_characters(zc: torch.Tensor, shifts: dict, p: int,
                        estimator: str = 'ls') -> list[dict]:
    """Blind per-feature frequency recovery from shift responses.

    Inputs: complex activations and index pairs only. Output per feature:
    frequency (k, l) mod p, angle-rounding error in steps, and group-law
    residuals against the exact quantized action.
    """
    if estimator not in ('ls', 'angle'):
        raise ValueError('unknown estimator')
    out = []
    for f in range(zc.shape[1]):
        col = zc[:, f]
        lam = {}
        for g in ('a', 'b'):
            i, j = shifts[g]
            x, y = col[i], col[j]
            if len(x) == 0:
                raise ValueError('no shift pairs')
            if estimator == 'ls':
                m = (x.conj() * y).sum() / x.abs().square().sum().clamp_min(1e-20)
            else:
                u = x.conj() * y
                m = (u / u.abs().clamp_min(1e-20)).mean()
            lam[g] = m
        step, freq, err = {}, {}, {}
        for g in ('a', 'b'):
            step[g] = float(torch.angle(lam[g])) * p / (2 * math.pi)
            freq[g] = int(round(step[g])) % p
            err[g] = abs(step[g] - round(step[g]))
        omega = lambda k: complex(math.cos(2 * math.pi * k / p), math.sin(2 * math.pi * k / p))
        res = {}
        for g, k in (('a', freq['a']), ('b', freq['b']), ('ab', freq['a'] + freq['b'])):
            i, j = shifts[g]
            x, y = col[i], col[j]
            res[g] = float((y - omega(k) * x).abs().square().mean() / x.abs().square().mean().clamp_min(1e-20))
        out.append({'k': freq['a'], 'l': freq['b'],
                    'multiplier_modulus': [float(lam['a'].abs()), float(lam['b'].abs())],
                    'angle_round_error_steps': [err['a'], err['b']],
                    'residual_a': res['a'], 'residual_b': res['b'],
                    'residual_composite': res['ab'],
                    'group_law_residual': max(res.values())})
    return out


def solve_exponents(freqs: list[tuple[int, int]], target: tuple[int, int], p: int):
    (k1, l1), (k2, l2) = freqs
    det = (k1 * l2 - k2 * l1) % p
    if det == 0:
        return None
    inv = pow(det, -1, p)
    return ((target[0] * l2 - target[1] * k2) * inv) % p, ((target[1] * k1 - target[0] * l1) * inv) % p


def group_decode(zc: torch.Tensor, freqs, labels: torch.Tensor, fit: torch.Tensor,
                 test: torch.Tensor, p: int, normalize: str, lift: str) -> dict:
    exps = solve_exponents(freqs, TARGET, p)
    if exps is None:
        return {'accuracy': None, 'recoverable': False, 'reason': 'frequencies dependent mod p'}
    if lift == 'signed':
        exps = tuple(c if c <= p // 2 else c - p for c in exps)
    elif lift != 'unsigned':
        raise ValueError('unknown lift')
    blocks = [zc[:, f] / zc[:, f].abs().clamp_min(1e-12) if normalize == 'unit' else zc[:, f]
              for f in (0, 1)]
    prod = blocks[0] ** exps[0] * blocks[1] ** exps[1]
    prod = prod / prod.abs().clamp_min(1e-12)
    truth = torch.exp((2j * math.pi / p) * labels.double())
    off = (truth[fit] * prod[fit].conj()).mean()
    phase = off / off.abs() if off.abs() > 1e-12 else torch.tensor(1 + 0j, dtype=torch.complex128)
    centers = torch.exp((2j * math.pi / p) * torch.arange(p, dtype=torch.double))
    pred = ((prod[test, None] * phase) * centers[None, :].conj()).real.argmax(1)
    return {'accuracy': float((pred == labels[test]).double().mean()), 'recoverable': True,
            'exponents': [int(c) for c in exps], 'offset_coherence': float(off.abs())}


def blind_method(z: torch.Tensor, labels: torch.Tensor, fit: torch.Tensor, test: torch.Tensor,
                 pool: torch.Tensor, cfg: dict, p: int, label_count: int | None = None) -> dict:
    """Discovery on `pool` activations, then decoding. Timing covers the whole fit."""
    t0, c0 = time.perf_counter(), time.process_time()
    zc = complex_features(z[:, :4])
    shifts = shift_pairs(p, pool, cfg['n_shift'])
    found = discover_characters(zc, shifts, p, cfg['estimator'])
    freqs = [(d['k'], d['l']) for d in found]
    use = fit if label_count is None else fit[:label_count]
    dec = group_decode(zc, freqs, labels, use, test, p, cfg['normalize'], cfg['lift'])
    return {'config': cfg, 'discovered': found, 'frequencies': [list(f) for f in freqs],
            'group_law_residual': max(d['group_law_residual'] for d in found),
            'decoder': dec, 'accuracy': dec['accuracy'],
            'fit_wall_seconds': time.perf_counter() - t0,
            'fit_cpu_seconds': time.process_time() - c0,
            'shift_pairs_per_generator': {k: len(v[0]) for k, v in shifts.items()},
            'labels_used': len(use), 'fitted_scalars': 2, 'discrete_integers': 4}


def generic_features(z: torch.Tensor, kind: str) -> torch.Tensor:
    z = z.double()
    base = z[:, :4] if kind.endswith('4') else z
    if kind.startswith('bilinear'):
        i, j = torch.triu_indices(base.shape[1], base.shape[1])
        return torch.cat((base, base[:, i] * base[:, j]), 1)
    return base


def generic_decoder(z, labels, fit, test, p, kind, penalty, label_count=None) -> dict:
    t0, c0 = time.perf_counter(), time.process_time()
    x = generic_features(z, kind)
    use = fit if label_count is None else fit[:label_count]
    acc = ridge_scores(x, labels, use, test, p, penalty)['accuracy']
    d = x.shape[1]
    return {'kind': kind, 'penalty': penalty, 'accuracy': acc, 'parameters': (d + 1) * p,
            'labels_used': len(use), 'fit_wall_seconds': time.perf_counter() - t0,
            'fit_cpu_seconds': time.process_time() - c0,
            'fit_multiply_adds': int(len(use) * (d + 1) ** 2 + (d + 1) ** 3 + len(use) * (d + 1) * p)}


def random_rotation_controls(z, labels, fit, test, pool, cfg, p, draws, seed) -> dict:
    g = torch.Generator().manual_seed(seed)
    accs, res, dets = [], [], 0
    for _ in range(draws):
        q, r = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=torch.double))
        q = q * torch.sign(torch.diagonal(r))
        zr = torch.cat((z[:, :4].double() @ q, z[:, 4:].double()), 1)
        out = blind_method(zr, labels, fit, test, pool, cfg, p)
        res.append(out['group_law_residual'])
        if out['accuracy'] is None:
            dets += 1
        else:
            accs.append(out['accuracy'])
    srt = sorted(accs)
    return {'draws': draws, 'dependent_frequency_failures': dets,
            'accuracy_mean': sum(accs) / max(len(accs), 1),
            'accuracy_max': max(accs) if accs else None,
            'accuracy_median': srt[len(srt) // 2] if srt else None,
            'group_law_residual_min': min(res), 'group_law_residual_median': sorted(res)[len(res) // 2]}


def train_condition(initial, condition, m, pairs, labels, ia, ib):
    model = copy.deepcopy(initial)
    opt = torch.optim.AdamW(model.parameters(), lr=m['learning_rate'], weight_decay=m['weight_decay'])
    for _ in range(m['steps']):
        opt.zero_grad(set_to_none=True)
        logits, hidden, z = model(pairs)
        loss = F.cross_entropy(logits, labels['sum'])
        if condition == 'task_and_a':
            loss = loss + F.cross_entropy(hidden, labels['a'])
        if condition in ('equivariance', 'eq_norm', 'eq_norm_cov'):
            st = components(z, m['modulus'], ia, ib)
            loss = loss + m['equivariance_weight'] * st['equivariance']
            if condition in ('eq_norm', 'eq_norm_cov'):
                loss = loss + m['norm_weight'] * st['norm']
            if condition == 'eq_norm_cov':
                loss = loss + m['covariance_weight'] * st['covariance']
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        _, _, z = model(pairs)
        st = components(z, m['modulus'], ia, ib)
    return z.detach(), float(st['equivariance'])


def fft_modes(z, p):
    """Evaluation-only ground truth: dominant (k,l) per complex block."""
    out = []
    for blk in spectrum_diagnostic(z[:, :4], p):
        top = blk['strongest_modes'][0]
        out.append({'mode': [top['k'], top['l']], 'power': top['power_fraction']})
    return out


def committed_equivariance(m):
    rows = {}
    with open(HERE / 'anti_collapse_summary.csv') as fh:
        for r in csv.DictReader(fh):
            rows[(int(r['seed']), r['condition'])] = float(r['equivariance_loss'])
    return rows


def split_pairs(m, n):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(m['split_seed']))
    test = perm[:n // 2 + 1].sort().values            # 481, untouched until freeze
    pool = perm[n // 2 + 1:]                          # 480
    return test, pool.sort().values, pool[:len(pool) // 2].sort().values, pool[len(pool) // 2:].sort().values


def calibrate(z, labels, A_fit, A_val, cfgs, penalties, p):
    """Seed-303 selection using only the calibration pool (never the test set)."""
    blind_rows = []
    for cfg in cfgs:
        out = blind_method(z, labels['shear'], A_fit, A_val, A_fit, cfg, p)
        blind_rows.append({'config': cfg, 'val_accuracy': out['accuracy'] if out['accuracy'] is not None else -1.0,
                           'frequencies': out['frequencies'], 'group_law_residual': out['group_law_residual']})
    # Pre-declared order: best validation accuracy, then larger n_shift, then grid order.
    order = sorted(range(len(cfgs)), key=lambda i: (-blind_rows[i]['val_accuracy'],
                                                    -(cfgs[i]['n_shift'] or 10 ** 9), i))
    best = cfgs[order[0]]
    generic = {}
    for kind in ('affine4', 'bilinear4', 'affine32', 'bilinear32'):
        scores = [(generic_decoder(z, labels['shear'], A_fit, A_val, p, kind, pen)['accuracy'], -i, pen)
                  for i, pen in enumerate(penalties)]
        generic[kind] = max(scores)[2]
    return best, generic, blind_rows


def run(mpath: Path, out: Path) -> dict:
    m = json.loads(mpath.read_text())
    if m.get('schema') != 'adict-blind-fourier/v0.1' or m['modulus'] != 31:
        raise ValueError('unexpected protocol')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    p = m['modulus']
    sources = {n: digest(HERE / n) for n in ('blind_fourier.py', 'test_blind_fourier.py',
                                             'anti_collapse.py', 'pilot.py', 'anti_collapse_manifest.json')}
    sources[mpath.name] = digest(mpath)
    (out / 'start.json').write_text(json.dumps({
        'schema': m['schema'], 'source_sha256': sources, 'manifest': m,
        'ordering': 'local pre-run write, unsigned; written before any encoder is trained'}, indent=2) + '\n')
    base = json.loads((HERE / 'anti_collapse_manifest.json').read_text())
    pairs, labels = states(p)
    ia, ib = shift_indices(p)
    test, A, A_fit, A_val = split_pairs(m, len(pairs))
    committed = committed_equivariance(base)
    t_start = time.monotonic()

    def encoders(seed):
        torch.manual_seed(seed)
        initial = Encoder(base)
        res = {}
        for cond in m['conditions']:
            z, eq = train_condition(initial, cond, base, pairs, labels, ia, ib)
            res[cond] = {'z': z, 'equivariance_loss': eq,
                         'abs_diff_vs_committed': abs(eq - committed[(seed, cond)])}
        return res, sum(q.numel() for q in initial.parameters())

    # ---- calibration on seed 303 (pool A only), then freeze ----
    cal_seed = m['calibration_seed']
    enc_cal, n_enc_params = encoders(cal_seed)
    cfgs = [dict(zip(m['grid'].keys(), v)) for v in itertools.product(*m['grid'].values())]
    z_cal = enc_cal['eq_norm_cov']['z']
    best, generic_pen, cal_rows = calibrate(z_cal, labels, A_fit, A_val, cfgs, m['ridge_penalties'], p)
    frozen = {'blind_config': best, 'generic_penalties': generic_pen,
              'calibration_seed': cal_seed, 'calibration_condition': 'eq_norm_cov',
              'calibration_pool': 'A_fit/A_val (480 pairs), disjoint from the 481 test pairs',
              'success_criteria': m['success_criteria'], 'source_sha256': sources}
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
    frozen_sha = digest(out / 'frozen.json')

    def evaluate(seed, enc):
        rows = []
        for cond, rec in enc.items():
            z = rec['z']
            blind = blind_method(z, labels['shear'], A, test, A, best, p)
            row = {'seed': seed, 'condition': cond, 'role': 'calibration' if seed == cal_seed else 'held_out_seed',
                   'equivariance_loss': rec['equivariance_loss'],
                   'encoder_abs_diff_vs_committed': rec['abs_diff_vs_committed'],
                   'fft_ground_truth_eval_only': fft_modes(z, p), 'blind': blind}
            row['character_recovery'] = [
                list(f) == g['mode'] for f, g in zip(blind['frequencies'], row['fft_ground_truth_eval_only'])]
            row['generic'] = {k: generic_decoder(z, labels['shear'], A, test, p, k, generic_pen[k])
                              for k in ('affine4', 'bilinear4', 'affine32', 'bilinear32')}
            row['oracle_assisted_basis'] = train_only_character_basis_decode(
                z[:, :4], pairs, labels['shear'], A, test, p)
            if cond == 'eq_norm_cov':
                row['random_rotation_control'] = random_rotation_controls(
                    z, labels['shear'], A, test, A, best, p, m['rotation_draws'], m['control_seed'] + seed)
                row['label_efficiency'] = {
                    str(n): {'blind': blind_method(z, labels['shear'], A, test, A, best, p, n)['accuracy'],
                             **{k: generic_decoder(z, labels['shear'], A, test, p, k, generic_pen[k], n)['accuracy']
                                for k in ('affine4', 'bilinear4', 'affine32', 'bilinear32')}}
                    for n in m['label_counts']}
            rows.append(row)
        return rows

    results = evaluate(cal_seed, enc_cal)
    enc_held, _ = encoders(m['evaluation_seed'])
    results += evaluate(m['evaluation_seed'], enc_held)

    primary = next(r for r in results if r['seed'] == m['evaluation_seed'] and r['condition'] == 'eq_norm_cov')
    c = m['success_criteria']
    best_generic = max(v['accuracy'] for v in primary['generic'].values())
    verdict = {'characters_recovered_both_blocks': all(primary['character_recovery']),
               'group_law_residual': primary['blind']['group_law_residual'],
               'residual_ok': primary['blind']['group_law_residual'] <= c['max_group_law_residual'],
               'blind_accuracy': primary['blind']['accuracy'], 'best_generic_accuracy': best_generic,
               'margin': (primary['blind']['accuracy'] or 0.0) - best_generic,
               'margin_ok': (primary['blind']['accuracy'] or 0.0) - best_generic >= c['min_accuracy_margin']}
    verdict['success'] = (verdict['characters_recovered_both_blocks'] and verdict['residual_ok']
                          and verdict['margin_ok'])
    result = {'schema': m['schema'], 'manifest': m, 'source_sha256': sources, 'frozen_sha256': frozen_sha,
              'python': platform.python_version(), 'torch': torch.__version__,
              'encoder_parameters': n_enc_params, 'split': {'test': len(test), 'pool': len(A),
              'fit': len(A_fit), 'val': len(A_val)}, 'calibration_grid': cal_rows,
              'frozen': frozen, 'results': results, 'verdict': verdict,
              'elapsed_seconds': round(time.monotonic() - t_start, 3),
              'scope': 'CPU; encoders re-derived from AC-01 manifest; decoder split only; not zero-shot'}
    (out / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (out / 'receipt.json').write_text(json.dumps({
        'schema': 'blind-fourier-receipt-v0.1', 'start_sha256': digest(out / 'start.json'),
        'frozen_sha256': frozen_sha, 'results_sha256': digest(out / 'results.json'),
        'sources': sources, 'n_records': len(results), 'success': verdict['success']},
        indent=2, sort_keys=True) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=HERE / 'blind_fourier_manifest.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    r = run(args.manifest, args.out)
    print(json.dumps(r['verdict'], indent=2))


if __name__ == '__main__':
    main()
