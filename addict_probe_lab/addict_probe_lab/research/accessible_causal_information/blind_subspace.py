"""BF-02: coordinate-free discovery of the shift representation in the 32-d code.

BF-01 assumed which coordinates pair into complex features. Here the discovery
sees only the real code matrix and opaque shift index pairs (x, Tx). It fits a
linear shift operator per generator, diagonalizes a generic combination, keeps
eigen-features that are invariant (small residual, non-trivial frequency, enough
variance), and reads each feature's frequency pair modulo p from its multipliers.
No coordinate pairing, operand values, labels or target frequencies are used.

Run from addict_probe_lab/addict_probe_lab:
    python -m research.accessible_causal_information.blind_subspace --out /tmp/bf02
Exact encoder checkpoints and activation hashes are written to the checkpoint dir.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from .anti_collapse import components, shift_indices
from .blind_fourier import (TARGET, complex_features, discover_characters, digest, fft_modes,
                            group_decode, shift_pairs, solve_exponents, split_pairs)
from .pilot import Encoder, ridge_scores, states

HERE = Path(__file__).resolve().parent


def canon_p(freq, p):
    k, l = freq
    return min((k % p, l % p), ((-k) % p, (-l) % p))


# ---------------------------------------------------------------- discovery

def fit_operators(z: torch.Tensor, shifts: dict, pool: torch.Tensor, ridge: float):
    """Ridge-fit B_g with z(Tx) ~ z(x) B_g on centered codes; returns centered codes and fit stats."""
    zc = z.double()
    zc = zc - zc[pool].mean(0)
    ops, rel = {}, {}
    eye = torch.eye(zc.shape[1], dtype=torch.double)
    for g in ('a', 'b'):
        i, j = shifts[g]
        x, y = zc[i], zc[j]
        gram = x.T @ x
        reg = ridge * gram.diagonal().mean() * eye
        ops[g] = torch.linalg.solve(gram + reg, x.T @ y)
        rel[g] = float((y - x @ ops[g]).square().sum() / y.square().sum().clamp_min(1e-20))
    return zc, ops, rel


def discover_subspace(z: torch.Tensor, pool: torch.Tensor, cfg: dict, p: int) -> dict:
    return discover_from_shifts(z, shift_pairs(p, pool, cfg['n_shift']), pool, cfg, p)


def discover_from_shifts(z: torch.Tensor, shifts: dict, pool: torch.Tensor, cfg: dict, p: int) -> dict:
    t0 = time.perf_counter()
    zc, ops, rel = fit_operators(z, shifts, pool, cfg['ridge'])
    combo = ops['a'] + cfg['mix'] * ops['b']
    evals, evecs = torch.linalg.eig(combo)
    total = zc.square().sum(1).mean()
    zcc = zc.to(torch.complex128)
    cands = []
    for idx in range(len(evals)):
        if evals[idx].imag <= 1e-9:
            continue
        w = evecs[:, idx]
        w = w / torch.linalg.vector_norm(w)
        f = zcc @ w
        d = discover_characters(f[:, None], shifts, p, 'ls')[0]
        cands.append({'w': w, 'feature': f, 'freq': (d['k'], d['l']),
                      'residual': d['group_law_residual'],
                      'variance_share': float(f.abs().square().mean() / total),
                      'eigenvalue_modulus': float(evals[idx].abs())})
    keep = [c for c in cands if c['residual'] <= cfg['tau'] and c['variance_share'] >= cfg['nu']
            and c['freq'] != (0, 0)]
    best = {}
    for c in sorted(keep, key=lambda c: c['residual']):
        best.setdefault(canon_p(c['freq'], p), c)       # one feature per frequency up to conjugation
    feats = sorted(best.values(), key=lambda c: c['residual'])
    comm = ops['a'] @ ops['b'] - ops['b'] @ ops['a']
    w_all = torch.stack([c['w'] for c in feats], 1) if feats else None
    return {'features': feats, 'n_candidates': len(cands), 'ops': ops, 'operator_rel_error': rel,
            'commutator_rel': float(comm.norm() / (ops['a'].norm() * ops['b'].norm()).clamp_min(1e-20)),
            'commutator_on_subspace': (float((comm.to(torch.complex128) @ w_all).norm() / w_all.norm())
                                       if w_all is not None else None),
            'shifts': shifts, 'zc': zc, 'wall_seconds': time.perf_counter() - t0}


def pick_pair(feats: list[dict], p: int):
    best = None
    for fa, fb in itertools.combinations(feats, 2):
        c = solve_exponents([fa['freq'], fb['freq']], TARGET, p)
        if c is None:
            continue
        cost = sum(abs(x if x <= p // 2 else x - p) for x in c)
        key = (cost, fa['residual'] + fb['residual'])
        if best is None or key < best[0]:
            best = (key, fa, fb)
    return best


def subspace_vs_truth(feats: list[dict], dim: int = 32) -> dict:
    """Evaluation-only: overlap of the discovered subspace with coordinates 0-3."""
    if not feats:
        return {'overlap_with_phase_coords': 0.0, 'leakage_outside_phase_coords': None, 'rank': 0}
    cols = []
    for c in feats:
        cols += [c['w'].real, c['w'].imag]
    q, _ = torch.linalg.qr(torch.stack(cols, 1))
    e = torch.eye(dim, dtype=torch.double)[:, :4]
    inside = float((e.T @ q).square().sum())
    return {'overlap_with_phase_coords': inside / 4.0,
            'leakage_outside_phase_coords': 1.0 - inside / q.shape[1], 'rank': int(q.shape[1])}


def blind_subspace_method(z, labels, fit, test, pool, cfg, p) -> dict:
    t0 = time.perf_counter()
    d = discover_subspace(z, pool, cfg, p)
    feats = d['features']
    out = {'config': cfg, 'n_candidates': d['n_candidates'], 'n_features': len(feats),
           'frequencies': [list(c['freq']) for c in feats],
           'feature_residuals': [c['residual'] for c in feats],
           'variance_shares': [c['variance_share'] for c in feats],
           'operator_rel_error': d['operator_rel_error'], 'commutator_rel': d['commutator_rel'],
           'commutator_on_subspace': d['commutator_on_subspace'],
           'shift_pairs_per_generator': {k: len(v[0]) for k, v in d['shifts'].items()},
           'subspace': subspace_vs_truth(feats, z.shape[1]),
           'group_law_residual': max((c['residual'] for c in feats), default=None)}
    pair = pick_pair(feats, p)
    if pair is None:
        out.update({'accuracy': None, 'decoder': {'recoverable': False, 'reason': 'no independent pair'}})
    else:
        _, fa, fb = pair
        zc2 = torch.stack((fa['feature'], fb['feature']), 1)
        dec = group_decode(zc2, [fa['freq'], fb['freq']], labels, fit, test, p, 'unit', 'signed')
        out.update({'accuracy': dec['accuracy'], 'decoder': dec,
                    'decoder_frequencies': [list(fa['freq']), list(fb['freq'])]})
    out['fit_wall_seconds'] = time.perf_counter() - t0
    out['labels_used'] = len(fit)
    return out


# ---------------------------------------------------------------- baselines

def exponent_search_decoder(z, labels, fit, test, p, span=3) -> dict:
    """Known complex pairing (coords 0-1, 2-3); frequency-agnostic search over monomials f1^a f2^b."""
    zc = complex_features(z[:, :4])
    blocks = [zc[:, f] / zc[:, f].abs().clamp_min(1e-12) for f in (0, 1)]
    truth = torch.exp((2j * math.pi / p) * labels.double())
    centers = torch.exp((2j * math.pi / p) * torch.arange(p, dtype=torch.double))
    best = None
    t0 = time.perf_counter()
    for a, b in itertools.product(range(-span, span + 1), repeat=2):
        prod = blocks[0] ** a * blocks[1] ** b
        off = (truth[fit] * prod[fit].conj()).mean()
        ph = off / off.abs() if off.abs() > 1e-12 else torch.tensor(1 + 0j, dtype=torch.complex128)
        pred = ((prod * ph)[:, None] * centers[None, :].conj()).real.argmax(1)
        fit_acc = float((pred[fit] == labels[fit]).double().mean())
        key = (fit_acc, -(abs(a) + abs(b)))
        if best is None or key > best[0]:
            best = (key, (a, b), float((pred[test] == labels[test]).double().mean()))
    return {'kind': 'exponent_search', 'accuracy': best[2], 'exponents': list(best[1]),
            'fit_accuracy': best[0][0], 'candidates': (2 * span + 1) ** 2,
            'fit_wall_seconds': time.perf_counter() - t0, 'labels_used': len(fit),
            'parameters': 2 + 2}


def monomial_features(z, span=2):
    zc = complex_features(z[:, :4])
    blocks = [zc[:, f] / zc[:, f].abs().clamp_min(1e-12) for f in (0, 1)]
    cols = []
    for a, b in itertools.product(range(-span, span + 1), repeat=2):
        if (a, b) == (0, 0):
            continue
        m = blocks[0] ** a * blocks[1] ** b
        cols += [m.real, m.imag]
    return torch.stack(cols, 1)


def monomial_ridge_decoder(z, labels, fit, test, p, penalty) -> dict:
    t0 = time.perf_counter()
    x = monomial_features(z)
    acc = ridge_scores(x, labels, fit, test, p, penalty)['accuracy']
    return {'kind': 'monomial_ridge', 'penalty': penalty, 'accuracy': acc,
            'parameters': (x.shape[1] + 1) * p, 'fit_wall_seconds': time.perf_counter() - t0,
            'labels_used': len(fit)}


def rff_decoder(z, labels, fit, test, p, sigma, penalty, n_features=1024, seed=17) -> dict:
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    x = z[:, :4].double()
    w = torch.randn(4, n_features, generator=g, dtype=torch.double) * sigma
    b = torch.rand(n_features, generator=g, dtype=torch.double) * 2 * math.pi
    feats = torch.cos(x @ w + b)
    acc = ridge_scores(feats, labels, fit, test, p, penalty)['accuracy']
    return {'kind': 'rff4', 'sigma': sigma, 'penalty': penalty, 'accuracy': acc,
            'parameters': (n_features + 1) * p, 'fit_wall_seconds': time.perf_counter() - t0,
            'labels_used': len(fit)}


# ---------------------------------------------------------------- training, checkpoints

def tensor_hash(arrs) -> str:
    h = hashlib.sha256()
    for name, t in arrs:
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_hash(model) -> str:
    return tensor_hash(sorted(model.state_dict().items()))


def train_model(initial, condition, m, pairs, labels, ia, ib):
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
    return model


def activations(model, pairs):
    with torch.no_grad():
        return model(pairs)[2].detach().double()


def save_checkpoint(model, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return {'file': path.name, 'file_sha256': digest(path), 'state_sha256': state_hash(model)}


def load_checkpoint(base_manifest, path: Path, expected: dict) -> Encoder:
    if digest(path) != expected['file_sha256']:
        raise ValueError(f'checkpoint file hash mismatch: {path.name}')
    model = Encoder(base_manifest)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    if state_hash(model) != expected['state_sha256']:
        raise ValueError(f'checkpoint state hash mismatch: {path.name}')
    return model


def encoder_set(seed, conditions, base, pairs, labels, ia, ib, ckpt_dir: Path, registry: dict):
    """Train, save, reload from disk and take activations from the reloaded weights."""
    torch.manual_seed(seed)
    initial = Encoder(base)
    out = {}
    for cond in conditions:
        model = train_model(initial, cond, base, pairs, labels, ia, ib)
        info = save_checkpoint(model, ckpt_dir / f'seed{seed}_{cond}.pt')
        reloaded = load_checkpoint(base, ckpt_dir / f'seed{seed}_{cond}.pt', info)
        z = activations(reloaded, pairs)
        if not torch.equal(z, activations(model, pairs)):
            raise ValueError('reloaded checkpoint changes activations')
        info['activation_sha256'] = tensor_hash([('z', z)])
        info['equivariance_loss'] = float(components(z, base['modulus'], ia, ib)['equivariance'])
        registry[f'{seed}/{cond}'] = info
        out[cond] = z
    return out


# ---------------------------------------------------------------- calibration and run

def true_modes(z, p):
    return [canon_p(tuple(t['mode']), p) for t in fft_modes(z, p) if tuple(t['mode']) != (0, 0)]


def recovery_summary(res, z, p):
    truth = set(true_modes(z, p))
    found = {canon_p(tuple(f), p) for f in res['frequencies']}
    return {'true_modes': [list(t) for t in sorted(truth)], 'found_modes': [list(t) for t in sorted(found)],
            'true_modes_recovered': truth <= found, 'spurious_modes': [list(t) for t in sorted(found - truth)]}


def calibrate(z, labels, fit, val, cfgs, p, penalties, rff_grid):
    rows = []
    for cfg in cfgs:
        out = blind_subspace_method(z, labels['shear'], fit, val, fit, cfg, p)
        rows.append({'config': cfg, 'val_accuracy': -1.0 if out['accuracy'] is None else out['accuracy'],
                     'n_features': out['n_features'], 'frequencies': out['frequencies']})
    order = sorted(range(len(cfgs)), key=lambda i: (-rows[i]['val_accuracy'],
                                                    -(cfgs[i]['n_shift'] or 10 ** 9), i))
    best = cfgs[order[0]]
    mono_pen = max((monomial_ridge_decoder(z, labels['shear'], fit, val, p, pen)['accuracy'], -i, pen)
                   for i, pen in enumerate(penalties))[2]
    rff_best = max((rff_decoder(z, labels['shear'], fit, val, p, s, pen)['accuracy'], -i, (s, pen))
                   for i, (s, pen) in enumerate(rff_grid))[2]
    return best, {'monomial_penalty': mono_pen, 'rff_sigma': rff_best[0], 'rff_penalty': rff_best[1]}, rows


def rotation_matrix(dim, seed):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.double))
    return q * torch.sign(torch.diagonal(r))


def evaluate_one(seed, cond, z, labels, pairs_pool, test, cfg, base_pen, p, m, with_controls):
    A = pairs_pool
    res = blind_subspace_method(z, labels['shear'], A, test, A, cfg, p)
    row = {'seed': seed, 'condition': cond, 'blind': res, 'recovery': recovery_summary(res, z, p),
           'fft_ground_truth_eval_only': fft_modes(z, p),
           'baselines': {
               'exponent_search': exponent_search_decoder(z, labels['shear'], A, test, p),
               'monomial_ridge': monomial_ridge_decoder(z, labels['shear'], A, test, p, base_pen['monomial_penalty']),
               'rff4': rff_decoder(z, labels['shear'], A, test, p, base_pen['rff_sigma'], base_pen['rff_penalty'])}}
    if with_controls:
        q = rotation_matrix(z.shape[1], m['control_seed'] + seed)
        zr = z @ q
        rres = blind_subspace_method(zr, labels['shear'], A, test, A, cfg, p)
        row['rotated_code'] = {
            'blind': {k: rres[k] for k in ('accuracy', 'frequencies', 'group_law_residual', 'n_features')},
            'same_modes_as_unrotated': {canon_p(tuple(f), p) for f in rres['frequencies']}
                                       == {canon_p(tuple(f), p) for f in res['frequencies']},
            'exponent_search': exponent_search_decoder(zr, labels['shear'], A, test, p)['accuracy'],
            'monomial_ridge': monomial_ridge_decoder(zr, labels['shear'], A, test, p, base_pen['monomial_penalty'])['accuracy'],
            'rff4': rff_decoder(zr, labels['shear'], A, test, p, base_pen['rff_sigma'], base_pen['rff_penalty'])['accuracy']}
        g = torch.Generator().manual_seed(m['control_seed'] + 100 + seed)
        shuffled = shift_pairs(p, A, cfg['n_shift'])
        zc, ops, rel = fit_operators(z, {k: (i, j[torch.randperm(len(j), generator=g)])
                                         for k, (i, j) in shuffled.items()}, A, cfg['ridge'])
        row['shuffled_shift_graph'] = {'operator_rel_error': rel,
            'note': 'targets of each shift pair permuted; operators should explain nothing'}
        row['label_efficiency'] = {}
        for n in m['label_counts']:
            row['label_efficiency'][str(n)] = {
                'blind': blind_subspace_method(z, labels['shear'], A[:n], test, A, cfg, p)['accuracy'],
                'exponent_search': exponent_search_decoder(z, labels['shear'], A[:n], test, p)['accuracy'],
                'monomial_ridge': monomial_ridge_decoder(z, labels['shear'], A[:n], test, p, base_pen['monomial_penalty'])['accuracy']}
    return row


def run(mpath: Path, out: Path, ckpt_dir: Path) -> dict:
    m = json.loads(mpath.read_text())
    if m.get('schema') != 'adict-blind-subspace/v0.1' or m['modulus'] != 31:
        raise ValueError('unexpected protocol')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    p = m['modulus']
    names = ('blind_subspace.py', 'test_blind_subspace.py', 'blind_fourier.py', 'anti_collapse.py',
             'pilot.py', 'anti_collapse_manifest.json')
    sources = {n: digest(HERE / n) for n in names}
    sources[mpath.name] = digest(mpath)
    (out / 'start.json').write_text(json.dumps({
        'schema': m['schema'], 'source_sha256': sources, 'manifest': m,
        'ordering': 'local pre-run write, unsigned; written before any encoder is trained'}, indent=2) + '\n')
    base = json.loads((HERE / 'anti_collapse_manifest.json').read_text())
    pairs, labels = states(p)
    ia, ib = shift_indices(p)
    test, A, A_fit, A_val = split_pairs(m, len(pairs))
    registry: dict = {}
    t_start = time.monotonic()
    cfgs = [dict(zip(m['grid'].keys(), v)) for v in itertools.product(*m['grid'].values())]
    rff_grid = [tuple(x) for x in m['rff_grid']]

    cal = encoder_set(m['calibration_seed'], m['conditions'], base, pairs, labels, ia, ib, ckpt_dir, registry)
    best, base_pen, cal_rows = calibrate(cal['eq_norm_cov'], labels, A_fit, A_val, cfgs, p,
                                         m['ridge_penalties'], rff_grid)
    frozen = {'blind_config': best, 'baseline_hyperparameters': base_pen,
              'calibration_seed': m['calibration_seed'], 'confirmation_seed': m['confirmation_seed'],
              'success_criteria': m['success_criteria'], 'source_sha256': sources,
              'calibration_checkpoint_hashes': {k: v for k, v in registry.items()}}
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
    frozen_sha = digest(out / 'frozen.json')

    results = [evaluate_one(m['calibration_seed'], c, z, labels, A, test, best, base_pen, p, m, c == 'eq_norm_cov')
               for c, z in cal.items()]
    for seed, conds, role in ((m['confirmation_seed'], m['conditions'], 'confirmation'),
                              (m['secondary_seed'], ['eq_norm_cov'], 'secondary_post_hoc')):
        enc = encoder_set(seed, conds, base, pairs, labels, ia, ib, ckpt_dir, registry)
        for c, z in enc.items():
            r = evaluate_one(seed, c, z, labels, A, test, best, base_pen, p, m, c == 'eq_norm_cov')
            r['role'] = role
            results.append(r)
    for r in results:
        r.setdefault('role', 'calibration')

    prim = next(r for r in results if r['role'] == 'confirmation' and r['condition'] == 'eq_norm_cov')
    c = m['success_criteria']
    rot = prim['rotated_code']
    best_base = max(v['accuracy'] for v in prim['baselines'].values())
    best_rot_base = max(rot['exponent_search'], rot['monomial_ridge'], rot['rff4'])
    acc = prim['blind']['accuracy'] or 0.0
    verdict = {
        'modes_recovered': prim['recovery']['true_modes_recovered'],
        'spurious_modes': prim['recovery']['spurious_modes'],
        'spurious_ok': len(prim['recovery']['spurious_modes']) <= c['max_spurious_modes'],
        'subspace_overlap': prim['blind']['subspace']['overlap_with_phase_coords'],
        'overlap_ok': prim['blind']['subspace']['overlap_with_phase_coords'] >= c['min_subspace_overlap'],
        'group_law_residual': prim['blind']['group_law_residual'],
        'residual_ok': (prim['blind']['group_law_residual'] or 9) <= c['max_group_law_residual'],
        'blind_accuracy': acc, 'accuracy_ok': acc >= c['min_accuracy'],
        'best_baseline_accuracy': best_base, 'non_inferior': acc >= best_base - c['non_inferiority_margin'],
        'rotated_blind_accuracy': rot['blind']['accuracy'],
        'rotation_invariant': (rot['blind']['accuracy'] or 0.0) >= acc - c['rotation_tolerance'],
        'best_rotated_baseline_accuracy': best_rot_base,
        'baselines_break_under_rotation': best_rot_base <= c['max_rotated_baseline_accuracy']}
    verdict['success'] = all(verdict[k] for k in ('modes_recovered', 'spurious_ok', 'overlap_ok', 'residual_ok',
                                                  'accuracy_ok', 'non_inferior', 'rotation_invariant',
                                                  'baselines_break_under_rotation'))
    (out / 'checkpoints.json').write_text(json.dumps(registry, indent=2, sort_keys=True) + '\n')
    result = {'schema': m['schema'], 'manifest': m, 'source_sha256': sources, 'frozen_sha256': frozen_sha,
              'python': platform.python_version(), 'torch': torch.__version__,
              'checkpoints': registry, 'calibration_grid': cal_rows, 'frozen': frozen,
              'results': results, 'verdict': verdict,
              'elapsed_seconds': round(time.monotonic() - t_start, 3),
              'scope': 'CPU; checkpoints saved and reloaded; decoder split only; not zero-shot'}
    (out / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True, default=list) + '\n')
    (out / 'receipt.json').write_text(json.dumps({
        'schema': 'blind-subspace-receipt-v0.1', 'start_sha256': digest(out / 'start.json'),
        'frozen_sha256': frozen_sha, 'results_sha256': digest(out / 'results.json'),
        'checkpoints_sha256': digest(out / 'checkpoints.json'), 'sources': sources,
        'n_records': len(results), 'success': verdict['success']}, indent=2, sort_keys=True) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=HERE / 'blind_subspace_manifest.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--checkpoints', type=Path, default=HERE / 'bf02_checkpoints')
    args = ap.parse_args()
    print(json.dumps(run(args.manifest, args.out, args.checkpoints)['verdict'], indent=2))


if __name__ == '__main__':
    main()
