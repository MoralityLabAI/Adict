"""C1 closeout: frozen BF-02 discovery on new seeds, target/rotation checks, native interchange.

Run from addict_probe_lab/addict_probe_lab:
    python -m research.accessible_causal_information.c1_closeout --out /tmp/c1
The BF-02 discovery configuration is loaded unchanged from blind_subspace_frozen.json.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch
import torch.nn.functional as F

from .anti_collapse import shift_indices
from .blind_fourier import (complex_features, digest, fft_modes, shift_pairs, solve_exponents,
                            split_pairs)
from .blind_subspace import (discover_subspace, encoder_set, exponent_search_decoder,
                             load_checkpoint, monomial_ridge_decoder, recovery_summary, rff_decoder,
                             rotation_matrix, state_hash, subspace_vs_truth)
from .pilot import states

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- artifact lock

def lf_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def verify_bf02_artifacts(base_manifest: dict, root: Path = HERE) -> dict:
    """Resolve BF-02 checkpoint files and verify file/state hashes, frozen file and sources."""
    registry = json.loads((root / 'blind_subspace_checkpoints.json').read_text())
    receipt = json.loads((root / 'blind_subspace_run_receipt.json').read_text())
    start = json.loads((root / 'blind_subspace_start.json').read_text())
    ck = {}
    for key, info in registry.items():
        path = root / 'bf02_checkpoints' / info['file']
        if not path.exists():
            ck[key] = 'missing'
            continue
        try:
            load_checkpoint(base_manifest, path, info)
            ck[key] = 'ok'
        except ValueError:
            ck[key] = 'hash_mismatch'
    src = {}
    for name, h in start['source_sha256'].items():
        p = root / name
        src[name] = 'ok' if p.exists() and digest(p) == h else (
            'ok_after_lf_normalization' if p.exists() and lf_digest(p) == h else 'mismatch_or_missing')
    return {'checkpoints': ck, 'n_checkpoints': len(ck),
            'frozen_file': 'ok' if digest(root / 'blind_subspace_frozen.json') == receipt['frozen_sha256'] else 'mismatch',
            'results_file': 'ok' if digest(root / 'blind_subspace_results.json') == receipt['results_sha256'] else 'mismatch',
            'sources': src, 'all_checkpoints_ok': all(v == 'ok' for v in ck.values())}


# ---------------------------------------------------------------- target-aware blind decoding

def pick_pair_for(feats, target, p):
    best = None
    for fa, fb in itertools.combinations(feats, 2):
        c = solve_exponents([fa['freq'], fb['freq']], target, p)
        if c is None:
            continue
        cost = sum(abs(x if x <= p // 2 else x - p) for x in c)
        key = (cost, fa['residual'] + fb['residual'])
        if best is None or key < best[0]:
            best = (key, fa, fb, c)
    return best


def decode_target(feats, target, labels, fit, test, p) -> dict:
    """Unit-normalized signed-exponent product, one fitted output phase; abstains if no pair spans the target."""
    pick = pick_pair_for(feats, target, p)
    if pick is None:
        return {'accuracy': None, 'abstained': True, 'reason': 'no independent pair'}
    _, fa, fb, c = pick
    exps = [x if x <= p // 2 else x - p for x in c]
    blocks = [f['feature'] / f['feature'].abs().clamp_min(1e-12) for f in (fa, fb)]
    prod = blocks[0] ** exps[0] * blocks[1] ** exps[1]
    prod = prod / prod.abs().clamp_min(1e-12)
    truth = torch.exp((2j * math.pi / p) * labels.double())
    off = (truth[fit] * prod[fit].conj()).mean()
    ph = off / off.abs() if off.abs() > 1e-12 else torch.tensor(1 + 0j, dtype=torch.complex128)
    centers = torch.exp((2j * math.pi / p) * torch.arange(p, dtype=torch.double))
    pred = ((prod[test, None] * ph) * centers[None, :].conj()).real.argmax(1)
    return {'accuracy': float((pred == labels[test]).double().mean()), 'abstained': False,
            'frequencies': [list(fa['freq']), list(fb['freq'])], 'exponents': exps,
            'max_abs_exponent': max(abs(x) for x in exps)}


def heldout_group_residual(feats, p, heldout):
    """Mean squared error of f(Tx) against the frozen frequency's exact action on held-out shift pairs."""
    if not feats:
        return None
    shifts = shift_pairs(p, heldout, None)
    worst = 0.0
    for c in feats:
        k, l = c['freq']
        f = c['feature']
        for g, kk in (('a', k), ('b', l), ('ab', k + l)):
            i, j = shifts[g]
            w = complex(math.cos(2 * math.pi * kk / p), math.sin(2 * math.pi * kk / p))
            worst = max(worst, float((f[j] - w * f[i]).abs().square().mean() / f[i].abs().square().mean()))
    return worst


# ---------------------------------------------------------------- native interchange

def interchange_pairs(pairs, test, n, seed, p):
    g = torch.Generator().manual_seed(seed)
    order = test[torch.randperm(len(test), generator=g)]
    inside = set(test.tolist())
    out = []
    for r in order.tolist():
        a, b = int(pairs[r, 0]), int(pairs[r, 1])
        cands = [a2 for a2 in range(p) if a2 != a and (a2 * p + b) in inside]
        if not cands:
            continue
        out.append((r, cands[int(torch.randint(len(cands), (1,), generator=g))] * p + b))
        if len(out) == n:
            break
    return torch.tensor([o[0] for o in out]), torch.tensor([o[1] for o in out])


def subspace_basis(feats, dim=32):
    if not feats:
        return torch.zeros(dim, 0, dtype=torch.double)
    cols = []
    for c in feats:
        cols += [c['w'].real, c['w'].imag]
    q, _ = torch.linalg.qr(torch.stack(cols, 1))
    return q


def patch(zr, zd, q):
    return zr + (zd - zr) @ q @ q.T


def head_argmax(model, z):
    with torch.no_grad():
        return model.read_sum(z.float()).argmax(1)


def score_patch(model, zp, zr, zd, target_out, rec_out):
    out = head_argmax(model, zp)
    return {'accuracy': float((out == target_out).double().mean()),
            'recipient_output_rate': float((out == rec_out).double().mean()),
            'off_target_rate': float(((out != target_out) & (out != rec_out)).double().mean()),
            'mean_patch_norm': float((zp - zr).norm(dim=1).mean()),
            'mean_displacement_from_counterfactual_code': float((zp - zd).norm(dim=1).mean())}


def native_interchange(model, z, feats, r_idx, d_idx, n_random, seed) -> dict:
    zr, zd = z[r_idx], z[d_idx]
    target_out, rec_out = head_argmax(model, zd), head_argmax(model, zr)
    dim = z.shape[1]
    q = subspace_basis(feats, dim)
    res = {'rank': int(q.shape[1]), 'pairs': len(r_idx)}
    res['no_edit'] = score_patch(model, zr, zr, zd, target_out, rec_out)
    res['zero_edit_identity'] = score_patch(model, patch(zr, zr, q), zr, zr, rec_out, rec_out)
    res['full_code_donor'] = score_patch(model, zd, zr, zd, target_out, rec_out)
    res['discovered'] = score_patch(model, patch(zr, zd, q), zr, zd, target_out, rec_out)
    comp = zd + (zr - zd) @ q @ q.T
    res['complement_replacement'] = score_patch(model, comp, zr, zd, target_out, rec_out)
    disc_norm = ((zd - zr) @ q @ q.T).norm(dim=1)
    g = torch.Generator().manual_seed(seed)
    matched, unscaled = [], []
    for _ in range(n_random):
        if q.shape[1]:
            qr, _ = torch.linalg.qr(torch.randn(dim, q.shape[1], generator=g, dtype=torch.double))
        else:
            qr = torch.zeros(dim, 0, dtype=torch.double)
        edit = (zd - zr) @ qr @ qr.T
        scale = disc_norm / edit.norm(dim=1).clamp_min(1e-12)
        matched.append(score_patch(model, zr + edit * scale[:, None], zr, zd, target_out, rec_out))
        unscaled.append(score_patch(model, zr + edit, zr, zd, target_out, rec_out))
    res['random_norm_matched'] = matched
    res['random_unscaled'] = unscaled
    best_m = max((x['accuracy'] for x in matched), default=0.0)
    res['best_random_norm_matched_accuracy'] = best_m
    res['best_random_unscaled_accuracy'] = max((x['accuracy'] for x in unscaled), default=0.0)
    res['lift_over_best_random'] = res['discovered']['accuracy'] - best_m
    return res


# ---------------------------------------------------------------- MLP comparator

def mlp_decoder(z, labels, fit, test, p, cfg) -> dict:
    t0 = time.perf_counter()
    zc = complex_features(z[:, :4])
    u = zc / zc.abs().clamp_min(1e-12)
    x = torch.stack((u[:, 0].real, u[:, 0].imag, u[:, 1].real, u[:, 1].imag), 1).float()
    torch.manual_seed(cfg['seed'])
    h = cfg['hidden']
    net = torch.nn.Sequential(torch.nn.Linear(4, h[0]), torch.nn.ReLU(), torch.nn.Linear(h[0], h[1]),
                              torch.nn.ReLU(), torch.nn.Linear(h[1], p))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for _ in range(cfg['steps']):
        opt.zero_grad()
        F.cross_entropy(net(x[fit]), labels[fit]).backward()
        opt.step()
    with torch.no_grad():
        pred = net(x).argmax(1)
    return {'kind': 'mlp', 'accuracy': float((pred[test] == labels[test]).double().mean()),
            'train_accuracy': float((pred[fit] == labels[fit]).double().mean()),
            'parameters': sum(q.numel() for q in net.parameters()), 'labels_used': len(fit),
            'steps': cfg['steps'], 'fit_wall_seconds': time.perf_counter() - t0}


# ---------------------------------------------------------------- run

def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=HERE).stdout.strip()
    except OSError:
        return None


def blind_all_targets(z, labels, fit, test, pool, cfg, p, targets):
    d = discover_subspace(z, pool, cfg, p)
    feats = d['features']
    out = {'n_features': len(feats), 'frequencies': [list(c['freq']) for c in feats],
           'feature_residuals': [c['residual'] for c in feats],
           'shift_pairs_per_generator': {k: len(v[0]) for k, v in d['shifts'].items()},
           'labels_used': len(fit), 'operator_rel_error': d['operator_rel_error'],
           'subspace': subspace_vs_truth(feats, z.shape[1]),
           'heldout_group_residual': heldout_group_residual(feats, p, test)}
    out['targets'] = {name: decode_target(feats, tuple(t), labels[name], fit, test, p)
                      for name, t in targets.items()}
    return out, feats


def gate_row(r, g):
    t = r['blind']['targets']
    ni = r['native_interchange']
    acc = lambda d: d['accuracy'] if d['accuracy'] is not None else 0.0
    chg = [abs(v) if v is not None else 1.0 for v in r['rotation_accuracy_change'].values()]
    gate = {'seed': r['seed'],
            'present_task': r['present_task_accuracy'] >= g['present_task_accuracy_min'],
            'small_exponent_decoder': all(acc(v) >= g['small_exponent_decoder_accuracy_min'] for v in t.values()),
            'heldout_group_residual': (r['blind']['heldout_group_residual'] is not None
                                       and r['blind']['heldout_group_residual'] <= g['heldout_group_residual_max']),
            'rotation_change': max(chg) <= g['rotation_accuracy_change_max'],
            'interchange_accuracy': ni['discovered']['accuracy'] >= g['interchange_accuracy_min'],
            'interchange_lift': ni['lift_over_best_random'] >= g['interchange_lift_over_best_random_min']}
    gate['all'] = all(v for k, v in gate.items() if k != 'seed')
    return gate


def run(mpath: Path, out: Path, ckpt_dir: Path) -> dict:
    m = json.loads(mpath.read_text())
    if m.get('schema') != 'adict-closeout-c1/v0.1' or m['modulus'] != 31:
        raise ValueError('unexpected protocol')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    p = m['modulus']
    names = ('c1_closeout.py', 'test_c1_closeout.py', 'blind_subspace.py', 'blind_fourier.py',
             'anti_collapse.py', 'pilot.py', 'anti_collapse_manifest.json', 'blind_subspace_frozen.json')
    sources = {n: digest(HERE / n) for n in names}
    sources[mpath.name] = digest(mpath)
    base = json.loads((HERE / 'anti_collapse_manifest.json').read_text())
    lock = verify_bf02_artifacts(base)
    frozen = json.loads((HERE / 'blind_subspace_frozen.json').read_text())
    cfg, bl = frozen['blind_config'], frozen['baseline_hyperparameters']
    (out / 'start.json').write_text(json.dumps({
        'schema': m['schema'], 'source_sha256': sources, 'manifest': m, 'git_head_at_start': git_head(),
        'bf02_artifact_lock': lock, 'frozen_config_used': {'blind_config': cfg, 'baselines': bl},
        'ordering': 'local pre-run write, unsigned; written before any encoder is trained'}, indent=2) + '\n')
    pairs, labels0 = states(p)
    a, b = pairs[:, 0], pairs[:, 1]
    labels = {'sum': labels0['sum'], 'shear_2_1': (2 * a + b) % p, 'shear_1_2': (a + 2 * b) % p}
    ia, ib = shift_indices(p)
    test, A, _, _ = split_pairs(m, len(pairs))
    r_idx, d_idx = interchange_pairs(pairs, test, m['interchange']['pairs'], m['interchange']['pair_seed'], p)
    targets = {k: tuple(v) for k, v in m['targets'].items()}
    t_start = time.monotonic()
    registry: dict = {}
    results = []
    for seed in m['seeds']:
        enc = encoder_set(seed, m['conditions'], base, pairs, labels0, ia, ib, ckpt_dir, registry)
        for cond, z in enc.items():
            model = load_checkpoint(base, ckpt_dir / f'seed{seed}_{cond}.pt', registry[f'{seed}/{cond}'])
            with torch.no_grad():
                present = float((model(pairs)[0].argmax(1) == labels['sum']).double().mean())
            blind, feats = blind_all_targets(z, labels, A, test, A, cfg, p, targets)
            rot = rotation_matrix(z.shape[1], m['rotation_seed_offset'] + seed)
            rblind, _ = blind_all_targets(z @ rot, labels, A, test, A, cfg, p, targets)
            row = {'seed': seed, 'condition': cond, 'present_task_accuracy': present,
                   'fft_ground_truth_eval_only': fft_modes(z, p),
                   'recovery': recovery_summary(blind, z, p), 'blind': blind, 'rotated_blind': rblind,
                   'rotation_accuracy_change': {
                       k: None if blind['targets'][k]['accuracy'] is None or rblind['targets'][k]['accuracy'] is None
                       else rblind['targets'][k]['accuracy'] - blind['targets'][k]['accuracy'] for k in targets},
                   'baselines': {t: {
                       'exponent_search': exponent_search_decoder(z, labels[t], A, test, p),
                       'monomial_ridge': monomial_ridge_decoder(z, labels[t], A, test, p, bl['monomial_penalty']),
                       'rff4': rff_decoder(z, labels[t], A, test, p, bl['rff_sigma'], bl['rff_penalty']),
                       'mlp': mlp_decoder(z, labels[t], A, test, p, m['mlp_comparator'])} for t in targets},
                   'native_interchange': native_interchange(model, z, feats, r_idx, d_idx,
                                                            m['interchange']['random_controls'],
                                                            m['control_seed'] + seed),
                   'checkpoint': registry[f'{seed}/{cond}']}
            results.append(row)
            print(json.dumps({'seed': seed, 'condition': cond, 'present': present,
                              'freqs': blind['frequencies'],
                              'acc': {k: v['accuracy'] for k, v in blind['targets'].items()},
                              'interchange': row['native_interchange']['discovered']['accuracy']}), flush=True)
    primary = [x for x in results if x['condition'] == 'eq_norm_cov']
    gates = [gate_row(r, m['g1_gates']) for r in primary]
    ni = lambda r, k: r['native_interchange'][k]
    verdict = {'per_seed': gates, 'G1_pass': all(x['all'] for x in gates),
               'native_gate_pass': all(x['interchange_accuracy'] and x['interchange_lift'] for x in gates),
               'native_interchange_accuracy': {str(r['seed']): ni(r, 'discovered')['accuracy'] for r in primary},
               'native_best_random_matched': {str(r['seed']): ni(r, 'best_random_norm_matched_accuracy') for r in primary},
               'native_full_code_donor': {str(r['seed']): ni(r, 'full_code_donor')['accuracy'] for r in primary},
               'native_complement': {str(r['seed']): ni(r, 'complement_replacement')['accuracy'] for r in primary}}
    result = {'schema': m['schema'], 'manifest': m, 'source_sha256': sources, 'bf02_artifact_lock': lock,
              'git_head_at_start': git_head(), 'python': platform.python_version(), 'torch': torch.__version__,
              'checkpoints': registry, 'interchange_pair_count': len(r_idx), 'results': results,
              'verdict': verdict, 'elapsed_seconds': round(time.monotonic() - t_start, 3),
              'scope': 'CPU; prospective seeds; decoder-split plus native-head interchange on a synthetic task'}
    (out / 'checkpoints.json').write_text(json.dumps(registry, indent=2, sort_keys=True) + '\n')
    (out / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (out / 'receipt.json').write_text(json.dumps({
        'schema': 'closeout-c1-receipt-v0.1', 'start_sha256': digest(out / 'start.json'),
        'results_sha256': digest(out / 'results.json'), 'checkpoints_sha256': digest(out / 'checkpoints.json'),
        'sources': sources, 'n_records': len(results), 'G1_pass': verdict['G1_pass'],
        'native_gate_pass': verdict['native_gate_pass']}, indent=2, sort_keys=True) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=HERE / 'c1_manifest.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--checkpoints', type=Path, default=HERE / 'c1_checkpoints')
    args = ap.parse_args()
    print(json.dumps(run(args.manifest, args.out, args.checkpoints)['verdict'], indent=2))


if __name__ == '__main__':
    main()
