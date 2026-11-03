"""CPU-only anti-collapse test for a prescribed representation of Z_p x Z_p.

The first two code coordinates carry the sum character, the next two the
first-operand character. Other hidden coordinates may support the task head.
Run from addict_probe_lab/addict_probe_lab:
    python -m research.accessible_causal_information.anti_collapse --out /tmp/ac01
All 961 input pairs are used to train the encoder. Held-out is only for the
frozen-readout diagnostic. Never interpret this as zero-shot model transfer.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from .pilot import Encoder, exact_risk, quantize_nested, ridge_scores, states

HERE = Path(__file__).resolve().parent


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rotations(z4: torch.Tensor, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prescribed shift actions on four phase coordinates.

    T_a: (a,b) -> (a+1,b): R on sum and operand a.
    T_b: (a,b) -> (a,b+1): R on sum and identity on a.
    """
    theta=2*math.pi/p
    cs,sn=math.cos(theta),math.sin(theta)
    def turn(t: torch.Tensor) -> torch.Tensor:
        return torch.stack((cs*t[:,0]-sn*t[:,1], sn*t[:,0]+cs*t[:,1]),dim=1)
    zsum,za = z4[:,:2],z4[:,2:4]
    return torch.cat((turn(zsum),turn(za)),1),torch.cat((turn(zsum),za),1)


def shift_indices(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    a=torch.arange(p).repeat_interleave(p)
    b=torch.arange(p).repeat(p)
    return ((a+1)%p)*p+b, a*p+(b+1)%p


def components(z: torch.Tensor, p: int, idx_a: torch.Tensor,
               idx_b: torch.Tensor) -> dict[str,torch.Tensor]:
    v=z[:,:4]
    want_a,want_b=rotations(v,p)
    eq=.5*((v[idx_a]-want_a).square().sum(dim=1).mean()+
           (v[idx_b]-want_b).square().sum(dim=1).mean())
    blocks=v.reshape(-1,2,2)
    lengths=torch.linalg.vector_norm(blocks,dim=2)
    # A sqrt norm has informative gradients near small, nonzero phases.
    unit_loss=(lengths-1.0).square().mean()
    centered=v-v.mean(dim=0)
    cov=centered.T@centered/v.shape[0]
    cov_target=0.5*torch.eye(4,dtype=z.dtype,device=z.device)
    cov_loss=(cov-cov_target).square().sum()
    return {"equivariance":eq,"norm":unit_loss,"covariance":cov_loss,
            "mean_phase_energy":blocks.square().sum(dim=2).mean(),
            "covariance_matrix":cov}


def ideal_phases(p:int) -> torch.Tensor:
    pairs,labels=states(p)
    s,a=labels['sum'],labels['a']
    th=2*math.pi/p
    return torch.stack((torch.cos(th*s),torch.sin(th*s),
                        torch.cos(th*a),torch.sin(th*a)),1)


def phase_offset_decode(v:torch.Tensor, labels:torch.Tensor,
                        train:torch.Tensor, test:torch.Tensor,
                        p:int) -> dict:
    """Fit ONE output-phase offset on train, predict with phase product on test.

    At zero-energy phases output is undefined; the small denominator is a
    declared numerical diagnostic rather than an information guarantee.
    """
    vs=torch.complex(v[:,0].double(),v[:,1].double())
    va=torch.complex(v[:,2].double(),v[:,3].double())
    prod=vs*va
    absprod=prod.abs()
    unit=prod/absprod.clamp_min(1e-12)
    truth=torch.exp((2j*math.pi/p)*labels.double())
    u=(truth[train]*unit[train].conj()).mean()
    if u.abs()<1e-12:
        offset=torch.tensor(1+0j,dtype=torch.complex128)
    else:
        offset=u/u.abs()
    pred_unit=unit[test]*offset
    centers=torch.exp((2j*math.pi/p)*torch.arange(p,dtype=torch.double))
    scores=(pred_unit[:,None]*centers[None,:].conj()).real
    pred=scores.argmax(dim=1)
    return {'heldout_accuracy':float((pred==labels[test]).double().mean()),
            'train_phase_offset_magnitude':float(u.abs()),
            'fitted_scalar_count':2,'fixed_phase_coefficients':2*p,
            'score_count_per_example':p}


def phase_alignment(v:torch.Tensor, labels:dict, train:torch.Tensor, test:torch.Tensor,p:int)->dict:
    """Train-only constant rotations for the two known phase axes."""
    coords=[]
    for j,key in enumerate(('sum','a')):
        z=torch.complex(v[:,2*j].double(),v[:,2*j+1].double())
        truth=torch.exp(2j*math.pi*labels[key].double()/p)
        rho=(z[train]*truth[train].conj()).mean()
        mag=rho.abs()
        aligned=z*(rho.conj()/mag.clamp_min(1e-12))
        lengths=z.abs()
        angular=(aligned[test]*truth[test].conj()).real/lengths[test].clamp_min(1e-12)
        coords.append({'target':key,'train_phase_coherence':float(mag),
                       'heldout_cosine':float(angular.mean()),
                       'heldout_norm':float(lengths[test].mean())})
    return {'axes':coords,'fitted_scalars':4}



def spectrum_diagnostic(v: torch.Tensor, p: int) -> list[dict]:
    """Exhaustive character spectrum, with states ordered by (a,b).

    Frequencies (k,l) identify chi(k*a + l*b), so the prescribed
    characters are (1,1) and (1,0). This is a post-fit diagnostic;
    no frequencies are selected or changed during optimization.
    """
    expected=((1,1),(1,0))
    out=[]
    for j,expected_mode in enumerate(expected):
        z=torch.complex(v[:,2*j].double(),v[:,2*j+1].double()).reshape(p,p)
        f=torch.fft.fft2(z)/(p*p)
        mass=f.abs().square()
        total=mass.sum().clamp_min(1e-20)
        top=torch.topk(mass.reshape(-1),k=3)
        modes=[{'k':int(i.item()//p),'l':int(i.item()%p),
                'power_fraction':float(val/total)} for val,i in zip(top.values,top.indices)]
        out.append({'block':('sum','operand_a')[j],
                    'expected_mode':list(expected_mode),
                    'expected_power_fraction':float(mass[expected_mode[0],expected_mode[1]]/total),
                    'dc_power_fraction':float(mass[0,0]/total),
                    'strongest_modes':modes})
    return out


def train_only_character_basis_decode(v: torch.Tensor, pairs: torch.Tensor,
                                      shear: torch.Tensor, train: torch.Tensor,
                                      test: torch.Tensor, p: int) -> dict:
    """Exploratory, post-hoc group-basis readout; modes selected only on train.

    The reader receives known input operands (a,b) for 480 fitting samples.
    It identifies a single cyclic character per complex block, inverts the
    two-character basis over Z_p, then fits ONE phase offset from the shear
    labels on those same training examples. It has an explicit group-law
    inductive bias and is not a generic compact decoder.
    """
    kk=torch.arange(p,dtype=torch.long).repeat_interleave(p)
    ll=torch.arange(p,dtype=torch.long).repeat(p)
    modes=torch.stack((kk,ll),1)
    xy=pairs[train].long()
    freqs=(xy[:,0,None]*kk[None,:]+xy[:,1,None]*ll[None,:])%p
    characters=torch.exp(2j*math.pi*freqs.double()/p)
    selected=[];fit_coherences=[]
    blocks=[]
    for j in (0,1):
        z=torch.complex(v[:,2*j].double(),v[:,2*j+1].double())
        blocks.append(z/z.abs().clamp_min(1e-12))
        correlation=(z[train,None]*characters.conj()).mean(dim=0)
        idx=int(correlation.abs().argmax())
        selected.append((int(kk[idx]),int(ll[idx])))
        fit_coherences.append(float(correlation[idx].abs()))
    (k1,l1),(k2,l2)=selected
    determinant=(k1*l2-k2*l1)%p
    if determinant==0:
        return {'heldout_accuracy':None,'recoverable_basis':False,
                'modes':selected,'fit_coherences':fit_coherences,
                'reason':'selected characters linearly dependent mod p'}
    # Solve c1*(k1,l1)+c2*(k2,l2)=(2,1) modulo p.
    inv=pow(int(determinant),-1,p)
    c1=((2*l2-k2)*inv)%p
    c2=((k1-2*l1)*inv)%p
    # Choose the smallest-magnitude integer lift of each residue. On an
    # approximate unit circle, raising to p-1 amplifies phase noise while
    # the equivalent group exponent -1 means complex conjugation.
    signed1=c1 if c1<=p//2 else c1-p
    signed2=c2 if c2<=p//2 else c2-p
    product=blocks[0]**signed1 * blocks[1]**signed2
    truth=torch.exp((2j*math.pi/p)*shear.double())
    offset=(truth[train]*product[train].conj()).mean()
    if offset.abs()<1e-12:
        phase=torch.tensor(1+0j,dtype=torch.complex128)
    else:
        phase=offset/offset.abs()
    center=torch.exp((2j*math.pi/p)*torch.arange(p,dtype=torch.double))
    pred=((product[test,None]*phase)*center[None,:].conj()).real.argmax(1)
    return {'heldout_accuracy':float((pred==shear[test]).double().mean()),
            'recoverable_basis':True,'modes':selected,
            'exponents':[c1,c2],'signed_exponents':[signed1,signed2],
            'determinant':int(determinant),
            'fit_coherences':fit_coherences,'fitted_offset_coherence':float(offset.abs()),
            'fitted_discrete_mode_bits_upper_bound':4*math.ceil(math.log2(p)),
            'fitted_scalar_count':2,
            'character_candidates_per_block':p*p}

def evaluate(model:Encoder,pairs:torch.Tensor,labels:dict, train:torch.Tensor,
             test:torch.Tensor, m:dict, seed:int, condition:str,
             ia:torch.Tensor,ib:torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        task_logits,_,z=model(pairs)
        st=components(z,m['modulus'],ia,ib)
        eig=torch.linalg.eigvalsh(st['covariance_matrix']).tolist()
        p=m['modulus']
        phase=phase_offset_decode(z[:,:4],labels['shear'],train,test,p)
        alignment=phase_alignment(z[:,:4],labels,train,test,p)
        ridge=ridge_scores(z,labels['shear'],train,test,p,m['ridge_penalty'])
        # Fit only the common train split. Oracle risk sees all held-out labels.
        q=quantize_nested(z,train,m['retained_dimensions'],m['coordinate_bits'])
        qr=[]
        for k,code in q.items():
            codes=code.tolist()
            qr.append({'bits':k*m['coordinate_bits'],'distinct_codes':len(set(map(tuple,codes))),
                       'sum_oracle_risk':str(exact_risk(codes,labels['sum'].tolist())),
                       'shear_oracle_risk':str(exact_risk(codes,labels['shear'].tolist()))})
        block=(z[:,:4].reshape(-1,2,2).square().sum(dim=2))
        return {'seed':seed,'condition':condition,
                'present_task_accuracy':float((task_logits.argmax(1)==labels['sum']).float().mean()),
                'phase_energy':float(st['mean_phase_energy']),
                'phase_block_energies':[float(i) for i in block.mean(dim=0)],
                'phase_spectrum':spectrum_diagnostic(z[:,:4],p),
                'equivariance_loss':float(st['equivariance']),
                'unit_norm_penalty':float(st['norm']),
                'covariance_penalty':float(st['covariance']),
                'covariance_eigenvalues':[float(t) for t in eig],
                'phase_product':phase,'phase_alignment':alignment,
                'posthoc_character_basis':train_only_character_basis_decode(z[:,:4],pairs,labels['shear'],train,test,p),
                'ridge_shear':ridge,'quantized_oracle':qr}


def run(mpath:Path,out:Path)->dict:
    m=json.loads(mpath.read_text())
    if m.get('schema')!='adict-phase-anti-collapse/v0.1':
        raise ValueError('unknown protocol')
    if m['modulus']!=31 or m['steps']<=0:
        raise ValueError('unexpected fixed benchmark')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True)
    sources={name:digest(HERE/name) for name in ('anti_collapse.py','test_anti_collapse.py','pilot.py')}
    sources[mpath.name]=digest(mpath)
    # Before training / inspecting results, locally persist a protocol record.
    (out/'start.json').write_text(json.dumps({'schema':m['schema'],'source_sha256':sources,
            'manifest':m,'ordering':'local pre-run write, unsigned'},indent=2)+'\n')
    pairs,labels=states(m['modulus'])
    perm=torch.randperm(len(pairs),generator=torch.Generator().manual_seed(m['diagnostic_split_seed']))
    cut=int(m['diagnostic_train_fraction']*len(pairs))
    train,test=perm[:cut].sort().values,perm[cut:].sort().values
    ia,ib=shift_indices(m['modulus'])
    start=time.monotonic()
    results=[]
    for seed in m['seeds']:
        torch.manual_seed(seed)
        initial=Encoder(m)
        for condition in m['conditions']:
            model=copy.deepcopy(initial)
            if condition!='initial':
                optimizer=torch.optim.AdamW(model.parameters(),lr=m['learning_rate'],weight_decay=m['weight_decay'])
                for step in range(m['steps']):
                    optimizer.zero_grad(set_to_none=True)
                    logits,hidden,z=model(pairs)
                    loss=F.cross_entropy(logits,labels['sum'])
                    if condition=='task_and_a':
                        loss=loss+F.cross_entropy(hidden,labels['a'])
                    if condition in ('equivariance','eq_norm','eq_norm_cov'):
                        st=components(z,m['modulus'],ia,ib)
                        loss=loss+m['equivariance_weight']*st['equivariance']
                        if condition in ('eq_norm','eq_norm_cov'):
                            loss=loss+m['norm_weight']*st['norm']
                        if condition=='eq_norm_cov':
                            loss=loss+m['covariance_weight']*st['covariance']
                    loss.backward()
                    optimizer.step()
            record=evaluate(model,pairs,labels,train,test,m,seed,condition,ia,ib)
            results.append(record)
            print(json.dumps({'seed':seed,'condition':condition,
                  'task_accuracy':record['present_task_accuracy'],
                  'equivariance_loss':record['equivariance_loss'],
                  'phase_energy':record['phase_energy'],
                  'phase_shear_accuracy':record['phase_product']['heldout_accuracy']}),flush=True)
    result={'schema':m['schema'],'manifest':m,'source_sha256':sources,
            'python':platform.python_version(),'torch':torch.__version__,
            'parameter_count':sum(p.numel() for p in initial.parameters()),
            'n_states':len(pairs),'probe_train_rows':len(train),
            'probe_test_rows':len(test),'results':results,
            'elapsed_seconds':round(time.monotonic()-start,3),
            'scope':'CPU full-state encoder training; fixed 480/481 decoder split; not zero-shot transfer'}
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'receipt.json').write_text(json.dumps({'schema':'anti-collapse-receipt-v0.1',
           'start_sha256':digest(out/'start.json'),
           'results_sha256':digest(out/'results.json'),
           'sources':sources,'n_records':len(results)},indent=2,sort_keys=True)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=HERE/'anti_collapse_manifest.json')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    run(args.manifest,args.out)


if __name__=='__main__':main()
