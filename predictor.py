"""Frozen Phase-2 ΔΔG inference backend.

FASTA mode: calibrated XGBoost.
PDB mode: calibrated XGBoost + CNN + GNN + normalized ensemble.
"""
from __future__ import annotations

import json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser

from models_phase2 import load_cnn_checkpoint, load_gnn_checkpoint

try:
    from torch_geometric.data import Data as GeoData
except ImportError:
    GeoData = None

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa:i for i,aa in enumerate(AA_LIST)}
CONTACT_CUTOFF = 8.0
CONTACT_MAP_SIZE = 128
HYDRO={'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
VOLUME={'A':88.6,'C':108.5,'D':111.1,'E':138.4,'F':189.9,'G':60.1,'H':153.2,'I':166.7,'K':168.6,'L':166.7,'M':162.9,'N':114.1,'P':112.7,'Q':143.8,'R':173.4,'S':89.0,'T':116.1,'V':140.0,'W':227.8,'Y':193.6}
CHARGE={'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0}
THREE_TO_ONE={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
BLOSUM62 = substitution_matrices.load("BLOSUM62")
PARSER = PDBParser(QUIET=True)

@dataclass(frozen=True)
class EnsembleWeights:
    xgb: float = 1/3
    cnn: float = 1/3
    gnn: float = 1/3
    def normalized(self, available: Iterable[str]) -> dict[str,float]:
        a=set(available)
        d={"xgb":max(self.xgb,0) if "xgb" in a else 0.0,"cnn":max(self.cnn,0) if "cnn" in a else 0.0,"gnn":max(self.gnn,0) if "gnn" in a else 0.0}
        s=sum(d.values())
        if s<=0: raise ValueError("At least one available weight must be positive")
        return {k:v/s for k,v in d.items()}

@dataclass
class LoadedModels:
    booster: xgb.Booster
    xgb_bias: float
    cnn: Any
    gnn: Any
    metadata: dict[str,Any]
    device: torch.device
    feature_names: list[str]

def parse_fasta(text: str) -> str:
    seq=''.join(x.strip() for x in str(text).splitlines() if x.strip() and not x.startswith('>')).replace(' ','').upper()
    if not seq: raise ValueError('No sequence provided')
    bad=sorted(set(seq)-set(AA_LIST))
    if bad: raise ValueError('Unsupported residues: '+', '.join(bad))
    return seq

def parse_mutation(mut: str) -> tuple[str,int,str]:
    m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',str(mut).strip().upper())
    if not m: raise ValueError("Use mutation notation such as N28I")
    wt,pos,mt=m.group(1),int(m.group(2)),m.group(3)
    if wt not in AA_INDEX or mt not in AA_INDEX: raise ValueError('Use standard amino acids')
    if wt==mt: raise ValueError('Mutant must differ from WT')
    return wt,pos,mt

def validate_sequence_mutation(seq: str, mut: str):
    seq=parse_fasta(seq); wt,pos,mt=parse_mutation(mut)
    if not 1<=pos<=len(seq): raise ValueError('Mutation position outside sequence')
    if seq[pos-1]!=wt: raise ValueError(f'Sequence has {seq[pos-1]} at {pos}, not {wt}')
    return wt,pos,mt

def blosum62_score(wt: str, mt: str) -> float:
    try: return float(BLOSUM62[wt,mt])
    except Exception: return 0.0

def mutation_vector(wt: str, mt: str) -> np.ndarray:
    return np.array([HYDRO[mt]-HYDRO[wt],VOLUME[mt]-VOLUME[wt],CHARGE[mt]-CHARGE[wt],blosum62_score(wt,mt)],dtype=np.float32)

def _load_metadata(path: Path) -> dict[str,Any]:
    if not path.exists(): raise FileNotFoundError(f'Missing exact training metadata: {path}')
    d=json.loads(path.read_text())
    for k in ('measurement_types','chain_types','datasets'):
        if not isinstance(d.get(k),list) or not d[k]: raise ValueError(f'Metadata key {k} must be a non-empty list')
    d.setdefault('defaults',{})
    return d

def load_models(artifacts_dir: str|Path, metadata_path: str|Path|None=None, device: str|torch.device|None=None, load_structure_models: bool=True) -> LoadedModels:
    a=Path(artifacts_dir)
    meta=_load_metadata(Path(metadata_path) if metadata_path else a.parent/'phase2_metadata.json')
    dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    booster=xgb.Booster()
    booster.load_model(str(a/'xgb_phase2_v4w_huber_weighted.json'))

    feature_file=a/'feature_order_73.csv'
    if not feature_file.exists():
        raise FileNotFoundError(f'Missing archived XGB feature order: {feature_file}')
    feature_table=pd.read_csv(feature_file)
    required_cols={'feature_index','feature_name'}
    if not required_cols.issubset(feature_table.columns):
        raise ValueError('feature_order_73.csv must contain feature_index and feature_name columns')
    feature_names=(
        feature_table.sort_values('feature_index')['feature_name']
        .astype(str).tolist()
    )
    if len(feature_names)!=int(booster.num_features()):
        raise ValueError(
            f'Feature-order file has {len(feature_names)} names, '
            f'but XGBoost expects {booster.num_features()} features'
        )

    bias=float((a/'xgb_v4w_bias.txt').read_text().strip()) if (a/'xgb_v4w_bias.txt').exists() else 0.0
    cnn=load_cnn_checkpoint(a/'cnn_phase2_v2_best.pt',dev) if load_structure_models else None
    gnn=load_gnn_checkpoint(a/'gnn_phase2_best.pt',dev) if load_structure_models else None
    return LoadedModels(booster,bias,cnn,gnn,meta,dev,feature_names)

def _one_hot(value: str, vocab: Sequence[str]) -> np.ndarray:
    v=np.zeros(len(vocab),dtype=np.float32)
    if str(value) in vocab: v[list(vocab).index(str(value))]=1
    return v

def get_residues_with_ca(pdb_path: str|Path, chain_id: str|None=None):
    structure=PARSER.get_structure('protein',str(pdb_path)); model=next(structure.get_models(),None)
    if model is None: raise ValueError('PDB has no model')
    residues=[]; keys=[]
    for chain in model:
        if chain_id is not None and chain.id!=chain_id: continue
        for res in chain:
            if res.has_id('CA') and res.get_resname().upper() in THREE_TO_ONE:
                residues.append(res); keys.append((chain.id,int(res.id[1])))
    if not residues: raise ValueError(f'No CA residues found for chain {chain_id}')
    return residues,keys

def sequence_from_pdb(pdb_path: str|Path, chain_id: str) -> str:
    residues,_=get_residues_with_ca(pdb_path,chain_id)
    return ''.join(THREE_TO_ONE[r.get_resname().upper()] for r in residues)

def _contact_map(residues, target_size=CONTACT_MAP_SIZE, cutoff=CONTACT_CUTOFF):
    n=min(len(residues),target_size); coords=np.array([r['CA'].coord for r in residues[:n]],dtype=np.float32)
    dist=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
    small=((dist<=cutoff)&(dist>0)).astype(np.float32)
    out=np.zeros((target_size,target_size),dtype=np.float32); out[:n,:n]=small
    return out

def compute_contact_map_and_mut_index(
    pdb_path,
    chain_id,
    mut_pos,
    cutoff=CONTACT_CUTOFF,
    target_size=CONTACT_MAP_SIZE
):
    residues, keys = get_residues_with_ca(pdb_path, chain_id)
    cmap = _contact_map(residues, target_size, cutoff)

    # Match the original training notebook:
    # mutations outside the first 128-residue CNN window use no mask.
    idx = next(
        (
            i for i, key in enumerate(keys[:target_size])
            if key == (chain_id, int(mut_pos))
        ),
        None
    )

    return cmap, idx

def build_cnn_inputs(pdb_path, chain_id, mutation):
    wt, pos, mt = parse_mutation(mutation)

    cmap, idx = compute_contact_map_and_mut_index(
        pdb_path,
        chain_id,
        pos
    )

    mask = np.zeros_like(cmap, dtype=np.float32)

    # Match the original notebook.
    # For positions outside the 128-residue crop, leave the mask empty.
    if idx is not None:
        mask[idx, :] = 1.0
        mask[:, idx] = 1.0

    cnn_input = np.stack([cmap, mask]).astype(np.float32)

    return (
        torch.from_numpy(cnn_input),
        torch.from_numpy(mutation_vector(wt, mt))
    )

def build_gnn_graph(pdb_path,chain_id,mutation,cutoff=CONTACT_CUTOFF):
    if GeoData is None: raise ImportError('torch-geometric required')
    wt,pos,mt=parse_mutation(mutation); residues,keys=get_residues_with_ca(pdb_path,chain_id)
    coords=np.array([r['CA'].coord for r in residues],dtype=np.float32); dist=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
    src,dst=np.where((dist<=cutoff)&(dist>0))
    if len(src)==0: src=dst=np.array([0],dtype=np.int64)
    edge_index=torch.tensor(np.vstack([src,dst]),dtype=torch.long)
    idx=next((i for i,k in enumerate(keys) if k==(chain_id,pos)),None)
    if idx is None: raise ValueError(f'Residue {chain_id}:{pos} not found')
    x=torch.zeros((len(residues),25),dtype=torch.float32)
    for i,res in enumerate(residues):
        aa=THREE_TO_ONE.get(res.get_resname().upper())
        if aa in AA_INDEX: x[i,AA_INDEX[aa]]=1
    x[idx,20]=1; x[idx,21:25]=torch.from_numpy(mutation_vector(wt,mt))
    return GeoData(x=x,edge_index=edge_index)

CHAIN_FEATURES = [
    "chain_0","chain_1","chain_A","chain_B","chain_E",
    "chain_H","chain_I","chain_L","chain_X","chain_unsigned"
]

def structural_features(pdb_path,mutation,chain_id):
    """Five XGB structural descriptors calculated from the complete selected chain."""
    seq_len = 0
    if pdb_path is None:
        return {
            "structure_size": 0.0,
            "contact_density": 0.0,
            "mean_degree": 0.0,
            "mutation_degree": 0.0,
            "mutation_distance_to_center": 0.0,
        }

    _,pos,_=parse_mutation(mutation)
    residues,keys=get_residues_with_ca(pdb_path,chain_id)
    coords=np.array([r['CA'].coord for r in residues],dtype=np.float32)
    n=len(coords)
    if n==0:
        raise ValueError('Cannot derive structure features from an empty chain')

    idx=next((i for i,k in enumerate(keys) if k==(chain_id,int(pos))),None)
    if idx is None:
        raise ValueError(f'Residue {chain_id}:{pos} not found')

    dist=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
    adjacency=((dist<=CONTACT_CUTOFF)&(dist>0)).astype(np.float32)
    degrees=adjacency.sum(axis=1)
    edges=float(adjacency.sum()/2.0)
    density=(2.0*edges/(n*(n-1))) if n>1 else 0.0
    center=coords.mean(axis=0)

    return {
        "structure_size": float(n),
        "contact_density": float(density),
        "mean_degree": float(degrees.mean()),
        "mutation_degree": float(degrees[idx]),
        "mutation_distance_to_center": float(np.linalg.norm(coords[idx]-center)),
    }

def _chain_encoding(chain_id):
    values={name:0.0 for name in CHAIN_FEATURES}
    candidate=f"chain_{'' if chain_id is None else str(chain_id).strip()}"
    if candidate in values:
        values[candidate]=1.0
    else:
        values["chain_unsigned"]=1.0
    return values

def _validate_pdb_mutation(pdb_path, chain_id, mutation):
    """Validate mutation against actual PDB residue numbering."""
    wt, pos, mt = parse_mutation(mutation)
    residues, keys = get_residues_with_ca(pdb_path, chain_id)

    idx = next(
        (i for i, key in enumerate(keys) if key == (chain_id, int(pos))),
        None,
    )
    if idx is None:
        raise ValueError(f"PDB residue {chain_id}:{pos} not found")

    actual = THREE_TO_ONE.get(residues[idx].get_resname().upper())
    if actual != wt:
        raise ValueError(
            f"PDB residue {chain_id}:{pos} is {actual}, not {wt}"
        )
    return wt, pos, mt


def build_xgb_feature_vector(models,sequence,mutation,*,temperature=None,ph=None,measurement_type=None,chain_id=None,dataset=None,pdb_path=None):
    """Build the exact archived 73-feature vector in feature_order_73.csv order."""
    seq = parse_fasta(sequence)

    if pdb_path is None:
        wt, pos, mt = validate_sequence_mutation(seq, mutation)
    else:
        if not chain_id:
            raise ValueError("chain_id required in PDB mode")
        wt, pos, mt = _validate_pdb_mutation(pdb_path, chain_id, mutation)

    d=models.metadata.get('defaults',{})
    temperature=float(d.get('temperature',25) if temperature is None else temperature)
    ph=float(d.get('ph',7) if ph is None else ph)

    dh,dv,dq,db=mutation_vector(wt,mt)
    ps=float(pos)/1000.0

    features={}
    for aa in AA_LIST:
        features[f"WT_{aa}"]=float(aa==wt)
    for aa in AA_LIST:
        features[f"MT_{aa}"]=float(aa==mt)

    features.update({
        "position": float(pos),
        "position_scaled": ps,
        "position_sin": float(math.sin(float(pos)/50.0)),
        "position_cos": float(math.cos(float(pos)/50.0)),
        "delta_hydrophobicity": float(dh),
        "delta_volume": float(dv),
        "delta_charge": float(dq),
        "BLOSUM62": float(db),
        "temperature": temperature,
        "pH": ph,
        "abs_delta_hydrophobicity": abs(float(dh)),
        "abs_delta_volume": abs(float(dv)),
        "abs_delta_charge": abs(float(dq)),
        "abs_BLOSUM62": abs(float(db)),
        "position_scaled_x_delta_hydrophobicity": ps*float(dh),
        "position_scaled_x_delta_charge": ps*float(dq),
        "measurement_thermal_stability": 1.0,
    })
    features.update(_chain_encoding(chain_id))
    features["dataset_thermomut"]=0.0

    if pdb_path is None:
        sf={
            "structure_size": float(len(seq)),
            "contact_density": 0.0,
            "mean_degree": 0.0,
            "mutation_degree": 0.0,
            "mutation_distance_to_center": 0.0,
        }
    else:
        sf=structural_features(pdb_path,mutation,chain_id)
    features.update(sf)

    missing=[name for name in models.feature_names if name not in features]
    if missing:
        raise ValueError('Cannot generate archived XGB features: '+', '.join(missing))

    feat=np.asarray([features[name] for name in models.feature_names],dtype=np.float32)
    expected=int(models.booster.num_features())
    if feat.size!=expected:
        raise ValueError(f'XGB feature mismatch: built {feat.size}, model expects {expected}')
    if not np.isfinite(feat).all():
        raise ValueError('XGB feature vector contains NaN or infinity')
    return feat

def xgb_predict_one(models,sequence,mutation,**opts):
    f=build_xgb_feature_vector(models,sequence,mutation,**opts)
    return float(models.booster.predict(xgb.DMatrix(f.reshape(1,-1)))[0]+models.xgb_bias)

@torch.no_grad()
def cnn_predict_one(models,pdb_path,chain_id,mutation):
    if models.cnn is None: raise RuntimeError('CNN not loaded')
    x,m=build_cnn_inputs(pdb_path,chain_id,mutation)
    return float(models.cnn(x.unsqueeze(0).to(models.device),m.unsqueeze(0).to(models.device)).detach().cpu().reshape(-1)[0])

@torch.no_grad()
def gnn_predict_one(models,pdb_path,chain_id,mutation):
    if models.gnn is None: raise RuntimeError('GNN not loaded')
    return float(models.gnn(build_gnn_graph(pdb_path,chain_id,mutation).to(models.device)).detach().cpu().reshape(-1)[0])

def predict_one(models,sequence,mutation,*,pdb_path=None,chain_id=None,weights=EnsembleWeights(),temperature=None,ph=None,measurement_type=None,dataset=None):
    seq=parse_fasta(sequence)

    if pdb_path is None:
        validate_sequence_mutation(seq,mutation)
    else:
        if not chain_id:
            raise ValueError('chain_id required in PDB mode')
        _validate_pdb_mutation(pdb_path,chain_id,mutation)

    opts=dict(temperature=temperature,ph=ph,measurement_type=measurement_type,chain_id=chain_id,dataset=dataset,pdb_path=pdb_path)
    px=xgb_predict_one(models,seq,mutation,**opts)
    out={'mutation':mutation.upper(),'mode':'PDB' if pdb_path else 'FASTA','xgb':px,'cnn':np.nan,'gnn':np.nan,'ensemble':px,'cnn_ok':False,'gnn_ok':False}
    if pdb_path is None:
        out['ddg_class']=ddg_class(px)
        return out

    available=['xgb']
    try:
        out['cnn']=cnn_predict_one(models,pdb_path,chain_id,mutation)
        out['cnn_ok']=True
        available.append('cnn')
    except Exception as e:
        out['cnn_error']=str(e)

    try:
        out['gnn']=gnn_predict_one(models,pdb_path,chain_id,mutation)
        out['gnn_ok']=True
        available.append('gnn')
    except Exception as e:
        out['gnn_error']=str(e)

    w=weights.normalized(available)
    out['weights_used']=w
    out['ensemble']=float(
        w['xgb']*out['xgb']
        + w['cnn']*(out['cnn'] if out['cnn_ok'] else 0)
        + w['gnn']*(out['gnn'] if out['gnn_ok'] else 0)
    )
    out['ddg_class']=ddg_class(out['ensemble'])
    return out

def scan_19aa_position(models,sequence,position,*,pdb_path=None,chain_id=None,weights=EnsembleWeights(),temperature=None,ph=None,measurement_type=None,dataset=None):
    seq=parse_fasta(sequence)
    position=int(position)

    if pdb_path is None:
        if not 1<=position<=len(seq):
            raise ValueError('Position outside sequence')
        wt=seq[position-1]
    else:
        if not chain_id:
            raise ValueError('chain_id required in PDB mode')
        residues,keys=get_residues_with_ca(pdb_path,chain_id)
        idx=next((i for i,k in enumerate(keys) if k==(chain_id,position)),None)
        if idx is None:
            raise ValueError(f'Residue {chain_id}:{position} not found')
        wt=THREE_TO_ONE[residues[idx].get_resname().upper()]

    rows=[]
    for mt in AA_LIST:
        if mt==wt:
            continue
        rows.append(
            predict_one(
                models,
                seq,
                f'{wt}{position}{mt}',
                pdb_path=pdb_path,
                chain_id=chain_id,
                weights=weights,
                temperature=temperature,
                ph=ph,
                measurement_type=measurement_type,
                dataset=dataset,
            )
        )
    col='ensemble' if pdb_path else 'xgb'
    return pd.DataFrame(rows).sort_values(col).reset_index(drop=True)

def ddg_class(v: float) -> str:
    if v<-1: return 'Strong stabilizing'
    if v<0: return 'Mild stabilizing'
    if v<1: return 'Near-neutral'
    return 'Destabilizing'

def confidence_label(row: Mapping[str,Any]) -> str:
    vals=[float(row[k]) for k in ('xgb','cnn','gnn') if k in row and pd.notna(row[k])]
    if len(vals)<2: return 'Single-model'
    s=float(np.std(vals)); return 'High agreement' if s<=0.35 else ('Moderate agreement' if s<=0.75 else 'Low agreement')

def prioritize_scan(df: pd.DataFrame) -> pd.DataFrame:
    o=df.copy(); o['confidence']=o.apply(confidence_label,axis=1); col='ensemble' if 'ensemble' in o else 'xgb'; o['rank']=o[col].rank(method='dense',ascending=True).astype(int)
    return o.sort_values(['rank','mutation']).reset_index(drop=True)
