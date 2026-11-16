"""Create fixed 2D Gaussian mixtures, training samples and exact terminal draws."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent.parent

def draw(rng,n,means,weights,variance):
    component=rng.choice(len(weights),size=n,p=weights)
    return (means[component]+rng.normal(0,np.sqrt(variance),(n,2))).astype(np.float32)

def main():
    for split in ['development','final']:(ROOT/'data'/split).mkdir(parents=True,exist_ok=True)
    cases=[]
    configs=[('development',r,s) for r in [.02,.08] for s in [.8,2.]]+[('final',.04,1.4),('final',.01,2.8)]
    for split,rarity,separation in configs:
      for seed in [940,941,942]:
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(rarity*1000),int(separation*10),int(split=='final')]))
        means=np.array([[-separation,0.],[separation,0.],[0.,separation]])
        weights=np.array([(1-rarity)/2,(1-rarity)/2,rarity]);basevar=.35**2;terminalvar=16.
        name=f'{split}_r{rarity:.2f}_s{separation:.1f}_seed{seed}'
        np.savez_compressed(ROOT/'data'/split/f'{name}.npz',means=means,weights=weights,
            base_variance=np.array(basevar),terminal_variance=np.array(terminalvar),
            train=draw(rng,6000,means,weights,basevar),
            reference=draw(rng,50000,means,weights,basevar),
            terminal=draw(rng,5000,means,weights,basevar+terminalvar))
        cases.append(dict(name=name,split=split,seed=seed,rare_weight=rarity,separation=separation,n_train=6000,n_reference=50000,n_terminal=5000))
    (ROOT/'data/manifest.json').write_text(json.dumps({'source':'Synthetic Gaussian mixtures with exact variance-exploding score.', 'cases':cases},indent=2))
    print(f'Wrote {len(cases)} mixture cases.')
if __name__=='__main__':main()
