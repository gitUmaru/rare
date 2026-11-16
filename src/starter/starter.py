"""Exact-score probability-flow ODE starter, using explicit Euler integration."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parent.parent.parent

def score(x,v,means,weights,base_variance):
    var=base_variance+v
    delta=x[:,None,:]-means[None,:,:]
    logmass=np.log(weights)[None,:]-.5*np.sum(delta**2,axis=-1)/var
    posterior=np.exp(logmass-logsumexp(logmass,axis=1,keepdims=True))
    return -(posterior[:,:,None]*delta).sum(axis=1)/var

def sample(terminal,grid,means,weights,base_variance):
    x=np.array(terminal,dtype=float,copy=True)
    for v,next_v in zip(grid[:-1],grid[1:]):
        # For additive Gaussian variance v, the probability-flow ODE is dx/dv=-score/2.
        x+=(next_v-v)*(-.5*score(x,v,means,weights,base_variance))
    return x

def rare_region(x,means):
    return np.argmin(((x[:,None,:]-means[None,:,:])**2).sum(axis=2),axis=1)==2

def wilson(hits,n):
    z=1.96;p=hits/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='development_r0.02_s0.8_seed940');p.add_argument('--steps',type=int,nargs='+',default=[8,16,32]);p.add_argument('--output',default='results.json')
    p.add_argument('--schedule',choices=['uniform','quadratic'],default='uniform');p.add_argument('--quick',action='store_true');a=p.parse_args()
    if a.quick:a.steps=[8,16]
    if any(s<1 for s in a.steps):p.error('steps must be positive')
    info=next((c for c in json.loads((ROOT/'data/manifest.json').read_text())['cases'] if c['name']==a.case),None)
    if info is None:p.error('Unknown case; consult data/manifest.json')
    d=np.load(ROOT/'data'/info['split']/f'{a.case}.npz');means,weights=d['means'],d['weights']
    reference=d['reference'];r=rare_region(reference,means)
    out={'case':a.case,'definition':'Probability of nearest-mean region 2, not latent mixture membership',
         'reference_rare_region_probability':float(r.mean()),'reference_wilson_95':wilson(int(r.sum()),len(r)),
         'latent_rare_weight':float(weights[2]),'runs':[]}
    for steps in a.steps:
        u=np.linspace(1,0,steps+1);grid=float(d['terminal_variance'])*(u if a.schedule=='uniform' else u**2)
        generated=sample(d['terminal'],grid,means,weights,float(d['base_variance']))
        hit=rare_region(generated,means)
        out['runs'].append({'score_evaluations':steps,'schedule':a.schedule,'rare_region_probability':float(hit.mean()),
          'rare_region_error':float(hit.mean()-r.mean()),'generated_wilson_95':wilson(int(hit.sum()),len(hit)),
          'mean_error_l2':float(np.linalg.norm(generated.mean(0)-reference.mean(0))),
          'covariance_error_frobenius':float(np.linalg.norm(np.cov(generated.T)-np.cov(reference.T))),
          'finite_samples':bool(np.isfinite(generated).all())})
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
