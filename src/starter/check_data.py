import sys
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from starter import ROOT,score,sample
cases=json.loads((ROOT/'data/manifest.json').read_text())['cases']
assert len(cases)==18
for c in cases:
 d=np.load(ROOT/'data'/c['split']/f"{c['name']}.npz")
 assert d['train'].shape==(6000,2) and d['reference'].shape==(50000,2) and d['terminal'].shape==(5000,2)
 assert np.isclose(d['weights'].sum(),1)
# Compare score to finite differences of the analytic log density.
means=np.array([[-1.,0],[1.,0],[0.,1.]]);weights=np.array([.49,.49,.02]);base=.1225;v=.7
x=np.random.default_rng(940).normal(size=(12,2))
def logp(x):return logsumexp(np.log(weights)[None,:]-.5*((x[:,None,:]-means)**2).sum(-1)/(base+v),axis=1)
eps=1e-5;fd=np.zeros_like(x)
for j in range(2):
 e=np.zeros_like(x);e[:,j]=eps;fd[:,j]=(logp(x+e)-logp(x-e))/(2*eps)
err=float(abs(score(x,v,means,weights,base)-fd).max());assert err<1e-7
# A one-Gaussian ODE has a closed-form endpoint. Finer Euler steps must approach it.
m=np.array([[.2,-.1]]);w=np.array([1.]);terminal=np.random.default_rng(941).normal(size=(20,2));vmax=2.
exact=m+(terminal-m)*np.sqrt(base/(base+vmax))
coarse=sample(terminal,np.linspace(vmax,0,33),m,w,base);fine=sample(terminal,np.linspace(vmax,0,1025),m,w,base)
e0=np.linalg.norm(coarse-exact);e1=np.linalg.norm(fine-exact)
assert e1<e0/10
print(f'PASS: 18 cases; score finite-difference error {err:.2g}; Gaussian ODE errors {e0:.3g} -> {e1:.3g}.')
