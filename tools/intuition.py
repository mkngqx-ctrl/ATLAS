"""Regenerates the three intuition figures from the reference implementations.

Run from the project root:  python3 reference_code/make_intuition_figures.py
Writes figures/atlas_single_covariate_intuition.pdf,
       figures/atlas_competing_risks_intuition.pdf,
       figures/atlas_walks_illustration.pdf
Needs numpy, scipy, matplotlib. Walks in the third figure are drawn from the
true single-covariate generator (not a trained model), landmark 20, x = 10.
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'archive/reference_code'))
import single_covariate_reference as sc
import competing_risks_generator_reference as cr
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'lines.linewidth':1.3})
K='#222222'; G='#888888'

# ---------- single-covariate intuition ----------
fig,ax=plt.subplots(1,3,figsize=(6.5,2.3))
x=np.linspace(7,13,400); F=sc.covariate_F(x,'Weibull')
a,b=(7-10)/1.5,(13-10)/1.5; dens=truncnorm.pdf(x,a,b,loc=10,scale=1.5)
ax[0].fill_between(x,0,dens/dens.max()*0.8,color='#dddddd',lw=0,label='density of $X$ (scaled)')
ax[0].plot(x,F,color=K,label='$F(x)$')
ax[0].set_xlabel('covariate $x$'); ax[0].set_ylim(0,1); ax[0].set_title('(a) covariate map',fontsize=8)
ax[0].legend(frameon=False,fontsize=6.5,loc='upper right')
base=sc.base_matrix(10.,'Weibull'); u=np.arange(1,13)
def worsen(s,u,ref):
    row=base[s].copy(); donors=[0,1] if s==1 else [1,2]; dest=2 if s==1 else 3
    lam,nu,cap=(.35,2.5,.90) if s==1 else (.97,3.,.99)
    omega=row[donors].sum(); extra=min(lam*(u/ref)**nu,cap,.95*omega); return row[dest]+extra
for s,lab,mk in [(1,'Sick $\\to$ Sicker','o'),(2,'Sicker $\\to$ Death','s')]:
    ax[1].plot(u,[worsen(s,k,5) for k in u],color=K,marker=mk,ms=3,ls='-',label=lab+', $t<20$')
    ax[1].plot(u,[worsen(s,k,3) for k in u],color=G,marker=mk,ms=3,ls='--',label=lab+', $t\\geq 20$')
ax[1].set_xlabel('run length $u$'); ax[1].set_ylim(0,1); ax[1].set_title('(b) worsening probability, $x=10$',fontsize=8)
ax[1].legend(frameon=False,fontsize=6,loc='lower right')
qs=truncnorm.ppf([.1,.5,.9],a,b,loc=10,scale=1.5)
for q,ls,lab in zip(qs,['-','--',':'],['10th pct','median','90th pct']):
    states,init,k=sc.kernels(q,'Weibull','semi'); _,surv=sc.forward(init,k)
    ax[2].step(np.arange(0,31),surv,where='post',color=K,ls=ls,label=f'{lab} ($x={q:.1f}$)')
for t in (20,24): ax[2].axvline(t,color=G,lw=.6)
ax[2].set_xlabel('step $t$'); ax[2].set_ylim(0,1); ax[2].set_title('(c) exact baseline survival',fontsize=8)
ax[2].legend(frameon=False,fontsize=6.5,loc='lower left')
fig.tight_layout(); fig.savefig(ROOT/'outputs/figures/atlas_single_covariate_intuition.pdf')

# ---------- competing-risks intuition ----------
fig,ax=plt.subplots(1,4,figsize=(6.5,2.2))
ca=np.linspace(8,13,200); ax[0].plot(ca,((ca-8)/5)**2.5,color=K); ax[0].set_xlabel('calcium $x_c$'); ax[0].set_ylabel('effect'); ax[0].set_ylim(0,1); ax[0].set_title('(a) $z$',fontsize=8)
ag=np.linspace(30,90,200); ax[1].plot(ag,np.maximum(ag-50,0)/40,color=K); ax[1].set_xlabel('age $x_a$'); ax[1].set_ylim(0,1); ax[1].set_title('(b) $a$',fontsize=8)
d=np.linspace(0,1,50); ax[2].plot(d,.06+.35*d,color=K,label='Sicker B $\\to$ Sick B'); ax[2].plot(d,.04+.30*d,color=K,ls='--',label='Sicker AB $\\to$ Sick AB')
ax[2].set_xlabel('drug $x_d$'); ax[2].set_ylim(0,.5); ax[2].set_title('(c) recovery',fontsize=8); ax[2].legend(frameon=False,fontsize=5.8,loc='upper left')
t=np.arange(1,31)
for dv,ls,dl in [(0.,'-','drug 0'),(1.,'--','drug 1')]:
    c=cr.reference_curves(np.array([[10.5,60.,dv]]))[0]
    ax[3].plot(t,c[0],color=K,ls=ls,label='$S$, '+dl)
    ax[3].plot(t,c[1],color=G,ls=ls,marker='o',ms=2.2,markevery=5,label='$F_A$, '+dl)
    ax[3].plot(t,c[2],color=K,ls=ls,marker='^',ms=2.4,markevery=5,label='$F_B$, '+dl)
ax[3].set_xlabel('step $t$'); ax[3].set_ylim(0,1); ax[3].set_title('(d) exact curves',fontsize=8); ax[3].legend(frameon=False,fontsize=5.4,loc='center left')
fig.tight_layout(); fig.savefig(ROOT/'outputs/figures/atlas_competing_risks_intuition.pdf')

# ---------- walks illustration ----------
x0=10.; base=sc.base_matrix(x0,'Weibull')
def row_for(s,u,t):
    row=base[s].copy()
    if s in (1,2):
        donors=[0,1] if s==1 else [1,2]; dest=2 if s==1 else 3
        lam,nu,cap=(.35,2.5,.90) if s==1 else (.97,3.,.99); ref=5 if t<20 else 3
        omega=row[donors].sum(); extra=min(lam*(u/ref)**nu,cap,.95*omega)
        if omega>0: row[donors]*=(omega-extra)/omega
        row[dest]+=extra
    return row
def walk(rng,s,u,t0,T=30):
    out=[]
    for t in range(t0+1,T+1):
        row=row_for(s,u,t); nxt=rng.choice(5,p=row)
        if t>=24 and s==2 and nxt==2: nxt=3
        out.append(nxt)
        if nxt>=3: return out,t,(1 if nxt==3 else 0)
        u=u+1 if nxt==s else 1; s=nxt
    return out,T,0
rng=np.random.default_rng(7); L=20
while True:   # a prefix alive at L=20, currently Sick with a short run, having visited Sicker
    s=rng.choice(3,p=[2/3,1/4,1/12]); u=1; pre=[s]; alive=True
    for t in range(2,L+1):
        row=row_for(s,u,t); nxt=rng.choice(5,p=row)
        if nxt>=3: alive=False; break
        u=u+1 if nxt==s else 1; s=nxt; pre.append(s)
    if alive and pre[-1]==1 and u<=2 and 2 in pre: break
sL,uL=s,u
states,init,k=sc.kernels(x0,'Weibull','semi'); idx=states.index((sL,uL)); ref=sc.conditional_curves(k,L,[idx])[0]
def km(D,J):
    S=[];cur=1.
    for t in range(L+1,31):
        R=np.sum(D>=t); E=np.sum((D==t)&(J>0))
        if R>0: cur*=1-E/R
        S.append(cur)
    return np.array(S)
rng2=np.random.default_rng(11); D=[];J=[];paths=[]
for m in range(10000):
    p,dt,c=walk(rng2,sL,uL,L); D.append(dt);J.append(c)
    if m<12: paths.append(p)
D=np.array(D);J=np.array(J); km_big=km(D,J); km_small=km(D[:200],J[:200])
fig,ax=plt.subplots(1,2,figsize=(6.5,2.4),gridspec_kw={'width_ratios':[1.35,1]})
names=['Healthy','Sick','Sicker','Death','Dropout']
ax[0].step(np.arange(1,L+1),pre,where='post',color=K,lw=2,label='observed prefix')
rngj=np.random.default_rng(3)
for i,p in enumerate(paths):
    tt=np.arange(L,L+len(p)+1); yy=np.array([pre[-1]]+p,dtype=float); jit=rngj.uniform(-.14,.14)
    ax[0].step(tt,yy+jit,where='post',color=G,lw=.8,alpha=.85,label='sampled futures' if i==0 else None)
    if p[-1]>=3: ax[0].plot(tt[-1],yy[-1]+jit,marker='x' if p[-1]==3 else '+',color=K,ms=4)
ax[0].axvline(L,color=K,lw=.6,ls=':'); ax[0].set_yticks(range(5)); ax[0].set_yticklabels(names); ax[0].set_xlabel('step $t$'); ax[0].set_xlim(0.5,30.5)
ax[0].set_title('(a) twelve of 10,000 walks from $L=20$',fontsize=8); ax[0].legend(frameon=False,fontsize=6.5,loc='upper left')
tt=np.arange(L,31)
ax[1].step(tt,np.r_[1,km_small],where='post',color=G,lw=1,label='KM from 200 walks')
ax[1].step(tt,np.r_[1,ref],where='post',color=K,lw=1.4,label='exact reference')
ax[1].plot(tt[1:],km_big,ls='none',marker='o',mfc='white',mec=K,ms=4,label='KM from 10,000 walks')
ax[1].axvline(24,color=G,lw=.6); ax[1].set_xlabel('step $t$'); ax[1].set_ylim(0,1.02); ax[1].set_title('(b) survival curve for this patient',fontsize=8)
ax[1].legend(frameon=False,fontsize=6.5,loc='lower left')
fig.tight_layout(); fig.savefig(ROOT/'outputs/figures/atlas_walks_illustration.pdf')
print('done')
