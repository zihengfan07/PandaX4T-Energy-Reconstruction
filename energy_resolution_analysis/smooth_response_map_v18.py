#!/usr/bin/env python3
"""Cross-fitted additive smooth response maps with an untouched time holdout."""

import argparse, itertools, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from anchored_smallmodels_v14 import chronological_folds, energy_cor, time_audit
from grouped_nested_physics_v9 import protocol_metrics
from safe_general_v12 import load_table, local_spike_score, robust_metrics

SETS = {
 "top_xy": ["dt","wS2CDF_max","xS2T_max","yS2T_max"],
 "corrected_xy": ["dt","wS2CDF_max","yS2Tcor_max","xS2Bcor_max"],
 "both_xy": ["dt","wS2CDF_max","xS2T_max","yS2T_max","yS2Tcor_max","xS2Bcor_max"],
}
GRIDS=(4,6,8); SHRINKS=(0.3,0.6,0.9); CAPS=(0.0025,0.005,0.0075)

def winterp(x, xp, fp): return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

def fit_map(x, energy, bins, shrink, cap):
 x=np.asarray(x,float); med=np.nanmedian(x,axis=0); med[~np.isfinite(med)]=0
 z=np.where(np.isfinite(x),x,med)
 target=np.log(np.clip(energy/2614.5,1e-12,None))
 weights=np.exp(-0.5*((energy-2614.5)/(0.10*2614.5))**2)
 center=float(np.average(target,weights=weights)); residual=target-center
 maps=[]; pred=np.zeros(len(x))
 for _ in range(2):
  new=[]
  for j in range(z.shape[1]):
   old=maps[j] if len(maps)==z.shape[1] else None
   if old is not None: pred-=winterp(z[:,j],old[0],old[1])
   r=residual-pred; edges=np.unique(np.quantile(z[:,j],np.linspace(0,1,bins+1)))
   if len(edges)<3: edges=np.array([np.min(z[:,j])-1,np.median(z[:,j]),np.max(z[:,j])+1])
   centers=(edges[:-1]+edges[1:])/2; vals=[]
   ids=np.clip(np.digitize(z[:,j],edges[1:-1]),0,len(centers)-1)
   for k in range(len(centers)):
    m=ids==k; vals.append(float(np.average(r[m],weights=weights[m])) if np.any(m) else 0.)
   vals=np.asarray(vals)*shrink
   if len(vals)>2: vals=np.convolve(np.pad(vals,1,mode='edge'),[.25,.5,.25],mode='valid')
   vals-=float(np.average(winterp(z[:,j],centers,vals),weights=weights))
   pred+=winterp(z[:,j],centers,vals); new.append((centers,vals))
  maps=new
 q25,q75=np.quantile(z,[.25,.75],axis=0); scale=q75-q25; scale[scale<1e-9]=1
 lower=np.quantile(z,.0005,axis=0)-scale; upper=np.quantile(z,.9995,axis=0)+scale
 pc=float(np.average(pred,weights=weights))
 return {"medians":med,"maps":maps,"lower":lower,"upper":upper,"prediction_center":pc,
         "bins":bins,"shrink":shrink,"cap":cap,"strong_raw_limit":1.5*cap}

def apply(state,x,base,valid):
 x=np.asarray(x,float); finite=np.isfinite(x); z=np.where(finite,x,state["medians"])
 count=np.sum((~finite)|(z<state["lower"])|(z>state["upper"]),axis=1)
 raw=np.zeros(len(x))
 for j,(xp,fp) in enumerate(state["maps"]): raw+=winterp(z[:,j],xp,fp)
 raw-=state["prediction_center"]; ood=count>max(1,int(np.floor(.2*x.shape[1])))
 strong=np.abs(raw)>state["strong_raw_limit"]; used=valid&~ood&~strong
 delta=np.zeros(len(x)); delta[used]=state["cap"]*np.tanh(raw[used]/state["cap"])
 out=np.full(len(x),np.nan); out[valid]=base[valid]*np.exp(-delta[valid])
 return {"energy":out,"delta":delta,"applied":used,"ood":ood,"strong":strong}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--calibration',required=True); ap.add_argument('--background',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 fs=sorted(set(sum(SETS.values(),[]))); cols=['runNumber','eventNumber','t','qS1ub_C','qS2Bdesub_C']+fs
 cal=load_table(a.calibration,cols); bg=load_table(a.background,cols); cb,cv=energy_cor(cal); bb,bv=energy_cor(bg)
 order=np.argsort(cal.t.to_numpy(float),kind='mergesort'); split=int(.8*len(cal)); dev=order[:split]; hold=order[split:]; dev=dev[cv[dev]]; hv=cv[hold]
 folds=chronological_folds(cal.t.to_numpy(float)[dev],4); br=robust_metrics(cb[dev]); bp=protocol_metrics(cb[dev]); bs=local_spike_score(bb); b26=local_spike_score(bb,width=5,lo=2450,hi=2750)
 rows=[]; saved={}
 for sn,names in SETS.items():
  x=cal[names].to_numpy(float); bx=bg[names].to_numpy(float)
  for bins,shrink,cap in itertools.product(GRIDS,SHRINKS,CAPS):
   oof=np.full(len(cal),np.nan)
   for f in range(4):
    te=dev[folds==f]; tr=dev[folds!=f]; st=fit_map(x[tr],cb[tr],bins,shrink,cap); oof[te]=apply(st,x[te],cb[te],cv[te])['energy']
   st=fit_map(x[dev],cb[dev],bins,shrink,cap); pred=apply(st,bx,bb,bv); rr=robust_metrics(oof[dev]); pp=protocol_metrics(oof[dev]); ratio=pred['energy'][bv]/bb[bv]
   sp=local_spike_score(pred['energy']); s26=local_spike_score(pred['energy'],width=5,lo=2450,hi=2750); rank=pd.Series(bb[bv]).corr(pd.Series(pred['energy'][bv]),method='spearman'); ta=time_audit(bg.t.to_numpy(float),bb,pred['energy'])
   row={"id":"{}_b{}_s{}_c{}".format(sn,bins,int(shrink*10),int(cap*10000)),"set":sn,"features":"|".join(names),"bins":bins,"shrink":shrink,"cap":cap,
    "sigma_gain":1-pp['sigma_median']/bp['sigma_median'],"r68_gain":1-rr['r68']/br['r68'],"r90_gain":1-rr['r90']/br['r90'],"protocol":pp['protocol_successes'],
    "bg_applied":float(np.mean(pred['applied'])),"bg_ood":float(np.mean(pred['ood'])),"bg_strong":float(np.mean(pred['strong'])),"rank":float(rank),
    "spike":float(sp['ratio']/max(bs['ratio'],1)),"spike2600":float(s26['ratio']/max(b26['ratio'],1)),"q01":float(np.quantile(ratio,.01)),"q99":float(np.quantile(ratio,.99)),"time_span":ta['median_ratio_span'],"time_coverage":ta['applied_fraction_span'],"blocks":ta['supported_blocks']}
   checks=[row['sigma_gain']>0,row['r68_gain']>=-.001,row['r90_gain']>=-.002,row['protocol']>=10,row['bg_applied']>=.1,row['bg_ood']<=.2,row['bg_strong']<=.85,row['rank']>=.9995,row['spike']<=1.1,row['spike2600']<=1.1,row['q01']>=.992,row['q99']<=1.008,row['time_span']<=.002,row['time_coverage']<=.25,row['blocks']>=8]
   row['safe']=bool(all(checks)); row['passes']=sum(checks); rows.append(row); saved[row['id']]=(st,names)
 tab=pd.DataFrame(rows).sort_values(['safe','passes','sigma_gain'],ascending=[False,False,False]); tab.to_csv(out/'response_map_candidates.csv',index=False); safe=tab[tab.safe]; win=(safe.iloc[0] if len(safe) else tab.iloc[0]).to_dict(); st,names=saved[win['id']]
 hp=apply(st,cal[names].to_numpy(float)[hold],cb[hold],hv); hbr=robust_metrics(cb[hold][hv]); hnr=robust_metrics(hp['energy'][hv]); hbp=protocol_metrics(cb[hold][hv]); hnp=protocol_metrics(hp['energy'][hv]); h={"n":int(np.sum(hv)),"sigma_gain":1-hnp['sigma_median']/hbp['sigma_median'],"r68_gain":1-hnr['r68']/hbr['r68'],"r90_gain":1-hnr['r90']/hbr['r90'],"applied":float(np.mean(hp['applied']))}; status='holdout_pass' if len(safe) and h['sigma_gain']>0 and h['r68_gain']>=-.002 and h['r90_gain']>=-.003 else 'diagnostic_only'
 bundle={"version":"v18_smooth_response_map","status":status,"features":names,"state":st,"dev":win,"holdout":h,"warning":"Cross-run and multi-line validation required."}; joblib.dump(bundle,out/'candidate_v18.joblib'); (out/'summary.json').write_text(json.dumps({"status":status,"winner":win,"holdout":h,"safe":int(len(safe)),"total":int(len(tab))},indent=2,default=lambda v:v.item()),encoding='utf8'); print(tab.head(12).to_string(index=False)); print('HOLDOUT',json.dumps(h)); print('STATUS',status)

if __name__=='__main__': main()
