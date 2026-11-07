# Logistic models for an event with >=5 landslides: AR flag, log footprint (log1p of land cells with 2-day rain >= local P99 within 4 deg),
# log peak 2-day rain at the event centre relative to the local P99, month fixed effects (and inventory fixed effects when pooled).
import sys; sys.path.insert(0,'fig_code')
from arcommon import pool_mask
import pandas as pd, numpy as np, statsmodels.formula.api as smf
EF=pd.read_parquet('work/out/g1/events_fp.parquet'); EF['arm']=(EF.ar_frac>=0.5).astype(int); EF['big']=(EF.n>=5).astype(int)
EF['lfp']=np.log1p(EF.fp); EF['lpk']=np.log(EF.pk.clip(0.05))
EF['mo']=((pd.Timestamp('1979-01-01')+pd.to_timedelta(EF.d0,'D')).dt.month).astype(str)
rows=[]
for s in ['NVE','USGS','ITALICA','GLC','GFLD','ALL']:
    q=EF[pool_mask(EF)] if s=='ALL' else EF[EF.src==s]
    fe='+C(src)' if s=='ALL' else ''
    for name,f in [('raw','big~arm+C(mo)'+fe),('+peak','big~arm+lpk+C(mo)'+fe),('+footprint','big~arm+lfp+C(mo)'+fe),('+both','big~arm+lfp+lpk+C(mo)'+fe)]:
        m=smf.logit(f,q).fit(disp=0); b=m.params['arm']; ci=m.conf_int().loc['arm']
        rows.append((s,name,len(q),q.big.sum(),np.exp(b),np.exp(ci[0]),np.exp(ci[1])))
R=pd.DataFrame(rows,columns=['src','model','n','nbig','OR','lo','hi']); print(R.round(2).to_string()); R.to_csv('work/out/g1/logit_footprint.csv',index=False)
