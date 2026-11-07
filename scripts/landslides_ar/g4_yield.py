import numpy as np, pandas as pd
O='work/out/g1'
A=pd.read_parquet(f'{O}/ar_episodes.parquet')
L=pd.read_parquet(f'{O}/ls_ar.parquet')
L=L[~((L.src=='USGS')&L.inv.isin(['USGS Earthquake-Triggered Ground Failure','USGS Seismogenic Mass Movements']))]
EV=pd.read_parquet(f'{O}/events.parquet')
win={'NVE':('2000','2022'),'USGS':('1979','2022'),'ITALICA':('1996','2021'),'GLC':('2007','2017'),'GFLD':('2004','2017')}
out=[]
for s,(a,b) in win.items():
    g=L[(L.src==s)&(L.date>=a)&(L.date<=b+'-12-31')]
    cells=set(zip(g.ci,g.cj))
    ep=A[(A.t0>=a)&(A.t1<=b+'-12-31T18')]
    ep=ep[[ (x,y) in cells for x,y in zip(ep.ci,ep.cj)]].copy()
    # landslide counts per cell-day
    cd=g.groupby(['ci','cj','date']).size()
    cdd={}
    for (x,y,d),n in cd.items(): cdd.setdefault((x,y),[]).append((d,n))
    nls=[]
    for x,y,t0,t1 in zip(ep.ci,ep.cj,ep.t0,ep.t1):
        lst=cdd.get((x,y),[]); lo=t0.floor('D'); hi=t1.floor('D')+pd.Timedelta(days=1)
        nls.append(sum(n for d,n in lst if lo<=d<=hi))
    ep['nls']=nls
    t=ep.groupby('cat').agg(eps=('k','size'),p_any=('nls',lambda v:(v>0).mean()),mean_n=('nls','mean'),p5=('nls',lambda v:(v>=5).mean()))
    t['src']=s; out.append(t.reset_index())
    ep.to_parquet(f'{O}/ep_{s}.parquet')
R=pd.concat(out); pd.set_option('display.width',200); print(R.round(4).to_string())
