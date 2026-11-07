# landslide events: connected components (same source; <=1 day apart and within 3x3 1-deg cells); AR share by event size
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
O='work/out/g1'
L=pd.read_parquet(f'{O}/ls_ar.parquet')
L=L[~((L.src=='USGS')&L.inv.isin(['USGS Earthquake-Triggered Ground Failure','USGS Seismogenic Mass Movements']))]
win={'NVE':('2000','2022'),'USGS':('1979','2024'),'ITALICA':('1996','2021'),'GLC':('2007','2017'),'GFLD':('2004','2017')}
EV=[]
for s,(a,b) in win.items():
    g=L[(L.src==s)&(L.date>=a)&(L.date<=b+'-12-31')].copy().reset_index(drop=True)
    g['dn']=(g.date-pd.Timestamp('1979-01-01')).dt.days
    # nodes = unique cell-days
    cd=g.groupby(['dn','ci','cj']).agg(n=('lat','size'),ar=('ar','max'),pc=('pc','mean')).reset_index()
    idx={k:i for i,k in enumerate(zip(cd.dn,cd.ci,cd.cj))}
    r=[];c=[]
    for i,(d,x,y) in enumerate(zip(cd.dn,cd.ci,cd.cj)):
        for dd in (0,1):
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    j=idx.get((d+dd,x+di,(y+dj)%360))
                    if j is not None and j!=i: r.append(i);c.append(j)
    A=coo_matrix((np.ones(len(r)),(r,c)),shape=(len(cd),len(cd)))
    nc,lab=connected_components(A,directed=False)
    cd['ev']=lab
    e=cd.groupby('ev').agg(n=('n','sum'),ncell=('n','size'),d0=('dn','min'),d1=('dn','max'),ar_any=('ar','max'),ar_frac=('ar','mean'),pc=('pc','mean'),ci=('ci','mean'),cj=('cj','mean')).reset_index()
    e['src']=s; EV.append(e)
E=pd.concat(EV); E.to_parquet(f'{O}/events.parquet')
E['sz']=pd.cut(E.n,[0,1,4,19,99,1e9],labels=['1','2-4','5-19','20-99','>=100'])
E['arm']=E.ar_frac>=0.5
t=E.groupby(['src','sz'],observed=True).agg(events=('n','size'),slides=('n','sum'),ar_share=('arm','mean'),pc=('pc','mean'))
pd.set_option('display.width',200); print(t.round(3).to_string())
# share of landslides that belong to AR events, by source
print((E.n*E.arm).groupby(E.src).sum()/E.n.groupby(E.src).sum())
