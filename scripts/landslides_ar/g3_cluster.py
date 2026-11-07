import numpy as np, pandas as pd
O='work/out/g1'
L=pd.read_parquet(f'{O}/ls_ar.parquet')
win={'NVE':('2000','2022'),'USGS':('1979','2024'),'ITALICA':('1996','2021'),'GLC':('2007','2017'),'GFLD':('2004','2017')}
out=[]
for s,(a,b) in win.items():
    g=L[(L.src==s)&(L.date>=a)&(L.date<=b+'-12-31')].copy()
    g['dn']=(g.date-pd.Timestamp('1979-01-01')).dt.days
    # cluster size: landslides of same source within 3x3 cells and +-1 day
    cnt=g.groupby(['dn','ci','cj']).size()
    d=cnt.to_dict()
    def nb(r):
        t=0
        for dd in (-1,0,1):
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    t+=d.get((r.dn+dd,r.ci+di,(r.cj+dj)%360),0)
        return t
    g['cl']=[nb(r) for r in g.itertuples()]
    g['clb']=pd.cut(g.cl,[0,1,4,19,1e9],labels=['1','2-4','5-19','>=20'])
    t=g.groupby('clb',observed=True).agg(n=('ar','size'),ar=('ar','mean'),pc=('pc','mean'))
    t['OR']=(t.ar/(1-t.ar))/(t.pc/(1-t.pc)); t['src']=s
    out.append(t.reset_index())
    g[['src','date','ci','cj','cl']].to_parquet(f'{O}/cl_{s}.parquet')
R=pd.concat(out); print(R.round(3).to_string())
