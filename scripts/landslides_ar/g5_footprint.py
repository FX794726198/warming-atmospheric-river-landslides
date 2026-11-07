import numpy as np, pandas as pd
E='work/data/era5'; O='work/out/g1'
lsm=np.load('work/out/a2/lsm.npy'); print(lsm.shape, lsm.dtype, lsm.mean())
p99=np.load(f'{O}/r2_p99.npy')
EV=pd.read_parquet(f'{O}/events.parquet'); EV=EV[EV.d1<=(pd.Timestamp('2022-12-31')-pd.Timestamp('1979-01-01')).days]
days=pd.date_range('1979-01-01','2022-12-31')
PR={y:np.load(f'{E}/pr_{y}.npy',mmap_mode='r') for y in range(1979,2023)}
def day(dn):
    d=days[dn]; return np.nan_to_num(np.asarray(PR[d.year][d.dayofyear-1],dtype=np.float32))
cache={}
def r2(dn):
    if dn not in cache:
        if len(cache)>400: cache.clear()
        cache[dn]=day(dn)+(day(dn-1) if dn>0 else 0)
    return cache[dn]
R=4; land=lsm>0.5
fp=[];pk=[]
EV=EV.sort_values('d0')
for e in EV.itertuples():
    i=int(round(e.ci)); j=int(round(e.cj))
    ii=slice(max(0,i-R),min(146,i+R+1)); jj=[(j+o)%360 for o in range(-R,R+1)]
    ex=np.zeros((ii.stop-ii.start,len(jj)),bool); mx=0
    for dn in range(int(e.d0),int(e.d1)+1):
        a=r2(dn)[ii][:,jj]; q=p99[ii][:,jj]
        ex|=(a>=q)&land[ii][:,jj]
        mx=max(mx,float(r2(dn)[i,j]/max(p99[i,j],1)))
    fp.append(int(ex.sum())); pk.append(mx)
EV['fp']=fp; EV['pk']=pk
EV.to_parquet(f'{O}/events_fp.parquet')
EV['arm']=EV.ar_frac>=0.5
EV['sz']=pd.cut(EV.n,[0,1,4,19,1e9],labels=['1','2-4','5-19','>=20'])
print(EV.groupby(['src','arm']).agg(n=('n','size'),fp=('fp','mean'),pk=('pk','median'),size=('n','mean')).round(2))
print(EV.groupby(['src','sz'],observed=True).agg(fp=('fp','mean'),pk=('pk','median')).round(2))
