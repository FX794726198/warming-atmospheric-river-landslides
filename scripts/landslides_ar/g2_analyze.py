import numpy as np, pandas as pd
O='work/out/g1'
P=np.load(f'{O}/panel.npz'); W=P['W']; R2=P['R2'].astype(np.float32); A5=P['A5'].astype(np.float32)
ci,cj=P['ci'],P['cj']; key={(a,b):k for k,(a,b) in enumerate(zip(ci,cj))}
days=pd.date_range('1979-01-01','2022-12-31'); mon=days.month.values
L=pd.read_parquet(f'{O}/ls_ar.parquet'); L=L[L.date<='2022-12-31']
win={'NVE':('2000','2022'),'USGS':('1979','2022'),'ITALICA':('1996','2021'),'GLC':('2007','2017'),'GFLD':('2004','2017')}
# percentile edges per cell for R2 (wet-day based on all days)
qs=[0,50,75,90,95,97.5,99,99.5,100]
rows=[]
for s,(a,b) in win.items():
    g=L[(L.src==s)&(L.date>=a)&(L.date<=b+'-12-31')].drop_duplicates(['ci','cj','date'])
    t0=np.searchsorted(days,np.datetime64(a+'-01-01')); t1=np.searchsorted(days,np.datetime64(b+'-12-31'))+1
    ks=np.array([key[(x,y)] for x,y in zip(g.ci,g.cj)]); ts=np.searchsorted(days,g.date.values)
    Y=np.zeros((t1-t0,len(ci)),bool); Y[ts-t0,ks]=True
    cols=np.unique(ks)
    Y=Y[:,cols]; w=W[t0:t1,cols]; r=R2[t0:t1,cols]; ant=A5[t0:t1,cols]
    edges=np.percentile(r,qs,axis=0); edges[-1]+=1e-3
    bins=np.zeros_like(r,dtype=np.int8)
    for k in range(1,len(qs)-1): bins+= (r>=edges[k][None,:])
    p_ls=w[Y].mean(); p_all=w.mean(); heavy=bins>=4; p_heavy=w[heavy].mean()
    # landslide-days share of heavy rain
    ls_heavy=heavy[Y].mean()
    # MH OR stratified by cell x rainbin x month
    m=np.broadcast_to(mon[t0:t1,None],w.shape)
    strat=(np.arange(len(cols))[None,:]*1000+bins.astype(int)*20+m).ravel()
    df=pd.DataFrame({'s':strat,'y':Y.ravel(),'w':w.ravel()})
    tab=df.groupby(['s','w']).y.agg(['sum','size']).unstack('w').fillna(0)
    a1=tab[('sum',True)]; n1=tab[('size',True)]; a0=tab[('sum',False)]; n0=tab[('size',False)]
    N=n1+n0; num=(a1*(n0-a0)/N).sum(); den=(a0*(n1-a1)/N).sum()
    # same without rain bins (cell x month)
    strat2=(np.arange(len(cols))[None,:]*100+m).ravel()
    df['s']=strat2; tab=df.groupby(['s','w']).y.agg(['sum','size']).unstack('w').fillna(0)
    a1=tab[('sum',True)]; n1=tab[('size',True)]; a0=tab[('sum',False)]; n0=tab[('size',False)]
    N=n1+n0; num2=(a1*(n0-a0)/N).sum(); den2=(a0*(n1-a1)/N).sum()
    # AR share among landslide days within rain bins
    bl=bins[Y]; wl=w[Y]
    byb={k:(round(wl[bl==k].mean(),2),round(w[bins==k].mean(),2),int((bl==k).sum())) for k in range(8)}
    rows.append(dict(src=s,n_lsdays=int(Y.sum()),cells=len(cols),p_ls=p_ls,p_all=p_all,p_heavy95=p_heavy,ls_on_heavy=ls_heavy,OR_cellmonth=num2/den2,OR_rainmatched=num/den))
    print(s,byb,flush=True)
pd.set_option('display.width',200)
print(pd.DataFrame(rows).round(3).to_string())
