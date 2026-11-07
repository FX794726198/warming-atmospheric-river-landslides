import numpy as np, pandas as pd, os
E='work/data/era5/'
Y0,Y1=1959,2022
def cell_series(ci,cj):
    """ci,cj arrays -> (times6h, ivt[ncell,nt]), (days, pr[ncell,nd])"""
    ci=np.asarray(ci); cj=np.asarray(cj)
    iv=[];pr=[]
    for y in range(Y0,Y1+1):
        a=np.load(f'{E}ivt_{y}.npy',mmap_mode='r'); iv.append(np.asarray(a[:,ci,cj]).T)
        b=np.load(f'{E}pr_{y}.npy',mmap_mode='r'); pr.append(np.asarray(b[:,ci,cj]).T.astype(np.float32))
    t6=pd.date_range(f'{Y0}-01-01',f'{Y1}-12-31T18',freq='6h'); td=pd.date_range(f'{Y0}-01-01',f'{Y1}-12-31',freq='D')
    IV=np.concatenate(iv,1); PR=np.concatenate(pr,1)
    assert IV.shape[1]==len(t6) and PR.shape[1]==len(td), (IV.shape,len(t6),PR.shape,len(td))
    return t6,IV,td,PR
def ar_events(t6,ivt,thr=250,maxgap=1):
    """point AR events: runs of ivt>=thr, merging gaps of <=maxgap steps. returns DataFrame"""
    on=ivt>=thr
    idx=np.flatnonzero(on)
    if len(idx)==0: return pd.DataFrame()
    br=np.flatnonzero(np.diff(idx)>maxgap+1)
    st=np.r_[idx[0],idx[br+1]]; en=np.r_[idx[br],idx[-1]]
    rows=[]
    for s,e in zip(st,en):
        seg=ivt[s:e+1]; dur=(e-s+1)*6
        mx=int(seg.max())
        cat=1+min(4,int((mx-250)//250))
        if dur>=48: cat+=1
        elif dur<24: cat-=1
        rows.append((t6[s],t6[e],dur,mx,float(seg.mean()),int(np.clip(cat,0,5))))
    return pd.DataFrame(rows,columns=['start','end','dur_h','ivt_max','ivt_mean','cat'])
