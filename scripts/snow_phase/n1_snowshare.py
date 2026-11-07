"""Temperature of AR extreme-precipitation cell-days, 1979-2022 (1 deg ERA5 via WB2): per year and cell, counts of
extreme-precipitation cell-days (2-day precip >= local P99), of those under an AR (day or day before), and of the AR ones
by daily-mean T2m class (<-4, -4..0, 0..4, >4 C). Cold season (Oct-Mar N, Apr-Sep S) and all year."""
import numpy as np, pandas as pd, sys
E='work/data/era5'; I='work/data/arcat'; O='work/out/g1'; N='work/out/n1'
Y0,Y1=int(sys.argv[1]),int(sys.argv[2]) if len(sys.argv)>2 else (1979,2022)
p99=np.load(f'{O}/r2_p99.npy'); lat=np.arange(-70,76)
years=list(range(Y0,Y1+1)); ny=len(years)
# [year, season(0 cold,1 warm), class] ; classes: 0 all extreme, 1 AR extreme, 2 AR<-4, 3 AR -4..0, 4 AR 0..4, 5 AR >4 ; plus T sum of AR extreme
cnt=np.zeros((ny,2,6,146,360),np.int16); tsum=np.zeros((ny,2,146,360),np.float32)
prev_pr=None; prev_ar=None
for k,y in enumerate(years):
    pr=np.nan_to_num(np.load(f'{E}/pr_{y}.npy').astype(np.float32)); nd=pr.shape[0]
    ar=np.unpackbits(np.load(f'{I}/ard_{y}.npy'),axis=0)[:nd].astype(bool)
    t=np.load(f'{E}/t2m_{y}.npy')[:,0].astype(np.float32)
    r2=pr.copy(); r2[1:]+=pr[:-1]
    if prev_pr is not None: r2[0]+=prev_pr
    w=ar.copy(); w[1:]|=ar[:-1]
    if prev_ar is not None: w[0]|=prev_ar
    prev_pr=pr[-1]; prev_ar=ar[-1]
    ex=r2>=p99[None]; mo=pd.date_range(f'{y}-01-01',periods=nd).month.values
    coldN=np.isin(mo,[10,11,12,1,2,3])
    cold=np.where(lat[None,:,None]>=0, coldN[:,None,None], ~coldN[:,None,None]); cold=np.broadcast_to(cold,(nd,146,360))
    arex=ex&w
    for s,sm in ((0,cold),(1,~cold)):
        cnt[k,s,0]+=(ex&sm).sum(0); a=arex&sm; cnt[k,s,1]+=a.sum(0)
        cnt[k,s,2]+=(a&(t<-4)).sum(0); cnt[k,s,3]+=(a&(t>=-4)&(t<0)).sum(0); cnt[k,s,4]+=(a&(t>=0)&(t<4)).sum(0); cnt[k,s,5]+=(a&(t>=4)).sum(0)
        tsum[k,s]+=np.where(a,t,0).sum(0)
    print(y,flush=True)
np.savez_compressed(f'{N}/arex_temp_{Y0}_{Y1}.npz',cnt=cnt,tsum=tsum,years=np.array(years))
print('saved')
