# For each landslide source: cell-day panel 1979-2022 over cells with landslides; AR window flag (d or d-1), 2-day ERA5 rain, 5-day antecedent rain
import numpy as np, pandas as pd, sys
I='work/data/arcat'; E='work/data/era5'; O='work/out/g1'
L=pd.read_parquet(f'{O}/ls_ar.parquet'); L=L[L.date<='2022-12-31']
cells=L[['ci','cj']].drop_duplicates().values
ci,cj=cells[:,0],cells[:,1]
AR=[];PR=[]
for y in range(1979,2023):
    nd=366 if y%4==0 else 365
    a=np.unpackbits(np.load(f'{I}/ard_{y}.npy'),axis=0)[:nd]
    AR.append(a[:,ci,cj]); PR.append(np.load(f'{E}/pr_{y}.npy',mmap_mode='r')[:,ci,cj].astype(np.float32))
AR=np.concatenate(AR).astype(bool); PR=np.nan_to_num(np.concatenate(PR))
days=pd.date_range('1979-01-01','2022-12-31')
W=AR.copy(); W[1:]|=AR[:-1]
R2=PR.copy(); R2[1:]+=PR[:-1]
cs=np.vstack([np.zeros((1,PR.shape[1])),np.cumsum(PR,0)])
A5=np.zeros_like(PR); A5[7:]=cs[6:-1-0][:len(PR)-7]-cs[1:len(PR)-6]  # sum d-6..d-2
np.savez_compressed(f'{O}/panel.npz',ci=ci,cj=cj,W=W,R2=R2.astype(np.float16),A5=A5.astype(np.float16),AR=AR)
print(W.shape,flush=True)
