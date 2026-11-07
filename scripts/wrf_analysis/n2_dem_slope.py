"""Steep-terrain exposure of the WRF events: Copernicus GLO-90 DEM tiles (S3) -> slope -> share of steep pixels per 3 km cell ->
steep land area inside the heavy-rain areas of the control and warmer runs."""
import numpy as np, pandas as pd, glob, os, subprocess, sys
import tifffile
from scipy.spatial import cKDTree
WP='work/data/wrf/post_fig'; D='work/data/dem'; os.makedirs(D,exist_ok=True); EDGE=10
EVS=['NOR','BC','ORE','LIG','RSB']
def tile_name(la,lo):
    ns='N' if la>=0 else 'S'; ew='E' if lo>=0 else 'W'
    return f'Copernicus_DSM_COG_30_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM'
def get_tile(la,lo):
    n=tile_name(la,lo); f=f'{D}/{n}.tif'
    if os.path.exists(f) and os.path.getsize(f)>1000: return f
    url=f'https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif'
    r=subprocess.run(['curl','-sS','-L','--max-time','120','-o',f+'.tmp',url],capture_output=True,text=True)
    if os.path.exists(f+'.tmp') and os.path.getsize(f+'.tmp')>1000 and open(f+'.tmp','rb').read(4) in (b'II*\x00',b'MM\x00*'):
        os.rename(f+'.tmp',f); return f
    if os.path.exists(f+'.tmp'): os.remove(f+'.tmp')
    return None   # ocean tile or missing
rows=[]
for ev in EVS:
    c=np.load(glob.glob(f'{WP}/{ev}_*/{ev}_*_ctl.npz')[0]); w=np.load(glob.glob(f'{WP}/{ev}_*/{ev}_*_pgw585.npz')[0])
    lat=c['lat'][EDGE:-EDGE,EDGE:-EDGE]; lon=c['lon'][EDGE:-EDGE,EDGE:-EDGE]; A=c['cell_area_km2'][EDGE:-EDGE,EDGE:-EDGE]; land=c['landmask'][EDGE:-EDGE,EDGE:-EDGE]>0.5
    Lc=c['liq_max48h'][EDGE:-EDGE,EDGE:-EDGE]; Lw=w['liq_max48h'][EDGE:-EDGE,EDGE:-EDGE]
    la0,la1=int(np.floor(lat.min())),int(np.floor(lat.max())); lo0,lo1=int(np.floor(lon.min())),int(np.floor(lon.max()))
    steep15=[]; steep25=[]; cc=[]
    for la in range(la0,la1+1):
        for lo in range(lo0,lo1+1):
            f=get_tile(la,lo)
            if f is None: continue
            z=tifffile.imread(f).astype(np.float32); n=z.shape[0]   # 1200 x 1200 (90 m)
            dy=111320.0/n; dx=111320.0*np.cos(np.deg2rad(la+0.5))/z.shape[1]
            gy,gx=np.gradient(z,dy,dx); s=np.degrees(np.arctan(np.hypot(gx,gy)))
            # aggregate to 0.025 deg blocks (40 x 40 blocks of 30 px)
            nb=40; by=z.shape[0]//nb; bx=z.shape[1]//nb; s=s[:nb*by,:nb*bx]
            s15=(s>=15).reshape(nb,by,nb,bx).mean((1,3)); s25=(s>=25).reshape(nb,by,nb,bx).mean((1,3))
            bl=la+1-(np.arange(nb)+0.5)/nb; bo=lo+(np.arange(nb)+0.5)/nb   # tile rows go from north to south
            BL,BO=np.meshgrid(bl,bo,indexing='ij'); steep15.append(s15.ravel()); steep25.append(s25.ravel()); cc.append(np.c_[BL.ravel(),BO.ravel()])
    S15=np.concatenate(steep15); S25=np.concatenate(steep25); C=np.concatenate(cc)
    t=cKDTree(np.c_[lat.ravel(),lon.ravel()]); d,i=t.query(C); ok=d<0.03
    f15=np.zeros(lat.size); f25=np.zeros(lat.size); cnt=np.zeros(lat.size)
    np.add.at(f15,i[ok],S15[ok]); np.add.at(f25,i[ok],S25[ok]); np.add.at(cnt,i[ok],1)
    with np.errstate(invalid='ignore'): f15=(f15/cnt).reshape(lat.shape); f25=(f25/cnt).reshape(lat.shape)
    f15=np.nan_to_num(f15); f25=np.nan_to_num(f25)
    np.savez_compressed(f'{D}/{ev}_steep.npz',f15=f15,f25=f25,lat=lat,lon=lon)
    p99=np.nanpercentile(Lc[land],99)
    for thr_name,thr in (('100 mm',100),('50 mm',50),('control P99',p99)):
        mc=land&(Lc>=thr); mw=land&(Lw>=thr)
        for nm,fr in (('slope>=15',f15),('slope>=25',f25)):
            rows.append(dict(event=ev,threshold=thr_name,steep=nm,area_c=float((A*mc).sum()),area_w=float((A*mw).sum()),steep_c=float((A*fr*mc).sum()),steep_w=float((A*fr*mw).sum()),steep_new=float((A*fr*(mw&~mc)).sum()),steep_land=float((A*fr*land).sum())))
    print(ev,'tiles done; steep15 land km2 %.0f'%(A*f15*land).sum(),flush=True)
R=pd.DataFrame(rows); R.to_csv(f'{D}/steep_exposure.csv',index=False); print(R.round(0).to_string())
