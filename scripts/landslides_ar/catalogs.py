# Build harmonised daily landslide tables: source, date, lat, lon, type
import pandas as pd, numpy as np
D='work/data/landslide/data/'
def nve():
    n=pd.read_csv(D+'nve/nve_skredhendelser_layer0.csv',low_memory=False)
    n['date']=pd.to_datetime(n.skredtidspunkt,unit='ms',errors='coerce')
    ok=n.noyskredtidspunkt.isin(['Eksakt','Ukjent nar pa dagen','30 min','12 timer','4 timer','1 time','15 min','6 timer','10 min','5 min','1 min','2 min','1 dag'])
    n=n[ok & n.skredtype.isin([140,141,142,143,144,145]) & (n.date>='1979-01-01')]
    # date: NVE times stored in UTC ms; convert to local date approx (UTC+1)
    n['date']=(n.date+pd.Timedelta(hours=1)).dt.floor('D')
    typ=np.where(n.skredtype==142,'debris_flow',np.where(n.skredtype==141,'quick_clay','soil_slide'))
    return pd.DataFrame(dict(src='NVE',date=n.date.values,lat=n.lat.values,lon=n.lon.values,typ=typ))
def usgs():
    u=pd.read_csv(D+'usgs/US_Landslide_v3point.csv',low_memory=False)
    u['d0']=pd.to_datetime(u.Date_Min.str.strip(),errors='coerce',format='%Y/%m/%d %H:%M:%S.%f')
    u['d1']=pd.to_datetime(u.Date_Max.str.strip(),errors='coerce',format='%Y/%m/%d %H:%M:%S.%f')
    same=u.d1.isna()|((u.d1-u.d0)<=pd.Timedelta(days=1))
    u=u[same & u.d0.notna() & (u.d0>='1979-01-01')]
    u=u[~u.LS_Type.astype(str).str.contains('Embank|Rockfall|Rock fall|Sinkhole',case=False)]
    return pd.DataFrame(dict(src='USGS',date=u.d0.dt.floor('D').values,lat=u.Lat_N.values,lon=u.Lon_W.values,typ=u.LS_Type.astype(str).values,inv=u.Inventory.astype(str).values))
def italica():
    it=pd.read_csv(D+'italica/ITALICA-v2.csv',sep=';')
    it=it[it.temporal_accuracy.isin(['T1','T2','T3'])]
    it['date']=pd.to_datetime(dict(year=it.year,month=it.month,day=it.day),errors='coerce')
    return pd.DataFrame(dict(src='ITALICA',date=it.date.values,lat=it.lat.values,lon=it.lon.values,typ=it.landslide_type.values))
def glc():
    g=pd.read_csv(D+'glc/Global_Landslide_Catalog_Export_rows.csv',low_memory=False)
    g['date']=pd.to_datetime(g.event_date,format='%m/%d/%Y %I:%M:%S %p',errors='coerce').dt.floor('D')
    g=g[g.landslide_trigger.isin(['rain','downpour','continuous_rain','monsoon','tropical_cyclone','flooding','snowfall_snowmelt'])]
    return pd.DataFrame(dict(src='GLC',date=g.date.values,lat=g.latitude.values,lon=g.longitude.values,typ=g.landslide_category.astype(str).values,fat=g.fatality_count.values,trig=g.landslide_trigger.values))
def gfld():
    g=pd.read_csv(D+'gfld/GFLD_fatal_landslide_points_2004_2017.csv',low_memory=False)
    g['date']=pd.to_datetime(dict(year=g.Year,month=g.Month,day=g.Day),errors='coerce')
    g=g[g.Trigger_.astype(str).str.contains('rain|Rain|storm|monsoon|flood|typhoon|cyclone|hurricane',regex=True)]
    return pd.DataFrame(dict(src='GFLD',date=g.date.values,lat=g.Latitude.values,lon=g.Longitude.values,typ='fatal',fat=g.Fatalities.values,trig=g.Trigger_.values))
if __name__=='__main__':
    allc=[]
    for f in [nve,usgs,italica,glc,gfld]:
        d=f(); d=d.dropna(subset=['date','lat','lon']); print(f.__name__,len(d),d.date.min(),d.date.max()); allc.append(d)
    A=pd.concat(allc,ignore_index=True)
    A['lon360']=np.mod(A.lon,360)
    A['ci']=np.round(A.lat+70).astype(int); A['cj']=np.round(A.lon360).astype(int)%360
    A=A[(A.ci>=0)&(A.ci<146)]
    A.to_parquet('work/out/landslides_all.parquet')
    print(A.groupby('src').size())
    print(A.groupby('src').apply(lambda x: x[['ci','cj']].drop_duplicates().shape[0]))
