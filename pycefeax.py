# 
"""
Created on Tue Jan  7 14:31:15 2025

@author: johna

Version 1.0.0
"""

# 
import numpy as np
import pandas as pd

from obspy.geodetics.base import gps2dist_azimuth
from scipy.spatial import ConvexHull, Delaunay
from geopy.distance import geodesic

from itertools import combinations
import statsmodels.api as sm
from sklearn.decomposition import PCA
import json
import time
from joblib import Parallel, delayed
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator 
from datetime import datetime, timedelta
import warnings


def get_feature(data):
    #Loading parameters...
    with open('input.json', 'r') as f:
        input_par = json.load(f)
    period = (datetime.fromisoformat(input_par['period']["begin"]),
              datetime.fromisoformat(input_par['period']["end"]))
    nlw = input_par['event-window']

    # select only events with depth greater than 0
    data = data[data['source_depth_km'] > 0]
    
    # Select events by time
    data.loc[:, 'source_origin_time'] = pd.to_datetime(data['source_origin_time'])
    data = data[(data['source_origin_time'] >= period[0]) & (data['source_origin_time'] < period[1])]
    
    # Sort and drop duplicates
    data = data.sort_values('source_origin_time').drop_duplicates()
    savetag=input_par["save_tag"] 
    output_path='output_'+savetag
   
   # Preprocessing:
    print("\n\nPreprocessing...\n")
    
    data, preprocess=dataset_preprocessing(data, input_par)

    # Features Computation:
    
    if not input_par["only_preprocess"]:
        print("\n\nFeatures Computation\n")
        def process_window(data, ia, input_par):
            nw2 = ia + input_par['event-window']
            return features_computation(data.iloc[ia:nw2,], input_par)
        
        print("\n\nParallelizing...\n")
        
        if not input_par["Bootstrap"]:
            
            features = Parallel(n_jobs=-1)(delayed(process_window)(data, ia, input_par) for ia in range(data.shape[0] - nlw + 1))
        else:
            features = []
            sig_features = []
            sig_features_d = []
            last_time = time.time()
            Nmean=100
            avg_per_window = 0

            # Check if there is a saved state to resume from
            save_path = 'features_checkpoint_'+savetag+'.pkl'
            if os.path.exists(save_path) and input_par["checkpoint"]:
                with open(save_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                start_index = checkpoint['start_index']
                features = checkpoint['features']
                sig_features = checkpoint['sig_features']
                sig_features_d = checkpoint['sig_features_d']
            else:
                start_index = 0
                features = []
                sig_features = []
                sig_features_d = []
            print(data.shape)
            print(start_index)
            for ia in range(start_index, data.shape[0] - nlw + 1):

                elapsed = time.time() - last_time            
                last_time = time.time()
                if ia> start_index:
                    fac=1/min(Nmean, ia)
                    avg_per_window = avg_per_window*(1-fac) + elapsed*fac           
                    remaining = avg_per_window * (data.shape[0] - nlw + 1 - ia ) - 1  

                    hrs = int(remaining // 3600)
                    mins = int((remaining % 3600) // 60)
                    secs = int(remaining % 60)

                    percent=ia/(data.shape[0] - nlw + 1)*100
                    print(
                    "\r"
                    + f"Progress:  {ia+1}/{data.shape[0] - nlw + 1} ({percent:.3f}%)"
                    + f" — ETA: {hrs:02d}:{mins:02d}:{secs:02d} {avg_per_window:.4f}s/window",
                    end="",
                    flush=True  
                    )
                else:
                    print(
                    "\r"
                    + f"Progress:  {ia+1}/{data.shape[0] - nlw + 1}"
                    ,end="",flush=True)
                
                ### Parallelized

                features_temp = Parallel(n_jobs=-1)(
                  delayed(features_computation)(
                       data.iloc[np.random.choice(range(ia, ia + nlw), nlw - 1, replace=True).tolist() + [ia + nlw - 1]], 
                       input_par
                   ) 
                   for _ in range(input_par["Bootstrap_repetitions"])
               )
                
                ###
                features_delta = delta_features_boot(data.iloc[ia:ia + nlw], input_par)

                median_feature = {field: np.nanmedian([feature[field] for feature in features_temp]) for field in list(features_temp[0].keys())[5:]}
                median_feature_d = {field: np.nanmedian([feature[field] for feature in features_delta]) for field in features_delta[0].keys()}
                
                # PCA analysis (bootstrap)

                features.append({"source_origin_time": data.source_origin_time.iloc[ia + nlw - 1], "source_latitude_deg": data.source_latitude_deg.iloc[ia + nlw - 1], "source_longitude_deg": data.source_longitude_deg.iloc[ia + nlw - 1], 
                                 "source_depth_km": data.source_depth_km.iloc[ia + nlw - 1], "source_magnitude": data.source_magnitude.iloc[ia + nlw - 1],  **median_feature, **median_feature_d, **get_pca(data.iloc[ia:ia + nlw],input_par)})
                

                sig_features.append({field: np.nanstd([feature[field] for feature in features_temp]) for field in list(features_temp[0].keys())[7:]})
                sig_features_d.append({field: np.nanstd([feature[field] for feature in features_delta]) for field in features_delta[0].keys()})

                # Save progress every 10 iterations
                if (ia + 1) % 10 == 0 and input_par["checkpoint"]:
                    checkpoint = {
                        'start_index': ia + 1,
                        'features': features,
                        'sig_features': sig_features,
                        'sig_features_d': sig_features_d
                    }
                    with open(save_path, 'wb') as f:
                        pickle.dump(checkpoint, f)
                    

            sig_features = pd.DataFrame(sig_features)
            sig_features.columns = ["sig_" + col for col in sig_features.columns]
            sig_features_d = pd.DataFrame(sig_features_d)
            sig_features_d.columns = ["sig_" + col for col in sig_features_d.columns]

            features = pd.DataFrame(features)
            n_features = len(list(features.columns))

            features = pd.concat([features.iloc[:,:n_features-12], sig_features, sig_features_d,features.iloc[:,n_features-12:]], axis=1)

            n_features = len(list(features.columns))

            # Rearranging columns
            
            column_order = (0, 1,2,3,4,5,6) + tuple(7 + i // 2 + (i % 2) * ((n_features - 19) // 2) for i in range(n_features - 19))+tuple(range(n_features - 12, n_features))
            features=features.iloc[:, list(column_order)]
            
            features.to_csv(output_path + "/features.csv", index=False)
    return preprocess, features

# 


# 

def dataset_preprocessing(data, input_par):
    # Preprocessing:
    Mc = input_par['Mc']

    Mmax = input_par['Mmax']
    savetag=input_par["save_tag"] 
    output_path='output_'+savetag
   

    GR_bin=input_par["GR_bin"]
    Mw1B = data['source_magnitude'].to_numpy()
    
    agr_values = []
    bgr_values = []
    Mc_values = []
    print("\nGR Computation...\n")

    
    if input_par["Bootstrap"]:
        GR_reps = input_par["Bootstrap_repetitions"]
        for irep in range(GR_reps):
            print("\r",irep+1,"/",GR_reps,end="")

            index=np.random.choice(range(len(Mw1B)),len(Mw1B),replace=True)
            a_temp, b_temp, Mc_temp = calculate_gr(Mw1B[index], GR_bin)
            agr_values.append(a_temp)
            bgr_values.append(b_temp)
            Mc_values.append(Mc_temp)

        agr = np.nanmedian(agr_values)
        sig_agr = np.nanstd(agr_values)
        bgr = np.nanmedian(bgr_values)
        sig_bgr = np.nanstd(bgr_values)
        Mc1 = np.nanmedian(Mc_values)
        sig_Mc = np.nanstd(Mc_values)
        
    else:
        agr, bgr, Mc1 = calculate_gr(Mw1B, GR_bin)
        sig_agr = 0
        sig_bgr = 0
        sig_Mc = 0
    if Mc is None:
        Mc = Mc1
    
    data = data[(Mw1B >= Mc)]
    Mw1B = Mw1B[Mw1B >= Mc]
    if Mmax is not None:
        data = data[(Mw1B <= Mmax)]
        Mw1B = Mw1B[Mw1B <= Mmax]
    
    data.reset_index(drop=False, inplace=True)

    
    print("\nNearest Neighbor analysis...")
    
    T, R, LEta,Dat = eta_computation(data,input_par["ETA-Computation"])
 
    preprocess=pd.DataFrame({'Date':Dat,'T':T,'R':R,'LEta':LEta,'agr':agr,'sig_agr':sig_agr, 'bgr':-bgr,'sig_bgr':sig_bgr,'Mc':Mc1,'sig_Mc':sig_Mc})
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    preprocess.to_csv(output_path + "/preprocess.csv", index=False)
    return data, preprocess



def features_computation(data, input_par):
    # Features computation:

    """
    Compute the features for data
    """
    step_she=input_par["Ngrid_Shannon_Entropy"]
    Dmax=None
    mu=input_par["mu_GPa"]*10**9

    perc=input_par["perc"]
    r_cint=input_par["r_cint_km"]
    data = data.sort_values(by='source_origin_time')

    ref_source_latitude_deg=data.source_latitude_deg.iloc[-1]
    ref_source_longitude_deg=data.source_longitude_deg.iloc[-1]

    x,y=convert_to_local(data.source_latitude_deg, data.source_longitude_deg, 
        ref_source_latitude_deg, ref_source_longitude_deg)
    z=data.source_depth_km*1000

    # Data spatial selection and r_perc
    P=np.vstack((x, y, z)).T
    r_perc,_=findcilinder(P, perc)
    
    IDperc=whoin(P, r_perc)
    data=data.iloc[IDperc,:]
    Nev=data.shape[0]


    if (Dmax is None) or Dmax>Nev:
        Dmax=Nev
    
    #DT and time rate
    DTk=(data.source_origin_time.iloc[-1]-data.source_origin_time.iloc[0]).total_seconds()
    try :
        rate2k=Nev/DTk
    except ZeroDivisionError:
        rate2k=np.nan
    
   

    #latlon coordinates to local
    x,y=convert_to_local(data.source_latitude_deg, data.source_longitude_deg, 
                         ref_source_latitude_deg, ref_source_longitude_deg)
    z=data.source_depth_km*1000


    # Area and Volume of Convex Hull
    delaunay_2d = Delaunay(np.vstack((x, y)).T)
    convex_hull_2d = ConvexHull(delaunay_2d.points)
    areak = convex_hull_2d.volume  # Area in m² (volume attribute for 2D corresponds to area)
    
    
    delaunay_3d = Delaunay(np.vstack((x , y, z)).T)
    convex_hull_3d = ConvexHull(delaunay_3d.points)
    VOL = convex_hull_3d.volume  # Volume in m³
    
    # Event spatial density
    try:
        ratek=Nev/areak

    except ZeroDivisionError:
        ratek=np.nan

 
    # GR computation
    GR_bin=np.round(input_par["GR_bin"],decimals=5)
    if sum(np.logical_not( np.isnan(data.source_magnitude)))>=50:
       
        aw1, bw1, Mcw1 = calculate_gr(data.source_magnitude,GR_bin)
    else:
        aw1=np.nan
        bw1=np.nan
        Mcw1=np.nan
        
    #  Moment rate Catalli et al. 2008
    bw1=-bw1
    Moik=10**(1.5*Mcw1+9.1);
    
    MwkM=max(data.source_magnitude);
    try:
        Mr=ratek*Moik*(bw1/(1.5-bw1))*(10**((1.5-bw1)*(MwkM-Mcw1))-1)
    except ZeroDivisionError:
        Mr=np.nan

    #  Correlation integral and Fractal dimension

    index=np.arange(data.shape[0])
  
    Combo =np.array( list(combinations(index, 2)))

    [Rij,Nij,cor_int]=myfract3(x/1000,y/1000,data.source_depth_km,Combo,Dmax,r_cint)
    log_Rij = np.log10(Rij)
    log_Nij = np.log10(Nij)
    X = sm.add_constant(log_Rij)
    
    rlm_model = sm.RLM(log_Nij, X, M=sm.robust.norms.TukeyBiweight(c=2))
    rlm_results = rlm_model.fit()
    Dc1=rlm_results.params[1]


    #SEff

    Mo1B=10**(1.5*(data.source_magnitude)+9.1)
    Mo1B = Mo1B[~np.isnan(Mo1B)]

    R3=((3/4)*VOL/np.pi)
    try:
        Streff=(7/16)*(sum(Mo1B))/((10**6)*R3)# in MPa
    except ZeroDivisionError:
        Streff=np.nan
    

    #  Kostrov strain (Kostrov, 1974)
    try:
        Dei=(1/(2*mu*VOL))*(sum(Mo1B))#;  11
    except ZeroDivisionError:
        Dei=np.nan

    #  Shannon Entropy
    E1B=1.96*data.source_magnitude+2.05
    E1B = E1B[~np.isnan(E1B)]

    Eslk=10**E1B;

    
    
    [ShEn,_]=shannon(data.source_latitude_deg,data.source_longitude_deg,Eslk,step_she)

    # 
    if not input_par["Bootstrap"]:
        # M and dt correlations (no bootstrap)
        dm = data['source_magnitude'].diff()
        dt=data['source_origin_time'].diff().dt.total_seconds()
        ddt=dt.diff()
        dm = dm[~np.isnan(dm)]
        dt = dt[~np.isnan(dt)]
        ddt = ddt[~np.isnan(ddt)]

        mcor=dm.sum()/(Nev-1)
        dtcor=ddt.sum()/(Nev-2)
        m2cor=dm.pow(2).sum()/(Nev-1)
        dt2cor=ddt.pow(2).sum()/(Nev-2)

        
        CoVdt= np.nanstd(dt)/np.nanmean(dt)
        
        # PCA analysis (no bootstrap)
        pca = PCA(n_components=3)
        pca.fit_transform(np.vstack((x, y, z)).T)
        PCA_eigval = pca.explained_variance_ratio_
        PCA_avec=pca.components_
    else:
        NUniq=data.drop_duplicates().shape[0]



    features={"source_origin_time":data.source_origin_time.iloc[-1],"source_latitude_deg":data.source_latitude_deg.iloc[-1],
              "source_longitude_deg":data.source_longitude_deg.iloc[-1],"source_depth_km": data.source_depth_km.iloc[-1],
              "source_magnitude":data.source_magnitude.iloc[-1],"N_window":Nev}
    if input_par["Bootstrap"]:
        features.update({"N_unique":NUniq})
    features.update({"DT":DTk,"rho_s":ratek,"rho_t":rate2k,"A":areak,"V":VOL,
                     "a_GR":aw1,"b_GR":bw1,"M_c":Mcw1,"M_r":Mr,"Dc":Dc1,"S_eff":Streff,"DEps":Dei,
                     "h":ShEn,"r_perc":r_perc/1000})
    for i, r in enumerate(r_cint):
        features[f'C_r_{r}km'] = cor_int[i] 
    if not input_par["Bootstrap"]:
        features.update({"CoVdt":CoVdt,"mcor":mcor,"dtcor":dtcor,
                "m2cor":m2cor,"dt2cor":dt2cor} )
        for i in range(3):
            features[f'PCA_eigval_{i+1}'] = PCA_eigval[i]
            for j, c in enumerate(['x', 'y', 'z']):
                features[f'PCA_{i+1}{c}'] = PCA_avec[i, j]
    
        
    return features

# 
def delta_features_boot(data,input_par):
    # M and dt correlations (bootstrap)

    ref_source_latitude_deg=data.source_latitude_deg.iloc[-1]
    ref_source_longitude_deg=data.source_longitude_deg.iloc[-1]
    perc=input_par["perc"]
    x,y=convert_to_local(data.source_latitude_deg, data.source_longitude_deg, 
        ref_source_latitude_deg, ref_source_longitude_deg)
    z=data.source_depth_km*1000

    P=np.vstack((x, y, z)).T
    r_perc,_=findcilinder(P, perc)
    
    IDperc=whoin(P, r_perc)
    data=data.iloc[IDperc,:]
    Nev=data.shape[0]

    Nbootstrap=input_par["Bootstrap_repetitions"]
        
    dm1 = data['source_magnitude'].diff()
    dt1=data['source_origin_time'].diff().dt.total_seconds()
   
    features=[]

    for _ in range(Nbootstrap):
        index=np.random.choice(range(len(data)),len(data),replace=True)
        dm = dm1.iloc[index]
        dt=dt1.iloc[index]

        ddt=dt.diff()
        dm = dm[~np.isnan(dm)]
        dt = dt[~np.isnan(dt)]
        ddt = ddt[~np.isnan(ddt)]

        mcor=dm.sum()/(Nev-1)
        dtcor=ddt.sum()/(Nev-2)
        m2cor=dm.pow(2).sum()/(Nev-1)
        dt2cor=ddt.pow(2).sum()/(Nev-2)


        CoVdt=np.nanstd(dt)/np.nanmean(dt)

        features_temp={"CoVdt":CoVdt,"mcor":mcor,"dtcor":dtcor,
                "m2cor":m2cor,"dt2cor":dt2cor}
        features.append(features_temp)

 
    return features

def get_pca(data,input_par):
    #PCA function
    ref_source_latitude_deg=data.source_latitude_deg.iloc[-1]
    ref_source_longitude_deg=data.source_longitude_deg.iloc[-1]
    perc=input_par["perc"]
    x,y=convert_to_local(data.source_latitude_deg, data.source_longitude_deg, 
        ref_source_latitude_deg, ref_source_longitude_deg)
    z=data.source_depth_km*1000

    P=np.vstack((x, y, z)).T
    r_perc,_=findcilinder(P, perc)
    
    IDperc=whoin(P, r_perc)
    data=data.iloc[IDperc,:]
    
    
    x,y=convert_to_local(data.source_latitude_deg, data.source_longitude_deg, 
                        ref_source_latitude_deg, ref_source_longitude_deg)
    z=-data.source_depth_km.to_numpy()
    x,y=x/1000, y/1000                                    

    pca = PCA(n_components=3)
    pca.fit_transform(np.vstack((x, y, z)).T)
    PCA_eigval = pca.explained_variance_ratio_
    PCA_avec=pca.components_
    features={}
    for i in range(3):
        features[f'PCA_eigval_{i+1}'] = PCA_eigval[i]
        for j, c in enumerate(['x', 'y', 'z']):
            features[f'PCA_{i+1}{c}'] = PCA_avec[i, j]
    return features

# 
def shannon(lata, lona, Esla,step_she):
    """
    Calculate Shannon entropy for given inputs.

    Parameters:
        lata (array-like): Latitude values.
        lona (array-like): Longitude values.
        ETOT (float): Total energy.
        Esla (array-like): Energy at each location.
        
    Returns:
        tuple: (ShEn, idx)
            ShEn (float): Shannon entropy value.
            idx (numpy array): Index values for grid points.
    """
    ETOT=sum(Esla)
    
    xxgrd=np.linspace(min(lona),max(lona),step_she,endpoint=True)
    yygrd=np.linspace(min(lata),max(lata),step_she,endpoint=True)
    # Initialize index arrays
    indx = np.zeros(len(lata), dtype=int)
    indy = np.zeros(len(lata), dtype=int)
    idx = np.zeros(len(lata), dtype=int)

    # Compute grid indices
    for ik in range(len(lata)):
        if lona.iloc[ik] < max(xxgrd):
            indx[ik] = np.searchsorted(xxgrd, lona.iloc[ik], side='right')
        else:
            indx[ik] = step_she

        if lata.iloc[ik] < max(yygrd):
            indy[ik] = np.searchsorted(yygrd, lata.iloc[ik], side='right')
        else:
            indy[ik] = step_she

        idx[ik] = np.ravel_multi_index((indx[ik] - 1, indy[ik] - 1), (step_she, step_she))

    # Unique indices
    unique_idx, inverse_idx = np.unique(idx, return_inverse=True)

    # Normalize energy
    ekn = (1 / (step_she **2)) * np.log(1 / (step_she **2))

    # Calculate Shannon entropy components
    ek = []
    for ei in unique_idx:
        matching_indices = np.where(idx == ei)[0]
        esla_sum = np.nansum(Esla.iloc[matching_indices])
        ek.append((esla_sum / ETOT) * np.log(esla_sum / ETOT))

    ek = np.array(ek)
    H = -np.nansum(ek)
    He = -(ekn * step_she **2)
    try:
        ShEn = H / He
    except ZeroDivisionError:
        ShEn = np.nan
    
    # print(xxgrd,yygrd)
    return ShEn, idx

# 
def eta_computation( data,input_par):
    
    
    """
    Nearest-neighbor analysis
     
     """
    datB=data['source_origin_time']
    elat=data['source_latitude_deg']
    elon=data['source_longitude_deg']
    edep=data['source_depth_km']
    Mw1B=data['source_magnitude']

    nwin=input_par["window-size"]
    npshift=input_par["window-step"]
    Dc = input_par["Dc"]
    bbw = input_par["b-value"]
    rmax = input_par["rmax_km"]

    selT = []
    selR = []
    selEta = []
    selDat=[]
    selSon=[]
    npts=datB.shape[0]
    
    winbeg=range(0,npts-nwin+1,npshift)
    iwin=range(0,len(winbeg))
    
    for ia,nw1 in zip(iwin,winbeg):
        print(f"\rProgress: {(ia+1) / len(winbeg) * 100:.2f}%", end="")
        nw2 = nw1 + nwin - 1
        if nw2 > npts:
            nw2 = npts


        Eta = []
        R = []
        T = []
        for iz in range(nw1, nw2):
            
            dista = geodesic((elat[iz], elon[iz]), (elat[nw2], elon[nw2])).kilometers
            Ripo = np.sqrt(dista ** 2 + (edep[iz] - edep[nw2]) ** 2)

            tdifZ = (datB[nw2 ] - datB[iz]).total_seconds()

            R_value = (Ripo ** Dc) * 10 ** (-0.5 * bbw * Mw1B[iz])
            T_value = tdifZ * 10 ** (-(1 - 0.5) * bbw * Mw1B[iz])

            if tdifZ > 0 and Ripo < rmax and Ripo != 0:
                Eta.append(R_value * T_value)
            else:
                Eta.append(float('inf'))

            R.append(R_value)
            T.append(T_value)
        if Eta:

            index1 = np.argmin(np.log10(Eta))

            selT.append(T[index1])
            selR.append(R[index1])
            selEta.append(Eta[index1])
        else:
            selT.append(float('inf'))
            selR.append(float('inf'))
            selEta.append(float('inf'))
        selDat.append(datB[nw2])
        selSon.append(nw1+index1)


    return np.array(selT), np.array(selR), np.array(selEta) , np.array(selDat)

# 
def convert_to_local(source_latitude_deg, source_longitude_deg, ref_source_latitude_deg, ref_source_longitude_deg):
    """
        Convert Latlon coordinates to local ones 
    """
    distances = [gps2dist_azimuth(source_latitude_deg_i, source_longitude_deg_i, ref_source_latitude_deg, ref_source_longitude_deg)[0] for source_latitude_deg_i, source_longitude_deg_i in zip(source_latitude_deg, source_longitude_deg)]
    azimuths = [gps2dist_azimuth(source_latitude_deg_i, source_longitude_deg_i, ref_source_latitude_deg, ref_source_longitude_deg)[1] for source_latitude_deg_i, source_longitude_deg_i in zip(source_latitude_deg, source_longitude_deg)]
    lly = [dist * np.cos(np.radians(az)) for dist, az in zip(distances, azimuths)]
    llx = [dist * np.sin(np.radians(az)) for dist, az in zip(distances, azimuths)]
    return np.array(llx), np.array(lly)
# 
def calculate_gr(mag,mybin = 0.1):
    # GR computation

    decimal_places = len(str(mybin).split('.')[1])
    if isinstance(mag, pd.Series):
        mag = mag.values
    mag = np.array([np.format_float_positional(m+10**-5, precision=decimal_places) for m in mag], dtype=float)
    # Gutenberg-Richter cumulative frequency
    imagi = np.arange(max(mag) - 0.2, min(mag), -mybin)
    neve = [np.nansum(mag >= threshold) for threshold in imagi]
    # Histogram for initial Mc calculation (maximum curvature)
    mmin=np.floor(min(mag))-1
    mmax=np.ceil(max(mag))+1
    hist, bin_edges = np.histogram(mag, bins=np.arange(mmin-mybin/2, mmax + mybin*3/2, mybin))
    max_idx = np.argmax(hist)
    fMc_start = (bin_edges[max_idx]+bin_edges[max_idx+1]) / 2

    m_data = []

    # eigvaluate possible Mc values
    for n_cnt in np.round(np.arange(fMc_start - 0.9, fMc_start + 1.5, mybin),decimal_places):
        selected = mag > (n_cnt - mybin / 2)
        n_events = np.nansum(selected)

        if n_events >= 25:
            _, fB_value, _, _ = calc_bmemag(mag[selected])
            start_mag = n_cnt
            v_mag = np.arange(start_mag, 15, mybin)
            v_number=10 ** (np.log10(n_events) - fB_value * (v_mag - start_mag))

            v_number = np.round(v_number)
            # Find last bin with events
            last_event_bin = np.nanmax(np.where(v_number > 0)[0]) if np.any(v_number > 0) else len(v_number) - 1
            pm = v_mag[:last_event_bin + 1]
            v_number = v_number[:last_event_bin + 1]

            PM2=np.array([ np.format_float_positional(pm1,decimal_places+1) for pm1 in np.append(pm, pm[-1] + mybin)-mybin/2], dtype=float)
            b_val, _ = np.histogram(mag[selected], bins=PM2)
            b3 = np.flip(np.cumsum(np.flip(b_val)))
            try:
                res2 = np.nansum(np.abs(b3 - v_number)) / np.nansum(b3) * 100
            except ZeroDivisionError:   
                res2 = np.nan
            m_data.append([n_cnt, res2])
            
        else:
            m_data.append([n_cnt, np.nan])

    m_data = np.array(m_data)
 
    # Determine Mc values based on residual thresholds
    fMc = np.nan
    for threshold in [10, 15, 20, 25]:
        idx = np.where(m_data[:, 1] < threshold)[0]
        if len(idx) > 0:
            fMc = m_data[idx[0], 0]
            break

    if np.isnan(fMc):
        fMc = np.nan

    # Compute final parameters
    selected = mag >= fMc
    if sum(selected) == 0:

        return np.nan, np.nan, np.nan
    _, fB_value, _, fA_value = calc_bmemag(mag[selected], mybin)

    B = -fB_value
    A = fA_value
 
 
    return A, B, fMc
def calc_bmemag(mag, f_binning=0.1):
    """
    calculate Maximum Likelihood b-value.

    Parameters:
        mag (array-like): Vector of magnitudes.
        f_binning (float): Bin size in magnitude (default 0.1).

    Returns:
        tuple: (f_mean_mag, f_b_value, f_std_dev, f_a_value)
            f_mean_mag (float): Mean magnitude.
            f_b_value (float): b-value.
            f_std_dev (float): Standard deviation of b-value.
            f_a_value (float): a-value.
    """
    n_len = len(mag)
    f_min_mag = np.nanmin(mag)
    f_mean_mag = np.nanmean(mag)
    
    # calculate the b-value (maximum likelihood)
    try:
        f_b_value = (1 / (f_mean_mag - (f_min_mag - (f_binning / 2)))) * np.log10(np.exp(1))
    except ZeroDivisionError:
        f_b_value = np.nan
    
    # calculate the standard deviation
    
    f_std_dev = np.nansum((mag - f_mean_mag) ** 2) / (n_len * (n_len - 1))
    f_std_dev = 2.30 * np.sqrt(f_std_dev) * f_b_value ** 2
    
    # calculate the a-value
    f_a_value = np.log10(n_len) + f_b_value * f_min_mag

    return f_mean_mag, f_b_value, f_std_dev, f_a_value


def myfract3(xa, ya, depa, Combo, Dmax,Di1):
    #Correlation integral and Fractal dimension
    step = np.arange(2, Dmax+1, 1, dtype=int)
    
    
    xcom1 = xa[Combo[:, 0]]
    xcom2 = xa[Combo[:, 1]]
    
    ycom1 = ya[Combo[:, 0]]
    ycom2 = ya[Combo[:, 1]]
    
    depa1 = depa.iloc[Combo[:, 0]].values
    depa2 = depa.iloc[Combo[:, 1]].values

    
    coords1 = np.vstack((xcom1, ycom1)).T
    coords2 = np.vstack((xcom2, ycom2)).T


    Repi = np.linalg.norm(coords1 - coords2, axis=1)
    
    
    Ripo = np.sqrt(Repi**2 + (depa1 - depa2)**2)
    Xkm = Ripo
    
    Di = Dmax / step
    Ne = len(Xkm)
    Nij = np.array([(1 / (Ne**2)) * np.nansum(Xkm < d) for d in Di])
    Rij = Di
    cor_int = np.array([(1 / (Ne**2)) * np.nansum(Xkm < d) for d in Di1])
    

    
    valid_indices = ~np.isinf(np.log10(Rij)) & ~np.isinf(np.log10(Nij))
    Rij = Rij[valid_indices]
    Nij = Nij[valid_indices]
    
    
    return Rij, Nij,cor_int

def findcilinder(P,perc):
    # Find radius of cilinder containing perc percentage of events
    dhyp = np.linalg.norm(P - P[-1, :], axis=1)
    r=(dhyp.max()+dhyp.min())/2
    Nin=np.round(P.shape[0]*perc)
    # print(Nin*1.01, Nin*0.99)
    dr=(dhyp.max()+dhyp.min())/4
    repet=0
    while howmanyin(P,r)!=Nin:
        if howmanyin(P,r)>Nin:
            r=r-dr
        else:
            r=r+dr
        dr=dr/2
        repet=repet+1
        if repet>10000:
            break
    if howmanyin(P,r)<2:
        print("Warning: less than 2 events in the cilinder")
    
    P=P[whoin(P,r),:]
    depi = np.linalg.norm(P[:,:2] - P[-1,:2], axis=1)
    dz=np.abs(P[:,2]-P[-1,2])
    r=max(depi.max(),dz.max())
            
    return r,P
def howmanyin(P,r):
    depi = np.linalg.norm(P[:,:2] - P[-1,:2], axis=1)
    dz=np.abs(P[:,2]-P[-1,2])
    Nin=np.nansum(np.logical_and(depi<=r,dz<=r))
     
    return Nin
def whoin(P, r):
    # try:
    depi = np.linalg.norm(P[:, :2] - P[-1, :2], axis=1)
    dz = np.abs(P[:, 2] - P[-1, 2])
    ID = np.logical_and(depi <= r, dz <= r)
    return ID

def make_df(source_origin_time,source_latitude_deg,source_longitude_deg,source_depth_km,source_magnitude):
    # Create DataFrame from source parameters
    data={"source_origin_time":source_origin_time,"source_latitude_deg":source_latitude_deg,"source_longitude_deg":source_longitude_deg,"source_depth_km":source_depth_km,"source_magnitude":source_magnitude}
    df=pd.DataFrame(data=data)
    return df



def plot_allfeatures(features,  start_date=datetime(1900,1,1), end_date=datetime(2099,12,31)):
    """
    Plot all features in one Figure
    """
    warnings.filterwarnings("ignore")

    features = features[(features['source_origin_time'] >= start_date) & (features['source_origin_time'] <= end_date)]


    plt.figure();
    plt.clf()

   
    features['PCA_linearity']=(features['PCA_eigval_1']-features['PCA_eigval_2'])/features['PCA_eigval_1']

    featuresname = features.columns[7:51:2]
    featnames=['DT', r'$\rho_s$', r'$\rho_r$', 'A', 'V', r'$a_{GR}$',  r'$b_{GR}$', r'$M_c$', r'$M_r$', 'Dc',
        r'$S_{eff}$', r'$\Delta\epsilon$', 'h', r'$r_{perc}$', r'$C_r(1 km)$', r'$C_r(2 km)$', r'$C_r(5 km)$',
        r'$CoV_{dt}$', r'$\langle m \rangle$', r'$\langle dt \rangle$', r'$\langle m^2 \rangle$', r'$\langle dt^2 \rangle$']
    # Create a figure with 5x4 subplots
    fig, axes = plt.subplots(8,4, figsize=(20, 20))
    axes = axes.flatten()

    # Loop through each feature and plot it in a subplot
    for i, feat in enumerate(featuresname):
        ax = axes[i]
        ax.errorbar(features['source_origin_time'], features[feat], yerr=features['sig_' + feat], fmt='b.', label='features with error', zorder=1)
        ax.plot(features['source_origin_time'], features[feat].rolling(window=20, min_periods=1).mean(), 'r', label='Rolling Mean', linewidth=2, zorder=2)
        if i in [1,8, 10, 11]:  # Set y-axis to log scale for specific indices
            ax.set_yscale('log')
        
      
        ax.set_title(f"Feature: {featnames[i]}")
        ax.set_ylabel(featnames[i])

    
    
    ax=axes[len(featnames)]
    ax.plot(features['source_origin_time'], features['PCA_eigval_1'], 'b.', label='PCA1', zorder=1)
    ax.plot(features['source_origin_time'], features['PCA_eigval_2'], 'g.', label='PCA2', zorder=2)
    ax.plot(features['source_origin_time'], features['PCA_eigval_3'], 'r.', label='PCA3', zorder=3)
    ax.set_ylabel("PCA eigenvalues")
    
    ax.set_title("Feature: PCA eigenvalues")
    
    
    pca_ratio1 = features['PCA_eigval_1'] / features['PCA_eigval_2']
    pca_ratio2 = features['PCA_eigval_1'] / features['PCA_eigval_3']
    pca_ratio3 = features['PCA_eigval_2'] / features['PCA_eigval_3']


    ax=axes[len(featnames)+1]
    ax.plot(features['source_origin_time'], pca_ratio1, 'b.', label='PCA1/PCA2', zorder=1)
    ax.plot(features['source_origin_time'], pca_ratio1.rolling(window=20, min_periods=1).mean() , 'r-', label='Rolling Mean', zorder=2)

    ax.set_ylabel("PCA eigenvalues")
    
    ax.set_title(f"Feature: PCA1/PCA2")
    
    
    ax=axes[len(featnames)+2]

    ax.plot(features['source_origin_time'], pca_ratio2, 'b.', label='PCA1/PCA3', zorder=1)
    ax.plot(features['source_origin_time'], pca_ratio2.rolling(window=20, min_periods=1).mean() , 'r-', label='Rolling Mean', zorder=2)

    ax.set_ylabel("PCA eigenvalues")
    
    ax.set_title(f"Feature: PCA1/PCA3")
    
    ax=axes[len(featnames)+3]

    ax.plot(features['source_origin_time'], pca_ratio3, 'b.', label='PCA2/PCA3', zorder=1)
    ax.plot(features['source_origin_time'], pca_ratio3.rolling(window=20, min_periods=1).mean() , 'r-', label='Rolling Mean', zorder=2)

    ax.set_ylabel("PCA eigenvalues")
    
    ax.set_title(f"Feature: PCA2/PCA3")
    
    



    
    azimuth1 = np.arctan2(features['PCA_1y'], features['PCA_1x']) % np.pi
    
    

    ax=axes[len(featnames)+4]

    ax.plot(features['source_origin_time'], azimuth1/np.pi*180, 'b.', label='Azimuth PCA1', zorder=1)
    ax.plot(features['source_origin_time'], azimuth1.rolling(window=20, min_periods=1).mean()/np.pi*180 , 'r-', label='Rolling Mean', zorder=2)
    

    ax.set_ylabel("PCA1 Azimuth")
    
    ax.set_title(f"Feature: PCA1 Azimuth")
    
    if start_date.time() != datetime.min.time():
        first_tick = datetime.combine(start_date.date() + timedelta(days=1), datetime.min.time())
    else:
        first_tick = start_date

    
    total_days = (end_date - first_tick).days
    max_ticks = 5
    dt_days = max(1, int(np.ceil(total_days / (max_ticks - 1))))

    
    tick_dates = []
    current_date = first_tick
    while current_date <= end_date:
        tick_dates.append(current_date)
        current_date += timedelta(days=dt_days)

    
    Nplot=len(featnames)+5
    for j in range(Nplot):
        ax=axes[j]
        ax.grid(True)
        ax.legend()

        ax.set_xlim([start_date, end_date])
        ax.set_ylim(None)

        ax.xaxis.set_major_locator(FixedLocator([mdates.date2num(d) for d in tick_dates]))  # Usando FixedLocator corretto
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=0)
    # Hide any unused subplots
    for j in range(Nplot, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    warnings.filterwarnings("default")

    return

