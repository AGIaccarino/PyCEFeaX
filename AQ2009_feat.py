# %%
import pandas as pd

import pycefeax as pfx
import numpy as np
from matplotlib.path import Path

# %%
data=pd.read_csv("./AQ2009_catalogue.csv",index_col=0)
data["source_Mw"]=1.066*data["source_magnitude_DD"]-0.164 #ML to Mw Gasperini et al 2013
data_fx=pfx.make_df(data["source_origin_time_DD"], data["source_latitude_deg_DD"], data["source_longitude_deg_DD"], data["source_depth_km_DD"],data["source_Mw"])

preprocess, features =pfx.get_feature(data_fx)

