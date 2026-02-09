# %%
import pandas as pd

import pycefeax as pfx
import numpy as np
from matplotlib.path import Path



# %%


data=pd.read_csv("../catalog_CentralItaly_v1.csv",delimiter=" ")

# Define the polygon coordinates
poly = np.array([[42.2, 13.8], [42.0, 13.5], [42.6, 12.9], [42.72, 13.15], [42.63, 13.35]])

# Create a Path object for the polygon
polygon_path = Path(poly)

# Check which points are inside the polygon
points = np.column_stack((data['Lat'], data['Lon']))
index = polygon_path.contains_points(points)

# Filter the data based on the index
data = data[index]

data['OT'] = data['date(y-m-d'] + ' ' + data['h:m:s)']

data_fx=pfx.make_df(data['OT'],data['Lat'],data['Lon'],data['Depth(km)'],data['Mw'])

preprocess, features =pfx.get_feature(data_fx)



