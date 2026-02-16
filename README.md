# PyCEFeaX 
## Python Code for Earthquake Features eXtraction
### Overview
PyCEFeaX is a Python-based software package designed to extract physics-oriented features from seismic catalogs.

The code provides a unified and reproducible framework to characterize the spatio-temporal, energetic, and statistical properties of seismicity, with a particular focus on the analysis of seismic sequences preceding and following moderate-to-large earthquakes.

PyCEFeaX has been developed to support feature-based studies of earthquake preparation, fault interaction, fluid diffusion, and aftershock evolution, and to provide physically interpretable inputs for machine learning applications in seismology.

The methodology and scientific background of PyCEFeaX are described in detail in:

Iaccarino, A.G., Picozzi, M. — PyCEFeaX: a Python Code for Earthquake Features eXtraction (submitted).
________________________________________
### Key Features
PyCEFeaX computes a comprehensive set of 26 catalog-driven features, including:

- Temporal organization
1. Window duration (DT)
2.	Event rate (ρr)
3.	Coefficient of variation of inter-event times (CoVdt)
4.	Linear and quadratic trends of inter-event times
-	Spatial organization
5.	Area (A) and Volume (V)
6.	Spatial density (ρs)
7.	Correlation integral (Cr)
8.	Fractal dimension (Dc)
9.	Percentage cylinder radius (rperc)
10.	Spatial PCA (eigenvalues and eigenvectors)
-	Energetic and stress-related properties
11.	Moment rate (Mr)
12.	Effective stress (Seff)
13.	Kostrov strain (Δε)
-	Statistical descriptors
14.	Gutenberg–Richter parameters (aGR, bGR)
15.	Magnitude of completeness (Mc)
16.	Normalized Shannon entropy (h)

All features are computed using moving windows of events, optionally combined with bootstrap resampling to quantify feature variability.
________________________________________
### Code Structure
PyCEFeaX is organized into two main modules:
#### Preprocessing

The preprocessing module performs:
 1.	Gutenberg–Richter analysis
     - Estimation of aGR, bGR, and magnitude of completeness (Mc)
     -	Based on maximum-likelihood estimation
     -	Selected routines adapted from the ZMAP toolbox (Wiemer, 2001) and reimplemented in Python
 2. Nearest-neighbor distance analysis
     -	Computation of rescaled time (T), space (R), and generalized distance (η)
     -	Based on Zaliapin et al. (2008)
     -	Allows separation of background and clustered seismicity
 
This step enables users to restrict feature computation to:
- events above Mc,
- background seismicity,
- clustered seismicity,
- or the full catalog.

#### Feature Computation
Features are computed on a moving window of Nwin events and associated with the last event of each window, allowing a causal interpretation of feature evolution.
Key options include:
-	selection of a percentage of events (Ev%) spatially associated with the last event,
-	bootstrap resampling (Nboot realizations),
-	user-defined physical parameters (e.g., rigidity μ).
________________________________________
### Input Data
PyCEFeaX requires a seismic catalog containing:
-	Origin time
-	Hypocentral coordinates (latitude, longitude, depth)
-	Magnitude (moment magnitude Mw)

Catalogs can include foreshocks, aftershocks, or background seismicity.

Examples provided in the repository include datasets related to the 2009 Mw 6.1 L’Aquila earthquake.
________________________________________
### Output
The code outputs:
-	Time series of features associated with catalog events
-	Bootstrap-based uncertainty estimates (median and standard deviation)
-	Intermediate preprocessing results (GR parameters, η distributions)
Outputs are suitable for:
-	direct scientific interpretation,
-	comparative studies across regions or sequences,
-	machine learning workflows.
________________________________________
### Installation
Clone the repository:
    
    git clone https://github.com/AGIaccarino/PyCEFeaX.git
    
    cd PyCEFeaX

Install required dependencies (recommended via virtual environment):

    pip install -r requirements.txt

with Conda, create the environment using:

    conda env create -f environment.yml

Once installed all the requirements, install pycefeax:

    pip install -e .
    
________________________________________

### Functions
     make_df(source_origin_time,source_latitude_deg,source_longitude_deg,source_depth_km,source_magnitude): return data

This function creates a Pandas dataframe (data) in a SeisBench-like format. The inputs are Pandas Series that contains, in the order: Origin time, Latitude, Longitude, Depth (km) and Magnitude for all the events of the catalogue.

     get_feature(data): return preprocess, features

get_feature is the main function of PyCEFeaX. It will read the configuration file and start Preprocessing and Feature computation (when requested) on “data”. “data” must be the dataframe created with make_df. The outputs are “preprocess”, a dataframe contains the results from the GR and nearest-neighbor analyses, and “features”, a dataframe that contains the features for the analyzed windows. The function also saves both outputs in a folder named “output/$save-tag$” with save-tag that can be set in the input.json file. 

    plot_allfeatures(features,  start_date=datetime(1900,1,1), end_date=datetime(2099,12,31)): return None

plot_allfeatures is function that plots all the features in time in one figure. The input are the features computed with get_feature, and the starting and ending dates that must be in datetime format.
________________________________________
### Configuration file (Input.json)
The configuration file, “input.json”, contains all the parameters to configurate PyCEFeaX. Here it is an example:
{
    
    "only_preprocess": false,
only_preprocess controls whether PyCEFeaX has to perform only the preprocessing module. When “false”, PyCEFeaX performs both Preprocessing and Features Computation.

    "checkpoint": true,
When checkpoint is true, PyCEFeaX saves the results in a temporary file each 10 windows. This is done to prevent completely losing the computed features due to system failures. When PyCEFeaX is run again with the same input.json, it will start from the last saved checkpoint.

    "save_tag": "FOREAQ",

save_tag is a string that will identify the output directory and files.

    "Mc":null,

Mc is the completeness magnitude of the dataset or, in general, the minimum magnitude you want to analyze. If null, PyCEFeaX will use the Mc value obtained from the GR computation

    "Mmax" : null,

Mmax is maximum magnitude that PyCEFeaX will analyze. If null, there will be no upper limit for magnitude.


    "GR_bin":0.1,
GR_bin is the magnitude binning used to compute the GR.

    "Bootstrap": true,

Bootstrap is a boolean variable that controls whether PyCEFeaX will perform a bootstrap analysis for the features computation.

    "Bootstrap_repetitions": 200,

Bootstrap_repetitions is the number of bootstrap samplings. If Bootstrap is false, this parameter is unused.

    "event-window" : 250,

event-window is the number of events in the moving window used to compute the features.

    "perc":0.8,

perc is the ratio of events in the window that will be effectively used to compute the features. The events will be selected finding the smallest cylinder centered on the last event in the window that contains round(perc* event-window) events.

    "r_cint_km":[1,2,5],

r_cint_km is a list of values that represent the radius at which the correlation integral will be saved. 

    
    "period" :{
            "begin": "2004-04-05T00:00:00",
            "end": "2009-04-06T01:33:00"

        begin and end are dates that limit the analysis, PyCEFeaX will consider only the events happened between begin and end. Both dates should be string in the format “yyyy-mm-ddTHH:MM:SS”.

        },
    "Ngrid_Shannon_Entropy"  : 20,
Ngrid_Shannon_Entropy is the number of points of the side of the spatial grid used to compute the Shannon entropy. 

    "mu_GPa":  30,

mu_GPa is the value of the rigidity module (in GPa) used to compute the Kostrov strain (Δε).

    "ETA-Computation": {
The next parameters are used only in the nearest-neighbor analysis and their values are set to classical default values from literature.

        "window-size":200,
window-size is the size of the window.

        "window-step":1,
window-step is the step between two consecutive windows.

        "Dc":1.6,
Dc is the value of fractal dimension.

        "b-value":1,
b-value is the value of b (Gutemberg-Richter relation).

        "rmax_km":150

rmax_km is the maximum radius in km
        

________________________________________
### Usage
A typical workflow consists of:
1.	Fill the file input.json with the wanted configurations
2.	Import PyCEFeaX with the command “import PyCEFeaX as pfx”.
3.	Load the seismic catalog and create the input dataframe using the in-built pfx.make_df function
4.	Run pfx.get_feature to compute the features.
5.	Plot the features in time using pfx.plot_all_features.
6.	Example scripts and configuration files are provided in the repository.
________________________________________
### Examples
Two examples are provided with the code: 
- FOREAQ, that uses a catalogue of the foreshock sequence preceding the 2009 L'Aquila earthquake.
- AQ2009, that uses a catalogue of the aftershock sequence preceding the 2009 L'Aquila earthquake.

For both examples, a set of preprocessing and features is already computed, and they can be reproduced executing the file *_feat.ipynb or the equivalent *_feat.py.

The codes  Pycefeax_notebook_*.ipynb (.py) plot the results. 

It is important to note that the examples as they are, reproduce the results and figures of Iaccarino and Picozzi (submitted in 2026), any change in input.json can change the resulting figures.

________________________________________
### Scientific Background
Most feature-extraction routines in PyCEFeaX were independently implemented by the authors based on the methodological descriptions in the original literature.

In a limited number of cases, Python implementations correspond to reimplementations of publicly released codes, with appropriate attribution.
PyCEFeaX has been tested on both foreshock and aftershock sequences, demonstrating its ability to characterize:

-	preparatory phases,
-	stress-release processes,
-	diffusion and localization of seismicity,
-	sequence segmentation and doublets.
________________________________________
### Reproducibility
All analyses presented in the associated publication can be fully reproduced using:
-	the catalogs provided in this repository,
-	the configuration files supplied with the examples.
________________________________________
### License
This project is released under the MIT License.

________________________________________
### Citation

Iaccarino, Antonio Giovanni and, Picozzi, Matteo. PyCEFeaX, Python Code for Earthquake Features eXtraction. (2026) Zenodo. DOI: 10.5281/zenodo.18548970
________________________________________
### Contact
For questions, feedback, or collaboration:
A.G. Iaccarino (antoniogiovanni.iaccarino@unina.it)
M. Picozzi

