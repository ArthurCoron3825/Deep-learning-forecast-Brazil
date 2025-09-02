# Deep-learning-forecast-Brazil

This github has the python files used in the internship about precipitation forecasting in the northen-east coast of Brazil, and in a precise pixel using deep learning models.


# Models

Folder that contains the different deep learning models and mathematical methods used throughout the intership.

## 1. Binary classification
### Purpose
This script trains and evaluates a binary classifier to predict rain (1) vs no rain (0) at a specific latitude/longitude point, using meteorological variables from NetCDF files. The model is a simple feedforward neural network (MLP) built with PyTorch.

### How it works
- Loads .nc files containing climate data (precipitation + other variables).
- Extracts features around the chosen target coordinates (TARGET_LAT, TARGET_LON).
- Prepares input sequences of a given length (n_steps).
- Converts precipitation into a binary label using a threshold (RAIN_THRESHOLD).
- Trains an MLP with fully connected layers and evaluates it on train/validation/test sets.
- Reports metrics (accuracy, precision, recall, F1) and displays confusion matrices + loss curves.

### Usage
1. Organize your data into directories:
- data/train
- data/val
- data/test

2. Adjust the global parameters at the top of the script depending on the need.

3. Run the script, with the outputs.


## 2. GRU pixel
### Purpose
This script trains a GRU-based regression model to predict precipitation values (mm) at a target latitude/longitude. Unlike the binary classifier, it performs continuous rainfall prediction, learning from multiple meteorological variables and temporal sequences.

### How it works
- Loads .nc files containing precipitation (tp) and other variables.
- Builds temporal sequences (n_steps) of data for a specific grid point (TARGET_LAT, TARGET_LON).
- Adds month-based one-hot encoding and a flag for first-sequence initialization.
- Normalizes inputs and outputs using StandardScaler.
- Uses a multi-layer GRU followed by dense layers to predict rainfall at the final timestep.
- Training loop includes:
  - ReduceLROnPlateau scheduler to adjust learning rate
  - Early stopping with patience and minimum delta
- Evaluates performance using MAE, RMSE, and R² score.
- Visualizes:
  - MAE curves during training
  - Optional histograms comparing normalized vs real predictions

### Usage
1. Organize your data into directories:
- data/train
- data/val
- data/test

2. Adjust the key parameters.

3. Run the script, and get the outputs.


## 3. IDW interpolation code
### Purpose
This script builds a precipitation interpolation pipeline using the Inverse Distance Weighting (IDW) method. It processes raw precipitation data from weather stations, aggregates daily totals, and interpolates the values onto a regular spatial grid. The final outputs are monthly NetCDF files suitable for further analysis or use in machine learning models.  

### How it works
- Data preparation:  
  - Loads multiple INMET CSV files containing station precipitation data.  
  - Extracts metadata (station name, coordinates, altitude).  
  - Cleans and converts hourly precipitation into valid numeric values.  
  - Aggregates precipitation to daily totals for each station.  
- Interpolation:  
  - Applies Inverse Distance Weighting (IDW) to project station values onto a fixed 2D grid.  
  - Loops over all days in the selected month to build a spatiotemporal dataset.  
- Outputs:  
  - Creates an xarray Dataset with daily precipitation on the grid.  
  - Saves one NetCDF file per interpolated month.  
  - Provides helper functions for validation (summaries, histograms, maps).  

### Usage  
1. Place raw precipitation CSV files in a directory (e.g., `precip/2011-2015/old/`).  
   - The files must follow INMET station CSV format.  

2. Adjust main parameters in the script.

3. Run the script.


## 4. LSTM script main part
### Purpose  
This script trains, validates, and tests an LSTM neural network to predict daily precipitation at individual pixels.  
It handles normalization/denormalization and provides visualizations for model performance and predictions.  

### How it works  
- Loads pixel-based precipitation sequences from preprocessed datasets (`PixelSequenceDataset`).  
- Trains an LSTM model to predict next-day precipitation using past sequences.  
- Evaluates performance with:  
  - Smooth L1 Loss  
  - Mean Absolute Error (MAE)  
  - R² score 
- Produces visualizations:  
  - Training/validation loss and MAE curves  
  - Histograms of predicted vs. true precipitation (normalized + real scale)  
  - Scatter plot comparing predicted vs. true precipitation  
  - Temporal sequence comparison (first 240 predictions) 

### Usage
1. Organize your data into directories:
- data/train
- data/val
- data/test

2. Adjust the key parameters if needed.

3. Run the script, and get the outputs.


## 5. LSTM script, model part
### Purpose  
This module provides the **data handling class** (`PixelSequenceDataset`) and the **LSTM model** (`LSTMModel`) used for precipitation prediction.  
It prepares pixel-based sequences from climate NetCDF files and defines the neural network architecture for training.  

### How it works  
- **PixelSequenceDataset**  
  - Loads and concatenates multiple NetCDF files containing gridded climate data.  
  - Extracts input variables (all except precipitation `"tp"`) and the precipitation target for a chosen pixel.  
  - Builds fixed-length sequences (`SEQ_LEN`, default: 24).  
  - Applies preprocessing:  
    - Skips sequences with missing values.  
    - Log-transforms precipitation targets to stabilize variance.  
    - Standardizes features and targets using `StandardScaler`.  

- **LSTMModel**  
  - Input: time series of pixel features.  
  - Core: multi-layer LSTM (`num_layers=2`, `hidden_dim=128`, dropout=0.3).  
  - Output: fully connected layers projecting hidden state to a precipitation value.  

### Usage  
1. Prepare your NetCDF climate data in the same previous directory.  

2. Initialize dataset and adjust the parameters.

3. Run the main script.


## 6. MLP Pixel-based Precipitation Prediction
### Purpose  
This script trains, validates, and tests a **Multi-Layer Perceptron (MLP)** model to predict precipitation at a **target pixel** using ERA5 reanalysis data.  
It focuses on single-point prediction with feature engineering, normalization, and evaluation metrics.

### How it works  
- **Dataset (`PixelPrecipDataset`)**  
  - Loads multiple NetCDF ERA5 files.  
  - Selects the nearest grid point to the target pixel (`TARGET_LAT`, `TARGET_LON`).  
  - Builds temporal sequences (`n_steps`), adds one-hot month encoding (Nov–Jun), and a seasonal flag.  
  - Standardizes inputs and precipitation target.  

- **MLP model (`MLP`)**  
  - Input: flattened feature sequences + extra features.  
  - Hidden layers: 32 → 8 neurons with **LeakyReLU** activations and **Dropout(0.3)**.  
  - Output: single regression value (precipitation).  

- **Training loop**  
  - Loss: `L1Loss` (MAE).  
  - Optimizer: `Adam` with learning rate scheduler.  
  - Early stopping by tracking best validation loss.  

- **Evaluation**  
  - Metrics: MAE, RMSE, R² (normalized and real scale).  
  - Visualization: loss curves, histograms, scatter plots, error distribution, and time-series comparison.  
  - Binary rain/no-rain evaluation with **F1-score**.  

### Usage  
1. Place your ERA5 NetCDF files in directories:  
   - `data/train/`, `data/val/`, `data/test/`.  
2. Adjust key parameters inside the script:  
   - `TARGET_LAT`, `TARGET_LON` → target location coordinates.  
   - `n_steps` → sequence length (default: `1`).  
   - `batch_size`, `epochs`, and optimizer settings if needed.  
3. Run the script.



# Plotting

## 7. INMET vs ERA5 Daily Precipitation Comparison
### Purpose  
This script compares **daily precipitation data** from INMET weather stations (CSV format) with **ERA5 reanalysis data** (NetCDF format) for a selected year.  
It highlights similarities and discrepancies in rainfall timing and magnitude, supporting the validation of reanalysis datasets.

### How it works  
- **INMET preprocessing**  
  - Loads multiple CSV files with station precipitation data.  
  - Converts hourly precipitation into daily totals.  
  - Filters target year (`TARGET_YEAR`) and excludes dry months (Jul–Oct).  
  - Selects up to `MAX_STATIONS` stations with sufficient daily coverage.  

- **ERA5 extraction**  
  - Loads ERA5 daily NetCDF files for the target year.  
  - Extracts precipitation from the nearest grid cell to each station’s coordinates (randomly assigned if not available).  
  - Converts cumulative values to daily totals and excludes dry months.  

- **Comparison and visualization**  
  - Merges INMET and ERA5 daily series for each station.  
  - Identifies maximum precipitation events (date + value).  
  - Produces time-series plots with:  
    - INMET vs ERA5 daily precipitation curves.  
    - Markers for maximum rainfall events.  
    - Station metadata in the title (lat, lon).  

### Usage  
1. Organize your input data:  
   - Place INMET CSV files in `precip/2011-2015/old/`.  
   - Place ERA5 daily NetCDF files in `data/train/`.  

2. Adjust key parameters at the top of the script:  
   - `TARGET_YEAR` → year to analyze (default: `2013`).  
   - `MAX_STATIONS` → maximum number of stations to process.  
   - Replace random coordinates with actual station lat/lon if available.  

3. Run the script.


## 8. INMET and ERA5 Daily Precipitation Visualization
### Purpose  
This script visualizes **daily precipitation** for selected INMET weather stations and ERA5 reanalysis data for a target year.  
It produces **time-series plots** and **histograms** for both datasets, allowing a direct visual comparison of rainfall patterns and distributions.

### How it works  
- **INMET station data**  
  - Loads CSV files for selected stations (`TARGET_STATIONS`).  
  - Converts precipitation values to numeric and parses dates.  
  - Filters the target year (`TARGET_YEAR`) and removes dry-season months (July–October).  
  - Aggregates hourly precipitation to daily totals.  
  - Plots:  
    - Daily precipitation time series.  
    - Histogram of daily precipitation distribution.  

- **ERA5 reanalysis data**  
  - Loads NetCDF files from ERA5 data folder.  
  - For each station, finds the nearest grid point (lat/lon).  
  - Extracts daily cumulative precipitation and converts it to daily totals.  
  - Filters for the target year and excludes dry-season months.  
  - Plots:  
    - Daily precipitation time series.  
    - Histogram of daily precipitation distribution.  

### Usage  
1. Organize your data:  
   - INMET CSVs: `precip/2011-2015/old/`  
   - ERA5 NetCDFs: `data/train/`  

2. Adjust parameters at the top of the script:  
   - `TARGET_STATIONS` → list of stations to plot.  
   - `TARGET_YEAR` → year to analyze.  
   - `ERA5_COORDS` → latitude and longitude of each station.  

3. Run the script.


## 9. Pixel-based Precipitation Analysis and Visualization
### Purpose  
This script performs **pixel-level analysis** of daily precipitation from NetCDF datasets.  
It provides **daily precipitation maps** and **histograms for individual pixels** within a specified spatial subset.

### How it works  
- **Data loading and preprocessing**  
  - Loads multiple NetCDF files from a specified directory (`nc_dir`).  
  - Concatenates datasets along the time dimension.  
  - Selects data for **January to June**.  
  - Extracts a spatial subset defined by `lat_range` and `lon_range`.  
  - Converts precipitation values from meters to millimeters.  

- **Visualization**  
  - **Daily precipitation maps**: plots precipitation across the selected region for the first N days.  
  - **Pixel histograms**: shows the distribution of precipitation values at each pixel.  

- **Functions**  
  - `plot_daily_maps(days=N)`: visualize maps for the first N days.  
  - `plot_all_pixel_histograms()`: generate histograms for every pixel in the selected area.  
  - `plot_daily_map(date_str)`: plot a single day’s precipitation map.  
  - `plot_pixel_histogram(lon, lat)`: plot histogram for a specific pixel.  

### Usage  
1. Place NetCDF precipitation files in a directory (e.g., `data/test/`).  

2. Adjust spatial subset if needed:  

3. Run the script.


## 10. Brazil Study Area Map
### Purpose  
This script generates a **map of Brazil** with key geographic features (coastlines, borders, rivers, lakes, etc.) using **Cartopy**.  
A **red rectangle** highlights a predefined study region specified by latitude/longitude boundaries.  
The visualization provides spatial context for the selected area used in precipitation prediction studies.

### How it works  
- Loads a Cartopy map with the Plate Carree projection.  
- Adds geographic features: borders, coastlines, land, ocean, lakes, and rivers.  
- Zooms into Brazil (`ax.set_extent`).  
- Draws a red rectangle defined by coordinates (`LAT_MAX, LON_MIN, LAT_MIN, LON_MAX`).  
- Displays the map with a title.  

### Usage  
1. Adjust the study area boundaries at the top of the script:  

2. Run the script


## 11. Statistical Analysis of Precipitation at a Target Pixel
### Purpose  
This script performs **statistical analysis** of daily precipitation at a **specific target location** using ERA5 (or similar NetCDF) datasets.  
It computes basic statistics, such as:
- Number of dry days
- Days with very light rain (0–0.2 mm)
- Days with light rain (0–1 mm)

It also generates multiple visualizations to explore precipitation behavior at the selected pixel.

### How it works  
- Loads multiple NetCDF files from a given directory.  
- Selects the grid point nearest to the target latitude/longitude.  
- Extracts daily precipitation values and classifies days into categories: zero, light, and very light rain.  
- Computes rolling statistics (7-day moving average).  
- Generates plots:  
  - Summary table of day counts per category  
  - Histograms for light rain (0–1 mm) and very light rain (0–0.2 mm)  
  - Raw precipitation time series  
  - Global and complete histograms of all precipitation values  
  - Smoothed time series using 7-day moving average  

### Usage  
1. Set the data directory and target coordinates.

2. Run the script.


## 12. Visualization of Study Area, ERA5 Pixels, and Station Location
### Purpose  
This script generates **geospatial maps** for the Brazil study area, highlighting:
- The predefined **study area bounding box**.
- Selected **ERA5 grid pixels** used in precipitation analysis.
- The location of **INMET station 82287 (Parnaíba)**.

It provides a clear visual context of the spatial layout for modeling and analysis.

### How it works  
- Defines the study area latitude/longitude boundaries.  
- Specifies ERA5 pixel coordinates and the INMET station coordinates.  
- Uses **Cartopy** to create maps with:
  - Coastlines, borders, land/ocean coloring, lakes, and rivers.  
  - Points for ERA5 pixels and station location.  
  - Dashed rectangle for the study area.  
- Generates two separate figures:
  1. Study area map with ERA5 pixels.
  2. Map showing INMET Parnaíba station.

### Usage  
1. Adjust coordinates if needed.

2. Run the script.


## 13. ERA5 Daily Weather Maps Visualization
### Purpose
This script visualizes **daily ERA5 reanalysis data** for a selected month and study area in Brazil.  
It helps explore spatial and temporal variability of:
- **Precipitation (surface `tp`)**
- **Other atmospheric variables** (e.g., relative humidity, geopotential height)

Additionally, a **context map** shows the study area bounding box within South America.

### How it works
1. **Configuration**  
   - Define the target month and data directory.  
   - Define the study area bounds (`LAT_BOUNDS`, `LON_BOUNDS`).  
   - Specify variables of interest (`SURFACE_VAR` for precipitation, `OTHER_VARS` for additional atmospheric fields).

2. **Data loading**  
   - Loads ERA5 NetCDF file for the target month using `xarray`.

3. **Plotting daily maps**  
   - Loops over each day in the dataset.  
   - Creates a subplot for precipitation and other atmospheric variables:
     - Maps use `Cartopy` for coastlines, borders, and proper geographic projection.  
     - Extent is clipped to the study area.
   - Generates a **context map** with the study area highlighted for reference.

### Usage
1. Set the data directory and target month.

2. Run the script








































