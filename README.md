# Satellite Orbit Prediction using Machine Learning

Publication: https://ceur-ws.org/Vol-3282/icaiw_waai_3.pdf

## Overview
We focus on predicting satellite orbits using a machine learning approach that combines **Curve Fitting** and **Long Short-Term Memory (LSTM)** models. Our objective is to improve the accuracy of satellite orbit predictions beyond traditional physics-based models like **SGP4** by leveraging historical orbital data. We specifically process **Two-Line Element (TLE)** data to predict future satellite positions and assess their potential collision risks.

## Features
- **TLE Data Processing**: Reads TLE data from text files.
- **SGP4 Propagation**: Uses SGP4 to propagate TLE data and generate satellite position and velocity.
- **Curve Fitting Models**: Fits periodic and linear parameters for accurate predictions.
- **LSTM Neural Network**: Trains on historical TLE data to enhance prediction accuracy.
- **TLE Synthesis**: Generates new TLEs based on predicted parameters.
- **Error Analysis**: Evaluates prediction accuracy by comparing generated TLEs with original data.
- **3D Trajectory Visualization**: Plots satellite trajectories in a 3D space.

## Dependencies
Ensure the following libraries are installed before running the project:
```bash
pip install sgp4 skyfield astropy numpy scipy tensorflow keras matplotlib seaborn statsmodels pandas
```

## Usage
### 1. Running the Notebook
Open `Satellite_orbit_prediction.ipynb` in Jupyter Notebook and execute the cells sequentially to:
- Import necessary libraries
- Load and preprocess TLE data
- Train LSTM and curve fitting models
- Predict future orbital parameters
- Generate new TLEs and compare their accuracy

### 2. Visualizing Satellite Parameters
The function `plot_satellite_parameters()` can be used to visualize key TLE parameters over time.
```python
plot_satellite_parameters(RASAT_dict)
```

### 3. Predicting Future Orbits
To fit TLE parameters using LSTM and curve fitting, run:
```python
n = 8000
a = fit(RASAT_dict)
RASAT_dict_fitted = a.fit_all(n)
plot_satellite_parameters(RASAT_dict, RASAT_dict_fitted, n)
```

### 4. Generating a New TLE
Use the `TLE_synthesizer()` function to create a new TLE from predicted parameters.
```python
TLE_synthesized = TLE_synthesizer(satnum, international_designator, epochyr, epochdays, ndot, nddot, bstar, inclo, nodeo, ecco, argpo, mo, no_kozai)
```

### 5. Evaluating TLE Accuracy
To measure the accuracy of a generated TLE, compare it against actual TLE data:
```python
difference_of_TLE(TLE_synthesized, [RASAT_firsts[i], RASAT_seconds[i]], sat_synth.jdsatepoch)
```

## Results & Findings
- The LSTM model significantly improves prediction accuracy over conventional physics-based models.
- Curve fitting works well for linear and periodic parameters like the right ascension of the ascending node and eccentricity.
- Our proposed ML approach achieves an **89.32% improvement in along-track error reduction** compared to traditional TLE/SGP4 models.
- The model successfully synthesizes new TLEs that closely match real satellite behavior.

## Future Work
- **Hybrid Models**: Combine physics-based models with machine learning for better accuracy.
- **Automated Error Correction**: Implement differential correction techniques to refine predictions.
- **Real-Time TLE Updates**: Streamline live TLE data to dynamically improve predictions.
