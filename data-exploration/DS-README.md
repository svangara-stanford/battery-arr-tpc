# Data Exploration README
This sub-directory contains code and notebooks for exploring and processing battery cycle life prediction from the paper:

**"Data-driven prediction of battery cycle life before capacity degradation"**  
*Severson, K.A., Attia, P.M., et al. (2019)*

## Data Source
Original repository: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

Raw data: https://data.matr.io/1/

## Notes
- IR = internal resistance (Ω): grows as battery degrades
- QC = charge capacity (Ah): how much charge you can put in
- QD = discharge capacity (Ah): how much you can get out (this is what degrades)
- Tavg, Tmin, Tmax = temperature metrics (°C)
- chargetime = time to charge (hours)
- cycle = cycle number

## Instructions to run program using Jupyter notebooks
Great news! The setup and dependency installation commands are automated in shell scripts.

### Quick Start
Simply run two commands:

#### [1] Initial Setup (first time only)
```bash
bash setup.sh
```
This creates the virtual environment and installs all dependencies.

#### [2] Launch Jupyter (every time)
```bash
bash start-jupyter.sh
```
Jupyter Lab will automatically open in your browser at http://localhost:8888.