# PyYield — CMOS Statistical EDA Platform

**A Python-based platform for CMOS circuit analysis, statistical process variation modeling, and semiconductor yield estimation.**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Domain-CMOS%20%7C%20VLSI-orange" alt="CMOS and VLSI">
  <img src="https://img.shields.io/badge/Analysis-Statistical%20EDA-purple" alt="Statistical EDA">
</p>

**Live Demo:** [Launch PyYield](https://pyyield-eda.streamlit.app/)

---

## Overview

PyYield is a collaborative engineering project developed to investigate the effects of semiconductor process variations and operating conditions on CMOS circuit performance.

The platform brings together Python-based analytical modeling, statistical sampling, Monte Carlo analysis, Ngspice integration, machine-learning-assisted prediction, and interactive visualization in a browser-accessible application.

Rather than considering only nominal device parameters, PyYield explores how variations in transistor characteristics, supply voltage, temperature, and other modeled parameters can influence circuit behavior, power consumption, delay, and estimated yield.

The project was developed by a team of four, with me serving as the **primary technical contributor**, responsible for most of the software implementation and technical integration.

## Live Application

Explore the deployed application:

**[Open PyYield →](https://pyyield-eda.streamlit.app/)**

The application is hosted on Streamlit Community Cloud and can be accessed through a web browser.

The availability of simulation functionality depends on the corresponding Python dependencies, model artifacts, and system-level simulation tools being available in the deployment environment.

---

## Key Features

### 1. Statistical Process Variation Modeling

Investigate how variations in transistor and fabrication parameters affect CMOS circuit performance.

- Model statistical variations in device parameters.
- Generate parameter samples for circuit analysis.
- Examine distributions and variability in simulated performance metrics.
- Investigate relationships between input parameters and circuit behavior.

### 2. Monte Carlo Simulation

Use Monte Carlo sampling to explore the statistical behavior of CMOS circuits under parameter uncertainty.

- Generate randomized parameter combinations.
- Evaluate circuit performance across multiple samples.
- Analyze distributions of delay, power, and other supported metrics.
- Estimate the proportion of simulated samples meeting specified acceptance criteria.

### 3. Latin Hypercube Sampling

Use Latin Hypercube Sampling (LHS) as an alternative method for exploring multidimensional parameter spaces.

- Generate structured samples across parameter distributions.
- Investigate interactions between varying device parameters.
- Compare statistical analysis workflows using different sampling approaches.

### 4. CMOS Logic Gate Analysis

PyYield supports analysis workflows for multiple CMOS logic gates:

- CMOS Inverter
- CMOS NAND Gate
- CMOS NOR Gate

These gates provide a basis for exploring circuit behavior under varying device and operating conditions.

### 5. Ngspice Integration

PyYield integrates Ngspice into its circuit-analysis workflow where SPICE-based simulation is enabled.

This integration connects Python-driven analysis with circuit simulation, supporting the investigation of electrical behavior under configured circuit and operating conditions.

SPICE-dependent functionality requires a compatible Ngspice installation and a correctly configured execution environment.

### 6. Process Corners and PVT Analysis

Explore circuit behavior under supported process and operating scenarios.

- Fast-Fast (FF)
- Typical-Typical (TT)
- Slow-Slow (SS)
- Supply-voltage variation
- Temperature variation

These analyses help investigate the sensitivity of circuit performance to modeled process, voltage, and temperature conditions.

### 7. Machine Learning-Assisted Analysis

Integrate machine-learning techniques into the circuit-analysis workflow.

- Random Forest regression for configured prediction tasks.
- Statistical exploration of relationships between parameters and performance.
- Machine-learning-assisted performance estimation.
- Comparison of predicted behavior with available simulation or analysis results.

Prediction quality depends on the training data, model configuration, and validation methodology. Machine-learning estimates should not be treated as a substitute for validated circuit simulation.

### 8. Yield Estimation and Pass/Fail Classification

Evaluate simulated circuit samples against configured acceptance criteria.

- Classify samples according to specified performance limits.
- Estimate the fraction of simulated samples that pass those limits.
- Investigate the relationship between parameter variation and estimated yield.
- Examine how operating conditions and model assumptions influence results.

The resulting yield is a model-based estimate, not a guarantee of manufacturing yield.

### 9. Advanced Statistical Analysis

Additional analytical and visualization capabilities include:

- Descriptive statistical analysis
- Correlation analysis
- Linear regression
- Principal Component Analysis (PCA)
- Parameter sensitivity analysis
- Tornado-style sensitivity visualization
- Wafer-level spatial variation visualization
- Aging-related analysis
- Power breakdown analysis
- Multi-gate performance comparison

The interpretation of each result depends on the implemented model, input assumptions, and selected analysis configuration.

---

## Technical Architecture

PyYield combines several components into a unified analysis workflow.

| Component | Purpose |
|---|---|
| Python | Application logic and computational workflows |
| Streamlit | Interactive web interface |
| NumPy | Numerical computation and statistical sampling |
| Pandas | Tabular data processing and analysis |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine-learning and statistical-learning workflows |
| Ngspice | SPICE-based circuit simulation |
| PySpice | Python-to-SPICE integration, if used by the configured workflow |

The exact dependency set depends on the modules included in the deployed version.

### Analysis Workflow

The general workflow follows these stages:

1. **Circuit configuration:** Select a supported gate and configure the available device and operating parameters.
2. **Variation definition:** Specify the distributions and variation ranges used in the experiment.
3. **Sample generation:** Generate parameter combinations through Monte Carlo or Latin Hypercube Sampling.
4. **Circuit evaluation:** Evaluate samples using the configured analytical models and SPICE-based workflows where enabled.
5. **Statistical processing:** Calculate summaries and examine the resulting performance distributions.
6. **Yield estimation:** Apply configured acceptance criteria to the simulated samples.
7. **Machine-learning analysis:** Use supported models for configured prediction tasks.
8. **Visualization:** Explore statistical results, parameter sensitivity, and other available analysis outputs.

This workflow brings together circuit modeling, statistical experimentation, and data analysis in a single interface.

---

## Technology Stack

- **Language:** Python
- **Web framework:** Streamlit
- **Numerical computing:** NumPy
- **Data processing:** Pandas
- **Statistical visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Circuit simulation:** Ngspice
- **Version control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

---

## Running PyYield Locally

### Prerequisites

Before running the application locally, ensure that you have:

- A compatible Python version.
- Git installed.
- The Python dependencies required by the project.
- Ngspice installed and accessible if using SPICE-dependent functionality.
- Any required saved machine-learning models and configuration files.

### 1. Clone the repository

```bash
git clone https://github.com/Saikat-00/PyYield.git
cd PyYield
```

Repository access is required. If the repository is private, authenticate with an authorized GitHub account.

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Ngspice

If your selected workflow requires Ngspice, confirm that the executable is available.

```bash
ngspice --version
```

If the command is not recognized, check the Ngspice installation and system PATH configuration.

### 5. Launch the application

Run the actual Streamlit entry-point file in the project. For example, if it is named `app.py`:

```bash
streamlit run app.py
```

Replace `app.py` with the correct filename if the project uses a different entry point.

Open the local URL displayed in the terminal.

---

## Applications

PyYield is intended for educational, research, and exploratory engineering work, including:

- Understanding statistical process variation in CMOS circuits.
- Exploring Monte Carlo methods in semiconductor analysis.
- Investigating circuit delay and power sensitivity.
- Comparing supported CMOS logic gates.
- Studying model-based yield estimation.
- Exploring machine-learning-assisted circuit analysis.
- Visualizing relationships between device parameters and performance.
- Learning how Python, statistical analysis, and SPICE simulation can be integrated into an EDA workflow.

---

## Project Scope and Limitations

PyYield is an educational and exploratory statistical EDA platform.

The accuracy of its outputs depends on the implemented models, device parameters, statistical assumptions, sampling strategy, simulation configuration, and available training data.

The platform is not intended to replace foundry-calibrated industrial EDA tools, sign-off verification, silicon characterization, or production manufacturing-yield analysis.

Results should be interpreted within the scope of the models and assumptions used in each experiment.

---

## Team and Individual Contributions

PyYield was developed collaboratively by a team of four.

I served as the **primary technical contributor**, taking responsibility for most of the software implementation and technical development. My work covered the implementation and integration of the platform's major technical components.

My contributions included:

- **Application development:** Implementing and integrating the Python-based application.
- **Statistical analysis:** Developing the supported sampling and statistical-analysis workflows.
- **Circuit analysis:** Working on CMOS circuit modeling and Ngspice integration.
- **Machine learning:** Implementing and integrating machine-learning-assisted analysis workflows.
- **Visualization:** Developing analysis outputs and interactive visualizations.
- **System integration:** Connecting the platform's analytical, simulation, and presentation components.
- **Deployment:** Preparing the application for browser-based access.

The other team members contributed to literature research, logistics, and other supporting project activities.

This project reflects collaborative development, with my primary responsibility being the technical implementation.

---

## Future Development

Potential directions for further development include:

- Validation against reference SPICE simulations and measured device data.
- Improved calibration of statistical device models.
- Expansion of the supported CMOS circuit library.
- More comprehensive PVT sweeps.
- Improved uncertainty quantification for machine-learning predictions.
- Optimization for larger statistical experiments.
- Reproducible experiment configurations and automated reporting.

These are potential extensions rather than claims about existing functionality.

---

## Acknowledgements

- [Ngspice](https://ngspice.sourceforge.io/) — Open-source circuit simulator.
- [Streamlit](https://streamlit.io/) — Framework for interactive Python applications.
- [NumPy](https://numpy.org/) — Numerical computing.
- [Pandas](https://pandas.pydata.org/) — Data analysis and manipulation.
- [Scikit-learn](https://scikit-learn.org/) — Machine-learning and statistical-learning tools.

---

**PyYield — Exploring CMOS circuit variability through statistical modeling and simulation.**
