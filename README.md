# RL4RRT

## Publication and Acknowledgements

**OPTIMIZING RENAL REPLACEMENT THERAPY DECISIONS IN INTENSIVE CARE WITH REINFORCEMENT LEARNING**

This repository contains the code accompanying our research on optimizing renal replacement therapy decisions in intensive care using reinforcement learning. Our work builds upon the Python implementation of the AI-Clinician algorithm originally developed by Komorowski et al. For additional details, please refer to:

[AI-Clinician-MIMICIV](https://github.com/cmudig/AI-Clinician-MIMICIV/tree/main)

**RL4RRT** is a modular pipeline for data extraction, preprocessing, and modeling using MIMIC-IV data. The repository includes scripts to seamlessly extract data via Google BigQuery, preprocess the datasets for analysis, and train models for clinical research applications.

## Installation

**Clone the repository:**
   ```bash
   git clone https://github.com/lorenzkap/RL4RRT.git
   cd RL4RRT
   ```

## Running the Pipeline

The pipeline is split into three steps: data extraction, preprocessing, and modeling. Each step consists of Python scripts, which can either be run individually (for greater flexibility) or through the single `run.sh` shell script.

### Prerequisites

**Package Requirements.** Create a conda or venv environment and activate it, if desired. Then install the package and its dependencies:

```
pip install -e .
```

**MIMIC Access.** Create a Google Cloud account that you will use to access the MIMIC-IV data through BigQuery. Get access to MIMIC-IV by going to [PhysioNet](https://physionet.org/content/mimiciv/3.1/). Then select "Request access using Google BigQuery".

<img src="./assets/access_to_mimiciv.png"/>

Setting up your client credentials if needed (see [this guide](https://cloud.google.com/bigquery/docs/authentication/end-user-installed) for using client credential to authenticate the API).

<img src="./assets/manually_creating_credentials.png"/>

You may want to save the client secret JSON file into this directory, and note the file path for later.

You should create a GCP "project", which serves as a container for the queries and analyses you perform. To do so, go to the project selector (to the left of the search bar in the GCP console) and click New Project. Save the name of this project, as you will need it in the data extraction step.

### Simple Run Instructions

Extract data into `data/` within this directory:

```
python run.sh extract <PATH_TO_CLIENT_SECRET> <GCP_PROJECT_NAME>
```

Preprocess data, filter AKI cohort, and generate full MIMIC dataset into a directory `data/<DATA_NAME>`:

```
python run.sh preprocess <DATA_NAME>
```

Train models from data directory `data/<DATA_NAME>` into models directory `data/<MODELS_NAME>`:

```
python run.sh model <DATA_NAME> <MODELS_NAME>
```
