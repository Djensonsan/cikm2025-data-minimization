# What Data is Really Necessary? A Feasibility Study of Inference Data Minimization for Recommender Systems

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
## About

Welcome to the official repository for our paper, **“What Data is Really Necessary? A Feasibility Study of Inference Data Minimization for Recommender Systems,”** published in the proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM ’25).

This repository contains the source code and experimental results necessary to reproduce the findings discussed in our paper.

---

## Repository Scope and Status

This repository contains the official code snapshot used for the experiments in our CIKM 2025 paper. It is a specific version of a larger, actively developed research framework named **MiniPack**.

Its primary purpose is to ensure **transparency and reproducibility** of the results presented in the paper. As such, this repository is an **archival snapshot**. It is provided "as is" and will not be actively maintained or supported.

For inquiries about the latest version of MiniPack or to discuss potential collaborations, please contact us directly. Access to the main framework is granted at the discretion of our research group.

---

## Key Features:
* **Correctness:** Core algorithms and pipelines are unit and integration tested using PyTest.
* **Reproducibility:** Experiments are designed for reproducibility through configurable seeding and clearly defined experimental setups.
* **Documentation:** This README, along with detailed inline code comments and docstrings, provides instructions for use and an overview of the codebase.
* **Ease of Setup:** Relies on standard Python packaging with `requirements.txt` for straightforward dependency management. Clear CLI scripts and configuration files facilitate experiment execution.

## Limitations:
* **Performance Optimization:** While functional, code performance has not been the primary focus.
    * **Parallelization:** Implemented using Ray, allowing experiments to scale based on available hardware.
    * **Memory:** Utilizes Scipy sparse data structures for memory efficiency with large datasets.
    * **Inference:** Currently, all operations are CPU-based; GPU acceleration is not supported.
    * **Low-level Optimizations:** Advanced optimizations (e.g., Cython, Numba) are not implemented; the framework primarily builds upon NumPy and SciPy.
* **Dependencies:** The codebase has a tight coupling with [RecPack](https://gitlab.com/recpack-maintainers/recpack/-/tree/master) (an experimentation framework for Recommender Systems), leveraging its structures and utilities.

---

## Getting Started
### Prerequisites
* Python 3.11+
* pip

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run Tests (Optional):**
    To ensure everything is set up correctly, you can run the test suite:
    ```bash
    pytest
    ```

---
## Repository Structure

The repository is organized as follows:

* `minipack/`: Contains the core source code used for the experiments.
    * `minipack/algorithms/`: Wrappers for recommendation algorithms.
    * `minipack/metrics/`: Implementations of evaluation metrics.
    * `minipack/minimizers/`: Implementations of data minimization algorithms.
    * `minipack/pipelines/`: Core logic for the experimental pipelines.
    * `minipack/scenarios/`: Definitions for dataset splitting and evaluation scenarios.
    * `minipack/target_estimators/`: Implementation details for ground truth estimation.
    * `minipack/tests/`: Unit and integration tests.
* `notebooks/`: Jupyter notebooks for data analysis and visualization of results.
* `scripts/`: CLI scripts for executing the experimental pipelines.
* `requirements.txt`: A list of Python package dependencies.

---

## How-To Guide

This section is divided into two parts. The first explains how to explore our results and analysis. The second provides a full guide to reproducing the experiments from scratch.

### Inspecting the Results

If your goal is to analyze our findings without re-running the entire computational pipeline, follow these steps.

1.  **Download Our Raw Results:** All raw outputs from our experiments (hyperparameter optimization, ground truth estimation, and minimization processes) are available for download. They are organized by dataset and algorithm.
    * **Download Link:** [Proton Drive Folder](https://drive.proton.me/urls/QNNR8A6PH8#McluK9peEOWQ)

2.  **Explore the Analysis Notebooks:** The `notebooks/` directory contains the Jupyter notebooks we used to analyze the raw results and generate the plots and tables for the paper. You can use these notebooks to explore the data you downloaded in the previous step.

### Reproducing the Experiments

This guide is for running the entire experimental pipeline from scratch to generate the results yourself.

**Execution Environment:** Be aware that our experiments were conducted on a High-Performance Computing (HPC) cluster. Running the full suite on a standard local machine is computationally intensive and time-consuming.

1.  **Download the Public Datasets:** Our code does not include the large public datasets. You must download them separately and provide the path to them when running our scripts.
    * **MovieLens:** [GroupLens website](https://grouplens.org/datasets/movielens/)
    * **Million Song Dataset:** [Official page](http://millionsongdataset.com/)
    * **Netflix Prize:** [Kaggle page](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

2.  **Review the Configurations:** Pre-defined configurations for all experiments are available in the `scripts/config/` directory. These files specify all parameters for data splitting, hyperparameter grids, and optimal model settings used in the paper.

3.  **Run the CLI Pipeline:** All experiments are executed via scripts in the `scripts/` folder. They should be run in the following sequence:

    * **Step 1: Hyperparameter Optimization**
        This step tunes the recommendation models.
        ```bash
        python scripts/hyperoptimization_pipeline.py --dataset <dataset_name> --data_path <path_to_data> ...
        ```
    * **Step 2: Model Training**
        This step fits the models using the optimized hyperparameters.
        ```bash
        python scripts/training_pipeline.py --dataset <dataset_name> --data_path <path_to_data> ...
        ```
    * **Step 3: Ground Truth Estimation**
        This script estimates the ground truth performance of the trained models.
        ```bash
        python scripts/target_estimation_pipeline.py --dataset <dataset_name> --data_path <path_to_data> ...
        ```
    * **Step 4: Dataset Minimization**
        This script runs the data minimization algorithms on the trained models.
        ```bash
        python scripts/minimization_pipeline.py --dataset <dataset_name> --data_path <path_to_data> ...
        ```
    For detailed arguments for each script, use the `--help` flag (e.g., `python scripts/training_pipeline.py --help`).

---

## Authors & Contact

* **Jens Leysen** - *University of Antwerp, Belgium*
* **Marco Favier** - *University of Antwerp, Belgium*
* **Bart Goethals** - *University of Antwerp, Belgium* and *Monash University, Australia*

For questions about the paper or this repository, please contact Jens Leysen.