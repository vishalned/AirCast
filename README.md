
# AirCast: Improving Air Pollution Forecasting Through Multi-Variable Data Alignment

## Abstract

Air pollution remains a leading global health risk, exacerbated by rapid industrialization and urbanization, contributing significantly to morbidity and mortality rates. In this paper, we introduce AirCast, a novel multi-variable air pollution forecasting model, by combining weather and air quality variables. AirCast employs a multi-task head architecture that simultaneously forecasts atmospheric conditions and pollutant concentrations, improving its understanding of how weather patterns affect air quality. Predicting extreme pollution events is challenging due to their rare occurrence in historic data, resulting in a heavy-tailed distribution of pollution levels. To address this, we propose a novel Frequency-weighted Mean Absolute Error (fMAE) loss, adapted from the class-balanced loss for regression tasks. Informed from domain knowledge, we investigate the selection of key variables known to influence pollution levels. Additionally, we align existing weather and chemical datasets across spatial and temporal dimensions. AirCast’s integrated approach, combining multi-task learning, frequency weighted loss and domain informed variable selection, enables more accurate pollution forecasts. Our source code and models will be publicly available.


<p align="center">
  <img src="" width="320px">
</p>

[![Paper](https://img.shields.io/badge/arXiv-2301.10343-blue)]()


This repository contains code for an adaption of the ClimaX model for Air Quality Forecasting. We follow similar structure as the original ClimaX repository. We start from the climax pre-trained model and fine-tune it on a regional air quality forecasting task (hence the training code and configs are relating to the regional forecast folder).

## Data

Our paper uses data that is sourced from both WeatherBench (for the weather variables) and the Copernicus Atmosphere Monitoring Service (CAMS) (for the air quality variables). We align the temporal and spatial domains of the two datasets to create a combined dataset of weather and air quality variables. 

The data can be found [here](https://zenodo.org/records/8326445).


## Installation
To install the requirements, run:
```bash
conda create -n climaX python=3.9.18
conda activate climaX 
conda install -c conda-forge esmpy
pip install -r requirements.txt
pip install -U 'jsonargparse[signatures]'
# install so the project is in PYTHONPATH
pip install -e .
```


## Training

To train the model, run:
```bash
python -u src/climax/regional_forecast/train.py --config configs/regional_forecast_climax_full.yaml
```

Alternatively, you can run the following command to train the model on the HPC cluster:
```bash
sbatch slurm_train.sh
```




For details about the original ClimaX model please see the [documentation](https://microsoft.github.io/ClimaX).

## Citation

If you find this repository useful in your research, please consider citing the following papers:

```bibtex
@article{
}

```
  
