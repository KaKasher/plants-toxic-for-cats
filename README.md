# 🌱 Cat-Safe Plant Classifier 🐱

## 🔍 Project Overview

This computer vision project aims to classify 47 popular houseplant species and provide information about their toxicity to cats. The project utilizes PyTorch and fine-tunes the PlantNet model on a custom dataset.

## 📊 Dataset

The dataset consists of 14,790 images across 47 plant species classes. Images sourced from web scraping (Bing Images) and then manualy curated by me.

Download the dataset and get more info on [Kaggle](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species)

**Note:** The current dataset is for personal use only due to copyright considerations.

## 🚀 Current Status: Work in Progress
- Working on: Hyperparameter tuning

## Best Results (as of 25.09.24)
* architecture: vit-base-patch16-224 | 25 epochs | 0.001 lr | 256 batch size | adamW optimizer
* balanced_accuracy: 0.93976
* macro_f1: 0.94438
* top3_accuracy: 0.98789
* train_loss: 0.04805
* test_loss: 0.18520

![Losses_plot](https://github.com/KaKasher/plants-toxic-for-cats/blob/main/models/plots/vit_b16_224_25e_256bs_0.001lr_adamW_transforms_plot.png?raw=true)

## TODO

- Complete hyperparameter tuning
- Try some new experiments
- Create descriptions and categorization for toxic and non-toxic plants
- Host model on hugging face


## Acknowledgements

- PlantNet for the base model: https://github.com/plantnet/PlantNet-300K
