# 🌱 Cat-Safe Plant Classifier 🐱

## 🔍 Project Overview

This computer vision project aims to classify 47 popular houseplant species and provide information about their toxicity to cats. The project utilizes PyTorch and fine-tunes the PlantNet model on a custom dataset.

## 📊 Dataset

The dataset consists of 14,790 images across 47 plant species classes. Images sourced from web scraping (Bing Images) and then manualy curated by me.

Download the dataset and get more info on [Kaggle](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species)

**Note:** The current dataset is for personal use only due to copyright considerations.

## 🚀 Current Status: Work in Progress
- Working on: Hyperparameter tuning

## Best Results (as of 05.09.24)
* balanced_accuracy: 0.85239
* macro_f1: 0.85051
* train_loss: 0.29411
* test_loss: 0.47510

[Losses_plot](https://github.com/KaKasher/plants-toxic-for-cats/blob/main/models/plots/plantnet_finetuned_resnet34_v5_losses_plot.png?raw=true)

## TODO

- Complete hyperparameter tuning
- Try some new experiments
- Create descriptions and categorization for toxic and non-toxic plants
- Host model on hugging face


## Acknowledgements

- PlantNet for the base model: https://github.com/plantnet/PlantNet-300K
