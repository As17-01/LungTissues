# Lung Tissues Classification

## Introduction

This study is based on the following paper: 

`https://www.nature.com/articles/s41591-018-0177-5`

Their Git repository:

`https://github.com/ncoudray/DeepPATH/`

## Data

Data could be downloaded from `https://disk.yandex.com/d/yEbNmJRmKvaXiA` (not guaranteed to be supported) using `./scripts/download/main.py`. There is a limitation on how many files you can download at a time, so it required to update the link every 100 files. Contact me if it is needed. TODO: Find an alternative to store the data.

Or the slides in the required format can be directly downloaded from `https://portal.gdc.cancer.gov/cart`. I downloaded 424 slides from there. They were chosen randomly from all slides of the same category. Example:

![Alt text](assets/download_data_example.jpg?raw=true "Download Data Example")

The resulting structure of the `./data` folder should be the following. Note that each image file should be placed inside `file_name/file_name` folder for consistency with the Yandex Drive downloading data method. TODO: Rework downloading data script:

![Alt text](assets/data_structure_example.png?raw=true "Data Structure Example")

## Installation

Install the project and requirements with the following command:

`poetry install`

Enter the virtual environment using the following command:

`poetry shell`

## Additional requirements

You will also need to run the following commands to install non-python requirements:

`sudo apt-get install libopenslide0`