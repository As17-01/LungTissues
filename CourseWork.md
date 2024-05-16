# Lung Tissues Classification

## Abstract

The integration of deep learning methodologies into the field of medical imaging has revolutionized the way healthcare professionals analyze and interpret complex image data. This course work explores the application of Liquid Neural Networks (LNNs) to medical images, starting with the MedMNIST datasets and transitioning to the analysis of whole slide images of lung tissues. Liquid Neural Networks, inspired by the dynamic behavior of liquid systems, offer a novel computational framework that enables continuous learning and adaptation to intricate data patterns. They tend to have significantly less parameters than normal convolutional neural networks, which allows them to train on smaller data, and the process is faster overall. In addition, they are capable of capturing the dynamics on multiple images, which may benefit to the model's quality. The course work begins by utilizing the MedMNIST datasets, a diverse collection of medical image data, to train and validate neural network models in a controlled setting. Subsequently, the focus shifts towards applying Liquid Neural Networks to the analysis of whole slide images of lung tissues, aiming to enhance the accuracy and efficiency of medical image analysis in the context of pulmonary diseases. Through this exploration, the course work seeks to contribute to the ongoing advancements in healthcare to improve diagnostic accuracy and patient outcomes in the medical image analysis.

## Introduction

In recent years, the field of medical imaging has experienced rapid advancements in deep learning technologies. These advancements have changed the way healthcare professionals analyze and interpret complex image data, offering automated frameworks to recognize diseases in medical images. Besides other popular approaches that have emerged, the application of Liquid Neural Networks stands out as a promising possibility to enhance the accuracy and efficiency of medical image analysis. Inspired by the dynamic behavior of liquid systems, LNNs are offering a unique framework that utilizes adaptive and self-organizing neural layers to make predictions. This course work shows the potential of Liquid Neural Networks as a new deep learning technique, with a specific focus on their application to the analysis of diverse medical images. The utilization of the MedMNIST datasets serves as a initial step in this research, providing a great source of medical image data for training and validating neural network models. As the paper progresses, It also dives into the application of Liquid Neural Networks to the analysis of whole slide images of lung tissues.

## Literature review

TODO

## Data

The data comes from two different sources: MedMNIST images of Chest CT and whole slide images of lung tissues from "Genomic Data Commons" Data Portal by  National Cancer Institute. The study starts with the application of Liquid neural networks to MedMNIST collection of medical images to create a solid foundation of the work and check the reliability of constructed models. There is a plenty of papers, which analyze datasets from this collection, and there are multiple benchmarks to compare the performance of our model with. Then it analyzes the whole slide images, which in comparison require major preprocessing and a lot of computational resources. One of the advantages of the dataset include that there are papers, which also use slides from the same source for their research. 

### MedMNIST images

MedMNIST collection of images contain 6 different 3D image sets to train a model on, such as OrganMNIST3D and VesselMNIST3D and etc. There is no strong preference which dataset to use, and in the paper I train the models on NoduleMNIST3D dataset because it is thematically close to the dataset in the second part of the study. It contains Chest CT images of size `28 × 28 × 28` with binary target. The images are already preprocessed and ready to use by a model. Also the size of the datset is relatively small. Overall, it allows to conduct fast experiments on the data to compare the quality of different architectures. 

### Whole slide Lung Tissue images

This dataset of high-resolution lung tissue images contains frozen slices of cancer tissue. There are two of tumor types in the dataset: lung adenocarcinoma (LUAD) and lung squamous cell carcinoma (LUSC). To perform the analysis, the slides are sliced into parts of the same size `2990 x 2990`. On this stage the images with a large share of no tissue are removed and the quality of the images is fixated. Then they are sliced again and packed into 3D batches of `299 x 299 x 100` with `3` channels total. The total size of the dataset is `424` random images out of around `900` from the Data Portal. Not all images were chosen for the study to reduce computational resources needed. 

## Models

### MedMNIST images

TODO

### Whole slide Lung Tissue images

TODO

## Training process

TODO

## Evaluation

TODO

## Results

TODO

## Conclusion

TODO
