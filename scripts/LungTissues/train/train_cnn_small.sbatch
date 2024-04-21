#!/bin/bash
#SBATCH --job-name=lung_tissues_train_cnn_small                # Название задачи 
#SBATCH --error=lung_tissues_train_cnn_small-%j.err                # Файл для вывода ошибок 
#SBATCH --output=lung_tissues_train_cnn_small-%j.log               # Файл для вывода результатов 
#SBATCH --time=96:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate lung_tissues_venv

poetry run python main.py dataset=single_dataset model=cnn_small
