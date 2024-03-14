#!/bin/bash
#SBATCH --job-name=lung_tissues_train_cnn_baseline                # Название задачи 
#SBATCH --error=lung_tissues_train_cnn_baseline-%j.err                # Файл для вывода ошибок 
#SBATCH --output=lung_tissues_train_cnn_baseline-%j.log               # Файл для вывода результатов 
#SBATCH --time=00:02:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate lung_tissues_venv

poetry run python main.py model=cnn_baseline
