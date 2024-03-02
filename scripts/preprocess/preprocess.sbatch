#!/bin/bash
#SBATCH --job-name=lung_tissues_preprocess                 # Название задачи 
#SBATCH --error=lung_tissues_preprocess-%j.err                # Файл для вывода ошибок 
#SBATCH --output=lung_tissues_preprocess-%j.log               # Файл для вывода результатов 
#SBATCH --time=0:30:00                      # Максимальное время выполнения 
#SBATCH --cpus-per-task=3                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=0                            # Требуемое кол-во GPU

module load Python
source activate lung_tissues_venv

poetry run python main.py
