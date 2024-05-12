#!/bin/bash
#SBATCH --job-name=mnist_3d_full_liquid_conf0                # Название задачи 
#SBATCH --error=mnist_3d_full_liquid_conf0-%j.err                # Файл для вывода ошибок 
#SBATCH --output=mnist_3d_full_liquid_conf0-%j.log               # Файл для вывода результатов 
#SBATCH --time=01:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate lung_tissues_venv

poetry run python 3d.py model=mnist3d_full_liquid_conf0
