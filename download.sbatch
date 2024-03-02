#!/bin/bash
#SBATCH --job-name=lung_tissues_download                 # Название задачи 
#SBATCH --error=lung_tissues_download-%j.err                # Файл для вывода ошибок 
#SBATCH --output=lung_tissues_download-%j.log               # Файл для вывода результатов 
#SBATCH --time=0:05:00                      # Максимальное время выполнения 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=0                            # Требуемое кол-во GPU

module load Python
conda create --name lung_tissues_venv python=3.10
source activate lung_tissues_venv
poetry install

python scripts/yandex_disk_download/main.py
