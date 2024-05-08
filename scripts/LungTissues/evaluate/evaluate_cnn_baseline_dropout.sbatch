#!/bin/bash
#SBATCH --job-name=lung_tissues_evaluate_cnn_baseline_dropout                # Название задачи 
#SBATCH --error=lung_tissues_evaluate_cnn_baseline_dropout-%j.err                # Файл для вывода ошибок 
#SBATCH --output=lung_tissues_evaluate_cnn_baseline_dropout-%j.log               # Файл для вывода результатов 
#SBATCH --time=10:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate lung_tissues_venv

poetry run python main.py dataset=single_dataset model=cnn_baseline_dropout
