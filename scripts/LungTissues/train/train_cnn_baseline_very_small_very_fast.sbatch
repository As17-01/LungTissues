#!/bin/bash
#SBATCH --job-name=lung_tissues_train_cnn_baseline_very_small_very_fast                # Название задачи 
#SBATCH --error=lung_tissues_train_cnn_baseline_very_small_very_fast-%j.err                # Файл для вывода ошибок 
#SBATCH --output=lung_tissues_train_cnn_baseline_very_small_very_fast-%j.log               # Файл для вывода результатов 
#SBATCH --time=24:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate lung_tissues_venv

poetry run python main.py dataset=single_dataset model=cnn_baseline_very_small training_params=very_fast
