#!/bin/bash
#
#SBATCH --job-name=ElectionAnalysis
#SBATCH --output=/mnt/nfs/work1/696ds-s21/hshukla/2020ElectionAnalysis/output_%j.txt
#SBATCH -e /mnt/nfs/work1/696ds-s21/hshukla/2020ElectionAnalysis/error_%j.txt
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

source /home/hshukla/anaconda3/bin/activate Aspect-Based-Sentiment-Analysis
python3 sentiment.py
