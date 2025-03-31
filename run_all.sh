#!/bin/bash

MODEL_PATH_PREFIX="./joint_optimization/experiment1/llama3-finetuned-experiment1"

mkdir -p logs

python -m joint_optimization.experiment1.finetune --direction p2d --joint_optimization_method pairing > logs/finetune_p2d_pairing.log
python -m joint_optimization.experiment1.evaluate --direction p2d --model_path $MODEL_PATH_PREFIX-pairing-p2d > logs/evaluate_p2d_pairing.log

python -m joint_optimization.experiment1.finetune --direction d2p --joint_optimization_method pairing > logs/finetune_d2p_pairing.log
python -m joint_optimization.experiment1.evaluate --direction d2p --model_path $MODEL_PATH_PREFIX-pairing-d2p > logs/evaluate_d2p_pairing.log

python -m joint_optimization.experiment1.finetune --direction p2d --joint_optimization_method contrastive > logs/finetune_p2d_contrastive.log
python -m joint_optimization.experiment1.evaluate --direction p2d --model_path $MODEL_PATH_PREFIX-contrastive-p2d > logs/evaluate_p2d_contrastive.log

python -m joint_optimization.experiment1.finetune --direction d2p --joint_optimization_method contrastive > logs/finetune_d2p_contrastive.log
python -m joint_optimization.experiment1.evaluate --direction d2p --model_path $MODEL_PATH_PREFIX-contrastive-d2p > logs/evaluate_d2p_contrastive.log
