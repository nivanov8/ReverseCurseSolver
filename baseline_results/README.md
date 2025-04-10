# Baseline Results
Baseline results for the paper

## Experiment 1
1. To finetune the model run ```python -m baseline_results.experiment1.finetune --direction ["p2d", "d2p"] --model_name $MODEL_NAME```
2. To run evaluation run ```python -m joint_optimization.experiment1.evaluate --direction ["p2d", "d2p"] --model_path $MODEL_PATH```

## Experiment 2
1. In the root directory of ReverseCurseSolver run ```python -m baseline_results.experiment2.test_experiment2_base``` this will generate the test results using Llama-3.2-1B
2. In the root directory of ReverseCurseSolver run ```python ./baseline_results/experiment2/plot_experiment2_base.py``` this will save a bar plot to the /figures directory