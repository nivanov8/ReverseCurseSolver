# ReverseCurseSolver
### Environment Setup
1. Ensure miniconda3 is installed
2. Source conda with ```source $YOUR_PATH/miniconda3/bin/activate```
3. If you are creating the environment for the first time, navigate to the ReverseCurseSolver root directory run ```conda env create -f environment.yml```, if the environment is already created go to step 4
4. Run ```conda activate LLMProject```

### Hugging Face Access Token
1. To be able to run Llama models create a local .env file in the root directory of ReverseCurseSolver
2. In the .env file create a variable ```HF_TOKEN = $YOUR_HF_TOKEN```

### Running Llama
1. To run standard Llama inference, in the root directory of ReverseCurseSolver run ```python inference/LlamaStandard.py```


### Reproducing Experiment #2
1. In the root directory of ReverseCurseSolver run ```python -m baseline_results.experiment2.test_experiment2_base``` this will generate the test results using Llama-3.2-1B
2. In the root directory of ReverseCurseSolver run ```python ./baseline_results/experiment2/plot_experiment2_base.py``` this will save a bar plot to the ```/figures``` directory