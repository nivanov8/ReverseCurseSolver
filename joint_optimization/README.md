# Joint Optimization
Optimize for the likelihood of generating both "A is B" and "B is A" simultaneously by pairing them together and ensuring that each loss used in the gradient computation takes acocunt of both directions.

## Experiment 1
Finetune: `python -m joint_optimization.experiment1.finetune`
Evaluate: `python -m joint_optimization.experiment1.evaluate`