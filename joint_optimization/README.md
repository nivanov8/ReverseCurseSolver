# Joint Optimization
Simultaneously optimize for the likelihood of both "A is B" and "B is A". 

## Experiment 1
Finetune: `python -m joint_optimization.experiment1.finetune --direction [p2d, d2p] --joint_optimization_method [pairing, contrastive]`

- `--direction` sets direction of training in the additional fine-tuning step
- `--joint_optimization` determines the joint optimization strategy. `pairing` pairs up complementary statements ("A is B" and "B is A") within the same batch so that both statements in the pair are used for each gradient update. `contrastive` tries adds an additional optimization objective to minimize the distance between the last hidden state embeddings between "A is B" and "B is A".

Evaluate: `python -m joint_optimization.experiment1.evaluate --direction [p2d, d2p] --model_path <path_to_model>`

Or simply run `./run_all.sh` to run all combinations.
