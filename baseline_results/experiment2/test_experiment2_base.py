import argparse
import math
import os
import pandas as pd
from tqdm import tqdm
from src.tasks.celebrity_relations.parent_reversals import ParentChildPair, PromptCompletionDataset
from src.models.model import Model
from torch.utils.data import DataLoader
from accelerate import Accelerator

DF_SAVE_PATH = "data/celebrity_relations/parent_child_pairs.csv"
SAVE_PATH = "data/celebrity_relations"

FEW_SHOT_PROMPT = """Below is a converation with a helpful and terse assistant. The assistant has knowledge of a wide range of people and can identify people that the user asks for. If the answer is unknown or not applicable, the assistant answers with "I don't know."

Q: Name a child of Barack Obama.
A: Malia Obama
Q: Who is Elon Musk's mother?
A: Maye Musk
Q: Who is Kathy Pratt's mother?
A: I don't know.
Q: Who is Chris Hemsworth's father?
A: Craig Hemsworth
Q: Name a child of Karen Lawrence.
A: Jennifer Lawrence
Q: Who is Aaron Taylor-Johnson's mother?
A: Sarah Johnson"""

accelerator = Accelerator()

def get_os_model_logits(model, dataloader):
    logprobs = []

    for inputs_batch, completions_batch in tqdm(dataloader):
        logprobs_batch = model.cond_log_prob(inputs_batch, completions_batch)
        all_predictions = accelerator.gather_for_metrics((logprobs_batch))

        logprobs.extend(all_predictions.cpu().tolist())

    return logprobs

def create_dataloader(prompts, completions, batch_size=1):
    assert all([len(completion) == 1 for completion in completions])
    completions = [completion[0] for completion in completions]

    dataset = PromptCompletionDataset(prompts, completions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def get_prompts_completions(reversals_df: pd.DataFrame, query_type: str) -> tuple[list, list]:
    prompts = []
    completions = []
    for _, row in list(reversals_df.iterrows()):
        if query_type == "parent":
            question = (
                "Q: " + ParentChildPair(child=row["child"], parent=row["parent"], parent_type=row["parent_type"]).ask_for_parent()
            )
            completion = " " + row["parent"]
        elif query_type == "child":
            question = ParentChildPair(child=row["child"], parent=row["parent"], parent_type=row["parent_type"]).ask_for_child()
            completion = " " + row["child"]
        else:
            raise ValueError(f"Invalid query_type: {query_type}")
        prompts.append("\n".join([FEW_SHOT_PROMPT, question, "A:"]))
        completions.append([completion])

    return prompts, completions

def test_can_reverse_complete(reversals_df, model_name) -> tuple[list, list]:
    prompts_parent, completions_parent = get_prompts_completions(reversals_df, "parent")
    prompts_child, completions_child = get_prompts_completions(reversals_df, "child")

    if model_name.startswith("meta-llama"):
        model = Model.from_id(model_name)
        batch_size = 20
        parent_dataloader = create_dataloader(prompts_parent, completions_parent, batch_size=batch_size)
        child_dataloader = create_dataloader(prompts_child, completions_child, batch_size=batch_size)

        model.model, parent_dataloader, child_dataloader = accelerator.prepare(model.model, parent_dataloader, child_dataloader)
        parent_logprobs = get_os_model_logits(model, parent_dataloader)
        child_logprobs = get_os_model_logits(model, child_dataloader)

    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    return parent_logprobs, child_logprobs

def reversal_test(model_name: str, reversals_df: pd.DataFrame) -> pd.DataFrame:
    parent_probs, child_probs = test_can_reverse_complete(reversals_df, model_name)
    return pd.DataFrame(
        {
            "child": reversals_df["child"],
            "parent": reversals_df["parent"],
            "parent_type": reversals_df["parent_type"],
            "child_prediction": reversals_df["child_prediction"],
            f"{model_name}_parent_logprob": parent_probs,
            f"{model_name}_child_logprob": child_probs,
        }
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()
    return args

def main(model_name: str):
    reversals_df = pd.read_csv(DF_SAVE_PATH)
    reversal_test_results = reversal_test(model_name, reversals_df)

    # save dataframe
    model_name = model_name.replace("/", "-")
    reversal_test_results.to_csv(os.path.join(SAVE_PATH, f"{model_name}_reversal_test_results.csv"), index=False)

    print(reversal_test_results.head())

if __name__ == "__main__":
    args = parse_args()

    main(model_name=args.model)