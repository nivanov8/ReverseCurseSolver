import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_PATH = "data/celebrity_relations"

# load dataframe from csv
df = pd.read_csv(os.path.join(SAVE_PATH, "parent_child_pairs.csv"))
sns.set(font_scale=1.5)

def model_name_to_official(model_name: str) -> str:
    if model_name.startswith("llama"):
        return "LLaMA" + model_name[len("llama"):]
    elif model_name.startswith("gpt"):
        return "GPT" + model_name[len("gpt"):]
    else:
        return model_name

def get_results_df(model_name: str) -> pd.DataFrame:
    path = os.path.join(SAVE_PATH, f"{model_name}_reversal_test_results.csv")
    results_df = pd.read_csv(path)

    #just modify the model_name string
    parts = model_name.split("-", 2)
    model_name = model_name_fixed = f"{parts[0]}-{parts[1]}/{parts[2]}"

    results_df[f"{model_name}_parent_prob"] = results_df[f"{model_name}_parent_logprob"].apply(lambda x: np.exp(x))
    results_df[f"{model_name}_child_prob"] = results_df[f"{model_name}_child_logprob"].apply(lambda x: np.exp(x))

    return results_df
    
def combine_completion_results(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combines completion results for multiple models."""
    while len(dfs) > 1:
        df1 = dfs.pop()
        df2 = dfs.pop()
        combined_df = pd.merge(df1, df2)
        dfs.append(combined_df)

    return dfs[0]

llama1b_df = get_results_df("meta-llama-Llama-3.2-1B")

# llama30b_df = get_results_df("llama-30b")
# llama65b_df = get_results_df("llama-65b")
# davinci_df = get_results_df("davinci")
# gpt35_df = get_results_df("gpt-3.5-turbo")

combined_df = combine_completion_results([llama1b_df])
print(combined_df.head())


def bar_plot_completions(df: pd.DataFrame, model_names: list[str], title: str = None, name: str = None):
    """
    Args:
        df: dataframe containing completion results
        model_names: names of models to plot
        title: title of plot
    """
    # sns.set(font_scale=1.2)
    sns.set_theme(style="white", font_scale=1.2)

    
    # get percentage of relations that can be found for each model
    percentages = []
    for model_name in model_names:
        parent_field = f"{model_name}_parent_prob"
        child_field = f"{model_name}_child_prob"
            
        parent_percentage = df[parent_field].mean() * 100
        child_percentage = df[child_field].mean() * 100
        percentages.append((parent_percentage, child_percentage))

    # create a bar plot
    barWidth = 0.35
    r1 = range(len(model_names))
    r2 = [x + barWidth for x in r1]

    # plot data
    plt.bar(r1, [i[0] for i in percentages], width=barWidth, label='Parent')
    plt.bar(r2, [i[1] for i in percentages], width=barWidth, label='Child')

    # Add xticks in the middle of the group bars
    plt.xlabel('Models')
    
    # Calculate midpoints for tick positions
    midpoints = [(a + b) / 2 for a, b in zip(r1, r2)]
    
    plt.xticks(midpoints, [model_name_to_official(m) for m in model_names])

    plt.ylabel("Accuracy (%)")
    if title:
        plt.title(title)

    # Create legend & Show graphic
    plt.legend()
    plt.tight_layout()
    # save plot
    if name:
        if not os.path.exists("figures"):
            os.makedirs("figures")
        plt.savefig(os.path.join("figures", f"{name}.pdf"), format="pdf")
    plt.show()

bar_plot_completions(combined_df, ["meta-llama/Llama-3.2-1B"], name="Experiment_2_figure_1")