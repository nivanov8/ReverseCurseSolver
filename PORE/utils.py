from datasets import load_dataset, Dataset
def preprocess(example):
    pairs = []
    for key in example:
        if key.startswith("qa_"):
            prompt = example[key]
            if '?' in prompt:
                question, answer = prompt.split('?', 1)
                prompt_clean = question.strip() + '?'
                answer_clean = answer.strip()
                if prompt_clean and answer_clean:
                    pairs.append({
                        "prompt": example["origin_prompt"]+" "+prompt_clean,
                        "completion": answer_clean
                    })
    return pairs

def get_formatted_dataset(path):
    dataset = load_dataset("json", data_files=path)['train']
    all_pairs = []
    for example in dataset:
        all_pairs.extend(preprocess(example))
    dataset = load_dataset("json", data_files=path.replace("dataset","dataset_1"))['train']
    for example in dataset:
        all_pairs.extend(preprocess(example))
    dataset = load_dataset("json", data_files=path.replace("dataset","dataset_2"))['train']
    for example in dataset:
        all_pairs.extend(preprocess(example))

    # Create new HF dataset
    formatted_dataset = Dataset.from_list(all_pairs)
    return formatted_dataset