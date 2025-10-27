import argparse
import os
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="self-label-zanette-lab/knight-knave-3")
    parser.add_argument("--save_name", default="train", help="saved parquet name")
    
    args = parser.parse_args()

    DATASET = args.dataset
    SAVE_NAME = args.save_name
    
    local_dir = os.path.join(os.environ['CACHE'], 'verl-data', f'knk{DATASET.split("-")[-1]}') 

    dataset = datasets.load_dataset(DATASET)
    current_dataset = dataset['train']

    instruction_following = 'Let\'s think step by step and output the final answer (should be a sentence) within \\boxed{{}}.'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = answer_raw
            data = {
                "data_source": DATASET,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    processed_dataset = current_dataset.map(function=make_map_fn('train'), with_indices=True)

    os.makedirs(local_dir, exist_ok=True)

    output_file = os.path.join(local_dir, f"{SAVE_NAME}.parquet")
    processed_dataset.to_parquet(output_file)
