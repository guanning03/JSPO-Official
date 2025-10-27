import argparse
import os
os.environ['CURL_CA_BUNDLE'] = ''
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--no_test", action="store_true")

    args = parser.parse_args()
    
    data_source = args.dataset
    
    local_dir = os.path.join(os.environ["CACHE"], "verl-data", data_source.split("/")[-1])

    dataset = datasets.load_dataset(data_source)

    train_dataset = None
    if not args.no_train:
        train_dataset = dataset["train"]
    
    test_dataset = None
    if not args.no_test:
        test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = question_raw + " " + instruction_following

            answer_raw = str(example.pop("answer"))
            solution = answer_raw
            data = {
                "data_source": data_source,
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

    if not args.no_train:
        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if not args.no_test:
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))