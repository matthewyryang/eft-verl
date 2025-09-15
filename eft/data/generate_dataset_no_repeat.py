import os
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional, Any
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir')
    parser.add_argument('--remote_dir')
    parser.add_argument('--split', default=['train'], nargs='+')
    parser.add_argument('--name')
    

    args = parser.parse_args()

    datasets_to_load = args.split
    datasets = []

    for dataset_name in datasets_to_load:
        datasets.append(load_dataset(args.remote_dir, split=dataset_name))

    dataset = concatenate_datasets(datasets)
    seen_row_ids = set()
    unique_indices = []
    for idx, row_id in enumerate(dataset["row_id"]):
        if row_id not in seen_row_ids:
            seen_row_ids.add(row_id)
            unique_indices.append(idx)
    dataset = dataset.select(unique_indices)
        
    print(f"Loaded dataset with {len(dataset)} examples.")

    def make_map_fn(split: str):
        """Create a mapping function to process dataset examples.

        Args:
            split: Dataset split name ('train' or 'test')

        Returns:
            Function that processes individual dataset examples
        """
        def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            data = {
                "data_source": "",
                "prompt": [{
                    "role": "user",
                    "content": example['problem']
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example['answer']
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn

    dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)
    print(f"Loaded dataset with {len(dataset)} examples.")


    dataset.to_parquet(os.path.join(args.local_dir, f'{args.name}.parquet'))

