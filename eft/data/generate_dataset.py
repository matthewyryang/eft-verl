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
        
    print(f"Loaded dataset with {len(dataset)} examples.")

    def make_map_fn(split: str, with_hint=False):
        """Create a mapping function to process dataset examples.

        Args:
            split: Dataset split name ('train' or 'test')

        Returns:
            Function that processes individual dataset examples
        """
        def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            data = {
                "data_source": "",
                "prompt": example['problem_prompt' if not with_hint else 'problem_with_hint_prompt'],
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
    hint_dataset = dataset.map(function=make_map_fn(args.split, with_hint=True), with_indices=True)
    print(dataset[0]['prompt'])
    print(dataset[0]['reward_model'])
    print(hint_dataset[0]['prompt'])
    print(hint_dataset[0]['reward_model'])
    dataset = concatenate_datasets([dataset, hint_dataset])
    print(f"Loaded dataset with {len(dataset)} examples.")


    dataset.to_parquet(os.path.join(args.local_dir, f'{args.name}.parquet'))

