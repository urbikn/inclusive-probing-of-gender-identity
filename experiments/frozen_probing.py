import random
import os
import argparse

import torch
import datasets
import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from probing.probes import ProbingClassifier, MDLProbingClassifier


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=int, default=0, help='Index of the model to use:' + str(model_list))
    argparser.add_argument('--groups',  nargs='+', default=['cis man', 'cis woman'], help='Name of the group to compare against (min 2). Possible:' + str(gender_identities))
    argparser.add_argument('--mdl',  type=bool, default=False, help='Whether to use the MDL probing classifier')
    argparser.add_argument('--save_to',  type=str, default='probing_results.csv', help='Path to save the results to')
    args = argparser.parse_args()

    model_list = "roberta-base,roberta-large".split(",") + \
        "microsoft/deberta-base,microsoft/deberta-large".split(",") + \
        "microsoft/deberta-v3-base,microsoft/deberta-v3-large,microsoft/deberta-v3-xsmall,microsoft/deberta-v3-small".split(",")

    gender_identities = [
        'cis man',
        'cis woman',
        'trans man',
        'trans woman',
        'non-binary',
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_name in model_list:
        compare_groups = sorted(args.groups)
        compare_groups_str = '-'.join(compare_groups).replace(' ', '_')

        if len(compare_groups) < 2 or any([not group in gender_identities for group in compare_groups]):
            raise ValueError('You must specify at least two valid groups to compare against')

        def merge_datasets(dataset, dataset_embeddings, label_to_id):
            data_dict = {}

            for key in dataset.keys():
                data = dataset[key].to_pandas()
                embedding = dataset_embeddings[key].to_pandas()

                merged_df = pd.concat([data, embedding], axis=1)
                merged_df = merged_df[~merged_df['embedding'].apply(lambda x: np.isnan(x).any())]
                merged_df['label'] = merged_df['label'].map(label_to_id)

                data_dict[key] = datasets.Dataset.from_pandas(merged_df)
            
            return DatasetDict(data_dict)


        groups_to_load = []
        if 'cis' in compare_groups_str:
            groups_to_load.append('cis-gender')
        if 'trans' in compare_groups_str or 'non-binary' in compare_groups_str:
            groups_to_load.append('trans_and_non-binary')


        dataset = []
        for group in groups_to_load:
            dataset_group = datasets.load_from_disk(f'../datasets/{group}.{NUM_EXAMPLES}')
            experiment_folder = f'datasets/{group}.{model_name}/'
            dataset_embeddings = datasets.load_from_disk(experiment_folder)

            labels = ['cis woman', 'cis man', 'trans man', 'trans woman', 'non-binary']
            label_to_id = {label: i for i, label in enumerate(labels)}
            id_to_label = {i: label for i, label in enumerate(labels)}

            dataset_group = merge_datasets(dataset_group, dataset_embeddings, label_to_id)
            dataset.append(dataset_group)

        dataset = DatasetDict({key:concatenate_datasets([dataset[key] for dataset in dataset])
            for key in dataset[0].keys()
            })
        
        # Filter out groups that are not in the compare_groups
        disable_progress_bar()

        for split in dataset.keys():
            compare_groups_id = [label_to_id[group] for group in compare_groups]
            dataset[split] = dataset[split].filter(lambda x: x['label'] in compare_groups_id)

            dataset[split] = dataset[split].shuffle(seed=42)

            def rename_column_values(example):
                example['label'] = compare_groups.index(id_to_label[example['label']])

                return example

            dataset[split] = dataset[split].map(rename_column_values)


        # Keep only the ones you want to compare 
        label_to_id = {label: i for i, label in enumerate(compare_groups)}
        enable_progress_bar()

        print(model_name)
        print(dataset)
        dataset.set_format(type='torch')

        dataset = dataset.class_encode_column('label')

        random_numbers = [42, 12, 24, 36, 48]
        results = []
        for i in range(5):
            random_number = random_numbers[i]
            random.seed(random_number)
            torch.manual_seed(random_number)

            print(f'===== Run {i+1} ====')
            if not args.mdl:
                probe = ProbingClassifier(
                    input_dim=dataset['train'][0]['embedding'].shape[0],
                    num_classes=len(label_to_id),
                    label_to_id=label_to_id,
                    id_to_label={v:k for k, v in label_to_id.items()},
                    device=device,
                    progress_bar=False
                )

                probe.train(train_dataset=dataset['train'], validation_dataset=dataset['validation'])
                loss, result = probe.evaluate(dataset['test'])
                results.append(result)

                # Remove the cached model
                if os.path.exists('.classifiers/' + probe.ID):
                    os.remove('.classifiers/' + probe.ID)
            else:
                mdl_probe = MDLProbingClassifier(
                        input_dim=dataset['train'][0]['embedding'].shape[0],
                        num_classes=len(label_to_id),
                        label_to_id=label_to_id,
                        id_to_label={v:k for k, v in label_to_id.items()},
                        device=device
                )

                report = mdl_probe.analyze(dataset['train'], dataset['validation'], dataset['test'])
                results.append({'mdl': [report['mdl']]})

                # Remove the cached model
                if os.path.exists('.classifiers/' + mdl_probe.probing_model.ID):
                    os.remove('.classifiers/' + mdl_probe.probing_model.ID)

        results_df = pd.concat([pd.DataFrame(result).reset_index() for result in results])

        results_df.to_csv(f"{args.save_to}.{model_name.replace('/', '_')}", index=False)


if __name__ == '__main__':
    main()