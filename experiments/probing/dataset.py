import os
import json
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch
from torch.utils.data import Dataset, DataLoader
import spacy

CACHE_DIR = "__models__"
FILEPATH_OF_REPRESENTATIVE_ITEMS = '../../variation_analysis/results/per_user_representative_items_threshold_1_norm.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Embeddings():
    def __init__(self, model_name, dataset_group_name='cis-gender') -> None:
        self.model_name = model_name
        self.group_name = dataset_group_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, use_fast=False) # Need to set use_fast to False to avoid error from pre-tokenized data
        self.model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)

        self.spacy_nlp = spacy.load('en_core_web_md', exclude=['ner', 'textcat'])

        with open(FILEPATH_OF_REPRESENTATIVE_ITEMS, 'r') as f:
            self.representative_terms_per_user = json.load(f)
        
        self.overall_tokens = 0
        self.masked_tokens = 0
        

    def __postprocess_embeddings(self, embedding, special_tokens_mask):
        """ Remove special tokens from the embeddings and average into a single vector
        """
        filtered_embedding = embedding[~special_tokens_mask.bool()]
        single_embedding = torch.mean(filtered_embedding, dim=0)

        return single_embedding

    def tokenize(self, examples):
        text_examples = []

        # Mask the representative terms in the text
        # - to do this, we first tokenize the text and then replace the tokens that are representative terms with the mask token
        for i, tokens in enumerate(self.spacy_nlp.pipe(examples['text'])):
            user_id = examples['user_id'][i]
            representative_terms = self.representative_terms_per_user.get(user_id, {'words': []})['words']

            self.overall_tokens += len(tokens)
            mask_indexes = []
            for i, token in enumerate(tokens):
                if token.lemma_.lower() in representative_terms:
                    self.masked_tokens += 1
                    mask_indexes.append(i)
            
            tokens = [token.text if i not in mask_indexes else self.tokenizer.mask_token for i, token in enumerate(tokens)]
            text_examples.append(tokens)

        tokenized_examples = self.tokenizer(
            text_examples, padding="max_length",
            truncation=True, return_tensors="pt",
            max_length=512, is_split_into_words=True,
            return_special_tokens_mask=True
        )

        # Because there were instances where the mask token was not recognized as a special token, 
        # we manually set the special tokens mask
        for i in range(len(examples)):
            input_ids = tokenized_examples['input_ids'][i]
            special_tokens_mask  = tokenized_examples['special_tokens_mask'][i]

            for l, input_id in enumerate(input_ids):
                if input_id == self.tokenizer.mask_token_id:
                    special_tokens_mask[l] = 1
            
            tokenized_examples['special_tokens_mask'][i] = special_tokens_mask

        return tokenized_examples

    
    def extract_embeddings(self, dataset, batch_size=128, desc=None):
        """ Extracts the embeddings from the given dataset using the model.

        params:
            dataset: Dataset object with the dataset (it should contain already tokenized data!)
            batch_size: batch size for the model
        """
        # Extract embeddings for each split
        if isinstance(dataset, datasets.DatasetDict):
            dataset_dict = {}
            for split in dataset.keys():
                dataset_dict[split] = self.extract_embeddings(dataset[split], batch_size=batch_size, desc=f"Extract {split}")
            
            dataset = DatasetDict(dataset_dict)
        else:
            embeddings = torch.zeros(size=(len(dataset), self.model.config.hidden_size))

            with torch.no_grad():
                # Get the embeddings for each sentence
                for i in tqdm(range(0, len(dataset), batch_size), desc=f"Extracting embeddings from model {self.model_name} for {self.group_name} ({desc})"):
                    input = torch.LongTensor(dataset["input_ids"][i:i+batch_size]).to(device)
                    attention_mask = torch.LongTensor(dataset["attention_mask"][i:i+batch_size]).to(device)
                    special_tokens_mask = torch.LongTensor(dataset["special_tokens_mask"][i:i+batch_size])

                    model_output = self.model(input, attention_mask=attention_mask)

                    # Get the last hidden state and detach from the graph
                    output = model_output.last_hidden_state.cpu()
                    input = input.cpu()

                    # Extract singe embedding representation for each input
                    for l in range(0, batch_size):
                        if l >= input.shape[0]:
                            break

                        embedding = self.__postprocess_embeddings(embedding=output[l], special_tokens_mask=special_tokens_mask[l])
                        embeddings[i+l] = embedding

            # Add embeddings to the dataset
            dataset = dataset.add_column('embedding', embeddings.tolist())

        return dataset



if __name__ == '__main__':

    model_list = "roberta-base,roberta-large".split(",") + \
        "microsoft/deberta-base,microsoft/deberta-large".split(",") + \
        "microsoft/deberta-v3-base,microsoft/deberta-v3-large,microsoft/deberta-v3-xsmall,microsoft/deberta-v3-small".split(",")

    group_name_list = [
        'cis-gender',
        'trans_and_non-binary'
    ]

    # Arg parse that gets index for model and dataset group
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=int, default=0, help='Index of the model to use:' + str(model_list))
    argparser.add_argument('--group', type=int, default=0, help='Index of the dataset group to use:\n' + str(group_name_list))
    argparser.add_argument('--batch_size', type=int, default=64, help='Batch size for extracting embeddings')
    args = argparser.parse_args()

    model_name = model_list[args.model]
    dataset_group_name = group_name_list[args.group]
    batch_size = args.batch_size
    embedding_class = Embeddings(model_name=model_name, dataset_group_name=dataset_group_name)

    print(f'Using model {model_name} and dataset group {dataset_group_name}')

    ## === Load dataset ===
    num_examples = 6478

    file_path = f'../datasets/{embedding_class.group_name}.{num_examples}'

    # Check if directory exists
    if not os.path.exists(file_path):
        # Then create it
        dataset_filepath = f'../../../data/processed/topic_filtered/{embedding_class.group_name}/segment_256_with_metadata.csv'
        if not os.path.exists(dataset_filepath):
            raise ValueError(f'Dataset file {dataset_filepath} does not exist')

        dataset = pd.read_csv(dataset_filepath, sep='|')
        dataset.columns = ['text', 'label', 'user_id', 'doc_id', 'topic_id']

        dataset = datasets.Dataset.from_pandas(dataset)

        # Sample equal number of examples from each class
        def sample_examples(dataset, N_samples):
            sorted_dataset = dataset.sort("label")
            unique_labels = set(sorted_dataset['label'])
            datasets_by_label = {label: sorted_dataset.filter(lambda x: x['label'] == label) for label in unique_labels}

            sampled_datasets = [dataset.shuffle(seed=12).select(range(N_samples)) for dataset in datasets_by_label.values()]
            balanced_dataset = concatenate_datasets(sampled_datasets)

            return balanced_dataset

        balanced_dataset = sample_examples(dataset, num_examples)

        # Split into train/val/test sets with 80/10/10 split
        train_test_dataset = balanced_dataset.train_test_split(test_size=0.2, seed=12)
        val_test_dataset = train_test_dataset['test'].train_test_split(test_size=0.5, seed=12)

        # Combine the splits into a DatasetDict
        balanced_dataset= DatasetDict({
            'train': train_test_dataset['train'],
            'validation': val_test_dataset['train'],
            'test': val_test_dataset['test'] 
        })

        # Save the dataset
        balanced_dataset.save_to_disk(file_path)
    else:
        print(f'Loading dataset from disk')
        balanced_dataset = datasets.load_from_disk(file_path)


    # === Extract embeddings ===
    dataset = balanced_dataset.map(embedding_class.tokenize, batched=True, desc="Tokenize")
    print('Num. overall tokens:', embedding_class.overall_tokens, 'Num. masked tokens:', embedding_class.masked_tokens)
    dataset = embedding_class.extract_embeddings(dataset, batch_size=batch_size)

    # Only keep the embeddings
    dataset = dataset.select_columns('embedding')

    # === Save dataset ===
    dataset.save_to_disk(f'datasets/{embedding_class.group_name}.{embedding_class.model_name}')