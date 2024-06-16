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
import variation

CACHE_DIR = "__models__"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Embeddings():
    def __init__(self, model_name, dataset_group_name='cis-gender', controlled=False, dataset_path) -> None:
        self.model_name = model_name
        self.group_name = dataset_group_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, use_fast=False) # Need to set use_fast to False to avoid error from pre-tokenized data
        self.model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)
        self.spacy_nlp = spacy.load('en_core_web_md', exclude=['ner', 'textcat'])

        self.controlled=controlled

        if self.controlled:
            dataset = analysis.load_data(save_cache='dataset_new.pkl', dataset_path=dataset_path, n_process=16, batch_size=100)"
            variation_model = analysis.LinguisticVariationPerUser(dataset)

            user_ids = dataset['user_id'].unique()
            user_most_representative = []

            for target_user_id in tqdm(user_ids):
                variation_model.run_sage(user_id=target_user_id)
                representative_items = variation_model.most_representative_target_words(beta_threshold=1)
                user_most_representative.append(representative_items)

            results = {}
            for user_id, representative_items in zip(user_ids, user_most_representative):
                if len(representative_items) == 0:
                    continue

                words = list(np.array(representative_items)[:, 0].astype(str))
                coef = list(np.array(representative_items)[:, 1].astype(float).round(4))

                results[user_id] = {
                    'words': words,
                    'coef': coef
                }

            self.representative_terms_per_user = results


        

    def __postprocess_embeddings(self, embedding, special_tokens_mask):
        """ Remove special tokens from the embeddings and average into a single vector
        """
        filtered_embedding = embedding[~special_tokens_mask.bool()]
        single_embedding = torch.mean(filtered_embedding, dim=0)

        return single_embedding

    def tokenize(self, examples):
        if not self.controlled:
            tokenized_examples = self.tokenizer(
                text_examples, padding="max_length",
                truncation=True, return_tensors="pt",
                max_length=512, is_split_into_words=False,
                return_special_tokens_mask=True
            )
        else:

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
    argparser.add_argument('--dataset_path', type=str, help='Path to dataset')
    args = argparser.parse_args()

    model_name = model_list[args.model]
    dataset_group_name = group_name_list[args.group]
    batch_size = args.batch_size

    # TODO: add 'controlled' parameter
    embedding_class = Embeddings(model_name=model_name, dataset_group_name=dataset_group_name)

    print(f'Using model {model_name} and dataset group {dataset_group_name}')

    ## === Load dataset ===
    balanced_dataset = datasets.load_from_disk(args.dataset_path)


    # === Extract embeddings ===
    dataset = balanced_dataset.map(embedding_class.tokenize, batched=True, desc="Tokenize")
    dataset = embedding_class.extract_embeddings(dataset, batch_size=batch_size)

    # Only keep the embeddings
    dataset = dataset.select_columns('embedding')

    # === Save dataset ===
    dataset.save_to_disk(f'datasets/{embedding_class.group_name}.{embedding_class.model_name}')