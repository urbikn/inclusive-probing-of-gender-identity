import argparse
import os
import random
import time
import json

import numpy as np
from sklearn import metrics
import pandas as pd
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch
from torch import nn
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import disable_progress_bar, enable_progress_bar
from transformers import Trainer, compute_metrics

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a task")
    parser.add_argument("--model", type=int, help="The model to fine-tune (0-7)", default=0)
    parser.add_argument( "--batch_size", type=int, help="The batch size to use", default=8)
    parser.add_argument( "--device", type=int, help="Device to train on", default=0)
    parser.add_argument( "--groups", nargs="+", default=[0, 1], help="Name of the group to compare against (min 2)")
    parser.add_argument( "--dataset_paths", nargs="+", help="Paths to the datasets to use")

    args = parser.parse_args()

    model_list = [
        "microsoft/deberta-large",
        "microsoft/deberta-v3-large",
        "roberta-large",
        "roberta-base",
        "microsoft/deberta-base",
        "microsoft/deberta-v3-base",
        "microsoft/deberta-v3-small",
        "microsoft/deberta-v3-xsmall",
    ]

    model_index = args.model
    model_name = model_list[model_index]

    focus_on = [
        "trans woman",
        "trans man",
        "cis woman",
        "cis man",
        "non-binary",
    ]
    # group_name_list = ["cis-gender", "trans_and_non-binary"]
    gender_identities = focus_on

    if len(args.groups) != 5:
        groups = [int(i) for i in args.groups]
        focus_on = [label for i, label in enumerate(focus_on) if i in groups]

    compare_groups = focus_on
    compare_groups_str = "-".join(compare_groups).replace(" ", "_")
    if len(compare_groups) < 2 or any(
        [not group in gender_identities for group in compare_groups]
    ):
        raise ValueError(
            "You must specify at least two valid groups to compare against"
        )

    groups_to_load = []
    if "cis" in compare_groups_str:
        groups_to_load.append("cis-gender")
    if "trans" in compare_groups_str or "non-binary" in compare_groups_str:
        groups_to_load.append("trans_and_non-binary")

    dataset, label_to_id, id_to_label = load_dataset(args.dataset_paths, compare_groups_str)

    cache_dir = "__models__"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for seed in [42, 12, 24, 36, 48]:
        model, tokenizer, tokenized_datasets = setup_model(
            model_name, label_to_id, cache_dir, seed
        )

        num_train_epochs = 3
        train_batch_size = args.batch_size
        eval_time_steps = 10
        gradient_acc_size = 1
        total_steps = int(
            (len(tokenized_datasets["train"]) / train_batch_size)
            * (num_train_epochs)
        ) // gradient_acc_size

        if len(args.groups) != 5:
            output_dir = "__models__/binary-" + model_name.replace("/", "_")
        else:
            output_dir = "__models__/fiveway-" + model_name.replace("/", "_")

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            eval_steps=total_steps // eval_time_steps,
            save_steps=total_steps // eval_time_steps,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_acc_size,
        )

        train_model(
            model,
            tokenizer,
            tokenized_datasets,
            training_args,
            label_to_id,
            id_to_label,
            cache_dir,
            seed,
        )

class ModelForSequenceClassification(torch.nn.Module):
    def __init__(self, model_name, num_classes, cache_dir, seed=12):
        super(ModelForSequenceClassification, self).__init__()
        self.llm = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        self.num_classes = num_classes
        
        print('Hidden size:', self.llm.config.hidden_size)
        print('Num classes:', num_classes)

        hidden_dim = 100
        self.classification_head = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, special_tokens_mask, labels):
        llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden state
        embeddings = llm_output.last_hidden_state

        # Remove special tokens from the embeddings and average into a single vector
        embeddings_avg = torch.stack([output[mask == 0].mean(dim=0) if mask.sum() > 0 else torch.zeros_like(output[0]) for output, mask in zip(embeddings, special_tokens_mask)])

        embeddings_avg = torch.nan_to_num(embeddings_avg, nan=0.0)

        # Get the classification output
        logits = self.classification_head(embeddings_avg)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def load_dataset(filepaths, compare_groups):
    dataset = []
    for filepath in filepaths:
        dataset_group = load_from_disk(filepath)

        labels = [
            "trans woman",
            "trans man",
            "cis woman",
            "cis man",
            "non-binary",
        ]
        label_to_id = {label: i for i, label in enumerate(labels)}
        id_to_label = {i: label for i, label in enumerate(labels)}

        data_dict = {}
        for key in dataset_group.keys():
            data = dataset_group[key].to_pandas()
            data["label"] = data["label"].map(label_to_id)
            data_dict[key] = Dataset.from_pandas(data)

        dataset_group = DatasetDict(data_dict)
        dataset.append(dataset_group)

    dataset = DatasetDict(
        {
            key: concatenate_datasets(
                [dataset[key] for dataset in dataset]
            )
            for key in dataset[0].keys()
        }
    )

    # Filter out groups that are not in the compare_groups
    disable_progress_bar()

    for split in dataset.keys():
        compare_groups_id = [label_to_id[group] for group in compare_groups]
        dataset[split] = dataset[split].filter(
            lambda x: x["label"] in compare_groups_id
        )

        # Shuffle the dataset
        dataset[split] = dataset[split].shuffle(seed=42)

        def rename_column_values(example):
            example["label"] = compare_groups.index(id_to_label[example["label"]])

            return example

        dataset[split] = dataset[split].map(rename_column_values)

    enable_progress_bar()
    dataset.set_format(type="torch")

    return dataset, label_to_id, id_to_label

def setup_model(model_name, label_to_id, cache_dir, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if "deberta" not in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=False
        )

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=512,
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = tokenized_datasets.map(tokenize, batched=True)

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    return ModelForSequenceClassification(
        model_name, len(label_to_id), cache_dir
    ).to(device), tokenizer, tokenized_datasets

def train_model(model, tokenizer, tokenized_datasets, training_args, label_to_id, id_to_label, cache_dir, seed):

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(f"{training_args.output_dir}/final_model")

    if os.path.exists(f"{training_args.output_dir}/checkpoint-500"):
        os.system(f"rm -r {training_args.output_dir}/checkpoint-*")

    test_predictions = trainer.predict(tokenized_datasets["test"])
    predictions = test_predictions.predictions.argmax(axis=1)
    references = test_predictions.label_ids

    with open('results.json', "w") as f:
        results = {
            "report": metrics.classification_report(
                references,
                predictions,
                target_names=[id_to_label[i] for i in range(len(label_to_id))],
                output_dict=True,
            ),
            "label2id": label_to_id,
            "predictions": [int(pred) for pred in predictions],
            "references": [int(ref) for ref in references],
        }

        f.write(json.dumps(results, indent=2))

    # delete the model and optimizer
    del model
    del trainer

    # empty the CUDA cache
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
