# Code for paper "Gender Identity in Pretrained Language Models: An Inclusive Approach to Data Creation and Probing"

This repository contains code related to creating datasets and conducting experiments on language models.

### Dataset Folder
The dataset folder contains scripts for recreating the TRANsCRIPT corpus. It includes modules Wikidata sampling, YouTube data extraction, and preprocessing. The `create_dataset.py` script is the entry point for recreating the final dataset.

### Experiments Folder
The experiments folder is dedicated to running experiments on PLMs. It contains two main components:

- **Probing**: The `probing.py` script includes code for probing the PLMs representations. As an input it uses the representations created from the `probing/dataset.py` script, either frozen or author-controlled using SAGE.

- **Fine-Tuning**: The `fine_tuning.py` script is responsible for fine-tuning.