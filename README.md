# Gender Identity in Pretrained Language Models 

This repository contains code associated with the paper Gender Identity in Pretrained Language Models: An Inclusive Approach to Data Creation and Probing.

> [!TIP]
> Please contact the authors to obtain access to the existing TRANsCRIPT corpus.

## Getting started

### Installation

Clone the repository, create a virtual environment, and install the requirements:

```bash
git clone https://github.com/urbikn/inclusive-probing-of-gender-identity.git
cd inclusive-probing-of-gender-identity
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project structure

### Dataset folder
The dataset folder contains scripts for recreating the TRANsCRIPT corpus. It includes modules Wikidata sampling, YouTube data extraction, and preprocessing. The `create_dataset.py` script is the entry point for recreating the final dataset.



### Experiments folder
The experiments folder is dedicated to running experiments on PLMs. It contains two main components:

- **Probing**: The `probing.py` script includes code for probing the PLMs representations. As an input it uses the representations created from the `probing/dataset.py` script, either frozen or author-controlled using SAGE.

- **Fine-Tuning**: The `fine_tuning.py` script is responsible for fine-tuning.

## Requirements
- Python 3.12 or higher
- CUDA

> [!WARNING]
> To recreate the TRANsCRIPT dataset, you first require access to the [speaker diarization pipeline](https://huggingface.co/pyannote/speaker-diarization-3.1) available on the [Hugging Face Hub](https://huggingface.co/). Follow these steps:
> 1. Create a [Hugging Face Access token](https://huggingface.co/docs/hub/en/security-tokens) on your account
> 2. Add your token to the `.env` file in the root of the project:
> ```json
>{
>    "huggingface_token": "YOUR_TOKEN_HERE"
>}
>```
>3. Agree to the terms for accessing [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Citing

If you use this code in your research, please use the following citation from the [ACL Anthology](https://aclanthology.org/):

```
@inproceedings{knuples-etal-2024-gender,
    title = "Gender Identity in Pretrained Language Models: An Inclusive Approach to Data Creation and Probing",
    author = "Knuple{\v{s}}, Urban  and
      Falenska, Agnieszka  and
      Mileti{\'c}, Filip",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.680",
    pages = "11612--11631"
}
```
