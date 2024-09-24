# Gender Identity in Pretrained Language Models 

This repository contains code associated with the paper Gender Identity in Pretrained Language Models: An Inclusive Approach to Data Creation and Probing.

> [!TIP]
> For access to the existing TRANsCRIPT dataset, please contact the authors.

## Getting started

This provided instruction on recreating the TRANsCRIPT dataset allong with running all the probing experiments on PLMs.

### Installation

Clone the repository, create a virtual environment, and install the requirements:

```bash
git clone https://github.com/urbikn/inclusive-probing-of-gender-identity.git
cd inclusive-probing-of-gender-identity
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset recreation setup 

To recreate the TRANsCRIPT dataset, you first require access to the [speaker diarization pipeline](https://huggingface.co/pyannote/speaker-diarization-3.1) available on the [Hugging Face Hub](https://huggingface.co/). Follow these steps:
1. Create a [Hugging Face Access token](https://huggingface.co/docs/hub/en/security-tokens) on your Hugging Face account
2. Create a `.env` file in the root of the project
3. Add the following line to the `.env` file and replace YOUR_HUGGING_FACE_TOKEN_HERE with your actual Hugging Face token:

```json
{
    "huggingface_token": "YOUR_HUGGING_FACE_TOKEN_HERE"
}
```
4. Accept the conditions to access [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Project structure

### Dataset folder
The dataset folder contains scripts for recreating the TRANsCRIPT corpus. It includes modules Wikidata sampling, YouTube data extraction, and preprocessing. The `create_dataset.py` script is the entry point for recreating the final dataset.

### Experiments folder
The experiments folder is dedicated to running experiments on PLMs. It contains two main components:

- **Probing**: The `probing.py` script includes code for probing the PLMs representations. As an input it uses the representations created from the `probing/dataset.py` script, either frozen or author-controlled using SAGE.

- **Fine-Tuning**: The `fine_tuning.py` script is responsible for fine-tuning.

## Requirements
- Python 3.6

## Citing

If you use this code in your research, please use the following citation from the [ACL Anthology](https://aclanthology.org/):

```
TBA
```

Code associated with the paper "Gender Identity in Pretrained Language Models: An Inclusive Approach to Data Creation and Probing".