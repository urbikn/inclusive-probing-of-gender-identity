import argparse
import uuid
import random
import os

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.init as init
import datasets
from datasets import DatasetDict, concatenate_datasets
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ProbingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=100, device='cpu', ID=None, label_to_id=None, id_to_label=None, seed=12, progress_bar=True):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        ).to(self.device)

        # Create a .classification folder if it doesn't exist
        if not os.path.exists('.classifiers'):
            os.makedirs('.classifiers')

        # The classification probes ID
        self.ID = ID if ID is not None else str(uuid.uuid4())[:8]

        self.progress_bar = progress_bar

        self.label_to_id = label_to_id
        self.id_to_label = id_to_label

        self.seed = seed

        random.seed(self.seed)
        torch.manual_seed(self.seed)

    
    def train(self, train_dataset, validation_dataset):
        """ 
        This method trains the probing classifier on a provided training dataset.

        The training is done using the Adam optimizer with a learning rate of 1e-3 and CrossEntropyLoss as the loss function.
        It also uses ReduceLROnPlateau as a learning rate scheduler. The scheduler reduces the learning rate by a factor of 0.5
        It also uses early stopping with a patience of 5 epochs.

        Parameters:
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.

        Returns:
            None
        """

        if type(train_dataset['label'][0]) == str:
            raise ValueError('Labels in ´train_dataset´ must be integers, not strings')

        if type(validation_dataset['label'][0]) == str:
            raise ValueError('Labels in ´train_dataset´ must be integers, not strings')

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

        # Define the loss function, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)

        model_path = '.classifiers/' + self.ID
        early_stopping = EarlyStopping(patience=5, verbose=False, path=model_path)

        val_loss = 0

        # Train the model
        for epoch in range(100):  # Arbitrary large number of epochs
            self.model.train()
            train_loss = 0.0

            if self.progress_bar:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}, Training")
                pbar.set_postfix({'val_loss': val_loss / len(validation_loader)})
            else:
                pbar = train_loader

            for batch in pbar:
                optimizer.zero_grad()
                inputs, targets = batch['embedding'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()


            # Validation phase
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch in validation_loader:
                    inputs, targets = batch['embedding'].to(self.device), batch['label'].to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)

                    val_loss += criterion(outputs, targets).item()

            scheduler.step(val_loss)
            early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                break

        #Load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)


    def evaluate(self, test_dataset):
        """
        This method evaluates the probing classifier on a provided test dataset.

        Parameters:
            test_dataset (Dataset): The training dataset.

        Returns:

        """

        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        test_dataloader = DataLoader(test_dataset, batch_size=32)

        predictions = np.array([])
        references = np.array([])

        if self.progress_bar:
            pbar = tqdm(test_dataloader, desc='Evaluation', leave=False)
        else:
            pbar = test_dataloader

        self.model.eval()
        
        for batch_index, batch in enumerate(pbar):
            with torch.no_grad():
                inputs, labels = batch['embedding'].to(self.device), batch['label'].to(self.device)

                output = self.model(inputs)

                # Compute loss between model output and actual targets
                loss = criterion(output, labels)
                total_loss += loss.item() # Update total_loss and num_steps

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)

            predictions = np.append(predictions, preds)
            references = np.append(references, labels.detach().cpu().numpy())
        
        if self.progress_bar:
            pbar.close()


        labels = [self.id_to_label[i] for i in range(self.num_classes)]
        results = metrics.classification_report(
            references,
            predictions,
            digits=4,
            output_dict=True,
            target_names=labels,
            zero_division=0,
        )
        
        return total_loss, results

    @staticmethod
    def initialize_weights(model):
        """
        Initializes the weights of a module using the Normal distribution with mean 0 and standard deviation 0.01.

        Parameters:
            model (nn.Module): The module to initialize.
        
        Returns:
            model (nn.Module): The re-initialized module.

        """
        # Intialize weights using Kaiming normal initialization
        def init_weights(m):
            if isinstance(model, nn.Linear):
                init.normal_(m.weight.data, mean=0.0, std=0.01)
                m.bias.data.fill_(0.01)
        
        model.apply(init_weights)
        return model

class MDLProbingClassifier():
    """
    Implements the MDL Probing Classifier with Online Coding evaluation.

    Implementation is adapted from Voita and Titov 2020 (https://arxiv.org/pdf/2003.12298.pdf)
    """
    def __init__(self, input_dim, num_classes, device='cpu', ID=None, seed=12, label_to_id=None, id_to_label=None):
        self.portion_ratios = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1.0]
        self.num_classes = num_classes

        model = ProbingClassifier(
            input_dim=input_dim,
            num_classes=self.num_classes,
            device=device,
            ID=ID,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            seed=seed,
            progress_bar=False
        )
        self.probing_model = model

        self.label_to_id = label_to_id
        self.id_to_label = id_to_label

        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
    

    @staticmethod
    def split_datasets(dataset, num_fractions, shuffle=True, seed=12):
        """
        Split a dataset into portions, by splitting into non-overlapping subsets. The splits are 
        always half the size of the previous split.

        The first split is always 0.001 of the dataset, and the last split is always the full dataset.

        Parameters:
            dataset (Dataset): The dataset to split.
            num_fractions (int): The number of fractions to split the dataset into.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Default is True.

        Returns:
            portions: A list of Subsets, of size len(fractions)
        """
        portions = [dataset]

        for i in range(0, num_fractions):
            # It's 0.5, because we're splitting the dataset in half
            subset = dataset.train_test_split(train_size=0.5, seed=seed, shuffle=shuffle, stratify_by_column='label')

            portions.append(subset['train'])
            dataset = subset['test']

        portions = list(reversed(portions))
        return portions
    
    @staticmethod
    def online_code_length(num_classes, t1, losses):
        r"""Calculate the online code length.

        Parameters:
            num_classes (int): Number of classes in the probing (classification) task.
            t1 (int): The size of the first training block (fraction) dataset.
            losses (List[float]): The list of (test) losses for each evaluation block (fraction)
            dataset, of size len(fractions).

        Returns:
            online_code_length (float): The online code length for the given training/evaluation parameters of
            the probe.
        """
        return t1 * np.log2(num_classes) + sum(losses) # Add the losses, not substract, since they are positive


    def analyze(self, train_dataset, val_dataset, test_dataset):
        r"""Analyze the probing classifier using the MDL Probing Classifier with Online Coding evaluation.

        Parameters:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The test dataset.

        Returns:
            final_report (dict): A dictionary containing information the compression ration, uniform code length, online code length, and information \\
                                 about the training of the probing classifier on the subsets.
        """
        assert self.portion_ratios[-1] == 1.0, 'The last portion ratio must be 1.0'

        # Split the training dataset into incrementally larger subsets based on the portion ratios
        dataset_subsets = self.split_datasets(train_dataset, len(self.portion_ratios) - 1)

        # A list to store information about the training of the probes on the online coding subsets
        online_coding_results = []

        # Create progess bar for training the probing classifiers on the subsets
        progress_bar = tqdm(
            enumerate(zip(dataset_subsets[:-1], dataset_subsets[1:])), # So subset_i is the train subset and subset_i+1 is the test subset
            desc=f'Training MDL probe',
            total=len(dataset_subsets[:-1])
        )

        # Train the probing classifier a subset
        for i, (train_subset, test_subset) in progress_bar:
            progress_bar.set_postfix({'Fraction': f'{self.portion_ratios[i]:.4f}'})

            # Reset the model weights
            self.probing_model = ProbingClassifier.initialize_weights(self.probing_model)

            # Train the probing classifier on the current subset
            self.probing_model.train(
                train_subset,
                val_dataset
            )
            
            # Evaluate the probing classifier on the test dataset
            test_loss, classification_report = self.probing_model.evaluate(test_subset)

            online_coding_results.append({
                'fraction': self.portion_ratios[i],
                'test_loss': test_loss,
                'classification_report': classification_report,
            })
        
        # Calculate the uniform code length
        online_codelength = self.online_code_length(
            num_classes=self.num_classes,
            t1=len(dataset_subsets[0]),
            losses=[result['test_loss'] for result in online_coding_results] # Except the last full dataset
        )
        
        final_report = {
            'mdl': online_codelength,
            'online_coding_results': online_coding_results
        }

        return final_report



if __name__ == '__main__':
    model_list = "roberta-base,roberta-large".split(",") + \
        "microsoft/deberta-base,microsoft/deberta-large".split(",") + \
        "microsoft/deberta-v3-base,microsoft/deberta-v3-large,microsoft/deberta-v3-xsmall,microsoft/deberta-v3-small".split(",")

    group_name_list = [
        'cis-gender',
        'trans_and_non-binary'
    ]

    gender_identities = [
        'cis man',
        'cis woman',
        'trans man',
        'trans woman',
        'non-binary',
    ]

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=int, default=0, help='Index of the model to use:' + str(model_list))
    argparser.add_argument('--groups',  nargs='+', default=['cis man', 'cis woman'], help='Name of the group to compare against (min 2). Possible:' + str(gender_identities))
    argparser.add_argument('--mdl',  type=bool, default=False, help='Whether to use the MDL probing classifier')
    argparser.add_argument('--save_to',  type=str, default='probing_results.csv', help='Path to save the results to')
    args = argparser.parse_args()

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