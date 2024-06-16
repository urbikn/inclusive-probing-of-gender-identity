    import numpy as np
import sage.sage as sage
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm
from itertools import chain
import pandas as pd
from IPython.display import display, HTML

tqdm.pandas()

def load_data(use_cache='', dataset_path='', save_cache='', batch_size=1000, n_process=8):
    if not len(use_cache):
        # Since SAGE compares the distribution from a background corpus, we need the full text
        # and see the distribution of the words in the text for trans
        full_corpus_df = pd.read_csv(dataset_path, sep='|')

        # Perform some preprocessing
        nlp = spacy.load('en_core_web_md', exclude=['ner', 'textcat'])

        docs_filtered = []
        docs = []

        for i, doc in tqdm(enumerate(nlp.pipe(full_corpus_df['sentence'], batch_size=batch_size, n_process=n_process)), total=len(full_corpus_df)):
            docs.append(doc)

            doc = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_punct]

            docs_filtered.append(doc)

        full_corpus_df['sentence'] = docs
        full_corpus_df['sentence_split'] = docs_filtered
        full_corpus_df.rename(columns={'sentence': 'sentence_full', 'sentence_split': 'sentence'}, inplace=True)

        if len(save_cache):
            full_corpus_df.to_pickle(save_cache, protocol=4)

    else:
        print('Using cached data from {}'.format(use_cache))
        full_corpus_df = pd.read_pickle(use_cache)

    return full_corpus_df

class LinguisticVariationPerUser:
    """ This class is used to analyze the linguistic variation of a user compared to a background group.
    """
    def __init__(self, corpus):
        self.corpus = corpus

        self.__init_sage()
    
    def __init_sage(self):
        full_text = list(chain.from_iterable(self.corpus['sentence'].tolist()))

        # Count the number of times each word appears in the target corpus
        self.base_counts = Counter(full_text)

    def __scale_array(self, arr, min_val=-3, max_val=3):
        pos_arr = arr[arr > 0]
        neg_arr = arr[arr < 0]

        # Take all positive values and scale them to the range [0, max_val]
        if pos_arr.size > 0:
            pos_min = np.min(pos_arr)
            pos_max = np.max(pos_arr)
            pos_arr = (pos_arr - pos_min) * (max_val / pos_max)

        if neg_arr.size > 0:
            neg_min = np.min(neg_arr)
            neg_max = np.max(neg_arr)
            neg_arr = (neg_arr - neg_max) * (min_val / neg_min)

        arr[arr > 0] = pos_arr
        arr[arr < 0] = neg_arr

        return arr
    
    def run_sage(self, user_id, n_most_common=5000):
        """ Run SAGE on the target corpus. This will compute the log-probabilities of each word in the target corpus.

        It will also create a self.vocab attribute that contains the vocabulary of the target corpus.
        It will also create a self.sage_eta attribute that contains the output of SAGE.
        """
        target_corpus = self.corpus[self.corpus['user_id'] == user_id]

        target_text = list(chain.from_iterable(target_corpus['sentence'].tolist()))

        # Count the number of times each word appears in the target corpus
        child_counts = Counter(target_text)

        # Build a vocabulary of the most common terms
        self.vocab = [word for word, count in Counter(child_counts).most_common(n_most_common)]

        x_child = np.array([child_counts[word] for word in self.vocab])
        x_base = np.array([self.base_counts[word] for word in self.vocab]) + 1.

        # Compute the base log-probabilities of each word
        mu = np.log(x_base) - np.log(x_base.sum())

        # Run SAGE
        self.sage_eta = sage.estimate(x_child, mu)
        
    
    def most_frequent_target(self, n=20):
        """ Return the most frequent words in the target corpus """
        return sage.topK(self.sage_eta, self.vocab, n)

    def most_representative_target_words(self, beta_threshold=1.5):
        """ Return the most frequent words in the target corpus based on the beta threshold """
        return [[self.vocab[idx], self.sage_eta[idx]] for idx in (-self.sage_eta).argsort() if self.sage_eta[idx] > beta_threshold]
   
    def examples_from_words(self, words, n=5, word_window=-1, verbose=False, random_state=42):
        """ Return n examples of sentences containing the given words """

        # Create a DataFrame with the target corpus and the words we want to find
        target_df = {column:[] for column in self.target_corpus.columns.to_list() + ['word']}
        for i, row in tqdm(self.target_corpus.iterrows(), total=len(self.target_corpus)):
            for word in words:
                if word in row['sentence']:
                    for column in self.target_corpus.columns:
                        target_df[column].append(row[column])
                    
                    target_df['word'].append(word)

        target_df = pd.DataFrame(target_df)

        if word_window != -1:
            target_df_copy = {column:[] for column in target_df.columns.to_list()}
            for i, row in tqdm(target_df.iterrows(), total=len(target_df)):
                word = row.word
                word_indexes = [i for i, word_ in enumerate(row.sentence_full) if word_.lemma_.lower() == word]

                spans = [
                    row.sentence_full[max(0, index-word_window) : min(index+word_window, len(row.sentence_full))]
                    for index in word_indexes
                ]

                for span in spans:
                    for column in target_df.columns:
                        if column == 'sentence_full':
                            target_df_copy[column].append(span.text)
                        else:
                            target_df_copy[column].append(row[column])

            target_df = pd.DataFrame(target_df_copy)

        values = None
        if verbose:
            print('Printing number of users per word')
            values = target_df.groupby(['word'])['user_id'].nunique().sort_values(ascending=False)
            # to_frame().T is a hack to display the Series as a DataFrame horizontally
            display(values.to_frame().T)
        
        def sample_sentences(group, n_samples):
            # Group by 'UserID' and sample one sentence from each user
            sampled_sentences = group.groupby('user_id').sample(1)
            # If there are less than 10 sentences, sample with replacement
            if len(sampled_sentences) < n_samples:
                additional_samples = n_samples - len(sampled_sentences)
                more_sentences = group.sample(additional_samples, replace=True, random_state=random_state)
                sampled_sentences = pd.concat([sampled_sentences, more_sentences])
            # If there are more than 10 sentences, sample without replacement
            elif len(sampled_sentences) > n_samples:
                sampled_sentences = sampled_sentences.sample(n_samples, random_state=random_state)
            return sampled_sentences

        # Group the DataFrame and apply the custom function
        target_df = target_df.groupby(['word']).apply(sample_sentences, n_samples=n)[['user_id', 'sentence_full']]

        # Drop duplicates because of the sampling strategy
        target_df = target_df.drop_duplicates('sentence_full')

        # target_df['sentence_full'] = target_df['sentence_full'].apply(lambda x: x.text)

        return target_df, values
