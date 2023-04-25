# References used: https://www.kaggle.com/code/trunganhdinh/hidden-markov-model-for-pos-tagging/notebook

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset import *

def train_hmm(dataset):
    """
        Train a Categorical HMM model for the given
        NER training dataset
    """
    dataset = dataset.explode(['tokens', 'ner_tags'])
    dataset.reset_index(inplace=True)
    dataset.rename(columns={"index": "sent"}, inplace=True)

    # Sample UNKNOWN tokens
    rand_idx = np.random.choice(dataset.index, size=int(0.1 * dataset.shape[0]), replace=False)
    dataset.loc[rand_idx, 'tokens'] = 'unk'

    # List of words and tags
    words = list(set(dataset['tokens']))
    tags = list(set(dataset['ner_tags']))
    word2id = {w: i for i, w in enumerate(words)}

    # Solve for MLE using Counts
    count_tags = dict(dataset['ner_tags'].value_counts())
    count_tags_to_words = dataset.groupby(['ner_tags']).apply(lambda grp: grp.groupby('tokens')['ner_tags'].count().to_dict()).to_dict()
    count_init_tags = dict(dataset.groupby('sent').first()['ner_tags'].value_counts())
    count_tags_to_next_tags = np.zeros((len(tags), len(tags)), dtype=int)
    sentences = list(dataset['sent'])
    ner_tags = list(dataset['ner_tags'])
    for i in range(len(sentences)) :
        if (i > 0) and (sentences[i] == sentences[i - 1]):
            prevtagid = ner_tags[i - 1]
            nexttagid = ner_tags[i]
            count_tags_to_next_tags[prevtagid][nexttagid] += 1

    mystartprob = np.zeros((len(tags),))
    mytransmat = np.zeros((len(tags), len(tags)))
    myemissionprob = np.zeros((len(tags), len(words)))
    num_sentences = sum(count_init_tags.values())
    sum_tags_to_next_tags = np.sum(count_tags_to_next_tags, axis=1)
    for tag in tags:
        floatCountTag = float(count_tags.get(tag, 0))
        mystartprob[tag] = count_init_tags.get(tag, 0) / num_sentences
        for word, wordid in word2id.items():
            myemissionprob[tag][wordid]= count_tags_to_words.get(tag, {}).get(word, 0) / floatCountTag
        for tag2 in tags:
            mytransmat[tag][tag2]= count_tags_to_next_tags[tag][tag2] / sum_tags_to_next_tags[tag]

    # Define HMM Model
    model = hmm.CategoricalHMM(n_components=len(tags), algorithm='viterbi', random_state=0)
    model.startprob_ = mystartprob
    model.transmat_ = mytransmat
    model.emissionprob_ = myemissionprob

    final_model = {
        "model": model,
        "vocab": words,
        "vocab_map": word2id
    }
    return final_model

def test_hmm(hmm_model, dataset):
    """
        Test the Categorical HMM model for the given
        NER test dataset
    """
    model = hmm_model["model"]
    words = hmm_model["vocab"]
    word2id = hmm_model["vocab_map"]

    lengths = dataset.apply(lambda x: len(x['tokens']), axis=1).to_list()

    dataset = dataset.explode(['tokens'], ignore_index=True)
    dataset.loc[~dataset['tokens'].isin(words), 'tokens'] = 'unk'
    samples = dataset['tokens'].apply(lambda x: word2id[x]).to_numpy().reshape(-1,1)

    y_pred = model.predict(samples, lengths)

    return y_pred

def eval_results_hmm(y_test, y_pred, clf_report=False, conf_matrix=False, exclude_0=True):
    """
        Report evaluation metrics for the NER model (HMM)
    """
    labels = np.arange(1,7) if exclude_0 else np.arange(0,7)
    print("Weighted F1 of HMM = {:.4f}".format(f1_score(y_test, y_pred, average='weighted', labels=labels)))
    if clf_report:
        print(classification_report(y_test, y_pred, labels=labels, digits=3, zero_division=0))
    if conf_matrix:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=labels)
        plt.show()
    print("================================================================\n")

if __name__ == "__main__":
    hi_ds = get_dataset("ai4bharat/naamapadam", "hi", use_pickle=True)
    te_ds = get_dataset("ai4bharat/naamapadam", "te", use_pickle=True)
    ml_ds = get_dataset("ai4bharat/naamapadam", "ml", use_pickle=True)
    sa_ds = get_dataset("wikiann", "sa", use_pickle=True)

    datasets = {
        "Hindi": hi_ds,
        "Telugu": te_ds,
        "Malayalam": ml_ds,
        "Sanskrit": sa_ds
    }

    lang_codes = {
        "Hindi": "hi",
        "Telugu": "te",
        "Malayalam": "ml",
        "Sanskrit": "sa"
    }

    print("Evaluating model performance on same language")
    print("================================================================\n")

    models = {}
    for lang, ds in datasets.items():
        print("{0} language HMM model evaluated on {1} language".format(lang, lang))
        train_dataset = ds['train']
        test_dataset = combine_dataset_splits(ds, ['val', 'test'])

        model = train_hmm(train_dataset)
        models[lang] = model

        y_test = test_dataset['ner_tags'].explode().to_list()
        y_pred = test_hmm(model, test_dataset)

        eval_results_hmm(y_test, y_pred, clf_report=True, conf_matrix=True, exclude_0=True)

    tgt_lang = "Sanskrit"
    print("Evaluating model performance on {0} language without transliteration".format(tgt_lang))
    print("================================================================\n")

    for lang, ds in datasets.items():
        if lang == tgt_lang:
            continue

        print("{0} language HMM model evaluated on Sanskrit language without transliteration".format(lang, tgt_lang))
        train_dataset = ds['train']
        test_dataset = combine_dataset_splits(datasets[tgt_lang], ['val', 'test'])

        model = models[lang]

        y_test = test_dataset['ner_tags'].explode().to_list()
        y_pred = test_hmm(model, test_dataset)

        eval_results_hmm(y_test, y_pred, clf_report=True, conf_matrix=False, exclude_0=True)

    tgt_lang = "Sanskrit"
    print("Evaluating model performance on {0} language with transliteration".format(tgt_lang))
    print("================================================================\n")

    for lang, ds in datasets.items():
        if lang == tgt_lang:
            continue

        print("{0} language HMM model evaluated on Sanskrit language with transliteration".format(lang, tgt_lang))
        train_dataset = ds['train']
        trans_dataset = transliterate_dataset(datasets[tgt_lang], lang_codes[tgt_lang], lang_codes[lang])
        test_dataset = combine_dataset_splits(trans_dataset, ['val', 'test'])

        model = models[lang]

        y_test = test_dataset['ner_tags'].explode().to_list()
        y_pred = test_hmm(model, test_dataset)

        eval_results_hmm(y_test, y_pred, clf_report=True, conf_matrix=False, exclude_0=True)
