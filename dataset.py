import numpy as np
import pandas as pd
from datasets import load_dataset
import string

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from indicnlp.langinfo import *
articulations_func = {
    "Dental": is_dental,
    "Fricative": is_fricative,
    "Labial": is_labial,
    "Nasal": is_nasal,
    "Palatal": is_palatal,
    "Retroflex": is_retroflex,
    "Velar": is_velar
}
phonetics_func = {
    "Aspirated": is_aspirated,
    "Unaspirated": is_unaspirated,
    "Voiced": is_voiced,
    "Unvoiced": is_unvoiced,
    "Vowel": is_vowel,
    "Consonant": is_consonant
}

def get_dataset(dataset, language, re_download=False, clean_cache=False):
    """
        Wrapper to prepare datasets for 'ai4bharat/naamapadam' and
        'wikiann' datasets for a given language code
        Returns dictionary of dataset with train, test and val splits
        Each split is a dataframe with two columns - tokens and ner_tags
    """
    if not re_download:
        ds = load_dataset(dataset, language)
    else:
        ds = load_dataset(dataset, language, download_mode='force_redownload')

    data = {}
    data["train"] = pd.DataFrame(ds['train'])
    data["test"] = pd.DataFrame(ds['test'])
    data["val"] = pd.DataFrame(ds['validation'])

    # Drop extra columns from 'Wikiann' dataset
    if dataset == "wikiann":
        for k in data.keys():
            data[k].drop(labels=["langs", "spans"], axis=1, inplace=True)

    # Remove empty instances if any
    for k in data.keys():
        miss_idx = np.arange(data[k].shape[0])[data[k]['tokens'].apply(lambda x: x == [])]
        data[k].drop(miss_idx, axis=0, inplace=True)

    if clean_cache:
        ds.cleanup_cache_files()
    return data

def clean_dataset_cache(dataset, language):
    """
        Delete cached files for the specified dataset
    """
    ds = load_dataset(dataset, language)
    ds.cleanup_cache_files()
    del ds

def dataset_stats(dataset):
    """
        Print stats for a dataset dictionary containing
        train, test and val splits
    """
    splits = ["train", "test", "val"]
    train_toks = set(dataset["train"]['tokens'].explode().unique())
    stats = {}
    for k in splits:
        stats[k] = []
        toks = dataset[k]['tokens'].explode()
        uniq_toks = train_toks if k == "train" else set(toks.unique())
        stats[k].append(toks.shape[0])
        stats[k].append(len(uniq_toks))
        stats[k].append(0 if k == "train" else len(uniq_toks) - len(uniq_toks.intersection(train_toks)))
        stats[k].extend(list(np.unique(dataset[k]['ner_tags'].explode(), return_counts=True)[1]))
    cols = ["Tokens", "Unique Tokens", "Unknown Tokens"] + ["Tag = {0}".format(i) for i in range(7)]
    stats = pd.DataFrame.from_dict(stats, orient='index', columns=cols)
    return stats

def get_sent_len_dist(dataset, split='train', frac=1, seed=0):
    """
        Plot the sentence length distribution
    """
    df = dataset[split].sample(frac=frac, random_state=seed).copy(deep=True)
    df['sent_len'] = df.apply(lambda x: len(x['tokens']), axis=1)
    df = df.groupby('sent_len', as_index=False).size()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
    ax = sns.barplot(data=df, x='sent_len', y='size', ax=ax)
    mn, mx = df['sent_len'].min(), df['sent_len'].max()
    step = (mx - mn) // 10
    ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(mn,mx+1,step)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.set(xlabel="Sentence Length", ylabel="Count")
    ax.set_title("Sentence Length Distribution")
    plt.show()

def get_len_ner_corr(dataset, len_limit=10, split='train', explode=True, frac=1, seed=0):
    """
        Plot the correlation between the label and the
        token length (till len_limit)
    """
    if explode:
        df = dataset[split].explode(['tokens', 'ner_tags'], ignore_index=True)
    else:
        df = dataset.copy(deep=True)
        if frac < 1:
            df = df.sample(frac=frac, random_state=seed)
    df['len'] = df.apply(lambda x: len(x['tokens']), axis=1)
    df['is_ner'] = (df['ner_tags'] > 0).astype(int)
    cnt_df = df[['len','is_ner']].groupby('len', as_index=False).value_counts()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    ax = sns.barplot(data=cnt_df[cnt_df['len'] <= len_limit], x='len', y='count', hue='is_ner', palette='deep', ax=ax)
    ax.set(xlabel="Word Length", ylabel="Count")
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.legend(handles=ax.get_legend().legendHandles, title="Is Named Entity?", labels=["No", "Yes"])
    ax.set_title("Correlation between Word Length and NER tags")
    plt.show()

def get_linguistic_ner_corr(dataset, lang, linguistic_type=0, split='train', explode=True, frac=1, seed=0):
    """
        Plot the correlation between the label and the
        linguistic features (phonetics or articulation based)
    """
    if explode:
        df = dataset[split].explode(['tokens', 'ner_tags'], ignore_index=True)
    else:
        df = dataset.copy(deep=True)
        if frac < 1:
            df = df.sample(frac=frac, random_state=seed)
    df['is_ner'] = (df['ner_tags'] > 0).astype(int)
    linguistic_type = 'Phonetic' if linguistic_type == 0 else 'Articulation'
    func_dict = phonetics_func if linguistic_type == 'Phonetic' else articulations_func
    cols = list(func_dict.keys())
    df[cols] = df.apply(lambda x: [1 if func_dict[k](x['tokens'][0], lang=lang) else 0 for k in cols], axis=1, result_type='expand')

    dfs = []
    for col in cols:
        cnt_df = df[[col,'is_ner']].groupby(col, as_index=False).value_counts()
        cnt_df = cnt_df[cnt_df[col] == 1]
        cnt_df['key'] = cnt_df.apply(lambda x: "Yes" if x["is_ner"] == 1 else "No", axis=1)
        # cnt_df['key'] = cnt_df.apply(lambda x: ("Yes | " if x[col] == 1 else "No  | ") + ("Yes" if x["is_ner"] == 1 else "No"), axis=1)
        cnt_df.drop([col, 'is_ner'], axis=1, inplace=True)
        cnt_df['type'] = col
        dfs.append(cnt_df)
    res_df = pd.concat(dfs)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
    ax = sns.barplot(data=res_df, x='type', y='count', hue='key', palette='deep', ax=ax)
    ax.set(xlabel="{0} Type".format(linguistic_type), ylabel="Count")
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.legend(title="Is Named Entity?")         # bbox_to_anchor=(1, 1, 0, 0)
    ax.set_title("Correlation between {0} Types and NER tags".format(linguistic_type))
    plt.show()

def get_punctuation_ner_corr(dataset, split='train', explode=True, frac=1, seed=0):
    """
        Plot the correlation between the label and the
        punctuation positions
    """
    if explode:
        df = dataset[split].explode(['tokens', 'ner_tags'], ignore_index=True)
    else:
        df = dataset.copy(deep=True)
        if frac < 1:
            df = df.sample(frac=frac, random_state=seed)
    df['is_ner'] = (df['ner_tags'] > 0).astype(int)
    df['punc'] = df.apply(lambda x: 1 if x['tokens'] in string.punctuation else 0, axis=1)
    df.loc[(df.index > 0), 'Prefix'] = df['punc'].to_list()[:-1]
    df.loc[0, 'Prefix'] = 0
    df['Prefix'] = df['Prefix'].astype(int)
    df.loc[(df.index < df.shape[0]-1), 'Suffix'] = df['punc'].to_list()[1:]
    df.loc[df.shape[0]-1, 'Suffix'] = 0
    df['Suffix'] = df['Suffix'].astype(int)

    dfs = []
    cols = ['Prefix', 'Suffix']
    for col in cols:
        cnt_df = df[[col,'is_ner']].groupby(col, as_index=False).value_counts()
        # cnt_df = cnt_df[cnt_df[col] == 1]
        # cnt_df['key'] = cnt_df.apply(lambda x: "Yes" if x["is_ner"] == 1 else "No", axis=1)
        cnt_df['key'] = cnt_df.apply(lambda x: ("Yes | " if x[col] == 1 else "No  | ") + ("Yes" if x["is_ner"] == 1 else "No"), axis=1)
        cnt_df.drop([col, 'is_ner'], axis=1, inplace=True)
        cnt_df['type'] = col
        dfs.append(cnt_df)
    res_df = pd.concat(dfs)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    ax = sns.barplot(data=res_df, x='type', y='count', hue='key', palette='deep', ax=ax)
    ax.set(xlabel="Punctuation Position", ylabel="Count")
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.legend(title="Is Punctuation? Is Named Entity?", bbox_to_anchor=(1, 1, 0, 0))
    ax.set_title("Correlation between Punctuation Position and NER tags")
    plt.show()

def get_relpos_ner_corr(dataset, split='train', frac=1, seed=0):
    """
        Plot the distribution of position of
        occurence of named entities
    """
    df = dataset[split].sample(frac=frac, random_state=seed).copy(deep=True)
    df['rel_pos'] = df.apply(lambda x: np.arange(1,len(x['tokens'])+1)/len(x['tokens']), axis=1)
    df = df.explode(['tokens','ner_tags','rel_pos'], ignore_index=True)
    df['is_ner'] = (df['ner_tags'] > 0).astype(int)
    dist = df.loc[(df['is_ner'] == 1), 'rel_pos']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
    ax = sns.histplot(dist, bins=10, shrink=0.8, ax=ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    ax.set(xlabel="Relative Position", ylabel="Count")
    ax.set_title("Distribution of Position of Occurence of Named Entities")
    plt.show()

# https://www.kaggle.com/code/trunganhdinh/hidden-markov-model-for-pos-tagging