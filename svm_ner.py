import os
from argparse import ArgumentParser
import warnings 
warnings.simplefilter("ignore")
from datasets import load_dataset 
import nltk 
from nltk.tag import tnt
from nltk.corpus import indian 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report


# Load dataset of the selected language
# Return a tuple of NER dataset and a POS tagger check if the specific language has a POS tagger or not
def load_ner(language):
    language = language.lower()
    has_pos = False
    if(language == 'hindi'):
        ner_dataset = load_dataset('ai4bharat/naamapadam', 'hi')
        has_pos = True
    elif(language == 'telugu'):
        ner_dataset = load_dataset('ai4bharat/naamapadam', 'te')
        has_pos = True
    elif(language == 'malayalam'):
        ner_dataset = load_dataset('ai4bharat/naamapadam', 'ml')
        has_pos = False
    elif(language == 'sanskrit'):
        ner_dataset = load_dataset('wikiann', 'sa')
        has_pos = False
    else:
        os.system('ENTER A VALID LANGUAGE: HINDI, TELUGU, MALAYALAM, OR SANSKRIT')
        os._exit()
    
    return ner_dataset, has_pos


def prepare_dataset(language, sentences, include_zero_ner, save_csv):
    language = language.lower()
    ner_dataset, has_pos = load_ner(language=language)
    df_train = ner_dataset['train'].to_pandas().iloc[:sentences]
    df_test = ner_dataset['test'].to_pandas()
    if(language == 'sanskrit'):
        df_train = df_train.drop(['langs', 'spans'], axis=1)
        df_test = df_test.drop(['langs', 'spans'], axis=1)

    train_dict = {}     # Stores the word token as key and associated list of values (ner_tag, pos, prefix, word_length) as value

    if has_pos == True:
        pos_type = language + str('.pos')
        pos_train_data = indian.tagged_sents(pos_type)
        tnt_pos_tagger = tnt.TnT()
        tnt_pos_tagger.train(pos_train_data)
        train_token_list = df_train['tokens'].tolist()
        train_ner_list = df_train['ner_tags'].tolist()

        # Generate Train set
        for i in range(sentences):
            tags = (tnt_pos_tagger.tag(train_token_list[i]))
            for j in range(len(train_token_list[i])):
                train_dict.update({train_token_list[i][j]:[tags[j][1], 
                                                           train_ner_list[i][j], 
                                                           train_token_list[i][j][0],
                                                           len(train_token_list[i][j])
                                                           ]})
        
        train_token_wl = []
        train_tag_wl = []
        train_pos_wl = []
        train_pref_wl = []
        train_len_wl = []

        for i in train_dict:
            train_token_wl.append(i)
            train_pos_wl.append(train_dict[i][0])
            train_tag_wl.append(train_dict[i][1])
            train_pref_wl.append(train_dict[i][2])
            train_len_wl.append(train_dict[i][3])

        df_svm_train = pd.DataFrame({
                'token':train_token_wl,
                'ner_tags':train_tag_wl,
                'pos':train_pos_wl,
                'prefix':train_pref_wl,
                'word_length':train_len_wl
        })

        le = LabelEncoder()
        df_svm_train['token'] = le.fit_transform(df_svm_train['token'])
        df_svm_train['pos'] = le.fit_transform(df_svm_train['pos'])
        df_svm_train['prefix'] = le.fit_transform(df_svm_train['prefix'])

        df_svm_train_for_test = df_svm_train.copy()

        if include_zero_ner == False:
            df_svm_train = df_svm_train.drop(df_svm_train[df_svm_train['ner_tags']==0].index)
            if save_csv == True:
                file_name = language + "_train_" + str(sentences) + "_WOut_Zero.csv" 
                df_svm_train.to_csv(file_name)
        else:
            if save_csv == True:
                file_name = language + "_train_" + str(sentences) + "_With_Zero.csv"
                df_svm_train.to_csv(file_name)

        Y_train = np.array(df_svm_train['ner_tags'].tolist())
        X_train = np.array([np.array(df_svm_train['token'].tolist()),
           np.array(df_svm_train['pos'].tolist()),
           np.array(df_svm_train['prefix'].tolist()),
           np.array(df_svm_train['word_length'].tolist())
           ])

        X_train = X_train.transpose()

        # Generate Test set 
        test_token = df_test['tokens']
        test_tags = df_test['ner_tags']

        # Create an encoded dictionary that maps label encoded values for tokens, pos, and prefix
        token_dict = {}
        temp_token_list = df_svm_train_for_test['token'].tolist()
        for i in range(len(train_token_wl)):
            token_dict.update({train_token_wl[i]:temp_token_list[i]})

        pos_dict = {}
        temp_pos_list = df_svm_train_for_test['pos'].tolist()
        for i in range(len(train_token_wl)):
            pos_dict.update({train_pos_wl[i]:temp_pos_list[i]})

        pref_dict = {}
        temp_prefix_list = df_svm_train_for_test['prefix'].tolist()
        for i in range(len(train_token_wl)):
            pref_dict.update({train_pref_wl[i]:temp_prefix_list[i]})

        ukn_token = max(token_dict.values()) + 1
        ukn_pos = max(pos_dict.values()) + 1
        ukn_pref = max(pref_dict.values()) + 1

        test_dict = {}      # Stores the word token as key and associated list of values (ner_tag, pos, prefix, word_length) as value

        for i in range(len(test_token)):
            tags = (tnt_pos_tagger.tag(test_token[i]))
            for j in range(len(test_token[i])):
                ner = test_tags[i][j]
                pos = tags[j][1]
                if len(test_token[i][j]) == 0:
                    pref = ukn_pref
                else:
                    pref = test_token[i][j][0]
                    word_len = len(test_token[i][j])
                    test_dict.update({test_token[i][j]:[ner, pos, pref, word_len]})

        test_token_wl = []
        test_tags_wl = []
        test_pos_wl = []
        test_prefix_wl = []
        test_len_wl = []

        for i in test_dict:
            test_token_wl.append(i)
            test_tags_wl.append(test_dict[i][0])
            test_pos_wl.append(test_dict[i][1])
            test_prefix_wl.append(test_dict[i][2])
            test_len_wl.append(test_dict[i][3])

        test_token_enc = []
        test_pos_enc = []
        test_prefix_enc = []

        for i in range(len(test_token_wl)):
            if test_token_wl[i] in token_dict:
                test_token_enc.append(token_dict.get(test_token_wl[i]))
            else:
                test_token_enc.append(ukn_token)
            
            if test_pos_wl[i] in pos_dict:
                test_pos_enc.append(pos_dict.get(test_pos_wl[i]))
            else:
                test_pos_enc.append(ukn_pos)
            
            if test_prefix_wl[i] in pref_dict:
                test_prefix_enc.append(pref_dict.get(test_prefix_wl[i]))
            else:
                test_prefix_enc.append(ukn_pref)

        test_data = {
            'token':test_token_enc,
            'ner_tags':test_tags_wl,
            'pos':test_pos_enc,
            'prefix':test_prefix_enc,
            'word_length':test_len_wl
        }

        df_svm_test = pd.DataFrame(test_data)

        if include_zero_ner == False:
            df_svm_test = df_svm_test.drop(df_svm_test[df_svm_test['ner_tags']==0].index)
            if save_csv == True:
                file_name = language + "_test_" + str(sentences) + "_WOut_Zero.csv" 
                df_svm_test.to_csv(file_name)
        else:
            if save_csv == True:
                file_name = language + "_test_" + str(sentences) + "_With_Zero.csv"
                df_svm_test.to_csv(file_name)

        Y_test = np.array(df_svm_test['ner_tags'].tolist())

        X_test = np.array([
            np.array(df_svm_test['token'].tolist()),
            np.array(df_svm_test['pos'].tolist()),
            np.array(df_svm_test['prefix'].tolist()),
            np.array(df_svm_test['word_length'].tolist())
        ])

        X_test = X_test.transpose()

    else:
        train_token_list = df_train['tokens'].tolist()
        train_ner_list = df_train['ner_tags'].tolist()

        # Generate Train set
        for i in range(sentences):
            for j in range(len(train_token_list[i])):
                train_dict.update({train_token_list[i][j]:[train_ner_list[i][j], 
                                                           train_token_list[i][j][0],
                                                           len(train_token_list[i][j])
                                                           ]})
        
        train_token_wl = []
        train_tag_wl = []
        train_pref_wl = []
        train_len_wl = []

        for i in train_dict:
            train_token_wl.append(i)
            train_tag_wl.append(train_dict[i][0])
            train_pref_wl.append(train_dict[i][1])
            train_len_wl.append(train_dict[i][2])

        df_svm_train = pd.DataFrame({
                'token':train_token_wl,
                'ner_tags':train_tag_wl,
                'prefix':train_pref_wl,
                'word_length':train_len_wl
        })

        le = LabelEncoder()
        df_svm_train['token'] = le.fit_transform(df_svm_train['token'])
        df_svm_train['prefix'] = le.fit_transform(df_svm_train['prefix'])

        df_svm_train_for_test = df_svm_train.copy()

        if include_zero_ner == False:
            df_svm_train = df_svm_train.drop(df_svm_train[df_svm_train['ner_tags']==0].index)
            if save_csv == True:
                file_name = language + "_train_" + str(sentences) + "_WOut_Zero.csv" 
                df_svm_train.to_csv(file_name)
        else:
            if save_csv == True:
                file_name = language + "_train_" + str(sentences) + "_With_Zero.csv"
                df_svm_train.to_csv(file_name)

        Y_train = np.array(df_svm_train['ner_tags'].tolist())
        X_train = np.array([np.array(df_svm_train['token'].tolist()),
           np.array(df_svm_train['prefix'].tolist()),
           np.array(df_svm_train['word_length'].tolist())
           ])

        X_train = X_train.transpose()

        # Generate Test set 
        test_token = df_test['tokens']
        test_tags = df_test['ner_tags']

        # Create an encoded dictionary that maps label encoded values for tokens, pos, and prefix
        token_dict = {}
        temp_token_list = df_svm_train_for_test['token'].tolist()
        for i in range(len(train_token_wl)):
            token_dict.update({train_token_wl[i]:temp_token_list[i]})

        pref_dict = {}
        temp_prefix_list = df_svm_train_for_test['prefix'].tolist()
        for i in range(len(train_token_wl)):
            pref_dict.update({train_pref_wl[i]:temp_prefix_list[i]})

        ukn_token = max(token_dict.values()) + 1
        ukn_pref = max(pref_dict.values()) + 1

        test_dict = {}      # Stores the word token as key and associated list of values (ner_tag, pos, prefix, word_length) as value

        for i in range(len(test_token)):
            for j in range(len(test_token[i])):
                ner = test_tags[i][j]
                if len(test_token[i][j]) == 0:
                    pref = ukn_pref
                else:
                    pref = test_token[i][j][0]
                    word_len = len(test_token[i][j])
                    test_dict.update({test_token[i][j]:[ner, pref, word_len]})

        test_token_wl = []
        test_tags_wl = []
        test_prefix_wl = []
        test_len_wl = []

        for i in test_dict:
            test_token_wl.append(i)
            test_tags_wl.append(test_dict[i][0])
            test_prefix_wl.append(test_dict[i][1])
            test_len_wl.append(test_dict[i][2])

        test_token_enc = []
        test_prefix_enc = []

        for i in range(len(test_token_wl)):
            if test_token_wl[i] in token_dict:
                test_token_enc.append(token_dict.get(test_token_wl[i]))
            else:
                test_token_enc.append(ukn_token)
        
            if test_prefix_wl[i] in pref_dict:
                test_prefix_enc.append(pref_dict.get(test_prefix_wl[i]))
            else:
                test_prefix_enc.append(ukn_pref)

        test_data = {
            'token':test_token_enc,
            'ner_tags':test_tags_wl,
            'prefix':test_prefix_enc,
            'word_length':test_len_wl
        }

        df_svm_test = pd.DataFrame(test_data)

        if include_zero_ner == False:
            df_svm_test = df_svm_test.drop(df_svm_test[df_svm_test['ner_tags']==0].index)
            if save_csv == True:
                file_name = language + "_test_" + str(sentences) + "_WOut_Zero.csv" 
                df_svm_test.to_csv(file_name)
        else:
            if save_csv == True:
                file_name = language + "_test_" + str(sentences) + "_With_Zero.csv"
                df_svm_test.to_csv(file_name)

        Y_test = np.array(df_svm_test['ner_tags'].tolist())

        X_test = np.array([
            np.array(df_svm_test['token'].tolist()),
            np.array(df_svm_test['prefix'].tolist()),
            np.array(df_svm_test['word_length'].tolist())
        ])

        X_test = X_test.transpose()
    
    return X_train, Y_train, X_test, Y_test
        


def svm_model_predict(cost_parameter, kernel, classifier_type, X_train, Y_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if(classifier_type == 'ovr'):
        if(kernel == 'linear'):
            svm = OneVsRestClassifier(LinearSVC(C=cost_parameter, random_state=0)).fit(X_train, Y_train)
        elif(kernel == 'rbf'):
            svm = OneVsRestClassifier(SVC(C=cost_parameter, kernel='rbf', random_state=0)).fit(X_train, Y_train)
        else:
            svm = OneVsRestClassifier(SVC(C=cost_parameter,kernel='poly', random_state=0)).fit(X_train, Y_train)
    else:
        if(kernel == 'linear'):
            svm = OneVsOneClassifier(LinearSVC(C=cost_parameter, random_state=0)).fit(X_train, Y_train)
        elif(kernel == 'rbf'):
            svm = OneVsOneClassifier(SVC(C=cost_parameter, kernel='rbf', random_state=0)).fit(X_train, Y_train)
        else:
            svm = OneVsOneClassifier(SVC(C=cost_parameter,kernel='poly', random_state=0)).fit(X_train, Y_train)

    Y_pred = svm.predict(X_test)

    return Y_pred


def print_classification_report(Y_test, Y_pred, include_zero_ner):
    if include_zero_ner == False: 
        target_names = ['B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG']
    else:
        target_names = ['0','B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG']

    print(classification_report(Y_test, Y_pred, target_names=target_names))

def inference(args):
    language = args.language
    sentences = args.sentences
    kernel = args.kernel
    classifier_type = args.classifier_type
    cost_parameter = args.cost_parameter
    include_zero_ner = args.include_zero_ner
    save_csv = args.save_csv

    X_train, Y_train, X_test, Y_test = prepare_dataset(language, sentences, include_zero_ner, save_csv)
    Y_pred = svm_model_predict(cost_parameter, kernel, classifier_type, X_train, Y_train, X_test)
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("CLASSIFICATION REPORT")
    print_classification_report(Y_test, Y_pred, include_zero_ner)
    print("---------------------------------------------------")
    print("---------------------------------------------------")


def main():
    parser = ArgumentParser()
    parser.add_argument("--language", type=str, default='hindi')
    parser.add_argument("--sentences", type=int, default=50000)
    parser.add_argument("--kernel", type=str, default="linear")
    parser.add_argument("--classifier_type", type=str, default="ovr")
    parser.add_argument("--cost_parameter", type=int, default=10)
    parser.add_argument("--include_zero_ner", type=bool, default=False)
    parser.add_argument("--save_csv", type=bool, default=False)

    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
