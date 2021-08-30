#!/usr/bin/env python3
""" Utilities for term-prediction tool """

import arff
import pandas as pd

def get_chunk_size(filepath):
    """ Returns chunk size. Assumes files named like .../term_chunk_1_train.arff """
    return filepath.split("_")[2]

def get_arff_features(filepath):
    """ Returns string of features used from arff file """
    attrs = arff.load(open(filepath, 'r'))['attributes']
    features = [attr[0] for attr in attrs]
    l = ["notes"] # always have notes?, put first
    for f in features:
        if f == "text-length":
            l.append(f)
        elif f == "note-type":
            l.append(f)
        elif f == "empath-cat-frail":
            l.append("empath") # don't list all empath features
        elif f == "anatomy-terms":
            l.append("anatomy-terms")
    return ", ".join(l)

def get_features(flist):
    """ Returns string of features based on list of columns from dataframe """
    l = []
    # list notes first
    if "notes" in flist:
        l.append("notes")
    for f in flist:
        if f == "text-length":
            l.append(f)
        elif f == "note-type":
            l.append(f)
        elif f == "empath-cat-frail":
            l.append("empath") # don't list all empath features
        elif f == "anatomy-terms":
            l.append("anatomy-terms")
    return ", ".join(l)

# unused
# def get_dataset_from_arff(arff_file):
#     """ Return a dataset as Bunch from the data in given arff file """
#     dataset = arff.load(open(arff_file, 'r'))
#     arff_data = np.array(dataset['data'])
#     return Bunch(data=arff_data[:, 1:], target=arff_data[:, 0])

# https://github.com/renatopp/liac-arff
# https://pythonhosted.org/liac-arff/#loading-an-object-with-encoded-labels
# https://github.com/renatopp/liac-arff/blob/master/docs/source/index.rst
def get_dataframe_from_arff(arff_file):
    """ Return a pandas dataframe from the data in given arff file """
    dataset = arff.load(open(arff_file, 'r'))
    cols=[item[0] for item in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=cols)
    return df

def save_arff_as_dataframe(arff_file, df_filename):
    """ Saves arff data as pickled pandas dataframe """
    df = get_dataframe_from_arff(arff_file)
    df.to_pickle(df_filename)
