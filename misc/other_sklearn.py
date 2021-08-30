#!/usr/bin/env python3
""" misc sklearn stuff """

import pandas as pd
import sklearn
import sklearn_crfsuite

import utils
import config

# misc code that won't run, and only for reference
# moved out of main program

# def transform_text(text):
#     """ Transform note's text based on settings """

#     # convert the text to named entities (with scispacy)
#     if config.ner_transform:
#         nlp = spacy.load(config.spacy_model)
#         doc = nlp(text)
#         # https://stackoverflow.com/questions/58712418/replace-entity-with-its-label-in-spacy
#         text = " ".join([t.text if t.ent_type_ else "" for t in doc])

#     return text


# https://sklearn-crfsuite.readthedocs.io/en/latest/
# http://acepor.github.io/2017/03/06/CRF-Python/
def run_crf(filepath, do_cv=True, do_predict=False):
    """ Run CRF classifier on filepath (not compat with sklearn 0.24+) """

    # running with 0.24+ gives error:
    # AttributeError: 'CRF' object has no attribute 'keep_tempfiles'

    # only checks specific version, fix
    if sklearn.__version__ == "0.24.2":
        print("crfsuite not compatible with sklearn 0.24")
        return

    clf = sklearn_crfsuite.CRF()

    # load training dataframe
    df_train = pd.read_pickle(filepath)

    # apply any oversampling if configured
    if config.oversample_amount and config.oversample_amount > 0:
        df_train = oversample_data(df_train, config.oversample_amount)

    # trainging target/y data
    y_train = df_train.pop('has-word')
    # maybe unnecessary to convert to 1/0 but can work better with some defaults
    y_train = (y_train == 'True').astype(int)

    # get ColumnTransformer -- this may set columns to drop depending on config
    col_tran = get_col_transformer(df_train)

    # list of non-dropped columns to display which features are enabled
    collist = [t[0] for t in col_tran.transformers if t[1] != 'drop']

    # apply transformations
    X_train = col_tran.fit_transform(df_train)

    # convert features to what crfsuite expectes...
    # very gross but gives some ideas until future improvements
    # [ [{word1}, {word2}, ...], [{word1}, {word2}, ...] ]
    # [ [{chunk1}], [{chunk2}], ...], ...
    Xm = X_train.toarray()
    X2 = [chunk2features(c) for c in Xm]
    y2 = [[str(label)] for label in y_train]

    # cross-validation
    if do_cv:
        # custom because roc_auc_score won't work
        scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
                   'fp': make_scorer(fp), 'fn': make_scorer(fn)}

        with warnings.catch_warnings():
            # ignore warnings about how crfsuite should be built for future versions
            warnings.filterwarnings("ignore", category=FutureWarning)
            scores = run_cv(clf, X2, y2, scoring=scoring)
            print_score_results(type(clf).__name__, utils.get_features(collist), utils.get_chunk_size(filepath), scores)

    # prediction
    if do_predict:
        print("Not implemented yet.")

def chunk2features(chunk):
    """ helper for run_crf """
    # return [col2features(chunk, i) for i in range(len(chunk))]
    features = {}
    for i in range(len(chunk)):
        fname = 'feat_' + str(i)
        if isinstance(chunk[i], int):
            features[fname] = str(chunk[i])
        else:
            features[fname] = chunk[i]
    return [features]


