#!/usr/bin/env python3
""" sklearn-based classification for term prediction """

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import Bunch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn import svm
from sklearn import ensemble
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import neural_network
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib import pyplot

import utils
import config

def transform_arff_data(dataset):
    """ Transform feature data that came from arff file """

    # features: note-type, text-length, empath-cat-*, anatomy-terms, notes
    # requires columns to be in strict order (and all present that are modified below!)

    # https://scikit-learn.org/stable/modules/compose.html#column-transformer
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    # https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features

    # all data in dataset are strings, convert number fields like text-length and booleans to 1/0
    # should result in a sparse matrix (due to CountVectorizer)
    np_str_to_num = FunctionTransformer(lambda x: x.astype(np.float64))
    tf_str_to_num = FunctionTransformer(lambda x: np.where(x == 'True', 1, 0))
    num_transformer = Pipeline(steps=[
        ('str_to_num', np_str_to_num),
        ('scaler', StandardScaler(with_mean=False))]) # don't use negative numbers if all positive
    ct = ColumnTransformer(
        [('note_cat', OneHotEncoder(dtype='int'), [0]),
         ('text_len', num_transformer, [1]),
         ('t_f_feats', tf_str_to_num, slice(2, -1)),
         ('notes_bow', CountVectorizer(), -1)],
        remainder='passthrough')
    X = ct.fit_transform(dataset.data)
    return X

# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
# https://scikit-learn.org/stable/modules/compose.html#column-transformer
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
def get_col_transformer(df):
    """ Returns ColumnTransformer based on data in dataframe and sets cols to drop if needed """

    # features: note-type, text-length, empath-cat-*, anatomy-terms, notes
    # class label has-word probably not present

    # Drop unused features/columns and add transformers based on config settings
    # Convert True/False to 1/0 (not sure if this matters? but can't hurt)
    # Standardize text-length and don't use negative numbers for it
    # Convert notes to word vectors (bag of words)
    # Should result in a sparse matrix (due to CountVectorizer)
    tf_str_to_num = FunctionTransformer(lambda x: np.where(x == 'True', 1, 0))
    t_list = []

    if config.use_feature_notetype:
        t_list.append(('note-type', OneHotEncoder(dtype='int'), ['note-type']))
    elif 'note-type' in df:
        t_list.append(('note-type', 'drop', 'note-type'))

    if config.use_feature_textlen:
        t_list.append(('text-length', StandardScaler(with_mean=False), ['text-length']))
    elif 'text-length' in df:
        t_list.append(('text-length', 'drop', 'text-length'))

    if config.use_feature_notes:
        if config.use_tfidf:
            pipe = Pipeline([('count', CountVectorizer(ngram_range=config.ngram_range, stop_words=config.stop_words)),
                             ('tfidf', TfidfTransformer())])
            t_list.append(('notes', pipe, 'notes'))
        else:
            t_list.append(('notes', CountVectorizer(ngram_range=config.ngram_range, stop_words=config.stop_words), 'notes'))
    elif 'notes' in df:
        t_list.append(('notes', 'drop', 'notes'))

    for ecat in config.empath_categories:
        ecat = "empath-cat-" + ecat
        if config.use_feature_empath:
            t_list.append((ecat, tf_str_to_num, [ecat]))
        elif ecat in df:
            t_list.append((ecat, 'drop', ecat))

    if config.use_feature_anatomy_terms:
        t_list.append(('anatomy-terms', tf_str_to_num, ['anatomy-terms']))
    elif 'anatomy-terms' in df:
        t_list.append(('anatomy-terms', 'drop', 'anatomy-terms'))

    ct = ColumnTransformer(t_list)
    return ct

def oversample_data(df, num_extra=4):
    """ Simple oversampling of values in dataframe by row duplication """
    hw = df.loc[df['has-word'] == 'True']
    df_os = df.append([hw]*num_extra, ignore_index=True)
    return df_os

def get_ratings(filepath):
    """ Get manual ratings of chunks """
    ratings = {}
    with open(filepath, 'r') as f:
        for line in f:
            ls = line.split(':')
            chunk = int(ls[0].split()[-1])
            rating = int(ls[1])
            ratings[chunk] = rating
    return ratings

def get_clf_by_name(name):
    """ Returns a classifier based on shorthand names """

    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    # weka knn uses 1 neighbor
    # weka SVC uses poly kernel (exp/degree 1), sk uses rbf, etc...
    rs = None
    return {
        'svc': svm.SVC(kernel='linear', C=1, random_state=rs),
        'lsvc': svm.LinearSVC(C=1, class_weight='balanced', random_state=rs),
        'randomforest': ensemble.RandomForestClassifier(random_state=rs),
        'gbc': ensemble.GradientBoostingClassifier(random_state=rs),
        'dtree': tree.DecisionTreeClassifier(random_state=rs),
        'naivebayes': naive_bayes.MultinomialNB(),
        'knn': neighbors.KNeighborsClassifier(n_neighbors=1),
        'perceptron': linear_model.Perceptron(random_state=rs),
        'logreg': linear_model.LogisticRegression(solver='lbfgs', C=1, max_iter=1000, random_state=rs),
        'mlp': neural_network.MLPClassifier(hidden_layer_sizes=(5), solver='lbfgs', random_state=rs)
        # 'crf': sklearn_crfsuite.CRF() # not working with sklearn >= 0.24
    }.get(name, None)

def run_on_all_files(clf_name="svc", do_cv=True, do_predict=False, do_holdout=False):
    """ Run on all default files """
    clf_names = [clf_name]
    if clf_name == "all":
        clf_names = ["svc", "randomforest", "gbc", "dtree", "naivebayes", "knn", "logreg", "perceptron", "mlp"]
    for name in clf_names:
        run_clf(name, config.train_pd_file_1, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)
        run_clf(name, config.train_pd_file_3, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)
        run_clf(name, config.train_pd_file_5, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)
        run_clf(name, config.train_pd_file_7, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)
        run_clf(name, config.train_pd_file_9, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)

def run(clf_name, filepath, do_cv=True, do_predict=False, do_holdout=False):
    """ Run classifier(s) on filepath """
    clf_names = [clf_name]
    if clf_name == "all":
        clf_names = ["svc", "randomforest", "gbc", "dtree", "naivebayes", "knn", "logreg", "perceptron", "mlp"]
    for name in clf_names:
        run_clf(name, filepath, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)

def run_clf(clf_name, filepath, do_cv=True, do_predict=False, do_holdout=False):
    """ Run classifier on filepath """

    clf = get_clf_by_name(clf_name)
    if clf is None:
        print("Unknown classifier name " + clf_name)
        sys.exit()

    # load training dataframe
    df_train = pd.read_pickle(filepath)

    # apply any oversampling if configured
    if config.oversample_amount and config.oversample_amount > 0:
        df_train = oversample_data(df_train, config.oversample_amount)

    # training target/y data
    y_train = df_train.pop('has-word')
    # maybe unnecessary to convert to 1/0 but can work better with some defaults
    y_train = (y_train == 'True').astype(int)

    # get ColumnTransformer -- this may set columns to drop depending on config
    col_tran = get_col_transformer(df_train)

    # list of non-dropped columns to display which features are enabled
    collist = [t[0] for t in col_tran.transformers if t[1] != 'drop']

    # apply transformations
    X_train = col_tran.fit_transform(df_train)

    chunk_size = utils.get_chunk_size(filepath)
    feature_list = utils.get_features(collist)

    if do_holdout:
        holdout_test(X_train, y_train, clf, feature_list, chunk_size)
        return

    # cross-validation
    if do_cv:
        # catch warnings? some warnings that should cause a halt do not
        scores = run_cv(clf, X_train, y_train)

        # option to show classification_report ?
        print_score_results(type(clf).__name__, feature_list, chunk_size, scores)
        if config.show_cv_train_scores:
            print_score_results(type(clf).__name__, feature_list, chunk_size,
                                scores, use_train_scores=True)

        return

    # prediction
    if do_predict:

        # get test dataframe and transform it with same transformations as training data
        test_filepath = filepath.replace("train", "test")
        df_test = pd.read_pickle(test_filepath)
        df_test.pop('has-word') # don't need

        # apply transformations without refitting
        X_test = col_tran.transform(df_test)

        # train classifier
        clf.fit(X_train, y_train)

        # decision tree plotting
        if isinstance(clf, tree.DecisionTreeClassifier) and config.plot_dtree:
            plot_dtree(clf, col_tran.get_feature_names(), chunk_size)

        predicted = clf.predict(X_test)

        # print_prediction_results(type(clf).__name__, utils.get_features(collist), test_filepath, df_test)

        # TODO: better check for ratings
        ratings_exist = True if os.path.isfile(config.ratings_first) else False

        # if not chunk 5 or no ratings files, do nothing with ratings, just display results
        if chunk_size != "5" or not ratings_exist:
            print("\nPositive predictions on data set " + test_filepath + " with classifier " + type(clf).__name__)
            print("Using features: " + feature_list)
            print("-------------------------------------------------------------------------------------------------")

            for i, (chunk, hasword) in enumerate(zip(df_test['notes'], predicted), start=1):
                if hasword:
                    print('Chunk %d => %r...' % (i, chunk[0:50]))
            return

        # only compare ratings / get scores for chunk 5 (only one with manual ratings)
        r1 = get_ratings(config.ratings_first)
        r2 = get_ratings(config.ratings_second)
        r3 = get_ratings(config.ratings_third)
        combined_avg = {}
        # all ratings should have same length, so this is safe
        for i in r1:
            total = r1[i] + r2[i] + r3[i]
            avg = total / 3.0
            combined_avg[i] = round(avg, 1)

        cm_dicts = [get_pred_cm(predicted, r1),
                    get_pred_cm(predicted, r2),
                    get_pred_cm(predicted, r3)]
        # sum all keys
        cm_all = {k: sum(cm[k] for cm in cm_dicts) for k in cm_dicts[0]}

        if config.sklearn_print_predtext:
            print("\nPositive predictions on data set " + test_filepath + " with classifier " + type(clf).__name__)
            print("Using features: " + utils.get_features(collist))
            print("-------------------------------------------------------------------------------------------------")

            for i, (chunk, hasword) in enumerate(zip(df_test['notes'], predicted), start=1):
                if hasword:
                    print('Chunk %d => %r...  Ratings: %d %d %d' % (i, chunk[0:50], r1[i], r2[i], r3[i]))

        print_table_header()
        print_score_results(type(clf).__name__, feature_list, chunk_size, None, cm_all)
        # ratings avg version
        cm_avg = get_pred_cm_from_avg(predicted, combined_avg)
        print_score_results(type(clf).__name__ + " AVG", feature_list, chunk_size, None, cm_avg)
        print_table_footer()
        print()

        return

def get_pred_cm_from_avg(predicted, r, weight=config.weight_ratings):
    """ Return a conf matrix from prediction and avg ratings """
    scores = {'test_tp': 0, 'test_tn': 0, 'test_fp': 0, 'test_fn': 0}

    for i, val in enumerate(predicted, start=1):
        if val == 1:
            if r[i] <= 1:
                scores['test_fp'] += 1
            elif r[i] >= 2:
                scores['test_tp'] += 1
            elif r[i] == 5 and not weight:
                scores['test_tp'] += 1
            elif r[i] == 5 and weight:
                scores['test_tp'] += 2
                print('ok')
            else:
                print("Error: invalid rating " + str(r[i]))
        elif val == 0:
            if r[i] <= 1:
                scores['test_tn'] += 1
            elif r[i] >= 2:
                scores['test_fn'] += 1
            elif r[i] == 5 and not weight:
                scores['test_fn'] += 1
            elif r[i] == 5 and weight:
                scores['test_fn'] += 2
                print('no')
            else:
                print("Error: invalid rating " + str(r[i]))
        else:
            print("Error: invalid prediction " + str(val))

    return scores

def get_pred_cm(predicted, r, weight=config.weight_ratings):
    """ Return a conf matrix from prediction and ratings """
    scores = {'test_tp': 0, 'test_tn': 0, 'test_fp': 0, 'test_fn': 0}

    for i, val in enumerate(predicted, start=1):
        if val == 1:
            if r[i] == 0:
                scores['test_fp'] += 1
            elif r[i] == 3:
                scores['test_tp'] += 1
            elif r[i] == 5 and not weight:
                scores['test_tp'] += 1
            elif r[i] == 5 and weight:
                scores['test_tp'] += 2
                print('ok')
            else:
                print("Error: invalid rating " + str(r[i]))
        elif val == 0:
            if r[i] == 0:
                scores['test_tn'] += 1
            elif r[i] == 3:
                scores['test_fn'] += 1
            elif r[i] == 5 and not weight:
                scores['test_fn'] += 1
            elif r[i] == 5 and weight:
                scores['test_fn'] += 2
                print('no')
            else:
                print("Error: invalid rating " + str(r[i]))
        else:
            print("Error: invalid prediction " + str(val))

    return scores

def holdout_test(X, y, clf, feature_list, chunk_size):
    """ Holdout test for overfitting """

    # get 12 notes of training data (30%) for hold out test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # train model
    clf.fit(X_train, y_train)

    # get predictions of model on holdout set
    y_pred = clf.predict(X_test)

    # get cm based on holdout values and predictions
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # tn, fp, fn, tp = cm.ravel()
    # print(tn, fp, fn, tp)
    cm_scores = {'test_tp': cm[1, 1], 'test_tn': cm[0, 0], 'test_fp': cm[0, 1], 'test_fn': cm[1, 0]}
    # print(scores)
    clf_name = type(clf).__name__ + ' (H)'
    print_score_results(clf_name, feature_list, chunk_size, None, cm_scores)

def plot_dtree(clf, feature_names, chunk_size):
    """ Plot decision tree fitted clf and save image. """

    fname = "img/dtree_chunk_" + str(chunk_size) + ".png"
    # print("Plotting decision tree to " + fname)
    fig = pyplot.figure(figsize=(20,15))
    _ = tree.plot_tree(clf, feature_names=feature_names, max_depth=2,
                       node_ids=True, proportion=True, impurity=False,
                       rounded=True, filled=True)
    fig.savefig(fname)

# https://scikit-learn.org/stable/modules/cross_validation.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
def run_cv(clf, X, y, scoring=None):
    """ Cross-validate with given classifier on data X, y and output results """

    if not scoring:
        scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
                   'fp': make_scorer(fp), 'fn': make_scorer(fn),
                   'roc_auc': make_scorer(roc_auc_score, needs_threshold=True)}
    # will be StratifiedKFold by default (assuming clf supports it)
    scores = cross_validate(clf, X, y, cv=10, scoring=scoring, return_train_score=config.show_cv_train_scores)
    # scores = cross_validate(clf, X, y, cv=10, scoring=conf_matrix_scorer) # requires 0.24+
    return scores

# fns for 0.23.2 scoring
# https://scikit-learn.org/0.23/modules/model_evaluation.html#using-multiple-metric-evaluation
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

# requires 0.24+ to use
def conf_matrix_scorer(clf, X, y):
    """ Scorer function for cross_validate results """

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # https://stackoverflow.com/questions/65645125/producing-a-confusion-matrix-with-cross-validate
    # https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn

    # clf.predict_proba(X)[:, 1]
    # [:, 1] means all rows, second column, which should be probability of being 1 (True)

    # https://scikit-learn.org/stable/modules/model_evaluation.html#roc-auc-multilabel
    # calculating roc score depends on classifier (decision_function vs predict_proba)

    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1],
            'roc_auc': roc_auc_score(y, clf.decision_function(X) if hasattr(clf, "decision_function")
                                     else clf.predict_proba(X)[:, 1])}

def print_score_results(clf_name, features, chunk_size, scores, cm=None, use_train_scores=False):
    """ Print out results of cross-validation or prediction scores """

    # Due to small sample size, some folds may no results of some classes, resulting in division by zero.
    # So instead of getting a mean of all those results (like precision), we compute the precision based on
    # the end result of TPs, FPs, etc

    # can use the normal scores object or a custom conf matrix dict
    if use_train_scores:
        tp = scores['train_tp'].sum() if scores else cm['train_tp']
        tn = scores['train_tn'].sum() if scores else cm['train_tn']
        fp = scores['train_fp'].sum() if scores else cm['train_fp']
        fn = scores['train_fn'].sum() if scores else cm['train_fn']
        clf_name = clf_name + " *"
    else:
        tp = scores['test_tp'].sum() if scores else cm['test_tp']
        tn = scores['test_tn'].sum() if scores else cm['test_tn']
        fp = scores['test_fp'].sum() if scores else cm['test_fp']
        fn = scores['test_fn'].sum() if scores else cm['test_fn']

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp != 0 else float('NaN')
    recall = tp / (tp + fn) # sensitivity, true positive rate
    specificity = tn / (tn + fp) # true negative rate
    balanced_accuracy = (recall + specificity) / 2
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) != 0 else float('NaN')
    kappa = (2 * (tp * tn - fn * fp)) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    # mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # some scores might not include roc_auc so just show nan
    roc_auc = float('NaN')
    if scores and 'test_roc_auc' in scores:
        roc_auc = scores['test_roc_auc'].mean()

    if config.sklearn_reduced_output:
        print("| {0:26} | {1:52} | {2:5} | ".format(clf_name, features, chunk_size), end='')
        print("{2:5.3f} | {3:6.3f} | {4:5.3f} |"
              .format(accuracy, balanced_accuracy, precision, recall, f1))
    else:
        print("| {0:26} | {1:52} | {2:5} | ".format(clf_name, features, chunk_size), end='')
        print("{0:3g} | {1:3g} | {2:3g} | {3:4g} | ".format(tp, fn, fp, tn), end='')
        print("{0:5.3f} | {1:9.3f} | {2:5.3f} | {3:6.3f} | {4:5.3f} | {5:7.3f} | {6:6.3f} |"
              .format(accuracy, balanced_accuracy, precision, recall, f1, roc_auc, kappa))

def print_table_header():
    """ Print out table header info about results """
    print()
    if config.sklearn_reduced_output:
        print("|{:-<116}|".format(""))
        print("| {:26} | {:52} | {:5} | {:5} | {:6} | {:5} |"
              .format("Classifier", "Features", "Chunk", "Prec.", "Recall", "F1"))
        print("|{:-<116}|".format(""))
    else:
        print("|{:-<180}|".format(""))
        print("| {:26} | {:52} | {:5} | {:3} | {:3} | {:3} | {:4} | {:5} | {:9} | {:5} | {:6} | {:5} | {:7} | {:6} |"
              .format("Classifier", "Features", "Chunk", "TP", "FN", "FP", "TN", "Acc.", "Bal. Acc.", "Prec.", "Recall", "F1", "ROC AUC", "Kappa"))
        print("|{:-<180}|".format(""))

def print_table_footer():
    """ Print out table footer for results """
    if config.sklearn_reduced_output:
        print("|{:-<116}|".format(""))
    else:
        print("|{:-<180}|".format(""))

def print_current_config(do_cv, do_predict):
    """ Print relevant config options at time of run """
    print("sklearn version " + str(sklearn.__version__))
    print(f"Using: {config.ngram_range=}, {config.stop_words=}, {config.oversample_amount=}, {config.use_tfidf=}")
    if do_cv:
        print("Performing cross-validation")
    if do_predict:
        print("Performing prediction")
    print()
    
def main(do_cv, do_predict, do_holdout, classifier=None, filepath=None):
    """ Main function to run cross-validation and/or prediction """

    if filepath and not classifier:
        print("Incorrect options")
        return

    if do_cv and do_predict:
        print("Incorrect options: choose either -c or -p")
        return

    print_current_config(do_cv, do_predict)
    if do_cv: print_table_header()

    if classifier and filepath:
        run(classifier, filepath, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)
    elif classifier:
        run_on_all_files(classifier, do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)
    else:
        run_on_all_files(do_cv=do_cv, do_predict=do_predict, do_holdout=do_holdout)

    if do_cv: print_table_footer()
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run sklearn classification for term prediction.")
    parser.add_argument("-c", "--cv", help="run cross-validation", action="store_true")
    parser.add_argument("-p", "--predict", help="run prediction", action="store_true")
    parser.add_argument("-o", "--holdout", help="run holdout test", action="store_true")
    parser.add_argument("classifier", nargs="?", default=None,
                        help="classifier name (svc, randomforest, gbc, dtree, naivebayes, knn, perceptron, logreg, mlp, crf)")
    parser.add_argument("filepath", nargs="?", default=None, help="filename for training data")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(args.cv, args.predict, args.holdout, args.classifier, args.filepath)
