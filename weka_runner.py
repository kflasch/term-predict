#!/usr/bin/env python3
""" Helper to run Weka with python-weka-wrapper3 """

import sys
import time
import argparse
from datetime import timedelta
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.stemmers import Stemmer
from weka.core.stopwords import Stopwords
from weka.core.tokenizers import Tokenizer
from weka.filters import Filter, StringToWordVector
from weka.classifiers import FilteredClassifier
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random

import utils
import config

# Links on using Weka & this library:
# https://waikato.github.io/weka-wiki/primer/
# http://fracpete.github.io/python-weka-wrapper3/weka.html
# http://fracpete.github.io/python-weka-wrapper3/examples.html

train_file_1 = config.train_arff_file_1
train_file_3 = config.train_arff_file_3
train_file_5 = config.train_arff_file_5

# command line Java example
# java weka.classifiers.meta.FilteredClassifier -F weka.filters.unsupervised.attribute.StringToWordVector -c 1 -t data/term_chunk_3_train.arff -W weka.classifiers.functions.SMO 

def cross_validate(classifier, filepath):
    """ Cross-validate the given ARFF file and print results """

    # load the ARFF data, class index (has-word) is first
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(filepath, class_index="first")

    if not config.weka_minimal_output:
        print("\nCross-validating " + filepath)
        print("Using classifier " + classifier)
        print("Features: " + str(data.attribute_names()))
        print("-------------------------------------------------------------")

    # classifier we are using
    # cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls = Classifier(classname=classifier)
    if classifier.endswith("MultilayerPerceptron"):
        # need to set number of layers otherwise default is way too many due to word vector features
        cls.options = ["-H", "5"]
    # cls.options = []

    # convert the notes (as String) to word vectors, using Filtered Classifier
    #  https://weka.sourceforge.io/doc.dev/weka/filters/unsupervised/attribute/StringToWordVector.html
    #  https://weka.sourceforge.io/doc.dev/weka/core/tokenizers/Tokenizer.html
    #  https://github.com/fracpete/python-weka-wrapper3-examples/blob/master/src/wekaexamples/filters/filters.py
    stemmer = Stemmer(classname=config.weka_stemmer)
    stopwords = Stopwords(classname=config.weka_stopwords)
    # tokenizer = Tokenizer(classname=config.weka_tokenizer, options=config.weka_tokenizer_options)
    s2wv = StringToWordVector()
    s2wv.options=config.weka_s2wv_options
    s2wv.stemmer = stemmer
    s2wv.stopwords = stopwords
    # s2wv.tokenizer = tokenizer
    
    fc = FilteredClassifier()
    fc.filter = s2wv
    fc.classifier = cls

    # cross-validate the model
    evl = Evaluation(data)
    start = time.process_time()
    evl.crossvalidate_model(fc, data, 10, Random(1))
    cross_validate_time = timedelta(seconds=(time.process_time() - start))
    
    # print out results in format for table and additional default weka data
    # | Classifier | features | chunk size | TP, FN, FP, TN | Accuracy | Precision | Recall | F-score | ROC Area |  Kappa
    accuracy = round(evl.percent_correct/100, 4)
    if config.weka_minimal_output:
        print("| {0} | {1} | {2} | ".format(classifier, utils.get_arff_features(filepath), utils.get_chunk_size(filepath)), end='')
        print("{0:g}, {1:g}, {2:g}, {3:g} | ".format(evl.num_true_positives(0), evl.num_false_negatives(0), evl.num_false_positives(0), evl.num_true_negatives(0)), end='')
        print("{0} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(accuracy, evl.precision(0), evl.recall(0), evl.f_measure(0), evl.area_under_roc(0), evl.kappa))   
    else:
        print("\nCross-validation finished in " + str(cross_validate_time))
        print()
        print("TP, FN, FP, TN | Accuracy | Precision | Recall | F-score | ROC Area | Kappa")
        print("{0:g}, {1:g}, {2:g}, {3:g} | ".format(evl.num_true_positives(0), evl.num_false_negatives(0), evl.num_false_positives(0), evl.num_true_negatives(0)), end='')
        print("{0} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f}".format(accuracy, evl.precision(0), evl.recall(0), evl.f_measure(0), evl.area_under_roc(0), evl.kappa))
        print()
        # print(evl.percent_correct)
        print(evl.summary())
        print(evl.class_details())
        print(evl.matrix())
    
    # jvm.stop()

def cross_val_all(classifier=".SMO"):
    """ Cross-validate all training files """
    start = time.process_time()

    cross_validate(classifier, config.train_arff_file_1)
    cross_validate(classifier, config.train_arff_file_3)
    cross_validate(classifier, config.train_arff_file_5)

    cross_validate_time = timedelta(seconds=(time.process_time() - start))
    print("\nTime elapsed: " + str(cross_validate_time))
    
def main(args):
    
    jvm.start(system_cp=True, packages=True)

    print()
    if args.classifier and args.filepath:
        # cross validate on given classifier and file
        cross_validate(args.classifier, args.filepath)
    elif args.classifier:
        # cross validate on given classifier on default files
        cross_val_all(args.classifier)
    else:
        cross_val_all()

    print()
    jvm.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Weka for term prediction cross-validation.")
    parser.add_argument("classifier", nargs="?", default=None,
                        help="Weka classifier name (RandomForest, .SMO, J48, IBk, etc.. )")
    parser.add_argument("filepath", nargs="?", default=None, help="filename for dataset")
    args = parser.parse_args()
    main(args)
