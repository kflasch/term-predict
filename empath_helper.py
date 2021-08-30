#!/usr/bin/env python3
""" Analyze .csv files with Empath """

import sys
import os
import csv
import re
import argparse
import nltk
from empath import Empath

import config

# https://github.com/Ejhfast/empath-client/
# http://empath.stanford.edu/

term = config.term
lexicon = Empath()
total_dict = {}
doNormalization=False

def create_categories():
    """ Create Empath categories based on words """
    # this creates category data in a location where empath is installed
    # like lib/python3.9/site-packages/empath/data (likely under venv)
    for cat in config.empath_categories:
        lexicon.create_category(cat, config.empath_cat_words[cat])

def parse_csv_file(csv_filename, dataset):
    """ Parses csv file based on dataset """
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if dataset == 'train':
            # training model data
            for row in reader:
                if re.search(term, row['deid_note_text'], re.IGNORECASE):
                    result = lexicon.analyze(row['deid_note_text'], categories=config.empath_categories, normalize=doNormalization)
                    update_dict(result)
                else:
                    print("Term '" + term + "' not found in note id " + row['note_id_encr'] + " for file " + csv_filename)
        elif dataset == 'test':
            # test data -- coded dataset -- term will not occur in this text
            for row in reader:
                result = lexicon.analyze(row['deid_note_text'], categories=config.empath_categories, normalize=doNormalization)
                update_dict(result)
            
def update_dict(cat_dict):
    """ Update dict of Empath analyze results """
    for key, val in cat_dict.items():
        if val > 0:
            total_dict[key] = total_dict.get(key, 0) + val
            # print(key + " : " + str(val))

def run(path, dataset):
    """ Run on this file/dataset """
    if os.path.isdir(path):
        print("Running on directory not implemented.")
        return
    if not os.path.isfile(path) or not path.endswith('.csv'):
        print(path + " not valid .csv")
        return

    parse_csv_file(path, dataset)
    # sorted_total_dict = {k: v for k, v in sorted(total_dict.items(), key=lambda item: item[1], reverse=True)}
    sorted_total_dict = dict(sorted(total_dict.items(), key=lambda item: item[1], reverse=True))
    print(sorted_total_dict)
    print()

def main(args):
    if args.train:
        run(config.train_data, 'train')
    elif args.test:
        run(config.test_data, 'test')
    elif args.createcat:
        create_categories()
    else:
        print("Unknown option.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Helper for Empath related functions.")
    parser.add_argument("-c", "--createcat", help="create Empath categories", action="store_true")
    parser.add_argument("-t", "--train", help="analyze training notes with Empath", action="store_true")
    parser.add_argument("-s", "--test", help="analyze test notes with Empath", action="store_true")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(args)
