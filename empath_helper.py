#!/usr/bin/env python3
""" Analyze .csv files with Empath """

import sys
import os
import csv
import re
import nltk
from empath import Empath

import config

# https://github.com/Ejhfast/empath-client/
# http://empath.stanford.edu/

term = config.term
lexicon = Empath()
total_dict = {}
term_categories = ["depleting", "muscskel", "gaitmobility", "fracture", "frail"]
doNormalization=False

def create_categories():
    """ Create Empath categories based on words """
    lexicon.create_category("depleting", ["depleting"])
    lexicon.create_category("muscskel", ["muscle", "skeletal", "musculoskeletal"])
    lexicon.create_category("gaitmobility", ["gait", "mobility"])
    lexicon.create_category("fracture", ["fracture"])
    lexicon.create_category("frail", ["frail", "frailty"])

def parse_csv_file(csv_filename, dataset):
    """ Parses csv file based on dataset """

    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if dataset == 'train':
            # training model data
            for row in reader:
                if re.search(term, row['deid_note_text'], re.IGNORECASE):
                    result = lexicon.analyze(row['deid_note_text'], categories=term_categories, normalize=doNormalization)
                    update_dict(result)                    
                else:
                    print("Term '" + term + "' not found in note id " + row['note_id_encr'] + " for file " + csv_filename)
        elif dataset == 'test':
            # test data -- coded dataset -- term will not occur in this text
            for row in reader:
                result = lexicon.analyze(row['deid_note_text'], normalize=doNormalization)
                update_dict(result)
            
def update_dict(cat_dict):
    """ """
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

def main():
    if len(sys.argv) < 2:
        run(config.train_data, 'train')
        # run(config.test_data, 'test')
    elif sys.argv[1] == "-c":
        create_categories()
    else:
        run(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
