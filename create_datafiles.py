#!/usr/bin/env python3
""" Convert .csv data to pkl or arff datafiles """

import sys
import os
import csv
import re
import argparse
import pandas as pd
import nltk
import spacy
import scispacy
# from scispacy.linking import EntityLinker
from empath import Empath

import config
import utils

term = config.term
lexicon = Empath()
empath_categories = config.empath_categories
anatomy_terms_file = config.anatomy_terms_file

spacy_nlp = None
if config.ner_transform:
    spacy_nlp = spacy.load(config.spacy_model)
    # spacy_nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "mesh"})

def create_dataframe(csv_filename, dataset, chunk_len=1):
    """ Creates and returns a dataframe from a csv file """
    data = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            note_text = row['deid_note_text']
            note_type = row['type_ip_name']
            # note_id = row['note_id_encr']
            if dataset == 'train' and not re.search(term, note_text, re.IGNORECASE):
                # skip this note if training dataset and term does not occur in it
                print("Term '" + term + "' not found on line " + reader.line_num +
                      " for training file " + csv_filename + " -- skipping note!")
                continue
            dl = get_datalist_for_note(note_text, chunk_len, note_type, dataset)
            data.extend(dl)
    # column names for the dataframe -- order needs to match what is in data
    cols = ['has-word', 'note-type', 'text-length', 'empath-cat-depleting',
            'empath-cat-muscskel', 'empath-cat-gaitmobility', 'empath-cat-fracture',
            'empath-cat-frail', 'anatomy-terms', 'notes']
    df = pd.DataFrame(data, columns=cols)
    return df

def get_datalist_for_note(note_text, chunk_len, note_type, dataset):
    """ Get list of lists that contains data for this note """
    datalist = []
    # separate note into sentences with nltk
    sentences = nltk.sent_tokenize(note_text)
    chunks = None
    # creating chunks of sentences depends on if the term exists in notes or not
    if dataset == 'train':
        chunks = divide_chunks_on_word(sentences, chunk_len, term)
    elif dataset == 'test':
        chunks = divide_chunks(sentences, chunk_len)
    for chunk in chunks:
        chunk_text = ' '.join(chunk)
        has_word = '?'
        if dataset == 'train':
            # term only occurs in training set
            if re.search(term, chunk_text, re.IGNORECASE):
                has_word = 'True'
                chunk_text = mask_text(chunk_text, term, config.mask)
            else:
                has_word = 'False'
        chunk_text = transform_text(chunk_text)
        chunk_text = chunk_text.replace("'", r"\'") # escape single quotes
        datarow = [has_word, note_type, len(chunk_text)] + get_empath_data(chunk_text) + [get_anatomy_terms_data(chunk_text), chunk_text]
        datalist.append(datarow)
    return datalist

def create_arff_header(filename):
    """ Creates the arff file and populates the header with features/etc """
    with open(filename, "w") as f:
        f.write("@relation " + term + "\n")
        f.write("@attribute has-word {True, False}\n")
        if config.create_arff_feature_notetype:
            f.write("@attribute note-type {'Care Plan Note', 'Consults', 'Discharge Summary', 'H&P', 'Progress Notes'}\n")
        if config.create_arff_feature_textlen:
            f.write("@attribute text-length numeric\n")
        if config.create_arff_feature_empath:
            for cat in empath_categories:
                f.write("@attribute empath-cat-" + cat + " {True, False}\n")
            # f.write("@attribute empath-any-cat {True, False}\n")
        # f.write("% @attribute clamp-label {'problem', 'test', 'treatment'}\n")
        if config.create_arff_feature_anatomy_terms:
            f.write("@attribute anatomy-terms {True, False}\n")
        f.write("@attribute notes string\n")
        f.write("% @attribute note-id string\n")
        f.write("@data\n")
    return filename

def create_arff_body(csv_filename, arff_filename, dataset, chunk_len=1):
    """ Create arff file body from csv dataset file """
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if dataset == 'train':
            # training model data
            for row in reader:
                note_text = row['deid_note_text']
                note_type = row['type_ip_name']
                if not re.search(term, note_text, re.IGNORECASE):
                    # skip this note if term does not occur in it
                    print("Term '" + term + "' not found in note id " + row['note_id_encr'] + " for file " + csv_filename)
                    continue
                write_to_arff_train_file(note_type, note_text, arff_filename, chunk_len)
        elif dataset == 'test':
            # test data -- coded dataset -- term will not occur in this text
            for row in reader:
                note_text = row['deid_note_text']
                note_type = row['type_ip_name']
                write_to_arff_test_file(note_type, note_text, arff_filename, chunk_len)

def write_to_arff_train_file(note_type, note_text, filename, chunk_len):
    """ Write the note_text divided into chunks with features to the arff file """
    with open(filename, "a") as f:
        # separate note into sentences with nltk
        sentences = nltk.sent_tokenize(note_text)
        chunks = divide_chunks_on_word(sentences, chunk_len, term)
        for chunk in chunks:
            chunk_text = ' '.join(chunk)
            has_word = 'False'
            if re.search(term, chunk_text, re.IGNORECASE):
                has_word = 'True'
                chunk_text = mask_text(chunk_text, term, config.mask)
            chunk_text = transform_text(chunk_text)
            chunk_text = chunk_text.replace("'", r"\'") # escape single quotes
            dataline = "{}".format(has_word)
            if config.create_arff_feature_notetype:
                dataline += ", '{}'".format(note_type)
            if config.create_arff_feature_textlen:
                dataline += ", {}".format(len(chunk_text))
            if config.create_arff_feature_empath:
                dataline += ", {}".format(get_empath_data(chunk_text, as_str=True))
            if config.create_arff_feature_anatomy_terms:
                dataline += ", {}".format(get_anatomy_terms_data(chunk_text))
            dataline += ", '{}'".format(chunk_text)
            dataline += "\n"
            f.write(dataline)

def write_to_arff_test_file(note_type, note_text, filename, chunk_len):
    """ Write the data line to the test set arff file """
    with open(filename, "a") as f:
        # separate note into sentences with nltk
        sentences = nltk.sent_tokenize(note_text)
        chunks = divide_chunks(sentences, chunk_len)
        for chunk in chunks:
            chunk_text = ' '.join(chunk)
            chunk_text = chunk_text.replace("'", r"\'") # escape single quotes
            dataline = "?"
            if config.create_arff_feature_notetype:
                dataline += ", '{}'".format(note_type)
            if config.create_arff_feature_textlen:
                dataline += ", {}".format(len(chunk_text))
            if config.create_arff_feature_empath:
                dataline += ", {}".format(get_empath_data(chunk_text, as_str=True))
            if config.create_arff_feature_anatomy_terms:
                dataline += ", {}".format(get_anatomy_terms_data(chunk_text))
            dataline += ", '{}'".format(chunk_text)
            dataline += "\n"
            f.write(dataline)

def get_empath_data(text, as_str=False):
    """ Add feature data for each empath category. """
    l = []
    cat_dict = lexicon.analyze(text, categories=empath_categories) #, normalize=doNormalization)
    for cat in empath_categories:
        val = cat_dict[cat]
        if val > 0:
            l.append("True")
        else:
            l.append("False")
    if as_str:
        return ", ".join(l)
    return l

def get_anatomy_terms_data(text):
    """ Add feature data based on anatomy terms. """
    with open(anatomy_terms_file, "r") as f:
        aterms = f.read().splitlines()
        for aterm in aterms:
            if re.search(r'\b{}\b'.format(aterm.strip()), text, re.IGNORECASE):
                return "True"
    return "False"

def mask_text(text, mask_word, mask):
    """ Changes mark_word to mask in text """
    maskre = re.compile(re.escape(mask_word), re.IGNORECASE)
    return maskre.sub(mask, text)

def transform_text(text):
    """ Transform note's text based on settings """

    # convert the text to named entities (with scispacy)
    if config.ner_transform:
        doc = spacy_nlp(text)

        # linker = spacy_nlp.get_pipe("scispacy_linker")
        # for ent in doc.ents:
        #     for mesh_desc in ent._.kb_ents:
        #         link_ent = linker.kb.cui_to_entity[mesh_desc[0]]
                # print("* Entity name: ", ent)
                # print("  " + link_ent.concept_id + " " + link_ent.canonical_name)
                # print(linker.kb.cui_to_entity[mesh_desc[0]])

        # https://stackoverflow.com/questions/58712418/replace-entity-with-its-label-in-spacy
        text = " ".join([t.text if t.ent_type_ else "" for t in doc])

    return text

def divide_chunks(l, n):
    """ Divide list l into n-sized chunks """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def divide_chunks_on_word(sentences, n, word):
    """ Divide list sentences into n-sized chunks based around word """
    # what to do if word occurs multiple times?
    # should use islice instead?
    h = n//2
    # index of the sentence that contains the word
    si = next(i for i,v in enumerate(sentences)
              if re.search(word, v, re.IGNORECASE))
    chunked_sents = list(divide_chunks(sentences[0:si-h], n))
    chunked_sents.append(sentences[si-h:si+h+1])
    chunked_sents.extend(list(divide_chunks(sentences[si+h+1:len(sentences)], n)))
    return chunked_sents

def create_rating_file(csv_filename, chunk_size=5):
    """ Create a readable file separated into chunks for manual rating purposes """
    out_filename = config.data_dir + config.term + "_chunk_" + str(chunk_size) + "_ratings.txt"
    print("Creating rating file " + out_filename)
    with open(csv_filename, newline='') as f_in, open(out_filename, "w") as f_out:
        f_out.write(term + " text chunk ratings\n\n")
        f_out.write("Note text from " + csv_filename + "\n\n")
        f_out.write("Ratings: 0 (does not suggest), 3 (unsure or may suggest), 5 (suggests)\n")
        f_out.write("----------------------------------------------------------------------------------------------------------\n\n\n")
        reader = csv.DictReader(f_in)
        chunk_num = 1
        for row in reader:
            note_text = row['deid_note_text']
            note_id = row['note_id']
            # separate note into sentences with nltk
            sentences = nltk.sent_tokenize(note_text)
            chunks = divide_chunks(sentences, chunk_size)
            note_chunk_num = 1
            for chunk in chunks:
                chunk_text = ' '.join(chunk)
                dataline = "Note " + note_id + "_" + str(note_chunk_num) + "\n\n"
                dataline += chunk_text + "\n\n"
                dataline += "Rating for chunk " + str(chunk_num) + ": \n\n"
                dataline += "-------------------------------------------------------\n\n"
                f_out.write(dataline)
                note_chunk_num += 1
                chunk_num += 1

def run(path, dataset, create_arff, create_pkl):
    """ Run code on this file/dataset to generate pkl and/or arff file. """
    if os.path.isdir(path):
        print("Running on directory not implemented.")
        return
    if not os.path.isfile(path) or not path.endswith('.csv'):
        print(path + " not valid .csv")
        return

    for chunk_size in config.chunk_sizes:
        basename = config.data_dir + config.term + "_chunk_" + str(chunk_size) + "_" + dataset
        if config.ner_transform:
            basename = config.data_dir + config.term + "_chunk_" + str(chunk_size) + "_ner_" + dataset

        if create_arff:
            arff_filename = basename + ".arff"
            print("Creating " + arff_filename)
            create_arff_header(arff_filename)
            create_arff_body(path, arff_filename, dataset, chunk_size)
        if create_pkl:
            pkl_filename = basename + ".pkl"
            print("Creating " + pkl_filename)
            df = create_dataframe(path, dataset, chunk_size)
            df.to_pickle(pkl_filename)

def main(args):
    if args.rating:
        create_rating_file(config.test_data)
        return

    if args.csvfile and args.dataset:
        run(args.csvfile, args.dataset, args.arff, args.pkl)
    elif args.csvfile or args.dataset:
        print("Please specify both csvfile and dataset name or neither.")
        return
    else:
        run(config.train_data, 'train', args.arff, args.pkl)
        run(config.test_data, 'test', args.arff, args.pkl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create data files from csv data.")
    parser.add_argument("-a", "--arff", help="create arff files", action="store_true")
    parser.add_argument("-p", "--pkl", help="create pkl files", action="store_true")
    parser.add_argument("-r", "--rating", help="create rating template file from test data", action="store_true")
    parser.add_argument("csvfile", nargs="?", default=None, help="filename for csv dataset")
    parser.add_argument("dataset", nargs="?", default=None, help="dataset type ('train' or 'test')")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(args)
