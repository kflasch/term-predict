#!/usr/bin/env python3
""" spaCy/scispaCy-based Notes Analysis."""

import os
import sys
import re
import csv
import spacy
import scispacy

import config

# https://allenai.github.io/scispacy/
# https://github.com/allenai/scispacy

# https://www.datacamp.com/community/blog/spacy-cheatsheet

term = config.term
num_sents_each_side = 3

# nlp = spacy.load("en_core_sci_sm")
# nlp = spacy.load("en_core_sci_scibert")
nlp = spacy.load("en_ner_bionlp13cg_md")

def parse_csv_file(csv_filename, dataset):
    """ Parses csv file based on dataset """
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if dataset == 'train':
            # training model data
            for row in reader:
                matches = re.findall(term, row['deid_note_text'], re.IGNORECASE)
                if len(matches) > 1:
                    print("Mulitple occurences of term '" + term + "' in " + row['note_id_encr'])
                elif len(matches) == 0:
                    print("Term '" + term + "' not found in note id " + row['note_id_encr'] + " for file " + csv_filename)
                else:
                    analyze_surrounding_sents(row['deid_note_text'], num_sents_each_side)
                #analyze_text(row['deid_note_text'], row['note_id_encr'])
        elif dataset == 'test':
            # test data -- coded dataset -- term will not occur in this text
            for row in reader:
                pass

def analyze_text(text, note_id):
    """ """
    doc = nlp(text)
    # print(list(doc.sents))
    #print(doc.ents)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # print(" ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc]))

    # only print text tagged as an entity
    # https://stackoverflow.com/questions/58712418/replace-entity-with-its-label-in-spacy
    # new_text = " ".join([t.text if t.ent_type_ else "" for t in doc])
    # print(new_text)
    
    # matches = re.findall(term, new_text, re.IGNORECASE)
    # if len(matches) == 0:
    #     print("Term '" + term + "' not recognized as entity in note id " + note_id)
    #     print(text)
    #print(new_text)
    
    # for ent in reversed(doc.ents):
    #     text = text[:ent.start_char] + ent.label_ + text[ent.end_char:]
    # print(text)
    
def analyze_surrounding_sents(note_text, num_sents):
    """ Analyze the num_sents surrouding sentences and sentences itself that contains term  """
    # separate note into sentences with nltk
    # sentences = nltk.sent_tokenize(note_text)
    doc = nlp(note_text)
    sentences = list(doc.sents)
    for si in range(len(sentences)):
        if re.search(term, sentences[si].text, re.IGNORECASE):
            chunk = ""
            for i in range(-num_sents, num_sents+1):
                sent_index = si + i
                # if sent_index < 0:
                #     print('START OF NOTE')
                # elif sent_index >= len(sentences):
                #     print('END OF NOTE')
                # else:
                #     print(sentences[sent_index])
                if sent_index >= 0 and sent_index < len(sentences):
                    sent = sentences[sent_index].text
                    chunk += sent
            print(chunk)
            print([(ent.text, ent.label_) for ent in nlp(chunk).ents])
            print()
                

def run(path, dataset):
    """ Runs the analyzer on path/dataset. """
    if os.path.isdir(path):
        # infolist = parse_data_dir(path)
        # display_extracted_info(infolist)
        print("Running on directory not implemented.")
    elif os.path.isfile(path):
        if path.endswith('.csv'):
            parse_csv_file(path, dataset)
        else:
            print("File format not supported.")
    else:
        print('Error: ' + path + ' not a regular file or directory.')

def main():
    if len(sys.argv) < 2:
        run(config.train_data, 'train')
        # run(config.test_data, 'test')
    else:
        run(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
