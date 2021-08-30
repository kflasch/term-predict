#!/usr/bin/env python3
""" transform text with NER """

import csv
import argparse
import spacy

import config

nlp = spacy.load(config.spacy_model)

def run(infilename, outfilename):
    """ Read in csvfile and write out transformed csvfile """
    print("Writing new data file " + outfilename + " with model " + config.spacy_model)
    with open(infilename, newline='') as f_in, open(outfilename, "w", newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        # header = ','.join(reader.fieldnames)
        # outfile.write(header)
        writer.writeheader()
        for row in reader:
            # replace note text with note text with only named entities from the config model
            row['deid_note_text'] = transform_text(row['deid_note_text'])
            writer.writerow(row)
            # note_text = row['deid_note_text']
            # ner_note = transform_text(row['deid_note_text'])

            # write_to_file(row['type_ip_name'], note_text, arff_filename, chunk_len)
            # # masked_note = mask_text(row['deid_note_text'], term, 'Problem')
            # # write_to_file(row['type_ip_name'], masked_note, arff_filename, chunk_len)

def transform_text(text):
    """ Transform note's text based on settings """

    # ignore or keep term?

    # convert the text to named entities (with scispacy)
    doc = nlp(text)
    # https://stackoverflow.com/questions/58712418/replace-entity-with-its-label-in-spacy
    text = " ".join([t.text if t.ent_type_ else "" for t in doc])

    return text

def main(infile, outfile):
    if infile and outfile:
        run(infile, outfile)
    else:
        run(config.train_data, config.ner_train_data)
        run(config.test_data, config.ner_test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform csv notes with ner for term prediction.")
    parser.add_argument("infile", nargs="?", default=None, help="input filename for csv dataset")
    parser.add_argument("outfile", nargs="?", default=None, help="output filename for transformed csv dataset")
    args = parser.parse_args()
    main(args.infile, args.outfile)
