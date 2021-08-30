#!/usr/bin/env python3
""" Stanza-based Notes Analysis."""

import os
import sys
import re
import sqlite3
from collections import defaultdict
import stanza

import config

term = config.term

# this may need to be run once
# stanza.download('en')

def read_data_file(filename):
    """ Reads all data from filename and returns it. """
    f = open(filename, 'r')
    text = f.read()
    f.close()
    text = text.strip()
    return text

def read_sql_file(filename):
    """ Reads sqlite file. """
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    c.execute('SELECT * FROM tmp_sarcopenenia_to_ship_hb')
    return c.fetchall()

def parse_data_dir(path):
    """Runs parse_data_file on each file in path and returns the list
    of extracted info."""
    infolist = []
    for filename in sorted(os.listdir(path)):
        filepath = os.path.join(path, filename)
        infolist.append(parse_data_file(filepath))
    return infolist

def parse_sql_file(filename):
    """Parses a sqlite file"""
    # https://docs.python.org/3/library/sqlite3.html
    all_freq = {}
    processed_notes = {}
    query_word = term
    nlp = stanza.Pipeline('en', processors='tokenize, pos')
    db = read_sql_file(filename)
    for row in db:
        patient_num = row[0]
        encounter_num = row[1]
        note_id = row[2]
        type_ip_c = row[3]
        type_ip_name = row[4]
        note_text = row[5]

        # skip duplicate data, assumming same encounter/note means the note text is the same
        if processed_notes.get(encounter_num) and note_id in processed_notes[encounter_num]:
            # print('Already processed encounter/note: ' + str(encounter_num) + ' / ' + str(note_id))
            continue

        # annotate the note text with stanza
        doc = nlp(note_text)
        
        # print("Encounter " + str(encounter_num))
        note_freq = {}

        # some notes may have no mention of the word,
        # e.g., Encounter 66396416
        found_query_word = False
        for i, sent in enumerate(doc.sentences):
            #found_in_sent = False
            for word in sent.words:
                # print("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(\
                #     word.text, word.lemma, word.upos, word.head, word.deprel))
                if word.text.lower() == query_word:
                    found_query_word = True
                    # TODO: found word in a sentence. should we look at this sentences more closely?
                    #found_in_sent = True
                    #print(sent.text)
                note_freq[word.text] = note_freq.get(word.text, 0) + 1

        # combine note_freq with all_freq?
        parse_freqs(note_freq, 6)
        for k, v in note_freq.items():
            all_freq[k] = all_freq.get(k, 0) + v
        
        if not found_query_word:
            print(" No mention of " + query_word + " found.")
            continue

        print()

        # track processed notes
        enc_note_ids = processed_notes.get(encounter_num)
        if enc_note_ids:
            processed_notes[encounter_num].append(note_id)
        else:
            processed_notes[encounter_num] = [note_id]

    print()
    parse_freqs(all_freq, 6)
    # print(all_freq)
    # for e in processed_notes:
    #     print(str(e) + ' ' + str(processed_notes[e]))

def parse_freqs(freqs, base_freq=2):
    """ Display frequncies of words. """
    print()
    for freq in sorted(freqs, key=freqs.get, reverse=True):
        if freqs[freq] > base_freq:
            print(freq, freqs[freq])
    print()

def run(path):
    """Runs the analyzer on path."""
    if os.path.isdir(path):
        print("Running on directory not implemented.")
    elif os.path.isfile(path):
        if path.endswith('.csv'):
            print("Parsing csv not implemented.")
        elif path.endswith('.db'):
            parse_sql_file(path)
    else:
        print('Error: ' + path + ' not a regular file or directory.')

def main():
    if len(sys.argv) == 1:
        datafile = "data/sarcopenia_export_hb_4_47_2020.db"
        run(datafile)
        # run(config.train_data)
    else:
        run(sys.argv[1])

if __name__ == '__main__':
    main()
