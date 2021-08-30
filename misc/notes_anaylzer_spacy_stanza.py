#!/usr/bin/env python3
""" spaCy/Stanza-based Notes Analysis."""

import os
import sys
import re
import sqlite3
import stanza
from spacy.matcher import Matcher
from spacy_stanza import StanzaLanguage

import config

# https://spacy.io/universe/project/spacy-stanza
# https://github.com/explosion/spacy-stanza

term = config.term

# One or more of these may need to be run once to download stanza's data
# stanza.download('en')
# stanza.download('en', package='craft')
# stanza.download('en', package='mimic')
# stanza.download('en', package='i2b2')

def read_data_file(filename):
    """Reads all data from filename and returns it."""
    f = open(filename, 'r')
    text = f.read()
    f.close()
    text = text.strip()
    return text

def read_sql_file(filename):
    """Reads sqlite file. """
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
    # snlp = stanza.Pipeline(lang='en')#, processors='tokenize, pos')
    # snlp = stanza.Pipeline(lang='en', package='craft', processors={'ner': 'i2b2'})
    snlp = stanza.Pipeline(lang='en', package='mimic', processors={'ner': 'i2b2'})
    nlp = StanzaLanguage(snlp)
    
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": query_word}]
    matcher.add("queryword", None, pattern)
    
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
        # for token in doc:
        #     print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
        # print(doc.ents)

        # print("Encounter " + str(encounter_num))
        
        note_freq = {}

        # some notes may have no mention of the word,
        # e.g., Encounter 66396416        
        found_query_word = False
        for token in doc:
            if token.text.lower() == query_word:
                found_query_word = True
            elif not token.is_punct and not token.is_space:
                # calc word freqs
                note_freq[token.text] = note_freq.get(token.text, 0) + 1

        # show freqs for this note
        parse_freqs(note_freq, 10)
        # combine note_freq with all_freq
        for k, v in note_freq.items():
            all_freq[k] = all_freq.get(k, 0) + v
        
        if not found_query_word:
            print(" No mention of " + query_word + " found.")
            continue

        print()

        # rule-based matcher (in place of concordance?)
        matches = matcher(doc)
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            #print(match_id, string_id, start, end, span.text)
            tokens = get_surrounding_n(doc, start, 5)
            for token in tokens:
                print(token.text + ' [' +  token.ent_type_ + ']', end=' ')
        
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

def get_surrounding_n(doc, pos, n):
    """Returns surrounding n tokens"""
    # TODO: what about punctuation?
    tokens = []
    for i in range(-n, 0):
        tokens.append(get_token(doc, pos + i))
    tokens.append(get_token(doc, pos))
    for i in range(1, n+1):
        tokens.append(get_token(doc, pos + i))
    return tokens
    
def get_token(doc, pos):
    """Returns token at pos, or empty string if out of boundaries"""
    if pos < 0 or pos >= len(doc):
        return ""
    return doc[pos]
    
def parse_freqs(freqs, base_freq=2):
    """ Display frequncies of words. """
    print()
    for freq in sorted(freqs, key=freqs.get, reverse=True):
        if freqs[freq] > base_freq:
            print(freq, freqs[freq])
    print()

def run(path):
    """Runs the Spacy-based analyzer on path."""
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
