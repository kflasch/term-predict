#!/usr/bin/env python3
""" Note text analysis for term prediction tool """

import sys
import os
import csv
import re
from collections import Counter
import argparse
import sqlite3
import string
import nltk
import pandas as pd
from wordcloud import WordCloud
from wordcloud import STOPWORDS as wcstopwords
import config

term = config.term
num_sents_each_side = 2
chunk_sent_char_count = []
total_sent_char_count = []
chunk_word_freq = {}
total_word_freq = {}
stop_words = set(nltk.corpus.stopwords.words('english'))

def parse_csv_file(csv_filename, dataset):
    """ Parses csv file based on dataset """
    num_total_notes = 0
    num_total_sents = 0
    note_ids = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if dataset == 'train':
            # training model data
            for row in reader:
                # patient_num,encounter_num,note_id_encr,type_ip_c,type_ip_name,deid_note_text
                note_ids.append(row['note_id_encr'])
                note_text = row['deid_note_text']
                matches = re.findall(term, note_text, re.IGNORECASE)
                if len(matches) == 0:
                    # this should not happen in the training set -- ignore
                    print("Term '" + term + "' + not found in note id " + row['note_id_encr'] +
                          " for file " + csv_filename)
                    continue
                elif len(matches) > 1:
                    # one note has term twice in the same 'sentence'
                    # do nothing different
                    print("Mulitple occurences of term '" + term + "' in " + row['note_id_encr'])
                analyze_surrounding_sents(note_text, num_sents_each_side)
                update_word_freq(total_word_freq, note_text)

                sentences = nltk.sent_tokenize(note_text)
                for sent in sentences:
                    total_sent_char_count.append(len(sent))
                num_total_sents += len(sentences)
                num_total_notes += 1
        elif dataset == 'test':
            # test data -- coded dataset -- term will not occur in this text
            for row in reader:
                # name,note_id,note_csn_id,deid_note_text
                note_ids.append(row['note_id'])
                note_text = row['deid_note_text']

                update_word_freq(total_word_freq, note_text)

                sentences = nltk.sent_tokenize(note_text)
                for sent in sentences:
                    total_sent_char_count.append(len(sent))
                num_total_sents += len(sentences)
                num_total_notes += 1

    # see if data has any duplicate notes based on id
    check_duplicate_ids(note_ids)

    # show character count info
    if chunk_sent_char_count:
        chunk_sent_char_count.sort()
        char_avg = round(sum(chunk_sent_char_count)/len(chunk_sent_char_count), 2)
        print("\nSentences char count in surrounding chunks min: {0} max: {1} avg: {2}"
              .format(chunk_sent_char_count[0], chunk_sent_char_count[-1], char_avg))

    if total_sent_char_count:
        total_sent_char_count.sort()
        char_avg = round(sum(total_sent_char_count)/len(total_sent_char_count), 2)
        print("\nSentences char count in all notes min: {0} max: {1} avg: {2}"
              .format(total_sent_char_count[0], total_sent_char_count[-1], char_avg))

    print()
    print("Number of unique words (filtered): " + str(len(total_word_freq)))
    print()
    print("Avg. number of sentences per note: " + str(round(num_total_sents/num_total_notes)))

    # show word frequencies around chunks
    if dataset == 'train':
        print()
        print("Most frequent words in surrounding chunks:")
        print_freqs(chunk_word_freq, 8)
    print()
    print("Most frequent words in all notes:")
    print_freqs(total_word_freq, 100)
    print()

def check_duplicate_ids(note_ids):
    """ check for any duplicate note ids """
    note_ids.sort()
    duplicates = [k for k,v in Counter(note_ids).items() if v>1]
    if len(duplicates) > 0:
        print("\nDuplicate note ids in data: " + duplicates)
    else:
        print("\nNo duplicate note ids in data found.")

# def analyze_all_sents(text):
#     """ values for analyzing entire note text """
#     sentences = nltk.sent_tokenize(text)
#     for sent in sentences:
#         total_sent_char_count.append(len(sent))
#     num_total_sents += len(sentences)

def analyze_surrounding_sents(note_text, num_sents):
    """ Analyze surrouding sentences and sentences itself that contains term and update vars """
    # separate note into sentences with nltk
    sentences = nltk.sent_tokenize(note_text)
    for si in range(len(sentences)):
        if re.search(term, sentences[si], re.IGNORECASE):
            for i in range(-num_sents, num_sents+1):
                sent_index = si + i
                # if sent_index < 0:
                #     print('START OF NOTE')
                # elif sent_index >= len(sentences):
                #     print('END OF NOTE')
                # else:
                #     print(sentences[sent_index])
                if sent_index >= 0 and sent_index < len(sentences):
                    sent = sentences[sent_index]
                    chunk_sent_char_count.append(len(sent))
                    update_word_freq(chunk_word_freq, sent)


def update_word_freq(freqs, sent):
    """ Update the word frequency dict with the words in sent """
    words = nltk.word_tokenize(sent)
    words = [word.lower() for word in words]
    for word in words:
        # filter out stop words, punctuation, and numbers
        if word not in stop_words and word not in string.punctuation and not word.isnumeric():
            freqs[word] = freqs.get(word, 0) + 1

def print_freqs(freqs, limit):
    """ Display frequncies of words. """
    for freq in sorted(freqs, key=freqs.get, reverse=True):
        if freqs[freq] > limit:
            print(freq, freqs[freq])
    print()

# def get_tagged_text(text):
#     word_tokens = nltk.word_tokenize(text)
#     return nltk.pos_tag(word_tokens)

# def get_tagged_sents(text):
#     """Tokenize and tag text as sentences"""
#     sentences = nltk.sent_tokenize(text)
#     sentences = [nltk.word_tokenize(sent) for sent in sentences]
#     sentences = [nltk.pos_tag(sent) for sent in sentences]
#     return sentences

def get_chunk_text(note_text, num_sents):
    """ Return chunk of text around term """
    # separate note into sentences with nltk
    sentences = nltk.sent_tokenize(note_text)
    chunk_text = ''
    for si in range(len(sentences)):
        if re.search(term, sentences[si], re.IGNORECASE):
            for i in range(-num_sents, num_sents+1):
                sent_index = si + i
                if sent_index >= 0 and sent_index < len(sentences):
                    sent = sentences[sent_index]
                    chunk_text = chunk_text + ' ' + sent
    return chunk_text

def get_test_rating_text():
    """ Get all text from chunks with ratings > 0 """
    chunk_text = ""
    combined_avg = get_ratings_averages()
    df = pd.read_pickle(config.test_pd_file_5)
    dfnotes = df['notes']
    for i, val in enumerate(dfnotes, start=1):
        if combined_avg[i] > 0:
            chunk_text = chunk_text + ' ' + val
    return chunk_text

def gen_wordcloud(train_file, test_file):
    """ Creates wordcloud image from training data """

    stopwords = ["XXXXX", "patient"] + list(wcstopwords)
    args = dict(background_color='white', colormap='inferno',
                max_words=1000, stopwords=stopwords,
                collocations=False, include_numbers=False)

    os.makedirs("img", exist_ok=True)

    # training dataset wordclouds
    with open(train_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        all_text = ''
        chunk_text = ''
        for row in reader:
            note_text = row['deid_note_text']
            matches = re.findall(term, note_text, re.IGNORECASE)
            if len(matches) == 0:
                # don't use note text if term missing
                pass
            all_text = all_text + ' ' + note_text
            chunk_text = chunk_text + ' ' + get_chunk_text(note_text, 2)


        fname = "img/wordcloud_train_all.png"
        print("Creating " + fname)
        wordcloud = WordCloud(**args).generate(all_text)
        wordcloud.to_file(fname)

        fname = "img/wordcloud_train_chunk_5.png"
        print("Creating " + fname)
        wordcloud_chunk = WordCloud(**args).generate(chunk_text)
        wordcloud_chunk.to_file(fname)

    # test dataset wordclouds
    with open(test_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        all_text = ''
        for row in reader:
            note_text = row['deid_note_text']
            all_text = all_text + ' ' + note_text

        fname = "img/wordcloud_test_all.png"
        print("Creating " + fname)
        wordcloud = WordCloud(**args).generate(all_text)
        wordcloud.to_file(fname)

        # TODO: better check
        if os.path.isfile(config.ratings_first):
            fname = "img/wordcloud_test_chunk_5_ratings.png"
            print("Creating " + fname)
            wordcloud_chunk = WordCloud(**args).generate(get_test_rating_text())
            wordcloud_chunk.to_file(fname)

def get_ratings_averages():
    """ Combine ratings into averages """
    combined_avg = {}
    r1 = get_ratings(config.ratings_first)
    r2 = get_ratings(config.ratings_second)
    r3 = get_ratings(config.ratings_third)

    # all ratings should have same length, so this is safe
    for i in r1:
        total = r1[i] + r2[i] + r3[i]
        avg = total / 3.0
        combined_avg[i] = round(avg, 1)
    return combined_avg

def print_all_ratings_text():
    """ print out all ratings and the text """
    r1 = get_ratings(config.ratings_first)
    r2 = get_ratings(config.ratings_second)
    r3 = get_ratings(config.ratings_third)
    df = pd.read_pickle(config.test_pd_file_5)
    dfnotes = df['notes']
    for i, val in enumerate(dfnotes, start=1):
        print(r1[i], r2[i], r3[i], val)

def show_ratings_info():
    """ Show ratings and dict of avg rating distribution """
    print_all_ratings_text()
    combined_avg = get_ratings_averages()
    dist = dict(Counter(combined_avg.values()))
    print(dist)

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

def run(path, dataset):
    """ Runs the NLTK-based analyzer on path/dataset. """
    if os.path.isdir(path):
        print("Running on directory not implemented.")
        return
    if not os.path.isfile(path) or not path.endswith('.csv'):
        print(path + " not valid .csv")
        return

    parse_csv_file(path, dataset)

# def main(datafile, dataset, on_test, on_train):
def main(args):
    if args.datafile and args.dataset:
        run(args.datafile, args.dataset)
    elif args.datafile or args.dataset:
        print("Please specify both datafile and dataset name or neither.")
        return
    elif args.test:
        run(config.test_data, 'test')
    elif args.train:
        run(config.train_data, 'train')
    elif args.wordcloud:
        gen_wordcloud(config.train_data, config.test_data)
    elif args.ratings:
        show_ratings_info()
    else:
        print("Unknown option.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze data notes used for term prediction.")
    parser.add_argument("-t", "--train", help="analyze training notes", action="store_true")
    parser.add_argument("-s", "--test", help="analyze test notes", action="store_true")
    parser.add_argument("-w", "--wordcloud", help="generate wordcloud", action="store_true")
    parser.add_argument("-r", "--ratings", help="show ratings info", action="store_true")
    parser.add_argument("datafile", nargs="?", default=None, help="filename for dataset")
    parser.add_argument("dataset", nargs="?", default=None, help="dataset name (typically 'train' or 'test')")
    args = parser.parse_args()
    main(args)
