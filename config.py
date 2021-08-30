#
# --- configuration file for term prediction tools ---
#

# the term to be predicted
term = "sarcopenia"

# what to mask the word with in training data, if anything
mask = ""

# how many sentences per chunk
# use odd numbers unless code changed to allow even (term is in the middle chunk)
chunk_sizes = [1, 3, 5, 7, 9]

data_dir = "data/"
train_data = data_dir + term + "_training_data.csv"
test_data = data_dir + term + "_test_data.csv"
# ner_train_data = data_dir + "ner_" + term + "_training_data.csv"
# ner_test_data = data_dir + "ner_" + term + "_test_data.csv"

# list of anatomy terms to match for feature
anatomy_terms_file = data_dir + "mesh_a02_2019.txt"

# ratings files with values filled out
ratings_dir = data_dir + "ratings/"
ratings_first = ratings_dir + term + "_chunk_5_ratings_first_rater.txt"
ratings_second = ratings_dir + term + "_chunk_5_ratings_second_rater.txt"
ratings_third = ratings_dir + term + "_chunk_5_ratings_third_rater.txt"

# Empath categories to be matched against as feature
empath_categories = ["depleting", "muscskel", "gaitmobility", "fracture", "frail"]

# generates NER transformed data when datafiles are created with selected model
ner_transform = False
spacy_model = "en_core_sci_sm"
# spacy_model = "en_core_sci_lg"
# spacy_model = "en_core_sci_scibert"
# spacy_model = "en_ner_bionlp13cg_md"

# oversample the positives in the training set by adding this num of duplicate rows (0 for none)
oversample_amount = 0

# which features to use in cross-validation and prediction in sklearn
use_feature_notetype = False
use_feature_textlen = False
use_feature_empath = False
use_feature_anatomy_terms = False
use_feature_notes = True
ngram_range = (1, 1) # (2, 2) for only bigrams, (1, 2) for unigrams and bigrams
stop_words = None # 'english' for sklearn built-in list
# max_df = 0.7 # maximum document frequency (unused)
use_tfidf = False

# add slight weighting to '5' values in ratings in prediction
weight_ratings = False

# only show a few performance measures in output
sklearn_reduced_output = False
# print out text of predicted samples
sklearn_print_predtext = False
# plot decision tree image to file when decision tree classifier is used
plot_dtree = False
# show cross-validation train scores (may be useful for holdout sets)
show_cv_train_scores = False

# pandas dataframe files (pickle format)
train_pd_file_1 = data_dir + term + "_chunk_1_train.pkl"
train_pd_file_3 = data_dir + term + "_chunk_3_train.pkl"
train_pd_file_5 = data_dir + term + "_chunk_5_train.pkl"
train_pd_file_7 = data_dir + term + "_chunk_7_train.pkl"
train_pd_file_9 = data_dir + term + "_chunk_9_train.pkl"
test_pd_file_5 = data_dir + term + "_chunk_5_test.pkl"


# used in create_datafiles.py to decide which features to include in arff
# mostly used for Weka 
create_arff_feature_notetype = True
create_arff_feature_textlen = True
create_arff_feature_empath = True
create_arff_feature_anatomy_terms = True

# --- Weka specific ---
# train_chunk_file = data_dir + term + "_chunk_train.arff"
# train_sent_file = data_dir + term + "_sent_train.arff"
train_arff_file_1 = data_dir + term + "_chunk_1_train.arff"
train_arff_file_3 = data_dir + term + "_chunk_3_train.arff"
train_arff_file_5 = data_dir + term + "_chunk_5_train.arff"
train_arff_file_7 = data_dir + term + "_chunk_7_train.arff"
train_arff_file_9 = data_dir + term + "_chunk_9_train.arff"
weka_minimal_output = False
weka_s2wv_options = ["-W", "8000", "-L", "-C"] # -I?
weka_stemmer = "weka.core.stemmers.LovinsStemmer"
weka_stopwords = "weka.core.stopwords.Rainbow"
# weka_tokenizer = "weka.core.tokenizers.CharacterNGramTokenizer"
# weka_tokenizer_options = ["-max", "2", "-min", "2"]
