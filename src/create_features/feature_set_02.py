"""
Author = Cory J. Cascalheira
Date = 06/17/2023

The purpose of this script is to create features for the LGBTQ MiSSoM dataset.

The core code is heavily inspired by the following resources:
- https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
- https://radimrehurek.com/gensim/

Issues with importing pyLDAvis.gensim, solved with: https://github.com/bmabey/pyLDAvis/issues/131

Resources for working with spaCy
- https://spacy.io/models
- https://stackoverflow.com/questions/51881089/optimized-lemmitization-method-in-python

# Regular expressions in Python
- https://docs.python.org/3/howto/regex.html
"""

#region LOAD AND IMPORT

# Load core dependencies
import os
import pandas as pd
import numpy as np
import time

# Load plotting tools
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import seaborn as sns

# Import tool for regular expressions
import re

# Import NLTK
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('amp')

# Load Gensim libraries
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.downloader as api

# Initialize spaCy language model
# Must download the spaCy model first in terminal with command: python -m spacy download en_core_web_sm
# May need to restart IDE before loading the spaCy pipeline
import importlib_metadata
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Load GSDMM - topic modeling for short texts (i.e., social media)
from gsdmm import MovieGroupProcess

# Set file path
my_path = os.getcwd()

# Import data
missom_coded = pd.read_csv(my_path + '/data/cleaned/features/missom_coded_feat01.csv')
missom_not_coded = pd.read_csv(my_path + '/data/cleaned/features/missom_not_coded_feat01.csv')

#endregion

#region WORD2VEC MODEL ------------------------------------------------------------------

# MISSOM CODED DATASET ------------------------------------------------------------------

# Create empty list
corpus_coded = []

# Set the stop words from NLTK
stop_words = set(stopwords.words('english'))

# Create a custom tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# Create corpus
for string in missom_coded['text'].astype(str).tolist():

    # Remove strange characters
    string = string.replace('\r', '')
    string = string.replace('*', '')

    # Get tokens (i.e., individual words)
    tokens = tokenizer.tokenize(string)

    # Set a list holder
    filtered_sentence = []

    # For each token, remove the stop words
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Save list of tokens (i.e., sentences) to preprocessed corpus
    corpus_coded.append(filtered_sentence)

# Load the Word2vec model
wv = api.load('word2vec-google-news-300')

# List embeddings for each post
post_embeddings = []

# For every word in every sentence within the corpus
for sentence in corpus_coded:

    # List of word embeddings
    w2v_embeddings = []

    # Get the word embeddings for each word
    for word in sentence:

        # See if there is a pretrained word embedding
        try:
            vector_representation = wv[word]
            w2v_embeddings.append(vector_representation)

        # If there is no pretrained word embedding
        except KeyError:
            vector_representation = np.repeat(0, 300)
            w2v_embeddings.append(vector_representation)

    # Save the word embeddings at the post level
    post_embeddings.append(w2v_embeddings)

# Set a holder variable
avg_post_embeddings = []

# Aggregate word embeddings
for post in post_embeddings:

    # Transform embedding into data frame where each row is a word and each column is the embedding dimension
    df = pd.DataFrame(post)

    # Square each element in the data frame to remove negatives
    df = df.apply(np.square)

    # Get the mean of each embedding dimension
    df = df.apply(np.mean, axis=0)

    # The average word embedding for the entire Reddit post
    avg_embedding = df.tolist()

    # Append to list
    avg_post_embeddings.append(avg_embedding)

# Create a dataframe with the average word embeddings of each post
embedding_df = pd.DataFrame(avg_post_embeddings)

# Rename the columns
embedding_df = embedding_df.add_prefix('w2v_')

# Add average word embeddings to the MiSSoM coded data set
missom_coded1 = pd.concat([missom_coded, embedding_df], axis=1)

# MISSOM NOT CODED DATASET --------------------------------------------------------

# Create empty list
corpus_not_coded = []

# Set the stop words from NLTK
stop_words = set(stopwords.words('english'))

# Create a custom tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# Create corpus
for string in missom_not_coded['text'].astype(str).tolist():

    # Remove strange characters
    string = string.replace('\r', '')
    string = string.replace('*', '')

    # Get tokens (i.e., individual words)
    tokens = tokenizer.tokenize(string)

    # Set a list holder
    filtered_sentence = []

    # For each token, remove the stop words
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Save list of tokens (i.e., sentences) to preprocessed corpus
    corpus_not_coded.append(filtered_sentence)

# Load the Word2vec model
wv = api.load('word2vec-google-news-300')

# List embeddings for each post
post_embeddings = []

# For every word in every sentence within the corpus
for sentence in corpus_not_coded:

    # List of word embeddings
    w2v_embeddings = []

    # Get the word embeddings for each word
    for word in sentence:

        # See if there is a pretrained word embedding
        try:
            vector_representation = wv[word]
            w2v_embeddings.append(vector_representation)

        # If there is no pretrained word embedding
        except KeyError:
            vector_representation = np.repeat(0, 300)
            w2v_embeddings.append(vector_representation)

    # Save the word embeddings at the post level
    post_embeddings.append(w2v_embeddings)

# Set a holder variable
avg_post_embeddings = []

# Aggregate word embeddings
for post in post_embeddings:

    # Transform embedding into data frame where each row is a word and each column is the embedding dimension
    df = pd.DataFrame(post)

    # Square each element in the data frame to remove negatives
    df = df.apply(np.square)

    # Get the mean of each embedding dimension
    df = df.apply(np.mean, axis=0)

    # The average word embedding for the entire Reddit post
    avg_embedding = df.tolist()

    # Append to list
    avg_post_embeddings.append(avg_embedding)

# Create a dataframe with the average word embeddings of each post
embedding_df = pd.DataFrame(avg_post_embeddings)

# Rename the columns
embedding_df = embedding_df.add_prefix('w2v_')

# Add average word embeddings to the MiSSoM not coded data set
missom_not_coded1 = pd.concat([missom_not_coded, embedding_df], axis=1)

# Export files
missom_coded1.to_csv(my_path + '/data/cleaned/features/missom_coded_feat02a.csv')
missom_not_coded1.to_csv(my_path + '/data/cleaned/features/missom_not_coded_feat02a.csv')

#endregion

#region TOPIC MODELING ----------------------------------------------------------

# HELPER FUNCTIONS --------------------------------------------------------------

def transform_to_words(sentences):

    """
    A function that uses Gensim's simple_preprocess(), transforming sentences into tokens of word unit size = 1 and removing
    punctuation in a for loop.

    Parameters
    -----------
    sentences: a list
        A list of text strings to preprocess
    """

    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(word_list):

    """
    A function to remove stop words with the NLTK stopword data set. Relies on NLTK.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in word_list]


def make_bigrams(word_list):
    """
    A function to transform a list of words into bigrams if bigrams are detected by gensim. Relies on a bigram model
    created separately (see below). Relies on Gensim.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [bigram_mod[doc] for doc in word_list]


def make_trigrams(word_list):
    """
    A function to transform a list of words into trigrams if trigrams are detected by gensim. Relies on a trigram model
    created separately (see below). Relies on Gensim.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    """
    return [trigram_mod[bigram_mod[doc]] for doc in word_list]


def lemmatization(word_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN']):
    """
    A function to lemmatize words in a list. Relies on spaCy functionality.

    Parameters
    ----------
    word_list: a list
        A list of words that represent tokens from a list of sentences.
    allowed_postags: a list
        A list of language units to process.
    """
    # Initialize an empty list
    texts_out = []

    # For everyone word in the word list
    for word in word_list:

        # Process with spaCy to lemmarize
        doc = nlp(" ".join(word))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    # Returns a list of lemmas
    return texts_out


def get_optimal_lda(dictionary, corpus, limit=30, start=2, step=2):
    """
    Execute multiple LDA topic models and computer the perplexity and coherence scores to choose the LDA model with
    the optimal number of topics. Relies on Gensim.

    Parameters
    ----------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    limit: an integer
        max num of topics
    start: an integer
        number of topics with which to start
    step: an integer
        number of topics by which to increase during each model training iteration

    Returns
    -------
    model_list: a list of LDA topic models
    coherence_values: a list
        coherence values corresponding to the LDA model with respective number of topics
    perplexity_values: a list
        perplexity values corresponding to the LDA model with respective number of topics
    """
    # Initialize empty lists
    model_list = []
    coherence_values = []
    perplexity_values = []

    # For each number of topics
    for num_topics in range(start, limit, step):

        # Train an LDA model with Gensim
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                                                update_every=1, chunksize=2000, passes=10, alpha='auto',
                                                per_word_topics=True)

        # Add the trained LDA model to the list
        model_list.append(model)

        # Compute UMass coherence score and add to list  - lower is better
        # https://radimrehurek.com/gensim/models/coherencemodel.html
        # https://www.os3.nl/_media/2017-2018/courses/rp2/p76_report.pdf
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        coherence_values.append(coherence)

        # Compute Perplexity and add to list - lower is better
        perplex = model.log_perplexity(corpus)
        perplexity_values.append(perplex)

    return model_list, coherence_values, perplexity_values


def top_words(cluster_word_distribution, top_cluster, values):
    """
    Print the top words associated with the GSDMM topic modeling algorithm.

    Parameters
    ----------
    cluster_word_distribution: a GSDMM word distribution
    top_cluster: a list of indices
    values: an integer
    """

    # For each cluster
    for cluster in top_cluster:

        # Sort the words associated with each topic
        sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]

        # Print the results to the screen
        print('Cluster %s : %s' % (cluster, sort_dicts))
        print('-' * 120)

# PREPROCESS THE TEXT --------------------------------------------------------------------

# Select the columns
missom_coded2 = missom_coded[['tagtog_file_id', 'post_id', 'how_annotated', 'text']]
missom_not_coded2 = missom_not_coded[['tagtog_file_id', 'post_id', 'how_annotated', 'text']]

# Combine the two data frames
missom_full = pd.concat([missom_coded2, missom_not_coded2])

# Convert text to list
missom_text_original = missom_full['text'].astype(str).values.tolist()

# Remove emails, new line characters, and single quotes
missom_text = [re.sub('\\S*@\\S*\\s?', '', sent) for sent in missom_text_original]
missom_text = [re.sub('\\s+', ' ', sent) for sent in missom_text]
missom_text = [re.sub("\'", "", sent) for sent in missom_text]

# Remove markdown links with multiple words
missom_text = [re.sub("\\[[\\S\\s]+\\]\\(https:\\/\\/[\\D]+\\)", "", sent) for sent in missom_text]

# Remove markdown links with single words
missom_text = [re.sub("\\[\\w+\\]\\(https:\\/\\/[\\D\\d]+\\)", "", sent) for sent in missom_text]

# Remove urls
missom_text = [re.sub("https:\\/\\/[\\w\\d\\.\\/\\-\\=]+", "", sent) for sent in missom_text]

# Transform sentences into words, convert to list
missom_words = list(transform_to_words(missom_text))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(missom_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[missom_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove stop words
missom_words_nostops = remove_stopwords(missom_words)

# Form bigrams
missom_words_bigrams = make_bigrams(missom_words_nostops)

# Lemmatize the words, keeping nouns, adjectives, verbs, adverbs, and proper nouns
missom_words_lemma = lemmatization(missom_words_bigrams)

# Remove any stop words created in lemmatization
missom_words_cleaned = remove_stopwords(missom_words_lemma)

# CREATE DICTIONARY AND CORPUS ------------------------------------------------------------------

# Create Dictionary
id2word = corpora.Dictionary(missom_words_cleaned)

# Create Corpus
texts = missom_words_cleaned

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# EXECUTE THE TOPIC MODELS WITH VANILLA LDA ----------------------------------------------------

# Get the LDA topic model with the optimal number of topics
start_time = time.time()
model_list, coherence_values, perplexity_values = get_optimal_lda(dictionary=id2word, corpus=corpus,
                                                                  limit=50, start=2, step=2)
end_time = time.time()
processing_time = end_time - start_time
print(processing_time / 60)
print((processing_time / 60) / 15)

# Plot the coherence scores
# Set the x-axis valyes
limit = 50
start = 2
step = 2
x = range(start, limit, step)

# Create the plot
plt.figure(figsize=(6, 4), dpi=200)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("UMass Coherence Score")
plt.xticks(np.arange(min(x), max(x)+1, 2.0))
plt.axvline(x=20, color='red')
plt.savefig('results/plots/lda_coherence_plot.png')
plt.show()

# From the plot, the best LDA model is when num_topics == 20
optimal_lda_model = model_list[10]

# Visualize best LDA topic model
# https://stackoverflow.com/questions/41936775/export-pyldavis-graphs-as-standalone-webpage
vis = pyLDAvis.gensim_models.prepare(optimal_lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'results/plots/lda.html')

# Get the Reddit post that best represents each topic
# https://radimrehurek.com/gensim/models/ldamodel.html

# Initialize empty lists
lda_output = []
topic_distributions = []

# For each post, get the LDA estimation output
for i in range(len(missom_text_original)):
    lda_output.append(optimal_lda_model[corpus[i]])

# For each output, select just the topic distribution
for i in range(len(missom_text_original)):
    topic_distributions.append(lda_output[i][0])

# Initialize empty dataframe
# https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/
list_topic_names = list(range(0, 22))
list_topic_names = [str(i) for i in list_topic_names]
list_topic_probs = [0] * 22
topic_dict = dict(zip(list_topic_names, list_topic_probs))
topic_df = pd.DataFrame(topic_dict, index=[0])

# For each post, assign topic probabilities as features
for i in range(len(topic_distributions)):

    # Initialize list of zeros
    post_topic_probs = [0] * len(topic_df.columns)

    # For each tuple holding topic probabilities
    for tup in range(len(topic_distributions[i])):

        # Get the topic in the tuple
        tup_topic = topic_distributions[i][tup][0]

        # Get the topic probability in the tuple
        tup_prob = topic_distributions[i][tup][1]

        # Change the list element for the post
        post_topic_probs[tup_topic] = tup_prob

    # Add the list as a new row in the dataframe
    topic_df.loc[len(topic_df)] = post_topic_probs
    print('Percent done: ', str(round(i / len(topic_distributions) * 100, 4)), '%')

# Extract top words
# https://stackoverflow.com/questions/46536132/how-to-access-topic-words-only-in-gensim
lda_top_words = optimal_lda_model.show_topics(num_topics=22, num_words=3)
lda_tup_words = [lda_tup_words[1] for lda_tup_words in lda_top_words]

# Initialize empty list
lad_topic_names = []

# For each topic
for topic in range(len(lda_tup_words)):

    # Extract the top 3 words
    my_words = re.findall("\\w+", lda_tup_words[topic])
    my_elements = [2, 5, 8]

    # Concatenate the top 3 words together and save to list
    my_name = ''.join([my_words[i] for i in my_elements])
    my_name1 = 'lda_' + my_name
    lad_topic_names.append(my_name1)

# Rename the LDA features
# https://sparkbyexamples.com/pandas/rename-columns-with-list-in-pandas-dataframe/?expand_article=1
topic_df.set_axis(lad_topic_names, axis=1, inplace=True)

# Join the two data frames by index
missom_full = missom_full.join(topic_df)

# Filter the dataframes
missom_coded2 = missom_full[missom_full['how_annotated'] == 'human']
missom_not_coded2 = missom_full[missom_full['how_annotated'] == 'machine']

# Export
missom_coded2.to_csv(my_path + '/data/cleaned/features/missom_coded_feat02b.csv')
missom_not_coded2.to_csv(my_path + '/data/cleaned/features/missom_not_coded_feat02b.csv')

# EXECUTE THE TOPIC MODELS WITH GSDMM --------------------------------------------------------

# Get the number of words per post
words_per_post = []

for i in range(len(missom_words_cleaned)):
    words_per_post.append(len(missom_words_cleaned[i]))

# Histogram of words per post
plt.hist(x=words_per_post)
plt.show()

# Descriptive statistic of words per post
print(np.mean(words_per_post))
print(np.std(words_per_post))
print(len([num for num in words_per_post if num <= 50]) / len(words_per_post))

# GSDMM ALGORITHM

# Create the vocabulary
vocab = set(x for doc in missom_words_cleaned for x in doc)

# The number of terms in the vocabulary
n_terms = len(vocab)

# Train the GSDMM models, changing the value of beta given its meaning (i.e., how similar topics need to be to cluster
# together). K is 30, the same number of topic to consider as the above vanilla LDA. Alpha remains 0.1, which reduces
# the probability that a post will join an empty cluster

# Train the GSDMM model, beta = 1.0
mgp_10 = MovieGroupProcess(K=30, alpha=0.1, beta=1.0, n_iters=40)
gsdmm_b10 = mgp_10.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_10 = np.array(mgp_10.cluster_doc_count)
print('Beta = 1.0. The number of posts per topic: ', post_count_10)

# Train the GSDMM model, beta = 0.9
mgp_09 = MovieGroupProcess(K=30, alpha=0.1, beta=0.9, n_iters=40)
gsdmm_b09 = mgp_09.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_09 = np.array(mgp_09.cluster_doc_count)
print('Beta = 0.9. The number of posts per topic: ', post_count_09)

# Train the GSDMM model, beta = 0.8
mgp_08 = MovieGroupProcess(K=30, alpha=0.1, beta=0.8, n_iters=40)
gsdmm_b08 = mgp_08.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_08 = np.array(mgp_08.cluster_doc_count)
print('Beta = 0.8. The number of posts per topic: ', post_count_08)

# Train the GSDMM model, beta = 0.7
mgp_07 = MovieGroupProcess(K=30, alpha=0.1, beta=0.7, n_iters=40)
gsdmm_b07 = mgp_07.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_07 = np.array(mgp_07.cluster_doc_count)
print('Beta = 0.7. The number of posts per topic: ', post_count_07)

# Train the GSDMM model, beta = 0.6
mgp_06 = MovieGroupProcess(K=30, alpha=0.1, beta=0.6, n_iters=40)
gsdmm_b06 = mgp_06.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_06 = np.array(mgp_06.cluster_doc_count)
print('Beta = 0.6. The number of posts per topic: ', post_count_06)

# Train the GSDMM model, beta = 0.5
mgp_05 = MovieGroupProcess(K=30, alpha=0.1, beta=0.5, n_iters=40)
gsdmm_b05 = mgp_05.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_05 = np.array(mgp_05.cluster_doc_count)
print('Beta = 0.5. The number of posts per topic: ', post_count_05)

# Train the GSDMM model, beta = 0.4
mgp_04 = MovieGroupProcess(K=30, alpha=0.1, beta=0.4, n_iters=40)
gsdmm_b04 = mgp_04.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_04 = np.array(mgp_04.cluster_doc_count)
print('Beta = 0.4. The number of posts per topic: ', post_count_04)

# Train the GSDMM model, beta = 0.3
start_time = time.time()
mgp_03 = MovieGroupProcess(K=30, alpha=0.1, beta=0.3, n_iters=40)
gsdmm_b03 = mgp_03.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_03 = np.array(mgp_03.cluster_doc_count)
print('Beta = 0.3. The number of posts per topic: ', post_count_03)
end_time = time.time()
processing_time = end_time - start_time
print(processing_time / 60)

# Train the GSDMM model, beta = 0.2
mgp_02 = MovieGroupProcess(K=30, alpha=0.1, beta=0.2, n_iters=40)
gsdmm_b02 = mgp_02.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_02 = np.array(mgp_02.cluster_doc_count)
print('Beta = 0.2. The number of posts per topic: ', post_count_02)

# Train the GSDMM model, beta = 0.1
mgp_01 = MovieGroupProcess(K=30, alpha=0.1, beta=0.1, n_iters=40)
gsdmm_b01 = mgp_01.fit(docs=missom_words_cleaned, vocab_size=n_terms)
post_count_01 = np.array(mgp_01.cluster_doc_count)
print('Beta = 0.1. The number of posts per topic: ', post_count_01)

# Remove topics with 0 posts assigned
beta_01 = [x for x in post_count_01 if x > 0]
beta_02 = [x for x in post_count_02 if x > 0]
beta_03 = [x for x in post_count_03 if x > 0]
beta_04 = [x for x in post_count_04 if x > 0]
beta_05 = [x for x in post_count_05 if x > 0]
beta_06 = [x for x in post_count_06 if x > 0]
beta_07 = [x for x in post_count_07 if x > 0]
beta_08 = [x for x in post_count_08 if x > 0]
beta_09 = [x for x in post_count_09 if x > 0]
beta_10 = [x for x in post_count_10 if x > 0]

# Make lists the same size, transform in array
beta_01 = np.sort(np.array(beta_01))
beta_02 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_02)]), beta_02))
beta_03 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_03)]), beta_03))
beta_04 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_04)]), beta_04))
beta_05 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_05)]), beta_05))
beta_06 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_06)]), beta_06))
beta_07 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_07)]), beta_07))
beta_08 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_08)]), beta_08))
beta_09 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_09)]), beta_09))
beta_10 = np.sort(np.append(np.repeat(0, [len(beta_01)-len(beta_10)]), beta_10))

# Append all topics
n_posts = np.append(beta_01, [beta_02, beta_03, beta_04, beta_05, beta_06, beta_07, beta_08, beta_09, beta_10])

# Create list of topic numbers
topic_numbers = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 10

# Create a list of beta values
beta_list = [[0.1] * 17] + [[0.2] * 17] + [[0.3] * 17] + [[0.4] * 17] + [[0.5] * 17] + [[0.6] * 17] + [[0.7] * 17] + [[0.8] * 17] + [[0.9] * 17] + [[1.0] * 17]
beta_values = [item for sublist in beta_list for item in sublist]

# Double check that the betas are same length as topic numbers
print(len(topic_numbers) == len(n_posts) == len(beta_values))

# Create data frame for plotting
list_of_tuples = list(zip(beta_values, topic_numbers, n_posts))
gsdmm_df = pd.DataFrame(list_of_tuples, columns=['beta', 'topic_numbers', 'n_posts'])

# Make grid plot
sns.set_theme(style="white")
gsdmm_plot = sns.FacetGrid(gsdmm_df, col='beta', col_wrap=2)
gsdmm_plot.map(sns.barplot, 'topic_numbers', 'n_posts', color='cornflowerblue')
gsdmm_plot.set_axis_labels("Topic Numbers", "Number of Posts")
gsdmm_plot.savefig('results/plots/gsdmm_topics.png')

# Optimal number of topics?
print('The optimal number of topics in GSDMM, based on average, is: ', (6 + 4 + 3 + 4 + 2 + 2 + 1 + 2 + 2 + 1) / 10)

# Since optimal number of plots is GSDMM is 2.7, round to 3---use model where beta = 0.3

# Rearrange the topics in order of importance
top_index = post_count_03.argsort()[-17:][::-1]

# Get the top 15 words per topic
top_words(mgp_03.cluster_word_distribution, top_cluster=top_index, values=15)

# Initialize empty list
gsdmm_topics = []

# Predict the topic for each set of words in a Reddit post
for i in range(len(missom_words_cleaned)):
    gsdmm_topics.append(mgp_03.choose_best_label(missom_words_cleaned[i]))

# Initialize empty lists
topic_classes = []
topic_probs = []

# For each post, extract the dominant topic from the topic distribution
for i in range(len(missom_text_original)):

    # Extract the dominant topic
    topic_class = gsdmm_topics[i][0]
    topic_classes.append(topic_class)

    # Extract the probability of the dominant topic
    topic_prob = gsdmm_topics[i][1]
    topic_probs.append(topic_prob)

# Prepare to merge with original dataframe
gsdmm_missom_df = missom_full.loc[:, ['author', 'body', 'permalink']]

# Add the dominant topics and strengths
gsdmm_missom_df['topic'] = topic_classes
gsdmm_missom_df['topic_probability'] = topic_probs

# Sort the data frame
gsdmm_missom_df = gsdmm_missom_df.sort_values(by=['topic', 'topic_probability'], ascending=[False, False])

# Select the 10 most illustrative posts per topic
topics_to_quote = gsdmm_missom_df.groupby('topic').head(10)

# Save the data frame for easy reading
topics_to_quote.to_csv("FINISH THIS")

# Percentage of posts with each top 3 topic
print(post_count_03[29] / len(missom_words_cleaned))
print(post_count_03[13] / len(missom_words_cleaned))
print(post_count_03[5] / len(missom_words_cleaned))

#endregion
