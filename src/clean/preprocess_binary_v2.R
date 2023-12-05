# CLEAN AND PREPROCESS THE EXTRACTED TAGTOG DATA --------------------------

# Author: Cory J. Cascalheira
# Date created: 05/08/2023

# This script cleans the Version 1 data and prepares it for feature
# engineering and modeling. It focuses on the data where post-level coding
# was performed and then transformed into a binary outcome. 

# The Version 2 datasets are used for features that require more cleaning.

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(textstem)
library(tidyverse)
library(textclean)
library(tidytext)

# Import data
missom_coded_1 <- read_csv("data/cleaned/version_1/missom_coded_v1.csv")
missom_not_coded_1 <- read_csv("data/cleaned/version_1/missom_not_coded_v1.csv")

# Import stop words
nltk_stopwords <- read_csv("data/util/NLTK_stopwords.csv")

# Add NLTK to tidytext stop words
stop_words <- stop_words %>%
  select(word) %>%
  bind_rows(nltk_stopwords) %>%
  distinct(word) %>%
  # Add words unique to this project
  bind_rows(data.frame(word = c("quot", "amp")))

# CLEAN THE DATA ----------------------------------------------------------

# ...1) MISSOM CODED ------------------------------------------------------

# Preprocess by removing contractions
missom_coded <- missom_coded_1 %>%
  select(post_id, text) %>%
  # Convert to lowercase
  mutate(text = tolower(text)) %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
  # Replace contractions
  mutate(text = replace_contraction(word)) %>% 
  select(-word) %>%
  rename(word = text)
missom_coded

# Preprocess by removing stop words and lemmatizing
missom_coded <- missom_coded %>%
  # Concatenate the words into a single string
  group_by(post_id) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%
  # Keep one instance of the text
  distinct(post_id, .keep_all = TRUE) %>%
  select(-word) %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
  # Replace unique punctuation
  mutate(word = str_replace_all(word, regex("’"), regex("'"))) %>%
  # Remove the stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words (no stemming)
  # https://cran.r-project.org/web/packages/textstem/readme/README.html
  mutate(word = lemmatize_words(word)) %>%
  # Concatenate the words into a single string
  group_by(post_id) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%
  # Keep one instance of the text
  distinct(post_id, .keep_all = TRUE) %>%
  select(-word)
missom_coded

# Combine with original df
missom_coded_1 <- missom_coded_1 %>%
  select(-text) %>%
  left_join(missom_coded) %>%
  select(tagtog_file_id, post_id, how_annotated, subreddit, text, everything())

# ...2) MISSOM NOT CODED --------------------------------------------------

# Preprocess by removing contractions
missom_not_coded <- missom_not_coded_1 %>%
  select(post_id, text) %>%
  # Convert to lowercase
  mutate(text = tolower(text)) %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
  # Replace contractions
  mutate(text = replace_contraction(word)) %>% 
  select(-word) %>%
  rename(word = text)
missom_not_coded

# Preprocess by removing stop words and lemmatizing
missom_not_coded <- missom_not_coded %>%
  # Concatenate the words into a single string
  group_by(post_id) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%
  # Keep one instance of the text
  distinct(post_id, .keep_all = TRUE) %>%
  select(-word) %>%
  # Unnest the tokens
  unnest_tokens(output = "word", input = "text") %>%
  # Replace unique punctuation
  mutate(word = str_replace_all(word, regex("’"), regex("'"))) %>%
  # Remove the stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words (no stemming)
  # https://cran.r-project.org/web/packages/textstem/readme/README.html
  mutate(word = lemmatize_words(word)) %>%
  # Concatenate the words into a single string
  group_by(post_id) %>%
  mutate(text = paste(word, collapse = " ")) %>%
  ungroup() %>%
  # Keep one instance of the text
  distinct(post_id, .keep_all = TRUE) %>%
  select(-word)
missom_not_coded

# Combine with original df
missom_not_coded_1 <- missom_not_coded_1 %>%
  select(-text) %>%
  left_join(missom_not_coded) %>%
  select(tagtog_file_id, post_id, how_annotated, subreddit, text, everything())

# EXPORT FILES ------------------------------------------------------------

write_csv(missom_coded_1, "data/cleaned/version_2/missom_coded_v2.csv")
write_csv(missom_not_coded_1, "data/cleaned/version_2/missom_not_coded_v2.csv")
