# FEATURES - CLINICAL KEY WORDS -------------------------------------------

# Author = Cory J. Cascalheira
# Date = 05/09/2023

# The purpose of this script is to create features for the LGBTQ+ MiSSoM dataset.

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(textstem)
library(tidyverse)
library(tidytext)
library(pdftools)

# Import data files
missom_coded <- read_csv("data/cleaned/version_2/missom_coded_v2.csv")
missom_not_coded <- read_csv("data/cleaned/version_2/missom_not_coded_v2.csv")

# Import the DSM-5 text
dsm5_anxiety <- read_file("data/util/dsm5_anxiety.txt")
dsm5_depression <- read_file("data/util/dsm5_depression.txt")
dsm5_ptsd <- read_file("data/util/dsm5_ptsd.txt")
dsm5_substance_use <- read_file("data/util/dsm5_substance_use.txt")
dsm5_gender_dysphoria <- read_csv("data/util/dsm5_gender_dysphoria.csv")

# Get AFINN sentiment lexicon
afinn <- get_sentiments("afinn")

# Get slangSD lexicon: https://github.com/airtonbjunior/opinionMining/blob/master/dictionaries/slangSD.txt
slangsd <- read_delim("data/util/slangSD.txt", delim = "\t", col_names = FALSE) %>%
  rename(word = X1, value = X2)

# Combine sentiment libraries
sentiment_df <- bind_rows(afinn, slangsd) %>%
  distinct(word, .keep_all = TRUE)

# Import hate speech lexicons
hate_lexicon_sexual_minority <- read_csv("data/hatespeech/hate_lexicon_sexual_minority.csv") %>%
  select(word) %>%
  mutate(hate_lexicon_sexual_minority = 1)

hate_lexicon_gender_minority <- read_csv("data/hatespeech/hate_lexicon_gender_minority.csv") %>%
  select(word) %>%
  mutate(hate_lexicon_gender_minority = 1)

hate_lexicon_woman_man <- read_csv("data/hatespeech/hatebase_woman_man.csv") %>%
  select(word) %>%
  mutate(hate_lexicon_woman_man = 1)

# Import theoretical lexicon of minority stress text
minority_stress_2003 <- read_file("data/util/minority_stress_text/minority_stress_2003.txt")

minority_stress_ethnicity <- read_file("data/util/minority_stress_text/minority_stress_ethnicity.txt")

minority_stress_1995 <- pdf_text("data/util/minority_stress_text/minority_stress_1995.pdf")

minority_stress_transgender <- pdf_text("data/util/minority_stress_text/minority_stress_transgender.pdf")

# Import pain lexicon
pain_lexicon <- read_csv("data/util/pain_lexicon.csv")

# Import LIWC
missom_coded_liwc <- read_csv("data/cleaned/features/liwc_features/missom_coded_v1_liwc.csv") %>%
  select(-text)

missom_not_coded_liwc <- read_csv("data/cleaned/features/liwc_features/missom_not_coded_v1_liwc.csv") %>%
  select(-text)

# 1) CLINICAL KEYWORDS ----------------------------------------------------

# ...1a) PREPARE THE DSM-5 TEXT -------------------------------------------

# Common DSM-5 words to remove
common_dsm5 <- c("attack", "social", "individual", "situation", "specific", "child", "substance", "occur", "adult", "include", "experience", "criterion", "onset", "generalize", "prevalence", "feature", "rate", "age", "due", "figure", "physical", "risk", "attachment", "month", "home", "event", "factor", "episode", "major", "meet", "persistent", "day", "period", "note", "excessive", "behavior", "adjustment", "response", "code", "mental", "effect", "significant", 'time', "develop", "unknown", "gender", "sex", "male", "female", "adolescent", "desire", "strong", "boy", "characteristic", "girl", "refer", "andor", "lateonset", "sexual", "express", "identity", "increase")

# Transdiagnostic clinical key words
dsm5_transdiagnostic <- data.frame(word = c("disorder", "symptom", "medical", "condition", "diagnosis", "diagnostic", "diagnose", "withdrawal", "impairment", "comorbid", "chronic", "acute", "disturbance", "severe", "treatment")) %>%
  tibble()

# DSM-5 anxiety disorders
dsm5_anxiety_df <- data.frame(dsm5 = dsm5_anxiety) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  filter(!(word %in% c("avoid", "object"))) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("anxiety", "worry", "phobic"))) %>%
  rename(dsm5_anxiety = word) 
dsm5_anxiety_df

# DSM-5 depressive disorders
dsm5_depression_df <- data.frame(dsm5 = dsm5_depression) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  filter(!(word %in% c("depression", "dysphoric", "depress"))) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("anxiety"))) %>%
  rename(dsm5_depression = word)
dsm5_depression_df

# DSM-5 PTSD and stress disorders
dsm5_ptsd_df <- data.frame(dsm5 = dsm5_ptsd) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  rename(dsm5_ptsd = word)
dsm5_ptsd_df

# DSM-5 substance use
dsm5_substance_use_df <- data.frame(dsm5 = dsm5_substance_use) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  head(n = 10) %>%
  select(-n) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("intoxicat"))) %>%
  rename(dsm5_substance_use = word)
dsm5_substance_use_df

# DSM-5 gender dysphoria
dsm5_gender_dysphoria_df <- data.frame(dsm5 = dsm5_gender_dysphoria) %>% 
  as_tibble() %>%
  # Remove punctuation and digits
  mutate(dsm5 = str_remove_all(dsm5, regex("[:punct:]|[:digit:]", ignore_case = TRUE))) %>%
  # Convert to lowercase
  mutate(dsm5 = tolower(dsm5)) %>%
  unnest_tokens(output = "word", input = "dsm5") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>% 
  # Remove common words
  filter(!(word %in% common_dsm5)) %>%
  # Remove DSM-5 transdiagnostic clinical key words
  filter(!(word %in% dsm5_transdiagnostic$word)) %>%
  head(n = 10) %>%
  select(-n) %>%
  mutate(word = stem_words(word)) %>%
  # Add other permutations of the top 10 key words for the NLP search
  bind_rows(data.frame(word = c("surgery", "dysphoric"))) %>%
  rename(dsm5_gender_dysphoria = word)
dsm5_gender_dysphoria_df

# ...1b) GENERATE THE FEATURES --------------------------------------------

# Features for coded dataset
missom_coded <- missom_coded %>%
  mutate(
    dsm5_anxiety = if_else(str_detect(text, regex(paste(dsm5_anxiety_df$dsm5_anxiety, collapse = "|"), 
                                                  ignore_case = TRUE)), 1, 0),
    dsm5_depression = if_else(str_detect(text, regex(paste(dsm5_depression_df$dsm5_depression, collapse = "|"), 
                                                     ignore_case = TRUE)), 1, 0),
    dsm5_ptsd = if_else(str_detect(text, regex(paste(dsm5_ptsd_df$dsm5_ptsd, collapse = "|"),
                                               ignore_case = TRUE)), 1, 0),
    dsm5_substance_use = if_else(str_detect(text, regex(paste(dsm5_substance_use_df$dsm5_substance_use, collapse = "|"), 
                                                        ignore_case = TRUE)), 1, 0),
    dsm5_gender_dysphoria = if_else(str_detect(text, regex(paste(dsm5_gender_dysphoria_df$dsm5_gender_dysphoria, collapse = "|"), 
                                                           ignore_case = TRUE)), 1, 0),
    dsm5_transdiagnostic = if_else(str_detect(text, regex(paste(dsm5_transdiagnostic$word, collapse = "|"), 
                                                          ignore_case = TRUE)), 1, 0)
  )

# Features for uncoded dataset
missom_not_coded <- missom_not_coded %>%
  mutate(
    dsm5_anxiety = if_else(str_detect(text, regex(paste(dsm5_anxiety_df$dsm5_anxiety, collapse = "|"), 
                                                  ignore_case = TRUE)), 1, 0),
    dsm5_depression = if_else(str_detect(text, regex(paste(dsm5_depression_df$dsm5_depression, collapse = "|"), 
                                                     ignore_case = TRUE)), 1, 0),
    dsm5_ptsd = if_else(str_detect(text, regex(paste(dsm5_ptsd_df$dsm5_ptsd, collapse = "|"),
                                               ignore_case = TRUE)), 1, 0),
    dsm5_substance_use = if_else(str_detect(text, regex(paste(dsm5_substance_use_df$dsm5_substance_use, collapse = "|"), 
                                                        ignore_case = TRUE)), 1, 0),
    dsm5_gender_dysphoria = if_else(str_detect(text, regex(paste(dsm5_gender_dysphoria_df$dsm5_gender_dysphoria, collapse = "|"), 
                                                           ignore_case = TRUE)), 1, 0),
    dsm5_transdiagnostic = if_else(str_detect(text, regex(paste(dsm5_transdiagnostic$word, collapse = "|"), 
                                                          ignore_case = TRUE)), 1, 0)
  )

# 2) LEXICONS -------------------------------------------------------------

# ...2a) SENTIMENT LEXICON ------------------------------------------------

# For the coded dataset
missom_coded <- missom_coded %>%
  # Reduce df size
  select(post_id, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Get sentiment of words
  left_join(sentiment_df) %>%
  # Recode missing to 0 sentiment
  mutate(value = if_else(is.na(value), 0, value)) %>%
  # Group by post
  group_by(post_id) %>%
  # Calculate total sentiment of post
  summarize(sentiment_lexicon = sum(value)) %>%
  # Join to main dataframe
  left_join(missom_coded) %>%
  # Rearrange the variables
  select(tagtog_file_id, post_id, how_annotated:label_minority_stress_new, everything())

# For the coded dataset
missom_not_coded <- missom_not_coded %>%
  # Reduce df size
  select(post_id, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Get sentiment of words
  left_join(sentiment_df) %>%
  # Recode missing to 0 sentiment
  mutate(value = if_else(is.na(value), 0, value)) %>%
  # Group by post
  group_by(post_id) %>%
  # Calculate total sentiment of post
  summarize(sentiment_lexicon = sum(value)) %>%
  # Join to main dataframe
  left_join(missom_not_coded) %>%
  # Rearrange the variables
  select(tagtog_file_id, post_id, how_annotated:label_minority_stress_new, everything())

# ...2b) HATE SPEECH LEXICONS ----------------------------------------------

# Human coded dataset
missom_coded <- missom_coded %>%
  # Reduce df size
  select(post_id, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Join the hate speech data frames
  left_join(hate_lexicon_sexual_minority) %>%
  left_join(hate_lexicon_gender_minority) %>%
  left_join(hate_lexicon_woman_man) %>%
  # Recode missing to 0
  mutate(
    hate_lexicon_sexual_minority = if_else(is.na(hate_lexicon_sexual_minority), 0,
                                           hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = if_else(is.na(hate_lexicon_gender_minority), 0,
                                           hate_lexicon_gender_minority),
    hate_lexicon_woman_man = if_else(is.na(hate_lexicon_woman_man), 0,
                                 hate_lexicon_woman_man)
  ) %>%
  # Group by post
  group_by(post_id) %>%
  # Calculate presence of hate speech term
  summarize(
    hate_lexicon_sexual_minority = sum(hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = sum(hate_lexicon_gender_minority),
    hate_lexicon_woman_man = sum(hate_lexicon_woman_man)
  ) %>%
  # Join to main dataframe
  left_join(missom_coded) %>%
  # Rearrange the variables
  select(tagtog_file_id, post_id, how_annotated:label_minority_stress_new, everything())

# Machine coded dataset
missom_not_coded <- missom_not_coded %>%
  # Reduce df size
  select(post_id, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Join the hate speech data frames
  left_join(hate_lexicon_sexual_minority) %>%
  left_join(hate_lexicon_gender_minority) %>%
  left_join(hate_lexicon_woman_man) %>%
  # Recode missing to 0
  mutate(
    hate_lexicon_sexual_minority = if_else(is.na(hate_lexicon_sexual_minority), 0,
                                           hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = if_else(is.na(hate_lexicon_gender_minority), 0,
                                           hate_lexicon_gender_minority),
    hate_lexicon_woman_man = if_else(is.na(hate_lexicon_woman_man), 0,
                                 hate_lexicon_woman_man)
  ) %>%
  # Group by post
  group_by(post_id) %>%
  # Calculate presence of hate speech term
  summarize(
    hate_lexicon_sexual_minority = sum(hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = sum(hate_lexicon_gender_minority),
    hate_lexicon_woman_man = sum(hate_lexicon_woman_man)
  ) %>%
  # Join to main dataframe
  left_join(missom_not_coded) %>%
  # Rearrange the variables
  select(tagtog_file_id, post_id, how_annotated:label_minority_stress_new, everything())

# ...2c) THEORETICAL MINORITY STRESS LEXICON ------------------------------

# ......2c1) PREPARE THE DATA ---------------------------------------------

# Common terms to filter
ms_common_terms <- c("measure", "event", "association", "gay", "journal", "york", "study", "sample", "relate", "table", "aid", "subject", "american", "effect", "social", "google", "scholar", "lgb", "pubmed", "lesbian", "bisexual", "prevalence", "research", "al", "individual", "people", "process", "person", "pp", "editor", "suicide", "lgbt", "lgbtpoc", "participant", "item", "sexual", "experience", "scale", "subscale", "white", "trans", "gender", "transgender")

# Meyer (1995) - early version of minority stress
minority_stress_1995_df <- data.frame(article = minority_stress_1995) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 13) %>%
  select(-n) %>%
  # Add other permutations and combinations of the top key words for the NLP search
  bind_rows(data.frame(word = c("minority stress", "homophobic", "violent", 
                                "mental health")))
minority_stress_1995_df

# Meyer (2003) - most popular version of minority stress
minority_stress_2003_df <- data.frame(article = minority_stress_2003) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 13) %>%
  select(-n) %>%
  # Add other permutations and combinations of the top key words for the NLP search
  bind_rows(data.frame(word = c("mental disorder")))
minority_stress_2003_df

# Balsam et al (2011) - minority stress adapted for ethnic minority LGBTQ+ folx
minority_stress_ethnicity_df <- data.frame(article = minority_stress_ethnicity) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 11) %>%
  select(-n) %>%
  # Add other permutations and combinations of the top key words for the NLP search
  bind_rows(data.frame(word = c("racial", "ethnic")))
minority_stress_ethnicity_df

# Hendricks & Testa (2012) - minority stress adaoted for transgender folx
minority_stress_transgender_df <- data.frame(article = minority_stress_transgender) %>% 
  as_tibble() %>%
  mutate(
    # Remove punctuation and digits
    article = str_remove_all(article, regex("[:punct:]|[:digit:]", ignore_case = TRUE)),
    # Remove white space character
    article = str_replace_all(article, regex("\n"), " "),
    # Remove padding
    article = str_trim(article),
    article = str_squish(article),
    # Covert to lowercase
    article = tolower(article)
  ) %>%
  # Extract single words
  unnest_tokens(output = "word", input = "article") %>%
  # Remove stop words
  filter(!(word %in% stop_words$word)) %>%
  # Lemmatize the words
  mutate(word = lemmatize_words(word)) %>%
  # Top words in the DSM-5
  count(word) %>%
  arrange(desc(n)) %>%
  # Filter out common words
  filter(!(word %in% ms_common_terms)) %>%
  # Get the top unique words related to minority stress
  head(n = 4) %>%
  select(-n)
minority_stress_transgender_df

# Bind all dfs
minority_stress_df <- bind_rows(minority_stress_1995_df, minority_stress_2003_df) %>%
  bind_rows(minority_stress_ethnicity_df) %>%
  bind_rows(minority_stress_transgender_df) %>%
  # Remove repeats
  distinct(word)

# ......2c2) GENERATE THE FEATURES ----------------------------------------

# For coded data
missom_coded <- missom_coded %>%
  mutate(
    theoretical_ms_lexicon = if_else(str_detect(text, regex(paste(minority_stress_df$word, collapse = "|"), ignore_case = TRUE)), 1, 0)
  )

# For not coded data
missom_not_coded <- missom_not_coded %>%
  mutate(
    theoretical_ms_lexicon = if_else(str_detect(text, regex(paste(minority_stress_df$word, collapse = "|"), ignore_case = TRUE)), 1, 0)
  )

# ...2d) PAIN LEXICON -----------------------------------------------------

# Add pain terms to coded data
missom_coded <- missom_coded %>%
  mutate(
    pain_lexicon = if_else(str_detect(text, regex(paste(pain_lexicon$keywords, collapse = "|"),
                                                  ignore_case = TRUE)), 1, 0)
  )

# Add pain terms to not coded data
missom_not_coded <- missom_not_coded %>%
  mutate(
    pain_lexicon = if_else(str_detect(text, regex(paste(pain_lexicon$keywords, collapse = "|"),
                                                  ignore_case = TRUE)), 1, 0)
  )

# 4) OPEN VOCABULARY ------------------------------------------------------

# ...4a) N-GRAMS ----------------------------------------------------------

# Top unigrams
unigram_df <- missom_coded %>%
  # Select key columns
  select(post_id, text, label_minority_stress) %>%
  # Generate unigrams
  unnest_tokens(word, text, drop = FALSE) %>%
  # Remove stop words
  count(label_minority_stress, word) %>%
  arrange(desc(n)) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word = if_else(str_detect(word, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove remaining stop words
  filter(stop_word == 0) %>%
  select(-stop_word)

# TF-IDF unigrams
unigram_vector <- unigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(word, label_minority_stress, n) %>%
  # Get top tf-idf of unigrams for minority stress posts
  arrange(desc(tf_idf)) %>%
  filter(label_minority_stress == 1) %>%
  # Remove words based on closed inspection of unigrams
  mutate(remove = if_else(str_detect(word, regex("’s|’d|'s|	
’ve|\\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lockdown|covid|grammatical|film|eh|could’ve|december|vehicle|paint|ness|bout|brown|animals|âˆ|weather|bike|maria|albeit|amd|matt|minecraft|freind|have|𝙸|𝚠𝚒𝚕𝚕|𝚢𝚘𝚞|ᴍʏ|can‘t|causally")), 1, 0)) %>%
  filter(remove == 0) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(word)

# Generate bigrams
bigram_df <- missom_coded %>%
  # Select key columns
  select(post_id, text, label_minority_stress) %>%
  unnest_ngrams(bigram, text, n = 2, drop = FALSE) %>%
  # Separate the bigrams into two columns
  separate(bigram, c("word1", "word2")) %>%
  # Remove stop words
  filter(!(word1 %in% stop_words$word)) %>%
  filter(!(word2 %in% stop_words$word)) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all|amp")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0)
  ) %>%
  filter(stop_word1 == 0, stop_word2 == 0) %>%
  unite("bigram", c("word1", "word2"), sep = " ") %>%
  # Count top bigrams
  count(label_minority_stress, bigram) %>%
  arrange(desc(n))

# TF-IDF bigrams
bigram_vector <- bigram_df %>%
  # Calculate tf-idf
  bind_tf_idf(bigram, label_minority_stress, n) %>%
  # Get top tf-idf of unigrams for minority stress posts
  arrange(desc(tf_idf)) %>%
  filter(label_minority_stress == 1) %>%
  # Remove words based on closed inspection of unigrams
  mutate(remove = if_else(str_detect(bigram, regex("’s|’d|'s|	
’ve|\\d|monday|tuesday|wednesday|thursday|friday|saturday|sunday|lockdown|covid|^ive |^lot |minutes ago|ame$")), 1, 0)) %>%
  filter(remove == 0) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(bigram)

# Generate trigrams
trigram_df <- missom_coded %>%
  # Select key columns
  select(post_id, text, label_minority_stress) %>%
  unnest_ngrams(trigram, text, n = 3, drop = FALSE) %>%
  # Separate into three columns
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  # Remove stop words
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
  # Clean up based on remaining stop words
  mutate(
    stop_word1 = if_else(str_detect(word1, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0),
    stop_word2 = if_else(str_detect(word2, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) ,
    stop_word3 = if_else(str_detect(word3, regex("^im$|that's|i’m|it’s|you’re|don’t|dont|It|can’t|lt|he’s|she’s|i’ve|doesn’t|didn’t|isn’t|there’s|that'll|how’s|they’ll|it’ll|would've|we’ll|they’ve|shouldn’t|that’s|i’ll|they’re|aren’t|i’d|won’t|what’s|you’ve|we’re|wouldn’t|haven’t|wasn’t|y'all|let’s|here’s|who’s|you’ll|couldn’t|weren’t|hasn’t|we’ve|ain’t|you’d|y’all")), 1, 0) 
  ) %>%
  # Remove contracted stop words
  filter(
    stop_word1 == 0,
    stop_word2 == 0,
    stop_word3 == 0
  ) %>%
  # Combine into trigrams
  unite("trigram", c("word1", "word2", "word3"), sep = " ") %>%
  count(label_minority_stress, trigram) %>%
  arrange(desc(n))

# TF-IDF Trigrams
trigram_vector <- trigram_df %>%
  # Manual remove of nonsense
  mutate(remove = if_else(str_detect(trigram, "\\d|ðÿ|^amp |amp | amp$|NA NA NA|poll$|jfe|_link|link_|playlist 3948ybuzmcysemitjmy9jg si|complete 3 surveys|gmail.com mailto:hellogoodbis42069 gmail.com|hellogoodbis42069 gmail.com mailto:hellogoodbis42069|comments 7n2i gay_marriage_debunked_in_2_minutes_obama_vs_alan|debatealtright comments 7n2i|gift card|amazon|action hirewheller csr|energy 106 fm|form sv_a3fnpplm8nszxfb width|â íœê í|âˆ âˆ âˆ"), 1, 0)) %>%
  filter(remove == 0) %>%
  # Calculate tf-idf
  bind_tf_idf(trigram, label_minority_stress, n) %>%
  # Get top tf-idf of unigrams for minority stress posts
  arrange(desc(tf_idf)) %>%
  filter(label_minority_stress == 1) %>%
  # Select the top 100 n-grams
  head(n = 100) %>%
  pull(trigram)

# ......4a1) CODED DATASET ------------------------------------------------

# Assign the unigrams as features
for (i in 1:length(unigram_vector)) {
  
  # Get the n-grams
  ngram <- unigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(missom_coded$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  missom_coded[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_vector)) {
  
  # Get the n-grams
  ngram <- bigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(missom_coded$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  missom_coded[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_vector)) {
  
  # Get the n-grams
  ngram <- trigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(missom_coded$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  missom_coded[[ngram]] <- as.integer(x)  
}

# ......4b1) NOT CODED DATASET --------------------------------------------

# Assign the unigrams as features
for (i in 1:length(unigram_vector)) {
  
  # Get the n-grams
  ngram <- unigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(missom_not_coded$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  missom_not_coded[[ngram]] <- as.integer(x)  
}

# Assign the bigrams as features
for (i in 1:length(bigram_vector)) {
  
  # Get the n-grams
  ngram <- bigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(missom_not_coded$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  missom_not_coded[[ngram]] <- as.integer(x)  
}

# Assign the trigrams as features
for (i in 1:length(trigram_vector)) {
  
  # Get the n-grams
  ngram <- trigram_vector[i]
  
  # Detect the n-gram with regular expressions
  x <- str_detect(missom_not_coded$text, regex(ngram))
  
  # Add the n-gram to the dataframe
  missom_not_coded[[ngram]] <- as.integer(x)  
}

# 5) ADD LIWC FEATURES ----------------------------------------------------

# Coded dataset
missom_coded <- left_join(missom_coded, missom_coded_liwc, 
                          by = c("tagtog_file_id", "post_id"))

# Not coded dataset
missom_not_coded <- left_join(missom_not_coded, missom_not_coded_liwc, 
                              by = c("tagtog_file_id", "post_id"))

# EXPORT ------------------------------------------------------------------

# Save coded
write_csv(missom_coded, "data/cleaned/features/missom_coded_feat01.csv")

# Save not coded
write_csv(missom_not_coded, "data/cleaned/features/missom_not_coded_feat01.csv")
