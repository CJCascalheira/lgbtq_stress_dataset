# FEATURES - CLINICAL KEY WORDS -------------------------------------------

# Author = Cory J. Cascalheira
# Date = 05/09/2023

# The purpose of this script is to create the following features:
# - clinical keywords
# - sentiment lexicons

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(textstem)
library(tidyverse)
library(tidytext)

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

hatebase_woman_man <- read_csv("data/hatespeech/hatebase_woman_man.csv") %>%
  select(word) %>%
  mutate(hatebase_woman_man = 1)

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

# 2) SENTIMENT LEXICON ----------------------------------------------------

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
  summarize(sentiment_valence = sum(value)) %>%
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
  summarize(sentiment_valence = sum(value)) %>%
  # Join to main dataframe
  left_join(missom_not_coded) %>%
  # Rearrange the variables
  select(tagtog_file_id, post_id, how_annotated:label_minority_stress_new, everything())

# 3) HATE SPEECH LEXICONS -------------------------------------------------

# Human coded dataset
missom_coded <- missom_coded %>%
  # Reduce df size
  select(post_id, text) %>%
  # Tokenize Reddit post
  unnest_tokens(word, text) %>%
  # Join the hate speech data frames
  left_join(hate_lexicon_sexual_minority) %>%
  left_join(hate_lexicon_gender_minority) %>%
  left_join(hatebase_woman_man) %>%
  # Recode missing to 0
  mutate(
    hate_lexicon_sexual_minority = if_else(is.na(hate_lexicon_sexual_minority), 0,
                                           hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = if_else(is.na(hate_lexicon_gender_minority), 0,
                                           hate_lexicon_gender_minority),
    hatebase_woman_man = if_else(is.na(hatebase_woman_man), 0,
                                 hatebase_woman_man)
  ) %>%
  # Group by post
  group_by(post_id) %>%
  # Calculate presence of hate speech term
  summarize(
    hate_lexicon_sexual_minority = sum(hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = sum(hate_lexicon_gender_minority),
    hatebase_woman_man = sum(hatebase_woman_man)
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
  left_join(hatebase_woman_man) %>%
  # Recode missing to 0
  mutate(
    hate_lexicon_sexual_minority = if_else(is.na(hate_lexicon_sexual_minority), 0,
                                           hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = if_else(is.na(hate_lexicon_gender_minority), 0,
                                           hate_lexicon_gender_minority),
    hatebase_woman_man = if_else(is.na(hatebase_woman_man), 0,
                                 hatebase_woman_man)
  ) %>%
  # Group by post
  group_by(post_id) %>%
  # Calculate presence of hate speech term
  summarize(
    hate_lexicon_sexual_minority = sum(hate_lexicon_sexual_minority),
    hate_lexicon_gender_minority = sum(hate_lexicon_gender_minority),
    hatebase_woman_man = sum(hatebase_woman_man)
  ) %>%
  # Join to main dataframe
  left_join(missom_not_coded) %>%
  # Rearrange the variables
  select(tagtog_file_id, post_id, how_annotated:label_minority_stress_new, everything())

# EXPORT ------------------------------------------------------------------

# Save coded
write_csv(missom_coded, "data/cleaned/features/missom_coded_feat01.csv")

# Save not coded
write_csv(missom_not_coded, "data/cleaned/features/missom_not_coded_feat01.csv")
