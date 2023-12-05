# COMBINE ALL FEATURES ----------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 06/27/2023

# The purpose of this script is to join all feature dataframes into one
# object.

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(tidyverse)

# Import feature data files
missom_coded_feat01 <- read_csv("data/cleaned/features/missom_coded_feat01.csv")
missom_coded_feat02a <- read_csv("data/cleaned/features/missom_coded_feat02a.csv") %>%
  select(-...1, -contains("quot"))
missom_coded_feat02b <- read_csv("data/cleaned/features/missom_coded_feat02b.csv") %>%
  select(-...1, -contains("quot"))

missom_not_coded_feat01 <- read_csv("data/cleaned/features/missom_not_coded_feat01.csv")
missom_not_coded_feat02a <- read_csv("data/cleaned/features/missom_not_coded_feat02a.csv") %>%
  select(-...1, -contains("quot"))
missom_not_coded_feat02b <- read_csv("data/cleaned/features/missom_not_coded_feat02b.csv") %>%
  select(-...1, -contains("quot"))

# Import BERT-CNN machine-annotated data files
bert_cnn_labels <- read_csv("data/cleaned/private/missom_bertcnn_annotated.csv") %>%
  select(-how_annotated, -text)

# Import raw text
tagtog <- read_csv("data/raw/combined/07_CMIMS_html_annotated_json_data-all_html.csv") %>%
  select(tagtog_file_id = File_id, text = Text)

# Get shared column names
missom_coded_names <- names(missom_coded_feat01)
missom_not_coded_names <- names(missom_not_coded_feat01)

# Preprocess the columns
missom_coded_feat02a <- missom_coded_feat02a %>%
  select(-missom_coded_names[6:419])

missom_not_coded_feat02a <- missom_not_coded_feat02a %>%
  select(-missom_not_coded_names[6:419])

# COMBINE -----------------------------------------------------------------

# Join the data files - human coded
missom_coded <- left_join(missom_coded_feat01, missom_coded_feat02a) %>%
  left_join(missom_coded_feat02b) %>%
  select(-`time trans`, -label_limitation, -label_minority_stress_new)

# Join the data files - machine coded
missom_not_coded <- left_join(missom_not_coded_feat01, missom_not_coded_feat02a) %>%
  left_join(missom_not_coded_feat02b) %>%
  select(-`time trans`, -label_limitation, -label_minority_stress_new)

# Add the machine-annotated labels
missom_bert_cnn <- missom_not_coded %>%
  select(-starts_with("label")) %>%
  left_join(bert_cnn_labels) %>%
  select(tagtog_file_id, post_id, how_annotated, subreddit, text, 
         starts_with("label"), everything())

# Merge the human- and machine-coded datasets
missom_plus <- bind_rows(missom_coded, missom_bert_cnn)

# Remove identifying information
missom <- missom_plus %>%
  select(-subreddit, -text)

# Add the raw text to MiSSoM+
missom_plus <- missom_plus %>%
  select(-text) %>%
  left_join(tagtog) %>%
  distinct(post_id, .keep_all = TRUE) %>%
  select(tagtog_file_id, post_id, how_annotated, subreddit, text, 
         starts_with("label"), everything())

# EXPORT ------------------------------------------------------------------

# Export the data for initial papers
write_csv(missom_coded, "data/cleaned/private/missom_coded.csv")
write_csv(missom_not_coded, "data/cleaned/private/missom_not_coded.csv")

# Export the full dataset
write_csv(missom_plus, "data/cleaned/private/missom_plus.csv")
write_csv(missom, "data/cleaned/public/missom.csv")
