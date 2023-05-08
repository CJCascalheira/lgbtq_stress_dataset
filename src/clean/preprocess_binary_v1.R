# CLEAN AND PREPROCESS THE EXTRACTED TAGTOG DATA --------------------------

# Author: Cory J. Cascalheira
# Date created: 04/23/2023

# This script cleans the original Tagtog data and prepares it for feature
# engineering and modeling. It focuses on the data where post-level coding
# was performed and then transformed into a binary outcome. 

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(tidyverse)

# Import data
original_reddit <- read_csv("data/raw/df_reddit_lgbtq.csv") %>%
  select(temp_id, subreddit)

tagtog_full <- read_csv("data/raw/combined/05_CMIMS_html_json_data-all_html.csv") %>%
  rename(temp_id = Temp_id, text = Text, file_id = File_id,
         label_minority_coping = e_2, label_prej_event = e_3, label_exp_reject = e_4, 
         label_identity_conceal = e_5, label_internal_stigma = e_6, label_dysphoria = e_7, 
         label_limitation = e_9, label_minority_stress = minority_stress) %>%
  # Remove unnecessary variables
  select(-e_8, -e_1)

# Add the subreddits
missom_full <- left_join(tagtog_full, original_reddit) %>%
  # Reorder the variables and rename ID variables
  select(tagtog_file_id = file_id, post_id = temp_id, subreddit, text, everything())

# SPLIT THE DATA ----------------------------------------------------------

# Select the posts coded by the team
missom_coded <- na.omit(missom_full)

# Select the posts not coded
missom_not_coded <- missom_full %>%
  filter(!(post_id %in% missom_coded$post_id))

# DOUBLE CHECK CODING OF MINORITY STRESS ----------------------------------

# Create a new minority stress variable
missom_coded <- missom_coded %>%
  mutate(
    ms_new = if_else(label_prej_event == 1, 1, 0),
    ms_new = if_else(label_exp_reject == 1, 1, ms_new),
    ms_new = if_else(label_identity_conceal == 1, 1, ms_new),
    ms_new = if_else(label_internal_stigma == 1, 1, ms_new)
  )
  
# Check distribution of minority stress coding
table(missom_coded$label_minority_stress)
table(missom_coded$ms_new)

# Add gender dysphoria to new minority stress variable
missom_coded <- missom_coded %>%
  mutate(
    ms_new = if_else(label_dysphoria == 1, 1, ms_new)
  ) %>%
  rename(label_minority_stress_new = ms_new)

missom_not_coded <- missom_not_coded %>%
  mutate(label_minority_stress_new = NA_character_)

# Check new distribution
table(missom_coded$label_minority_stress_new)

# CLEAN THE DATA ----------------------------------------------------------

# Clean the text column
missom_coded <- missom_coded %>%
  # Remove links / URLs
  mutate(text = str_remove_all(text, " ?(f|ht)tp(s?)://(.*)[.][a-z]+")) %>%
  # Replace whitespace characters
  mutate(text = str_replace_all(text, "\r\n\r\n", " ")) %>%
  mutate(text = str_replace_all(text, "\r", " ")) %>%
  mutate(text = str_replace_all(text, "\n", " ")) %>%
  # Recode characters
  mutate(text = recode(text, "&amp;" = "and", "Â´" = "'", "â€™" = "'"))

missom_not_coded <- missom_not_coded %>%
  # Remove links / URLs
  mutate(text = str_remove_all(text, " ?(f|ht)tp(s?)://(.*)[.][a-z]+")) %>%
  # Replace whitespace characters
  mutate(text = str_replace_all(text, "\r\n\r\n", " ")) %>%
  mutate(text = str_replace_all(text, "\r", " ")) %>%
  mutate(text = str_replace_all(text, "\n", " ")) %>%
  # Recode characters
  mutate(text = recode(text, "&amp;" = "and", "Â´" = "'", "â€™" = "'"))

# EXPORT ------------------------------------------------------------------

# Save as CSV files
write_csv(missom_coded, "data/cleaned/version_1/missom_coded_v1.csv")
write_csv(missom_not_coded, "data/cleaned/version_1/missom_not_coded_v1.csv")
