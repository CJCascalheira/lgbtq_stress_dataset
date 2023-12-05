# PULL SUBSETS OF DATA FOR RESEARCHERS ------------------------------------

# Author = Cory J. Cascalheira
# Date = 06/17/2023

# The purpose of this script is to pull data subsets for interested authors who
# may lack programming skills (e.g., psychologists).

# LOAD AND IMPORT ---------------------------------------------------------

# Load dependencies
library(textstem)
library(tidyverse)
library(tidytext)

# Import data files
missom_coded_v1 <- read_csv("data/cleaned/version_1/missom_coded_v1.csv")
missom_not_coded_v1 <- read_csv("data/cleaned/version_1/missom_not_coded_v1.csv")
missom_coded_v2 <- read_csv("data/cleaned/version_2/missom_coded_v2.csv")
missom_not_coded_v2 <- read_csv("data/cleaned/version_2/missom_not_coded_v2.csv")

# Combine data
missom_v2 <- bind_rows(missom_coded_v2, missom_not_coded_v2)

# EMILY M. LUND -----------------------------------------------------------

# Pull data for a project on asexuality and aromanticism
missom_ace_aro <- missom_v2 %>%
  mutate(ace_aro = if_else(str_detect(text, regex("asexual|aromantic|ace|ace-spec|aro|aro-spec", 
                                                  ignore_case = TRUE)), 1, 0)) %>%
  filter(ace_aro == 1)

# Export the data
write_csv(missom_ace_aro, 'data/pulled_subsets/missom_ace_aro.csv')

# SANTOSH CHAPAGAIN -------------------------------------------------------

# Select and export
missom_coded_v1 %>%
  select(post_id, how_annotated, text, starts_with("label")) %>%
  select(-label_limitation, -label_minority_stress_new) %>%
  write_csv("data/pulled_subsets/missom_annotated.csv")

# Select and export
missom_not_coded_v1 %>%
  select(post_id, how_annotated, text, starts_with("label")) %>%
  select(-label_limitation, -label_minority_stress_new) %>%
  write_csv("data/pulled_subsets/missom_not_annotated.csv")
