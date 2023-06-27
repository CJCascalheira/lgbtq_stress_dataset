# COMBINE ALL FEATURES ----------------------------------------------------

# Author = Cory J. Cascalheira
# Date = 06/27/2023

# The purpose of this script is to join all feature dataframes into one
# object.

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(tidyverse)

# Import data files
missom_coded_feat01 <- read_csv("data/cleaned/features/missom_coded_feat01.csv")
missom_coded_feat02a <- read_csv("data/cleaned/features/missom_coded_feat02a.csv") %>%
  select(-...1)
missom_coded_feat02b <- read_csv("data/cleaned/features/missom_coded_feat02b.csv") %>%
  select(-...1)

missom_not_coded_feat01 <- read_csv("data/cleaned/features/missom_not_coded_feat01.csv")
missom_not_coded_feat02a <- read_csv("data/cleaned/features/missom_not_coded_feat02a.csv") %>%
  select(-...1)
missom_not_coded_feat02b <- read_csv("data/cleaned/features/missom_not_coded_feat02b.csv") %>%
  select(-...1)

# Get shared column names
missom_coded_names <- names(missom_coded_feat01)
missom_not_coded_names <- names(missom_not_coded_feat01)

# Preprocess the columns
missom_coded_feat02a <- missom_coded_feat02a %>%
  select(-missom_coded_names[6:419])

missom_not_coded_feat02a <- missom_not_coded_feat02a %>%
  select(-missom_not_coded_names[6:419])

# COMBINE AND EXPORT ------------------------------------------------------

# Join the data files - human coded
missom_coded <- left_join(missom_coded_feat01, missom_coded_feat02a) %>%
  left_join(missom_coded_feat02b)

# Join the data files - machine coded
missom_not_coded <- left_join(missom_not_coded_feat01, missom_not_coded_feat02a) %>%
  left_join(missom_not_coded_feat02b)

# Export the data
write_csv(missom_coded, "data/cleaned/private/missom_coded.csv")
write_csv(missom_not_coded, "data/cleaned/private/missom_not_coded.csv")
