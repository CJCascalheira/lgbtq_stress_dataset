# CLEAN AND PREPROCESS THE EXTRACTED ANNOTATED TAGTOG DATA -----------------

# Author: Cory J. Cascalheira
# Date created: 04/23/2023

# This script cleans the original, annotated tagtog data and prepares it for
# modeling. It focuses on sentence-level coding on minority stressors.

# LOAD LIBRARIES AND IMPORT DATA ------------------------------------------

# Load dependencies
library(tidyverse)

# Import data
original_reddit <- read_csv("data/raw/df_reddit_lgbtq.csv")
tagtog <- read_csv("data/raw/combined/07_CMIMS_html_annotated_json_data-all_html.csv")