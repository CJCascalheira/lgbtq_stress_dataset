# SCAPE HATEBASE FOR HATE SPEECH LEXICON ----------------------------------

# Author = Cory J. Cascalheira
# Date = 06/14/2023

# The purpose of this script is to scrape hatebase for hate speech related to 
# being LGBTQ+. Note that hatebase is deprecated and the website will be taken
# down soon. This code may no longer work in a few years. In that case, please
# use the web archive. Hatebase terms are also added to Wikipedia LGBTQ+ slurs.

# Resources
# - https://rvest.tidyverse.org/reference/html_nodes.html
# - https://stackoverflow.com/questions/32400916/convert-html-tables-to-r-data-frame

# LOAD LIBRARIES ----------------------------------------------------------

# Dependencies
library(tidyverse)
library(rvest)

# HATEBASE SCREEN SCRAPING ------------------------------------------------

# ...1) ASEXUAL HATE SPEECH --------------------------------------------------

# Specify the URL
url <- "https://hatebase.org/search_results/orientation_id%3D1"

# Get the HTML code
hatebase_asexual <- read_html(url)

# Extract the hate speech terms
asexual_df <- html_nodes(hatebase_asexual, "table") %>%
  html_table()

asexual_df <- asexual_df[[1]] %>%
  select(word = X1, language = X2)

# ...2) HOMOSEXUAL HATE SPEECH -----------------------------------------------

# Specify the URL
url_base <- "https://hatebase.org/search_results/orientation_id%3D3"
url_pages <- "https://hatebase.org/search_results/orientation_id=3%7Cpage="

# Get the HTML code
hatebase_homosexual <- read_html(url_base)

# Extract the hate speech terms
homosexual_df <- html_nodes(hatebase_homosexual, "table") %>%
  html_table()

homosexual_df <- homosexual_df[[1]] %>%
  select(word = X1, language = X2)

# Loop through additional pages
for (i in 2:7) {
  
  # Update the page number of the url
  url_page <- paste0(url_pages, i)
  
  # Get the HTML code
  hatebase_homosexual <- read_html(url_page)
  
  # Extract the hate speech terms
  homosexual_df_temp <- html_nodes(hatebase_homosexual, "table") %>%
    html_table()
  
  homosexual_df_temp <- homosexual_df_temp[[1]] %>%
    select(word = X1, language = X2)
  
  # Combine with main dataframe
  homosexual_df <- bind_rows(homosexual_df, homosexual_df_temp)
}

# Clean up the hate speech terms
homosexual_df <- homosexual_df %>%
  mutate(word = str_extract(word, regex("\\w+")))  %>%
  # Remove repeated words
  distinct(word, .keep_all = TRUE) %>%
  # Convert to lowercase
  mutate(word = tolower(word))

# ...3) PANSEXUAL HATE SPEECH ------------------------------------------------

# Specify the URL
url <- "https://hatebase.org/search_results/orientation_id%3D4"

# Get the HTML code
hatebase_pansexual <- read_html(url)

# Extract the hate speech terms
pansexual_df <- html_nodes(hatebase_pansexual, "table") %>%
  html_table()

pansexual_df <- pansexual_df[[1]] %>%
  select(word = X1, language = X2)  %>%
  # Remove repeated words
  distinct(word, .keep_all = TRUE) %>%
  # Convert to lowercase
  mutate(word = tolower(word))

# ...4) TRANSGENDER HATE SPEECH ----------------------------------------------

# Specify the URL 
url_nonbinary <- "https://hatebase.org/search_results/gender_id%3D12"
url_transgender <- "https://hatebase.org/search_results/gender_id%3D14"

# Get the HTML code
hatebase_nonbinary <- read_html(url_nonbinary)
hatebase_transgender <- read_html(url_transgender)

# Extract the hate speech terms - nonbinary
nonbinary_df <- html_nodes(hatebase_nonbinary, "table") %>%
  html_table()

nonbinary_df <- nonbinary_df[[1]] %>%
  select(word = X1, language = X2) %>%
  # Clean up the word column
  mutate(word = str_extract(word, regex("\\w+")))  %>%
  # Remove repeated words
  distinct(word, .keep_all = TRUE) %>%
  # Convert to lowercase
  mutate(word = tolower(word))

# Extract the hate speech terms - transgender
transgender_df <- html_nodes(hatebase_transgender, "table") %>%
  html_table() 

transgender_df <- transgender_df[[1]] %>%
  select(word = X1, language = X2) %>%
  # Clean up the word column
  mutate(word = str_extract(word, regex("\\w+")))  %>%
  # Remove repeated words
  distinct(word, .keep_all = TRUE) %>%
  # Convert to lowercase
  mutate(word = tolower(word))

# ...5) MALE HATE SPEECH -----------------------------------------------------

# Specify the URL 
url <- "https://hatebase.org/search_results/gender_id%3D11"

# Get the HTML code
hatebase_male <- read_html(url)

# Extract the hate speech terms
male_df <- html_nodes(hatebase_male, "table") %>%
  html_table()

male_df <- male_df[[1]] %>%
  select(word = X1, language = X2) %>%
  # Clean up the word column
  mutate(word = str_extract(word, regex("\\w+"))) %>%
  # Remove repeated words
  distinct(word, .keep_all = TRUE) %>%
  # Convert to lowercase
  mutate(word = tolower(word))

# ...6) FEMALE HATE SPEECH ---------------------------------------------------

# Specify the URL
url_base <- "https://hatebase.org/search_results/gender_id%3D6"
url_pages <- "https://hatebase.org/search_results/gender_id=6%7Cpage="

# Get the HTML code
hatebase_female <- read_html(url_base)

# Extract the hate speech terms
female_df <- html_nodes(hatebase_female, "table") %>%
  html_table()

female_df <- female_df[[1]] %>%
  select(word = X1, language = X2)

# Loop through additional pages
for (i in 2:7) {
  
  # Update the page number of the url
  url_page <- paste0(url_pages, i)
  
  # Get the HTML code
  hatebase_female <- read_html(url_page)
  
  # Extract the hate speech terms
  female_df_temp <- html_nodes(hatebase_female, "table") %>%
    html_table()
  
  female_df_temp <- female_df_temp[[1]] %>%
    select(word = X1, language = X2)
  
  # Combine with main dataframe
  female_df <- bind_rows(female_df, female_df_temp)
}

# Clean up the hate speech terms
female_df <- female_df %>%
  mutate(word = str_extract(word, regex("\\w+")))  %>%
  # Remove repeated words
  distinct(word, .keep_all = TRUE) %>%
  # Convert to lowercase
  mutate(word = tolower(word))

# ...7) COMBINE HATEBASE DATAFRAMES ----------------------------------------

# Sexual minority
hatebase_sexual_minority <- bind_rows(asexual_df, homosexual_df, pansexual_df) %>%
  distinct(word, .keep_all = TRUE) %>%
  mutate(database = "Hatebase")

# Gender minority
hatebase_gender_minority <- bind_rows(transgender_df, nonbinary_df) %>%
  distinct(word, .keep_all = TRUE) %>%
  mutate(database = "Hatebase")

# Women and men
hatebase_woman_man <- bind_rows(female_df, male_df) %>%
  distinct(word, .keep_all = TRUE) %>%
  mutate(database = "Hatebase")

# WIKIPEDIA HATE SPEECH ---------------------------------------------------

# Pulled LGBTQ+ hate slurs from https://en.wikipedia.org/wiki/LGBT_slang
wikipedia <- read_delim("data/util/wikipedia_lgbtq_slurs.txt", delim = "\t")

# Hate speech for transgender folx
wikipedia_gender_minority <- head(wikipedia, n = 7) %>%
  mutate(language = "English") %>%
  mutate(database = "Wikipedia")

# Hate speech for sexual minority folx
wikipedia_sexual_minority <- wikipedia %>%
  filter(!(word %in% wikipedia_gender_minority$word)) %>%
  mutate(language = "English") %>%
  mutate(database = "Wikipedia")

# COMBINE HATEBASE AND WIKIPEDIA ------------------------------------------

# Combine for sexual minority
hate_lexicon_sexual_minority <- bind_rows(hatebase_sexual_minority, wikipedia_sexual_minority) %>%
  distinct(word, .keep_all = TRUE)
hate_lexicon_sexual_minority

# Combine gender minority
hate_lexicon_gender_minority <- bind_rows(hatebase_gender_minority, wikipedia_gender_minority) %>%
  distinct(word, .keep_all = TRUE)
hate_lexicon_gender_minority  

# EXPORT ------------------------------------------------------------------

# Save the data
write_csv(hate_lexicon_sexual_minority, "data/hatespeech/hate_lexicon_sexual_minority.csv")
write_csv(hate_lexicon_gender_minority, "data/hatespeech/hate_lexicon_gender_minority.csv")
write_csv(hatebase_woman_man, "data/hatespeech/hatebase_woman_man.csv")
