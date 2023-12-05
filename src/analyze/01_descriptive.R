# DESCRIPRIVE STATISTICS AND BASIC VISUALIZATIONS -------------------------

# Author = Cory J. Cascalheira
# Created = 06/27/2023

# The purpose of this script is to generate descriptive statistics and simple,
# basic visualizations of the MiSSoM dataset.

# LOAD AND IMPORT ---------------------------------------------------------

# Load dependencies
library(tidyverse)
library(tidytext)
library(wordcloud2)
library(psych)
library(lsr)
library(scales)

# Import data
missom_coded <- read_csv("data/cleaned/private/missom_coded.csv")
missom_not_coded <- read_csv("data/cleaned/private/missom_not_coded.csv")
missom_coded_raw <- read_csv("data/cleaned/version_1/missom_coded_v1.csv")
post_timestamps <- read_csv("data/raw/post_timestamps.csv") %>%
  filter(post_id %in% c(missom_coded$post_id, missom_not_coded$post_id))

# Set the seed for reproducibility
set.seed(1234567)

# DESCRIPTIVE STATISTICS --------------------------------------------------

# Total number of posts
length(c(missom_coded$post_id, missom_not_coded$post_id))

# Get time frame of data pull 
post_timestamps %>%
  summarize(
    max_time = max(post_time),
    min_time = min(post_time)
  )

# Number of posts coded by hand
nrow(missom_coded)

# Number of posts coded by machine
nrow(missom_not_coded)

# Describe the word count
describe(missom_coded$WC)

# Describe the number of characters
describe(nchar(missom_coded_raw$text))

# Separate the two forms of minority stress
yes_ms <- missom_coded %>%
  filter(label_minority_stress == 1)

no_ms <- missom_coded %>%
  filter(label_minority_stress == 0)

# Number of posts with minority stress and without - original theory
nrow(yes_ms)
nrow(no_ms)

# Number of posts with minority stress and without - w/ gender dysphoria
missom_coded %>%
  filter(label_minority_stress_new == 1) %>%
  nrow()

missom_coded %>%
  filter(label_minority_stress_new == 0) %>%
  nrow()

# Types of minority stress
missom_coded %>%
  filter(label_minority_coping == 1) %>% 
  nrow()

missom_coded %>%
  filter(label_prej_event == 1) %>% 
  nrow()

missom_coded %>%
  filter(label_identity_conceal == 1) %>% 
  nrow()

missom_coded %>%
  filter(label_exp_reject == 1) %>% 
  nrow()

missom_coded %>%
  filter(label_internal_stigma == 1) %>% 
  nrow()

missom_coded %>%
  filter(label_dysphoria == 1) %>% 
  nrow()

# DESCRIPTIVE VISUALS -----------------------------------------------------

# ...1) PIECHART OF SUBREDDITS --------------------------------------------

# Create a blank theme
blank_theme <- theme_minimal()+
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.border = element_blank(),
    panel.grid=element_blank(),
    axis.ticks = element_blank(),
    plot.title=element_text(size=14, face="bold")
  )

# Select just the subreddits
missom_coded_sub <- missom_coded %>%
  select(post_id, subreddit)

missom_subreddits <- missom_not_coded %>%
  select(post_id, subreddit) %>%
  bind_rows(missom_coded_sub) %>%
  count(subreddit) %>%
  mutate(
    my_sum = sum(n),
    percent = round((n / my_sum) * 100, 2),
    subreddit = paste0("r/", subreddit, ": ", percent, "%"),
    subreddit = fct_reorder(as.factor(subreddit), percent)
  ) %>%
  rename(Subreddit = subreddit)

# Piechart of subreddits
subreddit_piechart <- missom_subreddits %>%
  ggplot(aes(x = "", y = percent, fill = Subreddit)) +
  geom_bar(stat = "identity") + 
  coord_polar("y", start = 0) + 
  scale_fill_viridis_d(option = "turbo") + 
  blank_theme +
  theme(axis.text.x = element_blank(),
        text = element_text(family = "serif"))
subreddit_piechart

# Save to file
ggsave(filename = "results/plots/subreddit_piechart.png", 
       plot = subreddit_piechart,
       width = 4,
       height = 4)

# WORDCLOUDS --------------------------------------------------------------

# References
# - https://towardsdatascience.com/create-a-word-cloud-with-r-bde3e7422e8a
# - http://rstudio-pubs-static.s3.amazonaws.com/564823_960901304f4e4853ba7dbc93eb4bc499.html
# - https://r-graph-gallery.com/196-the-wordcloud2-library.html

# ...1) PREPARE WITH TF-IDF ----------------------------------------------

# References
# - https://www.tidytextmining.com/tfidf.html

# Unnest tokens 
missom_coded_abrv <- missom_coded %>%
  unnest_tokens("word", "text") %>%
  select(tagtog_file_id, post_id, label_minority_stress, word) %>%
  # Additional cleaning
  mutate(
    word = str_remove_all(word, regex("_|\\d|[:punct:]")),
    remove = if_else(word == "", 1, 0)
  ) %>%
  filter(remove == 0) %>%
  select(-remove)

# Count tokens
missom_coded_words <- missom_coded_abrv %>%
  count(word)
missom_coded_words

# Join dataframes to create term-frequency dataframe
missom_coded_tf <- left_join(missom_coded_words, missom_coded_abrv) %>%
  distinct(word, .keep_all = TRUE) %>%
  arrange(desc(n)) %>%
  select(tagtog_file_id, post_id, label_minority_stress, word, n)

# Compute TF-IDF
missom_coded_tf_idf <- missom_coded_tf %>%
  bind_tf_idf(term = word, document = label_minority_stress, n = n) %>%
  arrange(desc(tf_idf))

# Separate the two forms of minority stress
yes_ms <- missom_coded_tf_idf %>%
  filter(label_minority_stress == 1)

no_ms <- missom_coded_tf_idf %>%
  filter(label_minority_stress == 0)

# ...2) YES MINORITY STRESS -----------------------------------------------

# Transform into reduced dataframe
yes_ms_df <- yes_ms %>%
  select(word, freq = tf_idf) %>%
  head(n = 200)

# Set the colors
n <- ceiling(nrow(yes_ms_df)/4)

color_yes <- sapply(sapply(seq(from = 1, to = 0.25, by = -0.25), rep, n), function(x) adjustcolor("#E4460A", x))

# Generate the wordcloud
yes_ms_wordcloud <- wordcloud2(data = yes_ms_df, shape = "circle", size = 1, color = color_yes)
yes_ms_wordcloud

# Take screenshot

# ...3) NO MINORITY STRESS ------------------------------------------------

# Transform into reduced dataframe
no_ms_df <- no_ms %>%
  select(word, freq = tf_idf) %>%
  head(n = 200)

# Set the colors
n <- ceiling(nrow(no_ms_df)/4)

color_no <- sapply(sapply(seq(from = 1, to = 0.25, by = -0.25), rep, n), function(x) adjustcolor("#4686FB", x))

# Generate the wordcloud
no_ms_wordcloud <- wordcloud2(data = no_ms_df, shape = "circle", size = 1, color = color_no)
no_ms_wordcloud

# Take screenshot

# GROUP TESTS & VISUALIZATIONS --------------------------------------------

# References
# - http://www.sthda.com/english/wiki/ggplot2-box-plot-quick-start-guide-r-software-and-data-visualization
# - https://stackoverflow.com/questions/5677885/ignore-outliers-in-ggplot2-boxplot

# ...1) SENTIMENT ---------------------------------------------------------

# Separate the two groups 
yes_ms_sentiment <- missom_coded %>% 
  filter(label_minority_stress == 1) %>%
  select(sentiment_lexicon)

no_ms_sentiment <- missom_coded %>% 
  filter(label_minority_stress == 0) %>%
  select(sentiment_lexicon)

# Descriptives of sentiment
describe(yes_ms_sentiment$sentiment_lexicon)
describe(no_ms_sentiment$sentiment_lexicon)

# Perform t test
t.test(yes_ms_sentiment, no_ms_sentiment, paired = FALSE)

# Visualize the group comparison with a boxplot - full version w/ outliers
sentiment_boxplot_full <- missom_coded %>%
  mutate(label_minority_stress = if_else(label_minority_stress == 1, "Yes", "No")) %>%
  ggplot(aes(x = label_minority_stress, y = sentiment_lexicon, fill = label_minority_stress)) + 
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = c("#4686FB", "#E4460A")) + 
  scale_y_continuous(breaks = pretty_breaks(n = 10)) + 
  theme_minimal() +
  theme(
    text = element_text(size = 20)
  ) +
  labs(
    x = "Minority Stress",
    y = "Sentiment Value"
  )
sentiment_boxplot_full

# Take screenshot for editing as one image 

# Visualize boxplot - zoomed in version
ylim1 = boxplot.stats(missom_coded$sentiment_lexicon)$stats[c(1, 5)]

sentiment_boxplot_zoomed <- missom_coded %>%
  mutate(label_minority_stress = if_else(label_minority_stress == 1, "Yes", "No")) %>%
  ggplot(aes(x = label_minority_stress, y = sentiment_lexicon, fill = label_minority_stress)) + 
  geom_boxplot(show.legend = FALSE) +
  coord_cartesian(ylim = ylim1*1.0) +
  scale_fill_manual(values = c("#4686FB", "#E4460A")) + 
  scale_y_continuous(breaks = pretty_breaks(n = 10)) + 
  theme_minimal() +
  theme(
    text = element_text(size = 20)
  ) +
  labs(
    x = "Minority Stress",
    y = "Sentiment Value"
  )
sentiment_boxplot_zoomed

# Take screenshot for editing as one image 

# ...2) PSYCHOLINGUISTIC ATTRIBUTES ---------------------------------------

# General preprocessing of the LIWC data
missom_coded_liwc <- missom_coded %>%
  # Make long format
  pivot_longer(cols = WC:OtherP, names_to = "liwc_names", values_to = "liwc_values") %>%
  # Add LIWC categories
  mutate(
    liwc_categories = if_else(liwc_names %in% c("Analytic", "Clout", "Authentic", 
                                                "Tone", "WPS", "Sixltr", "Dic"), 
                              "Summary", "None"),
    liwc_categories = if_else(liwc_names %in% c("function", "pronoun", "ppron", "i", "we", "you",
                                                "shehe", "they", "ipron", "article", "prep",
                                                "auxverb", "adverb", "conj", "negate"), 
                              "Linguistic", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("verb", "adj", "compare", "interrog", "numbers",
                                                "quant"), 
                              "Grammar", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("affect", "posemo", "negemo", "anx", "anger", "sad"), 
                              "Affect", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("social", "family", "friend", "female", "male"), 
                              "Social", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("cogproc", "insight", "cause", "discrep", "tentat", "certain",
                                                "differ"), 
                              "Cognitive", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("percept", "see", "hear", "feel"), 
                              "Perceptual", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("bio", "body", "health", "sexual", "ingest"), 
                              "Biological", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("drives", "affiliation", "achieve", "power", "reward", "risk"), 
                              "Drives", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("TimeOrient", "focuspast", "focuspresent", "focusfuture"), 
                              "Time Orientation", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("relativ", "motion", "space", "time"), 
                              "Relativity", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("work", "leisure", "home", "money", "relig", "death"), 
                              "Personal Concerns", liwc_categories),
    liwc_categories = if_else(liwc_names %in% c("informal", "swear", "netspeak", "assent", "nonflu", "filler"), 
                              "Informal Language", liwc_categories)
  )
missom_coded_liwc

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(missom_coded_liwc, missom_coded_liwc$liwc_names), 
                       function(x) t.test(liwc_values~label_minority_stress, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(missom_coded_liwc, missom_coded_liwc$liwc_names), 
                       function(x) cohensD(liwc_values~label_minority_stress, x)) %>%
  as_vector()

# Get LIWC vars names
liwc_names <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(liwc_names, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# Separate the data
missom_yes <- missom_coded %>%
  filter(label_minority_stress == 1)

missom_no <- missom_coded %>%
  filter(label_minority_stress == 0)

# Calculate Cohen's d 95% CI
cohen_d_ci_df <- data.frame()

for (i in 1:length(test_signf_results$effect_sizes)) {
  cohen_d_ci <- cohen.d.ci(d = test_signf_results$effect_sizes[i], n = nrow(missom_coded),
                           n1 = nrow(missom_yes), n2 = nrow(missom_no))
  
  cohen_d_ci_df <- bind_rows(cohen_d_ci_df, as.data.frame(cohen_d_ci)) 
}

# Rename column
cohen_d_ci_df <- cohen_d_ci_df %>%
  rename(effect_sizes = effect)

# Add the confidence intervals to the dataframe
test_signf_results <- left_join(test_signf_results, cohen_d_ci_df) %>%
  rename(d_upper = upper, d_lower = lower)

# Add LIWC categories
test_signf_results1 <- missom_coded_liwc %>%
  select(liwc_categories, liwc_names) %>%
  distinct(liwc_names, .keep_all = TRUE) %>%
  right_join(test_signf_results) %>% 
  # Round values
  mutate(
    t_stat = round(t_stat, 3),
    effect_sizes = round(effect_sizes, 3)
  ) %>%
  arrange(liwc_categories, desc(effect_sizes)) %>%
  mutate(higher = if_else(t_stat < 0, "Minority Stress is Present", "Minority Stress is Absent"))
test_signf_results1

# Save data to csv
write_csv(test_signf_results1, "results/liwc_significance_tests.csv")

# Visualize
liwc_forest_plot <- ggplot(test_signf_results1, 
       aes(y = reorder(liwc_names, effect_sizes), 
           x = effect_sizes,
           color = higher)) +
  geom_errorbar(aes(xmin = d_lower, xmax = d_upper), color = "black") +
  geom_point(size = 3) +
  scale_color_manual(values = c("#4686FB", "#E4460A"), name = "Group with\nHigher Mean") +
  theme_minimal() +
  labs(
    x = "Effect Size (Cohen's d)",
    y = "Psycholinguistic Attributes (LIWC)"
  ) + 
  theme(legend.position="bottom")
liwc_forest_plot

# ...3) LDA TOPICS MODELS -------------------------------------------------

# Preprocessing
missom_coded_lda <- missom_coded %>%
  # Make long format
  pivot_longer(cols = lda_postcommentsubreddit:lda_wearbodydress, 
               names_to = "lda_topics", values_to = "lda_values")

# Execute the independent samples t-tests w/o variance equality assumption
test_results <- lapply(split(missom_coded_lda, missom_coded_lda$lda_topics), 
                       function(x) t.test(lda_values~label_minority_stress, x))
print(test_results)

# Bonferroni correction value
bonf_value <- 0.05 / length(names(test_results))

# Extract data from list into df, next few lines

# Prepare vectors
t_stat <- c()
p_val <- c()
deg_free <- c()

# For loop to extract data
for (i in 1:length(names(test_results))) {
  t_stat <- c(t_stat, test_results[[i]]$statistic)
  p_val <- c(p_val, test_results[[i]]$p.value)
  deg_free <- c(deg_free, test_results[[i]]$parameter)
}

# Effect size using Cohen's D
# https://rcompanion.org/handbook/I_03.html
effect_sizes <- lapply(split(missom_coded_lda, missom_coded_lda$lda_topics), 
                       function(x) cohensD(lda_values~label_minority_stress, x)) %>%
  as_vector()

# Get LIWC vars names
lda_topics <- names(test_results)

# Add all vectors into df
test_signf_results <- data.frame(lda_topics, t_stat, deg_free, effect_sizes, p_val) %>%
  as_tibble() %>%
  # Set Bonferroni value and determine if p_val is beyond/more extreme
  mutate(bonf_value = bonf_value) %>%
  mutate(signf_at_bonf = p_val < bonf_value) %>%
  # Keep only significant differences
  filter(signf_at_bonf == TRUE)
test_signf_results

# No significant differences in LDA topics when Bonferroni correction is applied
