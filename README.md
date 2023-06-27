# The LGBTQ+ Minority Stress on Social Media Dataset (LGBTQ+ MiSSoM)

The LGBTQ+ **Mi**nority **S**tress on **So**cial **Me**dia (LGBTQ+ MiSSoM) dataset is the largest text-based, natural language processing (NLP) dataset on expressions of minority stress. The data are posts and comments from Reddit.com.

## Order of Script Execution

1. src/extract_tagtog/
2. src/clean/
    2a. preprocess_annotated.R
    2b. preprocess_binary_v1.R
    2c. preprocess_binary_v2.R
3. src/create_features/
4. src/analyze/
