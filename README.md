# The LGBTQ+ Minority Stress on Social Media Dataset (LGBTQ+ MiSSoM)

The LGBTQ+ **Mi**nority **S**tress on **So**cial **Me**dia (LGBTQ+ MiSSoM) dataset is the largest text-based, natural language processing (NLP) dataset on expressions of minority stress. The data are posts and comments from Reddit.com.

Some of the initial code in earlier phases of dataset creation, such as downloading the data via PushShift and establishing inter-coder reliability, can be found (here)[https://github.com/CJCascalheira/rise-ml-ms].

## Versions of the Dataset

1. **MiSSoM** = the public dataset with features and labels, but no raw text. You can access the public dataset here.

2. **MiSSoM+** = the private dataset with raw text. You can access the private dataset by emailing cjcascalheira@gmail.com, registering your study idea, and signing an agreement to keep the private dataset off on public-facing servers. 

## Order of Script Execution

1. src/extract_tagtog/
2. src/clean/
    1. preprocess_binary_v1.R
    2. preprocess_binary_v2.R
    3. preprocess_annotated.R
3. src/machine_annotate/
4. src/create_features/
5. src/analyze/

### Other Scripts

* src/util/ *scripts for utility*
* src/pull_subsets/ *scripts to manage data for other scientists*

## Projects Related to / Using the MiSSoM Datasets

* https://github.com/chapagaisa/BERT-CNN
* https://github.com/chapagaisa/BERT_BiGRU
