library(tidyverse)

text_data <- read_csv("chunked_policy_corpus.csv")

library(reticulate)

transformers <- import("transformers")

########## PY ENV SETUP IF NEEDED ##########
#
# install.packages("reticulate")
#
# python3 -m venv ~/.virtualenvs/r-reticulate
# source ~/.virtualenvs/r-reticulate/bin/activate
#
# pip install transformers torch
#
# use_virtualenv("~/.virtualenvs/r-reticulate", required = TRUE)
#
# reticulate::py_config()
#
# transformers <- import("transformers")
# 
# test it out:
tokenizer <- transformers$AutoTokenizer$from_pretrained("bert-base-uncased")
tokens <- tokenizer$encode("Hello world!", add_special_tokens = TRUE)
tokens

##########

tokenizer <- transformers$AutoTokenizer$from_pretrained("gpt2")


sample_text <- text_data$chunk_text[1]
sample_text


tokens <- tokenizer$encode(sample_text)
tokens


token_strings <- tokenizer$convert_ids_to_tokens(tokens)
token_strings


text_data <- text_data %>%
  mutate(token_count = map_int(chunk_text, ~ length(tokenizer$encode(.x))))


summary(text_data$token_count)


write_csv(text_data, "tokenized_policy_corpus.csv")

