library(tidyverse)
library(tm)

policy_texts <- c(
  "The agency shall provide access to public records.",
  "Requests must be processed within 20 business days.",
  "Data transparency is critical for public accountability.",
  "The FOIA establishes procedures for access to information.",
  "Privacy considerations must be balanced against transparency."
)

policy_corpus <- VCorpus(VectorSource(policy_texts))


tdm <- TermDocumentMatrix(policy_corpus, control = list(weighting = weightTfIdf))
inspect(tdm)


library(text2vec)

tokens <- space_tokenizer(policy_texts)
it <- itoken(tokens)

vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)

tcm <- create_tcm(it, vectorizer, skip_grams_window = 5)

glove_model <- GlobalVectors$new(rank = 50, x_max = 10)
word_vectors <- glove_model$fit_transform(tcm, n_iter = 10)


word_vectors["transparency", ]


library(reticulate)

sentence_transformers <- import("sentence_transformers")
model <- sentence_transformers$SentenceTransformer('all-MiniLM-L6-v2')

embeddings <- model$encode(policy_texts)
embeddings[[1]]
