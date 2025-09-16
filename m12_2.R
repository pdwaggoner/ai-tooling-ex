library(tidyverse)
library(reticulate)

policy_corpus <- read_csv("tokenized_policy_corpus.csv")
head(policy_corpus)


sentence_transformers <- import("sentence_transformers")
model <- sentence_transformers$SentenceTransformer('all-MiniLM-L6-v2')
policy_corpus$embedding <- map(policy_corpus$chunk_text, function(text) model$encode(text))


policy_corpus$embedding[[1]][1:10]


cosine_sim <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

vec1 <- c(1, 2, 3)
vec2 <- c(1, 2, 3)
cosine_sim(vec1, vec2)


user_query <- "How can I request public records?"
query_embedding <- model$encode(user_query)
query_embedding


policy_corpus <- policy_corpus %>%
  mutate(similarity = map_dbl(embedding, ~ cosine_sim(., query_embedding)))


policy_corpus %>%
  arrange(desc(similarity)) %>%
  select(chunk_text, similarity) %>%
  head(5)


top_k <- 3

get_top_k <- function(query_text, corpus_df, model, k = 3) {
  query_embed <- model$encode(query_text)
  corpus_df <- corpus_df %>%
    mutate(similarity = map_dbl(embedding, ~ cosine_sim(., query_embed))) %>%
    arrange(desc(similarity)) %>%
    slice_head(n = k)
  return(corpus_df)
}


top_results <- get_top_k("How do I request public records?", policy_corpus, model, top_k)
top_results$chunk_text


query <- "How do I request public records?"
query_embed <- model$encode(query)

policy_corpus <- policy_corpus %>%
  mutate(similarity = map_dbl(embedding, ~ cosine_sim(., query_embed)))

ggplot(policy_corpus, aes(x = similarity)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(
    title = "Distribution of Cosine Similarity Scores",
    x = "Cosine Similarity",
    y = "Number of Document Chunks"
  )


top_k <- 5

top_k_results <- policy_corpus %>%
  arrange(desc(similarity)) %>%
  slice_head(n = top_k)

ggplot(policy_corpus, aes(x = reorder(chunk_text, similarity), y = similarity)) +
  geom_point(alpha = 0.4) +
  geom_point(data = top_k_results, color = "red", size = 3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Cosine Similarity Scores Across Policy Corpus",
    x = "Document Chunk",
    y = "Cosine Similarity"
  )


