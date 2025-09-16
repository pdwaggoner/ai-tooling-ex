library(tidyverse)
library(reticulate)

openai <- import("openai")
openai$api_key <- Sys.getenv("OPENAI_API_KEY")  # Store your key in an .Renviron file for security

generate_response <- function(query_text, retrieved_docs, model_name = "gpt-3.5-turbo") {
  prompt_text <- paste(
    "You are a public policy assistant. Answer the following question using only the information below.\n",
    paste(retrieved_docs$chunk_text, collapse = "\n---\n"),
    "\nQuestion:", query_text
  )
  
  response <- openai$ChatCompletion$create(
    model = model_name,
    messages = list(list(role = "user", content = prompt_text)),
    temperature = 0
  )
  
  return(response$choices[[1]]$message$content)
}


# define a query to pass
query <- "How do I request public records?"

# retrieve relevant documents
top_docs <- get_top_k(query, policy_corpus, model, k = 3)

# generate a response
rag_response <- generate_response(query, top_docs)
cat(rag_response)


##########


sentence_transformers <- import("sentence_transformers")
model <- sentence_transformers$SentenceTransformer('all-MiniLM-L6-v2')

policy_corpus <- read_csv("tokenized_policy_corpus.csv")

policy_corpus$embedding <- map(policy_corpus$chunk_text, function(text) model$encode(text))


cosine_sim <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

get_top_k <- function(query_text, corpus_df, model, k = 3) {
  query_embed <- model$encode(query_text)
  corpus_df <- corpus_df %>%
    mutate(similarity = map_dbl(embedding, ~ cosine_sim(., query_embed))) %>%
    arrange(desc(similarity)) %>%
    slice_head(n = k)
  return(corpus_df)
}


generate_response <- function(query_text, retrieved_docs) {
  context_text <- paste(retrieved_docs$chunk_text, collapse = "\n\n")
  
  response <- paste(
    "Based on the following documents:\n", 
    context_text, 
    "\n\nAnswer:\n", 
    "The question you asked was: ", query_text, 
    ". Please review the above excerpts for relevant information."
  )
  
  return(response)
}


# query
query <- "How do I request public records?"

# retrieve top-3 chunks
top_docs <- get_top_k(query, policy_corpus, model, k = 3)

# sim response; take a look
rag_response <- generate_response(query, top_docs)
cat(rag_response)


query_embed <- model$encode(query)

policy_corpus <- policy_corpus %>%
  mutate(similarity = map_dbl(embedding, ~ cosine_sim(., query_embed)))

ggplot(policy_corpus, aes(x = reorder(chunk_text, similarity), y = similarity)) +
  geom_point(alpha = 0.4) +
  geom_point(data = top_docs, color = "red", size = 3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = paste("Cosine Similarity for Query:", query),
    x = "Document Chunk",
    y = "Similarity"
  )
