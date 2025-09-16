library(tidyverse)
library(reticulate)

policy_corpus <- read_csv("tokenized_policy_corpus.csv")

head(policy_corpus)

transformers <- import("transformers")
torch <- import("torch")
nn <- torch$nn
optim <- torch$optim


tokenizer <- transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")
model <- transformers$AutoModelForSequenceClassification$from_pretrained(
  "distilbert-base-uncased",
  num_labels = 2
)


device <- torch$device("cpu")
model$to(device)


set.seed(123)

train_data <- policy_corpus %>%
  sample_n(17) %>%
  mutate(label = sample(0:1, n(), replace = TRUE))


encode <- function(text) {
  tokens <- tokenizer$encode_plus(
    text,
    max_length = 15L,
    truncation = TRUE,
    padding = "max_length",
    return_tensors = "pt"
  )
  list(
    input_ids = tokens$input_ids$to(device)$long(),
    attention_mask = tokens$attention_mask$to(device)$long()
  )
}

encoded <- map(train_data$chunk_text, encode)


input_ids <- torch$cat(lapply(encoded, function(x) x$input_ids), dim = 0L)
attention_mask <- torch$cat(lapply(encoded, function(x) x$attention_mask), dim = 0L)
labels <- torch$tensor(train_data$label, dtype = torch$long)$to(device)


optimizer <- optim$AdamW(model$parameters(), lr = 2e-5)
criterion <- nn$CrossEntropyLoss()


model$train()
epochs <- 5

for (epoch in 1:epochs) {
  optimizer$zero_grad()
  outputs <- model(
    input_ids = input_ids,
    attention_mask = attention_mask
  )
  logits <- outputs$logits
  loss <- criterion(logits, labels)
  cat(sprintf("Epoch %d - Loss: %.4f\n", epoch, loss$item()))
  loss$backward()
  optimizer$step()
}


##### EVAL

test_text <- train_data$chunk_text[[1]]
true_label <- train_data$label[[1]]


encoded_test <- tokenizer$encode_plus(
  test_text,
  max_length = 15L,
  truncation = TRUE,
  padding = "max_length",
  return_tensors = "pt"
)


input_ids <- encoded_test$input_ids$to(device)$long()
attention_mask <- encoded_test$attention_mask$to(device)$long()


model$eval()
with_no_grad <- torch$no_grad()



with_no_grad$`__enter__`()
output <- model(
  input_ids = input_ids,
  attention_mask = attention_mask
)
with_no_grad$`__exit__`(NULL, NULL, NULL)


logits <- output$logits
probs <- torch$nn$functional$softmax(logits, dim = 1L)
predicted_class <- torch$argmax(probs, dim = 1L)$item()



cat("Text:\n", test_text, "\n\n")
cat("True Label:    ", true_label, "\n")
cat("Predicted Label:", predicted_class, "\n")
cat("Class Probabilities:\n")
print(probs$detach()$numpy())


