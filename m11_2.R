library(tidyverse)
library(reticulate)

transformers <- import("transformers")

tokenizer <- transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")

model <- transformers$AutoModelForSequenceClassification$from_pretrained("distilbert-base-uncased")


sample_text <- "The agency shall provide access to public records within 20 business days."

tokens <- tokenizer$encode(sample_text)
tokens


train_data <- tibble(
  text = c(
    "I request the agency’s 2022 budget report under FOIA.",
    "What is the timeline for FOIA processing?",
    "Please provide a copy of the environmental assessment for the new pipeline.",
    "What are the new hiring policies for public sector employees?"
  ),
  label = c(1, 1, 1, 0)  # 1 = FOIA-related, 0 = not
)


encode <- function(text) {
  tokenizer$encode_plus(
    text, 
    max_length = as.integer(64), 
    truncation = TRUE, 
    padding = "max_length", 
    return_tensors = "pt"
  )
}


encoded_train <- map(train_data$text, encode)


encoded_train[[1]]$input_ids
encoded_train[[1]]$attention_mask


########## IN PY ##########
#
# from torch.optim import AdamW
# optimizer = AdamW(model.parameters(), lr=2e-5)
#
# for epoch in range(3):
#   for batch in train_loader:
#   outputs = model(**batch)
# loss = outputs.loss
# loss.backward()
# optimizer.step()
# optimizer.zero_grad()
#
##########


model$save_pretrained("foia_classifier_model")

tokenizer$save_pretrained("foia_classifier_model")
