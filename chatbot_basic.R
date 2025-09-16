library(text2vec)
library(tidyverse)

corpus <- tibble(
  id = 1:10,
  doc = c(
    "The Freedom of Information Act (FOIA) ensures that citizens have the right to request access to records from any federal agency. Agencies are required to disclose information unless it falls under one of nine exemptions, which protect interests such as personal privacy, national security, and law enforcement. FOIA promotes transparency and accountability within the federal government by providing a mechanism for the public to access government documents and data that inform policy and decision-making.",
    
    "The Artificial Intelligence Bill of Rights outlines five guiding principles to protect the public from the potential harms of algorithmic systems. These include the right to safe and effective systems, algorithmic discrimination protections, data privacy, notice and explanation, and human alternatives. While not legally binding, the framework encourages federal agencies and private sector entities to adopt responsible AI development practices, especially when used in critical decision-making domains such as housing, employment, and criminal justice.",
    
    "Executive Order 13960, 'Promoting the Use of Trustworthy Artificial Intelligence in the Federal Government,' mandates that all federal agencies adopt principles of fairness, transparency, and accountability when deploying AI technologies. Agencies must also inventory AI use cases, evaluate risk levels, and ensure oversight mechanisms are in place. The order establishes a governance process led by agency Chief Data Officers and Chief AI Officers to monitor deployment and ensure compliance with ethical AI standards.",
    
    "The Digital Services Modernization Act of 2022 seeks to transform federal digital infrastructure by mandating the adoption of user-centered design, open data standards, and interoperable systems. Under the Act, agencies must transition legacy services to modern, cloud-based platforms and ensure mobile accessibility across digital government portals. Funding is allocated to support public engagement and usability testing for all major public-facing digital tools used by citizens and businesses.",
    
    "The Privacy Act of 1974 establishes rules for the collection, use, and dissemination of personally identifiable information (PII) by federal agencies. It grants individuals the right to access and amend records about themselves and sets limitations on how data can be shared without consent. Agencies are required to publish notices about their data systems and implement safeguards against unauthorized disclosure or misuse of sensitive information, particularly in automated systems.",
    
    "The Federal Algorithmic Risk Reduction Act introduces a tiered risk framework for evaluating algorithmic systems used in government decision-making. High-risk systems — including those used in public benefits eligibility, criminal justice assessments, and border enforcement — must undergo third-party audits and publish risk mitigation strategies. The Act also calls for public comment periods and participatory design processes for algorithms that have significant public impact.",
    
    "The Cloud Security and Resilience Act mandates that any cloud service provider working with federal agencies meet enhanced cybersecurity and data redundancy standards. Agencies must classify data into sensitivity tiers and adopt zero-trust architecture to minimize breach risk. Annual disaster recovery exercises and third-party penetration testing are required for continued compliance. This Act aims to strengthen government continuity and service reliability in digital operations.",
    
    "The National Data Literacy and Inclusion Initiative focuses on reducing disparities in digital literacy across demographic groups. The initiative funds local organizations to host workshops, develop culturally relevant learning materials, and train community data ambassadors. Federal agencies are also required to publish plain-language summaries of technical documents and integrate data literacy modules into all public engagement programs, especially those targeting underserved populations.",
    
    "The Public Algorithm Accountability Directive creates standardized requirements for algorithmic transparency and impact disclosures in the public sector. Agencies must publish detailed model documentation (so-called 'model cards') for any AI tool used in decision-making, including inputs, limitations, performance benchmarks, and bias mitigation strategies. Internal review boards are tasked with ensuring the accuracy and fairness of these tools and with fielding citizen complaints and appeals.",
    
    "The Autonomous Systems Governance Framework lays out procedures for overseeing the deployment of autonomous systems — including drones, self-driving vehicles, and robotic process automation — within government operations. Agencies must complete environmental and social impact assessments before implementation and provide clear channels for stakeholder feedback. The framework promotes equitable access, public safety, and ethical design in autonomous deployments used for surveillance, logistics, or citizen services."
  )
)


tokens <- itoken(corpus$doc, progressbar = FALSE)
vocab <- create_vocabulary(tokens)
vectorizer <- vocab_vectorizer(vocab)
dtm <- create_dtm(tokens, vectorizer)
corpus_matrix <- dtm

highlight_keywords <- function(text, keywords) {
  for (word in keywords) {
    word_pattern <- paste0("\\b", word, "\\b")
    text <- str_replace_all(text, regex(word_pattern, ignore_case = TRUE),
                            paste0("**", word, "**"))
  }
  text
}

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_file <- paste0("policy_bot_log_", timestamp, ".csv")

if (!file.exists(log_file)) {
  write_csv(tibble(time = character(), query = character(), matched_text = character(), matched_id = integer()), log_file)
}

ask_policy_bot <- function(user_input) {
  input_tokens <- itoken(user_input, progressbar = TRUE)
  input_dtm <- create_dtm(input_tokens, vectorizer)
  
  # compute similarity and coerce to vector
  similarity <- sim2(input_dtm, corpus_matrix, method = "cosine")
  similarity_vector <- as.numeric(similarity[1, ])
  
  if (all(similarity_vector == 0)) {
    cat("\n ❌ Policy Bot says:\n")
    cat("Sorry, I couldn't find anything relevant to your question.\n")
    return()
  }
  
  best_match <- which.max(similarity_vector)
  response_text <- corpus$doc[best_match]
  
  keywords <- unlist(str_extract_all(tolower(user_input), "\\w+"))
  
  highlighted <- highlight_keywords(response_text, keywords)
  
  cat("\n ✅ Policy Bot says:\n")
  cat(highlighted, "\n\n")
  
  log_entry <- tibble(
    time = Sys.time(),
    query = user_input,
    matched_text = response_text,
    matched_id = best_match
  )
  write_csv(log_entry, log_file, append = TRUE)
}


# good / working examples
ask_policy_bot("How can I access government records?")
ask_policy_bot("What protects our privacy from AI?")
ask_policy_bot("What is the paperwork act?")


# bad / working examples
ask_policy_bot("what did you say?")


##
##
## 


## now, put it on a shiny server to test out:

# app.R


library(shiny)
library(text2vec)
library(tidyverse)
library(pdftools)
library(officer)
library(readtext)

highlight_keywords <- function(text, keywords) {
  for (word in keywords) {
    word_pattern <- paste0("\\b", word, "\\b")
    text <- str_replace_all(text, regex(word_pattern, ignore_case = TRUE),
                            paste0("<b>", word, "</b>"))
  }
  text
}

extract_text_from_file <- function(file_path) {
  ext <- tools::file_ext(file_path)
  if (ext == "pdf") {
    return(paste(pdf_text(file_path), collapse = " "))
  } else if (ext == "docx") {
    doc <- read_docx(file_path)
    text <- docx_summary(doc)
    return(paste(text$text[text$content_type == "paragraph"], collapse = " "))
  } else {
    return("")
  }
}

chunk_text_by_sentences <- function(text, n_sentences = 2) {
  sentences <- unlist(strsplit(text, "(?<=[.!?])\\s+", perl = TRUE))
  sentences <- str_trim(sentences)
  sentences <- sentences[sentences != ""]
  chunks <- split(sentences, ceiling(seq_along(sentences) / n_sentences))
  chunked_text <- sapply(chunks, paste, collapse = " ")
  return(chunked_text)
}

ui <- fluidPage(
  titlePanel("📄 Policy Bot – Ask a Question About Your Uploaded Document"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload a PDF or Word Document", accept = c(".pdf", ".docx")),
      textInput("user_input", "Ask a question:", ""),
      actionButton("submit", "Ask"),
      br(), br(),
      downloadButton("download_log", "Download Log")
    ),
    mainPanel(
      uiOutput("response"),
      verbatimTextOutput("doc_status")
    )
  )
)

server <- function(input, output, session) {
  uploaded_text <- reactiveVal(NULL)
  corpus_matrix <- reactiveVal(NULL)
  vectorizer <- reactiveVal(NULL)
  
  log_data <- reactiveVal(tibble(
    time = character(),
    query = character(),
    matched_text = character(),
    matched_id = integer()
  ))
  
  observeEvent(input$file, {
    req(input$file)
    text <- extract_text_from_file(input$file$datapath)
    
    if (nchar(text) == 0) {
      uploaded_text(NULL)
      output$doc_status <- renderText("❌ Could not extract text from file.")
      return()
    }
    
    text_chunks <- chunk_text_by_sentences(text, n_sentences = 2)
    text_chunks <- text_chunks[nchar(text_chunks) > 20]
    
    if (length(text_chunks) == 0) {
      output$doc_status <- renderText("❌ No usable content found in the document.")
      return()
    }
    
    chunks <- tibble(id = seq_along(text_chunks), doc = text_chunks)
    uploaded_text(chunks)
    
    tokens <- itoken(chunks$doc, progressbar = FALSE)
    vocab <- create_vocabulary(tokens)
    vec <- vocab_vectorizer(vocab)
    dtm <- create_dtm(tokens, vec)
    
    if (nrow(dtm) == 0 || ncol(dtm) == 0) {
      output$doc_status <- renderText("❌ Document content could not be processed into usable form.")
      return()
    }
    
    vectorizer(vec)
    corpus_matrix(dtm)
    output$doc_status <- renderText("✅ Document successfully processed.")
  })
  
  observeEvent(input$submit, {
    req(uploaded_text(), vectorizer(), corpus_matrix(), input$user_input)
    
    input_tokens <- itoken(input$user_input, progressbar = FALSE)
    input_dtm <- create_dtm(input_tokens, vectorizer())
    
    similarity <- sim2(input_dtm, corpus_matrix(), method = "cosine")
    similarity_vector <- as.numeric(similarity[1, ])
    
    if (all(similarity_vector == 0)) {
      output$response <- renderUI({
        HTML("<p><b>❌ Policy Bot says:</b><br>No relevant content found in the uploaded document.</p>")
      })
      return()
    }
    
    best_match <- which.max(similarity_vector)
    response_text <- uploaded_text()$doc[best_match]
    
    keywords <- unlist(str_extract_all(tolower(input$user_input), "\\w+"))
    highlighted <- highlight_keywords(response_text, keywords)
    
    output$response <- renderUI({
      HTML(paste0("<p><b>✅ Policy Bot says:</b></p><p>", highlighted, "</p>"))
    })
    
    log_entry <- tibble(
      time = as.character(Sys.time()),
      query = input$user_input,
      matched_text = response_text,
      matched_id = best_match
    )
    
    log_data(bind_rows(log_data(), log_entry))
  })
  
  output$download_log <- downloadHandler(
    filename = function() {
      paste0("policy_bot_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
    },
    content = function(file) {
      write_csv(log_data(), file)
    }
  )
}

shinyApp(ui, server)


