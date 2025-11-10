# Profanity_filter_model
Building a ML model in order to filter out the profane comments and non-profane comments (e.g. vulgar, toxic, abusive, etc.)
## Profanity & Toxic Comment Filter
This is a machine learning model designed to detect and filter toxic, profane, and abusive language from online comments. It uses Natural Language Processing (NLP) to classify text as either "Profane/Toxic" or "Not Profane."

✨ Live Demo
You can test the live profanity filter here:  https://profanityfiltermodel-by-anshulkumarchauhan.streamlit.app/

## Project Overview
This project's goal was not just to build a classifier, but to build a fair and effective one in the face of a major real-world problem: a highly unbalanced dataset.

The Data: The training data (from Kaggle's Jigsaw Challenge) is over 90% "Not Profane" (class 0) and less than 10% "Profane" (class 1).

The Problem: A "lazy" model (like a standard Naive Bayes) can get 90% accuracy by always guessing "Not Profane." This results in a model with terrible recall, meaning it misses most of the actual toxic comments. Our initial model had a recall of only 21%.

The Solution: This model was built using a LogisticRegression classifier with the class_weight='balanced' parameter. This tells the model to "pay more attention" to the rare, profane class.

The Result: We successfully traded a small amount of (misleading) accuracy to boost our recall from 21% to 85%. This means the model now correctly identifies 85% of all toxic comments, making it a much more effective filter.

## How It Works: The ML Pipeline
The entire process, from a raw comment to a final prediction, is handled by a scikit-learn Pipeline.

1. Text Vectorization (TF-IDF)
First, we must convert the text comments into numbers.

The pipeline cleans the text (removes stop words, etc.) and uses a TfidfVectorizer.

This "smart" counter finds words that are highly indicative of toxicity (e.g., specific slurs or insults) by giving them a high numerical score.

## Classification (Logistic Regression)
The numerical data from the TF-IDF vectorizer is fed into our classifier.

We used a LogisticRegression model, which is a powerful and efficient classifier for text.

The most important parameter, class_weight='balanced', was set. This automatically adjusts the model's internal math to "punish" it more for misclassifying a "Profane" comment than for misclassifying a "Not Profane" one. This is what fixed our recall problem.

## Tech Stack
Python 3

Pandas: For loading and preprocessing the data.

Scikit-learn: For building the complete ML pipeline (TfidfVectorizer, LogisticRegression).

Pickle: For saving (serializing) the trained model object.

Streamlit: For creating and deploying the interactive web app.

## Dataset
This model was trained on the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle.

Link: Kaggle: Toxic Comment Classification Challenge

Preprocessing: The dataset's six label columns (toxic, severe_toxic, obscene, threat, insult, identity_hate) were combined into a single binary is_profane label. A comment was labeled 1 (Profane) if any of those six columns were 1.
