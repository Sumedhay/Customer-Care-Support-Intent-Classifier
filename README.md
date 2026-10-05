# Customer-Care-Support-Intent-Classifier# Customer Care Support Intent Classifier

## Project Overview

This project uses Machine Learning to automatically identify the intent behind customer support messages.

For example:

- "Where is my parcel?" → `shipping_query`
- "I want my money back" → `refund_request`
- "Please cancel my order" → `cancellation_request`
- "My app is not working" → `technical_issue`

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes

## How It Works

1. Customer messages are collected with their corresponding intent labels.
2. Intent labels are encoded into numerical values.
3. TF-IDF converts text into numerical features.
4. A Multinomial Naive Bayes classifier is trained.
5. The model predicts the intent of new customer messages.
6. Accuracy, precision, recall, and F1-score are evaluated using a classification report.

## Project Structure

```text
customer_intent_classifier/
│
├── customer_intent.py
├── README.md
└── .gitignore
