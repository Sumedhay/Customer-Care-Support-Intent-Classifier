import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sympy import im


# ==========================================
# CUSTOMER CARE SUPPORT INTENT CLASSIFIER
# ==========================================

# 1. Customer support messages
data = {
    "message": [
        # Shipping
        "Where is my parcel?",
        "Where is my order?",
        "When will my package arrive?",
        "Track my shipment",
        "My delivery has not arrived",

        # Refund
        "I want a refund",
        "I want my money back",
        "Please refund my order",
        "Can I get my payment refunded?",
        "I need a refund",

        # Cancellation
        "I want to cancel my order",
        "Please cancel my order",
        "Can I cancel my purchase?",
        "Cancel my order",
        "I placed the order by mistake",

        # Product
        "Tell me about this product",
        "What are the features of this product?",
        "Is this product available?",
        "What is the price of this item?",
        "I need product information",

        # Technical
        "The app is not working",
        "I cannot login",
        "The website keeps crashing",
        "I have a technical problem",
        "My account is not working",

        # Billing
        "Why was I charged extra?",
        "I have a billing problem",
        "My invoice is incorrect",
        "Why is my payment amount different?",
        "I was charged twice"
    ],

    "intent": [
        # Shipping
        "shipping_query",
        "shipping_query",
        "shipping_query",
        "shipping_query",
        "shipping_query",

        # Refund
        "refund_request",
        "refund_request",
        "refund_request",
        "refund_request",
        "refund_request",

        # Cancellation
        "cancellation_request",
        "cancellation_request",
        "cancellation_request",
        "cancellation_request",
        "cancellation_request",

        # Product
        "product_query",
        "product_query",
        "product_query",
        "product_query",
        "product_query",

        # Technical
        "technical_issue",
        "technical_issue",
        "technical_issue",
        "technical_issue",
        "technical_issue",

        # Billing
        "billing_query",
        "billing_query",
        "billing_query",
        "billing_query",
        "billing_query"
    ]
}


# 2. Create DataFrame
df = pd.DataFrame(data)

print("========================================")
print("Customer Care Support Intent Classifier")
print("========================================")

print("\nDataset loaded successfully!")
print("Total messages:", len(df))


# 3. Encode the intent labels
label_encoder = LabelEncoder()

df["intent_encoded"] = label_encoder.fit_transform(df["intent"])

print("\nIntent categories:")

for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)


# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["intent_encoded"],
    test_size=0.2,
    random_state=42,
    stratify=df["intent_encoded"]
)


# 5. Convert text into numerical values using TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nText vectorization completed!")


# 6. Train Machine Learning model
model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("Model trained successfully!")


# 7. Evaluate the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy, 2))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# 8. Test new customer messages
print("\n========================================")
print("        CUSTOMER CARE CHATBOT")
print("========================================")
print("Enter a customer message.")
print("Type 'exit' to close the program.\n")


while True:

    message = input("Customer: ")

    if message.lower() == "exit":
        print("\nThank you! Goodbye.")
        break

    # Convert the new message into numbers
    message_vector = vectorizer.transform([message])

    # Predict the intent
    prediction = model.predict(message_vector)

    # Convert number back to intent name
    intent = label_encoder.inverse_transform(prediction)[0]

    print("Predicted Intent:", intent)
    print()