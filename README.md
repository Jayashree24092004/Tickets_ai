# AI Ticket Classification System

This project implements an **AI-powered ticket classification system**. It can automatically categorize incoming support tickets and provide appropriate responses.  

The system is built using **Python**, **NLTK** for text preprocessing, and **Scikit-learn** for machine learning.

---

## **Problem Statement**

Given user support tickets such as:

1. "I forgot my password, how to reset it?"
2. "I can't log in, as password is incorrect."
3. "How to see leave balance?"

The task is to **group tickets into categories** (like Authentication, Leave Management, etc.) and provide relevant responses.  

**Example Grouping:**

| Ticket | Category |
|--------|----------|
| "I forgot my password, how to reset it?" | Authentication |
| "I can't log in, as password is incorrect." | Authentication |
| "How to see leave balance?" | Leave Management |

The AI model should be able to classify **new tickets** into these categories automatically.

---

## **Features**

- Preprocesses tickets: **lowercasing, punctuation removal, stopword removal, lemmatization**
- Converts text to **numeric features** using **TF-IDF**
- Trains a **Logistic Regression classifier** to categorize tickets
- Saves the trained model for reuse with **Joblib**
- Provides **predefined responses** for each ticket category
- Interactive command-line interface for entering tickets

---

## **Requirements**

- Python 3.8+
- Libraries:

bash
pip install nltk scikit-learn joblib
## **How It Works**
-Preprocessing:
Converts text to lowercase
Removes punctuation
Removes stopwords
Lemmatizes words
-Feature Extraction:
Uses TF-IDF vectorization (unigrams + bigrams)
-Classification:
Logistic Regression predicts ticket category
-Response Generation:
Randomly selects a response from predefined category responses
## **Command Window Output**
<img width="591" height="343" alt="image" src="https://github.com/user-attachments/assets/0346aa32-2157-42f5-a7b7-8cef603980d2" />

