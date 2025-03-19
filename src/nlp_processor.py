"""
NLP Processor for Skill Classification

This script provides natural language processing capabilities for:
1. Processing and analyzing text data related to skills and occupations
2. Classifying skills based on predefined categories or keywords
3. Creating vector representations of skills for similarity analysis

This is a generic tool that can be adapted for different research contexts.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing non-alphanumeric characters,
    and standardizing format.
    
    Parameters:
    - text: String to preprocess
    
    Returns:
    - Preprocessed text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def classify_skills_rule_based(skills, target_keywords):
    """
    Classify skills using a rule-based approach based on keyword matching.
    
    Parameters:
    - skills: String containing skills (comma or space separated)
    - target_keywords: Set of keywords to match against
    
    Returns:
    - 1 if any skill matches target keywords, 0 otherwise
    """
    if not isinstance(skills, str):
        return 0
        
    skills_lower = skills.lower()
    return 1 if any(keyword in skills_lower for keyword in target_keywords) else 0


def calculate_similarity_tfidf(texts, query, top_n=10):
    """
    Calculate similarity between texts and a query using TF-IDF vectors.
    
    Parameters:
    - texts: List of text documents to compare against
    - query: Query text to find similar documents to
    - top_n: Number of top similar documents to return
    
    Returns:
    - List of (index, similarity_score) tuples for top_n similar documents
    """
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Transform the query to the same TF-IDF space
    query_tfidf = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and all texts
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Get indices of top_n most similar documents
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Return (index, score) tuples
    return [(idx, similarity_scores[idx]) for idx in top_indices]


def train_skill_classifier(df, skill_column, label_column, model_type='random_forest'):
    """
    Train a model to classify skills based on provided labels.
    
    Parameters:
    - df: DataFrame containing skill data
    - skill_column: Column name containing skill text
    - label_column: Column name containing labels (0/1)
    - model_type: Type of model to train ('random_forest', 'svm', or 'logistic')
    
    Returns:
    - Trained model, vectorizer, and performance metrics
    """
    # Preprocess skills
    df['processed_skills'] = df[skill_column].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_skills'], 
        df[label_column], 
        test_size=0.2, 
        random_state=42,
        stratify=df[label_column]
    )
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Select and train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = LinearSVC(random_state=42)
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return model, vectorizer, metrics


def predict_skill_labels(model, vectorizer, new_skills):
    """
    Predict labels for new skills using a trained model.
    
    Parameters:
    - model: Trained classifier model
    - vectorizer: Fitted TF-IDF vectorizer
    - new_skills: List of new skill texts to classify
    
    Returns:
    - Array of predicted labels
    """
    # Preprocess new skills
    processed_skills = [preprocess_text(skill) for skill in new_skills]
    
    # Transform to TF-IDF vectors
    X_new = vectorizer.transform(processed_skills)
    
    # Predict labels
    return model.predict(X_new)


def main():
    """Main execution function for demonstration."""
    try:
        # Load sample data - replace with your actual data
        print("Loading sample data...")
        data_path = "data/input/skills_data.csv"
        df = pd.read_csv(data_path)
        
        print(f"Data loaded. Shape: {df.shape}")
        
        # Define keywords for rule-based classification
        target_keywords = {
            'machine learning', 'data science', 'neural networks', 
            'natural language processing', 'nlp', 'ai', 'artificial intelligence',
            'deep learning', 'computer vision', 'data analysis'
        }
        
        # Apply rule-based classification
        print("Applying rule-based classification...")
        df['rule_based_label'] = df['skills'].apply(
            lambda x: classify_skills_rule_based(x, target_keywords)
        )
        
        # Train a model
        print("Training skill classifier...")
        model, vectorizer, metrics = train_skill_classifier(
            df, 'skills', 'rule_based_label', model_type='random_forest'
        )
        
        # Print performance metrics
        print("\nClassifier Performance:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Save outputs
        output_path = "output/classified_skills.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    main()