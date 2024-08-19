from llama_index import GPTSimpleVectorIndex, Document
import coreferee
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

def process_text(text: str) -> str:
    doc = nlp(text)
    resolved = doc._.coref_resolved
    return resolved

def create_embeddings(text: str) -> GPTSimpleVectorIndex:
    documents = [Document(text)]
    index = GPTSimpleVectorIndex.from_documents(documents)
    return index

def query_embeddings(index: GPTSimpleVectorIndex, query: str) -> str:
    response = index.query(query)
    return response.response

def create_transformer_embeddings(text: str, model_name: str = "bert-base-uncased") -> torch.Tensor:
    """
    Create embeddings for the input text using a pre-trained transformer model.
    
    Args:
        text (str): Input text to create embeddings for.
        model_name (str): Name of the pre-trained model to use (default: "bert-base-uncased").
    
    Returns:
        torch.Tensor: Text embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the mean of the last hidden state as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def extract_entities(text: str) -> list:
    """
    Extract named entities from the input text using spaCy.
    
    Args:
        text (str): Input text to extract entities from.
    
    Returns:
        list: List of tuples containing entity text and entity label.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def sentiment_analysis(text: str) -> dict:
    """
    Perform sentiment analysis on the input text using a pre-trained model.
    
    Args:
        text (str): Input text to analyze.
    
    Returns:
        dict: Sentiment analysis result containing label and score.
    """
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result

def extract_keywords(text: str, top_n: int = 5) -> list:
    """
    Extract keywords from the input text using TF-IDF.
    
    Args:
        text (str): Input text to extract keywords from.
        top_n (int): Number of top keywords to extract (default: 5).
    
    Returns:
        list: List of top keywords.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    keyword_scores = list(zip(feature_names, tfidf_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [keyword for keyword, _ in keyword_scores[:top_n]]
