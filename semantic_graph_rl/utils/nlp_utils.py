from llama_index import GPTSimpleVectorIndex, Document
import coreferee
import spacy

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
