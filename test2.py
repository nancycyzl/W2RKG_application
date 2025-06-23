from sentence_transformers import SentenceTransformer

model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
print("Model initialized.")