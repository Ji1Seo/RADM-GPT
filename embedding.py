# Calculate Embeddings (actions, vision_text, audio_text)

from client import client
import numpy as np

def get_embedding(text, model = "text-embedding-3-small"):
    return client.embeddings.create(input = [text], model = model).data[0].embedding

def cosine_similarity(emb_1, emb_2):
    dot = np.dot(emb_1, emb_2)
    norm_1 = np.linalg.norm(emb_1)
    norm_2 = np.linalg.norm(emb_2)
    return dot / (norm_1 * norm_2)

def action_embedding(action_ls, audio_text_embed, image_text_embed):
    action_embs = [get_embedding(action) for action in action_ls]
    test_similarities = [cosine_similarity(audio_text_embed, action_emb) for action_emb in action_embs]
    image_similarities = [cosine_similarity(image_text_embed, action_emb) for action_emb in action_embs]
    combined_similarities = np.multiply(test_similarities, image_similarities) # Multiply
    # Using Softmax
    exps = np.exp(combined_similarities)
    probs = exps / np.sum(exps)
    return probs, action_embs





