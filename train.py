# Probs Policy Training with GPT Action Make

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from embedding import get_embedding, cosine_similarity

from robot_action import action_list
from prob_policy_net import Policy
import os

learning_rate = 0.01 # Para 1
action_embs = [get_embedding(action) for action in action_list]
action_dim = len(action_list) # Maybe, 6
state_dim = 1536 * 2 # Embedding Length
policy_net = Policy(input_dim=state_dim, output_dim=action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

model_save_path = './Model/model_checkpoint.pth'

def save_model(model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
    }, model_save_path)

def load_model(model, optimizer):
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss
    else:
        return 0, None

def train(probs, audio_text_emb, vision_text_emb):
    epoch, _ = load_model(policy_net, optimizer)
    epi_num = epoch 
    state_embed = torch.tensor(np.concatenate([audio_text_emb, vision_text_emb]), dtype=torch.float32) # Input to NN
    predicted_probs = policy_net(state_embed) # NN Output

    answer_text = input(f"Episode {epi_num + 1}: Enter the feedback text: ") # User Feedback
    answer_emb = get_embedding(answer_text)
    answer_similarities = [cosine_similarity(answer_emb, action_emb) for action_emb in action_embs] # Action Embs are Fixed
    exps = np.exp(answer_similarities)
    feed_probs = exps / np.sum(exps)

    posterior_numerator = predicted_probs.detach().numpy() * feed_probs # Bayesian
    posterior_probs = posterior_numerator / np.sum(posterior_numerator)
    target_probs = torch.tensor(posterior_probs, dtype=torch.float32) # Target
    loss_fn = nn.KLDivLoss(reduction='batchmean') # KL Divergence
    loss = loss_fn(torch.log(predicted_probs), target_probs) # Loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    save_model(policy_net, optimizer, epi_num + 1, loss.item())
    print(f"Episode {epi_num + 1}: Loss = {loss.item():.4f}, Prior probabilities: {probs}, Updated probabilities: {posterior_probs}") # Compare
