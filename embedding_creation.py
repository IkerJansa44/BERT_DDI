import torch
import numpy as np

from dataset import *
from codemaps import *
from transformers import BertModel, BertTokenizer

trainfile='train.pck'
validationfile='devel.pck'

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

# create indexes from training data
max_len = 150
embeddings = "Bert"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch.cuda.empty_cache()

bert = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Currently allocated GPU memory: {torch.cuda.memory_allocated(device=device) / 1024**3:.2f} GB")
bert.to(device)
print(f"Currently allocated GPU memory: {torch.cuda.memory_allocated(device=device) / 1024**3:.2f} GB")
Xt = []
print("Calculating Embeddings for train data")
for s in traindata.sentences():
    sentence_list = [w['form'] for w in s['sent']]
    sentence = ' '.join(sentence_list)
    inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=max_len)
    # Move inputs to the GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)

    embeddings = outputs.last_hidden_state.cpu().numpy()
    del inputs # Avoid CUDA out of memory
    del outputs # Avoid CUDA out of memory
    Xt.append(embeddings)

Xt = np.array(Xt)
np.save(f'Xt_{max_len}_.npy', Xt)

print(f"Currently allocated GPU memory: {torch.cuda.memory_allocated(device=device) / 1024**3:.2f} GB")
torch.cuda.empty_cache()
print(f"Currently allocated GPU memory: {torch.cuda.memory_allocated(device=device) / 1024**3:.2f} GB")
Xv = []
for s in valdata.sentences():
    sentence_list = [w['form'] for w in s['sent']]
    sentence = ' '.join(sentence_list)
    inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=max_len)
    # Move inputs to the GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)

    embeddings = outputs.last_hidden_state.cpu().numpy()
    del inputs # Avoid CUDA out of memory
    del outputs # Avoid CUDA out of memory
    Xv.append(embeddings)

Xv = np.array(Xv)
np.save(f'Xv_{max_len}_.npy', Xv)
