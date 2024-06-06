import torch
import neptune 
import numpy as np
import matplotlib.pyplot as plt
import util.evaluator as evaluator
import nltk
nltk.download('punkt')

from dataset import *
from codemaps import *
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset

class BertWithSoftmax(nn.Module):
    def __init__(self, num_labels):
        super(BertWithSoftmax, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        #outputs.last_hidden_state.shape is [batch_size, seq_len, embedding_size]
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]  # We use the [CLS] token representation
        cls_hidden_state = self.dropout(cls_hidden_state)
        logits = self.classifier(cls_hidden_state)
        return logits
    
def output_interactions(data, preds, outfile):
    outf = open(outfile, 'w')
    for exmp, tag in zip(data.sentences(), preds):
        sid = exmp['sid']
        e1 = exmp['e1']
        e2 = exmp['e2']
        if tag != 'null':
            print(sid, e1, e2, tag, sep="|", file=outf)
    outf.close()

def evaluation(datadir, outfile, run):
    run = evaluator.evaluate("DDI", datadir, outfile, run)
    sF1 = run["eval/f1_score"].fetch()
    sP = run["eval/precision"].fetch()
    sR = run["eval/recall"].fetch()
    return sF1, sP, sR


def tokenize_sentences(sentences, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for s in sentences:
        sentence_list = [w['form'] for w in s['sent']]
        sentence = ' '.join(sentence_list)
        #inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', truncation=True, max_length=max_len)
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
    return torch.cat(input_ids), torch.cat(attention_masks)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, run, traindir, valdir):
    for epoch in range(num_epochs):
        print("Epoch: ",epoch+1)
        model.train()
        train_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        total_predicted_val = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                total_predicted_val.extend(predicted.cpu().numpy())
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        Y = [codes.idx2label(p) for p in total_predicted_val]
        output_interactions(valdata, Y, outfile)
        F1_val, P_val, R_val = evaluation(valdir, outfile, run)
        run['Train Loss'].append(train_loss/len(train_loader))
        run['Validation Loss'].append(val_loss/len(val_loader))
        run['Validation F1'].append(F1_val)
        run['Validation Precision'].append(P_val)
        run['Validation Recall'].append(R_val)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}, Validation F1: {F1_val}")
        checkpoint_path = f"checkpoints_bert_155/model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
    return model

trainfile = 'train.pck'
validationfile = 'devel.pck'
modelname = 'model.pth'
outfile = 'out.txt'
testfile = 'test.pck'
transformer = "bert"
validationdir = "data/devel"
testdir = "data/test"

max_len = 100
num_class = 5
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)
testdata = Dataset(testfile)
codes = Codemaps(traindata, max_len)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

print("Tokenizing Training Data")
Xt, Xt_masks = tokenize_sentences(traindata.sentences(), tokenizer, max_len)

Yt = torch.tensor(codes.encode_labels(traindata))
print("Tokenizing Val Data")
Xval, Xval_masks = tokenize_sentences(valdata.sentences(), tokenizer, max_len)
Yval = torch.tensor(codes.encode_labels(valdata))

# Xtest, Xtest_masks = tokenize_sentences(testdata.sentences(), tokenizer, max_len)
# Ytest = torch.tensor(codes.encode_labels(testdata))

batch_size=150
epochs=10

train_dataset = TensorDataset(Xt, Xt_masks, Yt)
val_dataset = TensorDataset(Xval, Xval_masks, Yval)
#test_dataset = TensorDataset(Xtest, Xtest_masks, Ytest)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
#test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertWithSoftmax(5).to(device)
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5).to(device)
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5).to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

run = neptune.init_run(
    project="ikerjansa/DDI-MUD",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNmY4OGIzZC00YTAwLTQyNzctYTIxOC0yMjBlYzkwMDljYzAifQ==",
)
params = {"transformer": transformer,"max_length": max_len,"optimizer": "AdamW", "batch_size": batch_size, "epochs": epochs}
run["parameters"] = params

print("Training Model")
#model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, run)
model.load_state_dict(torch.load('checkpoints_bert_155/model_epoch_10.pth'))

model.eval()
preds = []
with torch.no_grad():
    for input_ids, attention_mask, labels in val_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask)
        preds.extend(outputs.cpu().numpy())
preds = np.array(preds)
Y = [codes.idx2label(np.argmax(p)) for p in preds]
output_interactions(valdata, Y, outfile)
evaluation(validationdir, outfile, run)
