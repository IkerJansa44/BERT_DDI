from dataset import *
from codemaps import *
from tensorflow.keras import Input
from contextlib import redirect_stdout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Flatten, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, Softmax

import sys
import neptune
import numpy as np
import matplotlib.pyplot as plt
import util.evaluator as evaluator
import nltk
nltk.download('punkt')


def build_network(n_labels):
    embedding_dim = 768
    inputs = Input(shape=(max_len,embedding_dim))
    conv = Conv1D(filters=30, kernel_size=2, strides=1, activation='relu', padding='same')(inputs)
    flat = Flatten()(conv)
    outputs = Dense(n_labels, activation='softmax')(flat)
    #y = Bidirectional(LSTM(units=200, return_sequences=False))(model)  #  biLSTM
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Predict
def output_interactions(data, preds, outfile) :

   #print(testdata[0])
   outf = open(outfile, 'w')
   for exmp,tag in zip(data.sentences(),preds) :
      sid = exmp['sid']
      e1 = exmp['e1']
      e2 = exmp['e2']
      if tag!='null' :
         print(sid, e1, e2, tag, sep="|", file=outf)

   outf.close()

def evaluation(datadir,outfile,run) :
    run = evaluator.evaluate("DDI", datadir, outfile,run)
    # Data for the barplot

    sF1 = run["eval/f1_score"].fetch()
    sP = run["eval/precision"].fetch()
    sR = run["eval/recall"].fetch()
    labels = ['F1', 'Precision', 'Recall']
    values = [sF1, sP, sR]

    # Creating the barplot
    plt.bar(labels, values)

    # Adding labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Evaluation Metrics')
    plt.ylim(0, 1)
    run["eval/plot"].upload(plt.gcf())
    plt.show()

trainfile='train.pck'
validationfile='devel.pck'
modelname ='model.keras'
outfile ='out.txt'
testfile='test.pck'
validationdir='/home/jaik194/07-DDI-nn/data/devel'
embedding_file_Xt = 'Xt_150_.npy' 
embedding_file_Xv = 'Xv_150_.npy' 
embeddings = "Bert"

max_len = int(embedding_file_Xt.split("_")[1])
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)
codes = Codemaps(traindata, max_len)
Xt = np.load(embedding_file_Xt)
Yt = codes.encode_labels(traindata)
Xv = np.load(embedding_file_Xv)
Yv = codes.encode_labels(valdata)

Xt = np.squeeze(Xt, axis=1)
Xv = np.squeeze(Xv, axis=1)
print("Shape input: ",Xt.shape)
print("Shape labels: ",Yt.shape)

model = build_network(5)
optimizer = 'adam'
model.compile(optimizer=optimizer ,metrics=["accuracy"], loss="categorical_crossentropy")
model.build([(None,max_len),(None,max_len),(None,max_len)])
with redirect_stdout(sys.stderr) :
   model.summary()

batch_size=32
epochs=2

run = neptune.init_run(
    project="ikerjansa/DDI-MUD",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNmY4OGIzZC00YTAwLTQyNzctYTIxOC0yMjBlYzkwMDljYzAifQ==",
)
params = {"embeddings": embeddings,"max_length": max_len,"optimizer": optimizer, "batch_size": batch_size, "epochs": epochs,"model_summary": model.summary}
run["parameters"] = params
with redirect_stdout(sys.stderr) :
   history = model.fit(Xt, Yt, batch_size=batch_size, epochs=epochs, validation_data=(Xv,Yv), verbose=1)
model.save(modelname)

epoch_values = history.history

# Print the history data for each epoch
for epoch, metrics in enumerate(zip(epoch_values['loss'], epoch_values['val_loss'], epoch_values.get('accuracy', []), epoch_values.get('val_accuracy', []))):
    run["train/epoch_loss"].append(metrics[0])
    run["train/epoch_val_loss"].append(metrics[1])
    run["train/epoch_accuracy"].append(metrics[2])
    run["train/epoch_val_accuracy"].append(metrics[3])

Y = model.predict(Xv)
Y = [codes.idx2label(np.argmax(s)) for s in Y]

# extract entities
output_interactions(valdata, Y, outfile)

# evaluate
evaluation(validationdir,outfile,run)

run.stop()  