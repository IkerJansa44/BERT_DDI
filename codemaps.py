import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import wordnet

from dataset import *

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet')

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None) :

        if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        words = set([])
        lc_words = set([])
        sufs = set([])
        suf3 = set([])
        suf4 = set([])
        pref3 = set([])
        pref4 = set([])
        labels = set([])
        mtoc_l = set([])
        oc_l = set([])
        punctuations = set([])
        digits = set([])
        postags = set([])  
        #drugs = set([])
        #brands = set([])
        #groups = set([])
        #hsdbs = set([])
        
        # Maybe ficar si el token es al final de la sentence o al principi
        # O ficar un pretrained model
        
        for s in data.sentences() :
            for t in s :
                form = t['form']
                lc_form = t['lc_form']
                
                words.add(form)
                sufs.add(lc_form[-self.suflen:])
                labels.add(t['tag'])
                lc_words.add(lc_form)
                suf3.add(lc_form[-3:])
                suf4.add(lc_form[-4:])
                pref3.add(lc_form[:3])
                pref4.add(lc_form[:4:])
                
                if sum(1 for c in form if c.isupper()) > 1:
                    mtoc_l.add(form)
                elif sum(1 for c in form if c.isupper()) == 1:
                    oc_l.add(form)
                    
                if any(c in string.punctuation for c in form):
                    punctuations.add(form)
                    
                if any(c.isdigit() for c in form):
                    digits.add(form)
                
                pos_tag = nltk.pos_tag([form])[0][1]
                # Map POS tag to WordNet POS tag
                if pos_tag.startswith('J'):
                    wn_pos = wordnet.ADJ
                elif pos_tag.startswith('V'):
                    wn_pos = wordnet.VERB
                elif pos_tag.startswith('N'):
                    wn_pos = wordnet.NOUN
                elif pos_tag.startswith('R'):
                    wn_pos = wordnet.ADV
                else:
                    wn_pos = wordnet.NOUN  # Noun by default
                postags.add(wn_pos)
                

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes
        
        self.lc_index = {lc: i+2 for i,lc in enumerate(list(lc_words))}
        self.lc_index['PAD'] = 0
        self.lc_index['UNK'] = 1
        
        ############################################################################################
        
        self.suf3_index = {s: i+2 for i,s in enumerate(list(suf3))}
        self.suf3_index['PAD'] = 0
        self.suf3_index['UNK'] = 1
        
        self.suf4_index = {s: i+2 for i,s in enumerate(list(suf4))}
        self.suf4_index['PAD'] = 0
        self.suf4_index['UNK'] = 1
        
        self.pref3_index = {p3: i+2 for i,p3 in enumerate(list(pref3))}
        self.pref3_index['PAD'] = 0
        self.pref3_index['UNK'] = 1
        
        self.pref4_index = {p4: i+2 for i,p4 in enumerate(list(pref4))}
        self.pref4_index['PAD'] = 0
        self.pref4_index['UNK'] = 1
        
        ############################################################################################
        
        self.mtoc_index = {capital: i+2 for i,capital in enumerate(list(mtoc_l))}
        self.mtoc_index['PAD'] = 0
        self.mtoc_index['UNK'] = 1
        
        self.oc_index = {capital: i+2 for i,capital in enumerate(list(oc_l))}
        self.oc_index['PAD'] = 0
        self.oc_index['UNK'] = 1
        
        self.punctuations_index = {punc: i+2 for i,punc in enumerate(list(punctuations))}
        self.punctuations_index['PAD'] = 0
        self.punctuations_index['UNK'] = 1
        
        self.digit_index = {dig: i+2 for i,dig in enumerate(list(digits))}
        self.digit_index['PAD'] = 0
        self.digit_index['UNK'] = 1
        
        self.postag_index = {pos: i+2 for i,pos in enumerate(list(postags))}
        self.postag_index['PAD'] = 0
        self.postag_index['UNK'] = 1
        
        ############################################################################################

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)


    ## --------- encode X from given data ----------- 
    def encode_words(self, data) :        
        # encode and pad sentence words
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])
        # encode and pad suffixes
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])
        # encode and pad lowercase form words
        Xlc = [[self.lc_index[w['lc_form']] if w['lc_form'] in self.lc_index else self.lc_index['UNK'] for w in s] for s in data.sentences()]
        Xlc = pad_sequences(maxlen=self.maxlen, sequences=Xlc, padding="post", value=self.lc_index['PAD'])
        # encode and pad suf3
        Xs3 = [[self.suf3_index[w['lc_form'][-3:]] if w['lc_form'][-3:] in self.suf3_index else self.suf3_index['UNK'] for w in s] for s in data.sentences()]
        Xs3 = pad_sequences(maxlen=self.maxlen, sequences=Xs3, padding="post", value=self.suf3_index['PAD'])
        # encode and pad suf4
        Xs4 = [[self.suf4_index[w['lc_form'][-4:]] if w['lc_form'][-4:] in self.suf4_index else self.suf4_index['UNK'] for w in s] for s in data.sentences()]
        Xs4 = pad_sequences(maxlen=self.maxlen, sequences=Xs4, padding="post", value=self.suf4_index['PAD'])
        # encode and pad pref3
        Xp3 = [[self.pref3_index[w['lc_form'][:3]] if w['lc_form'][:3] in self.pref3_index else self.pref3_index['UNK'] for w in s] for s in data.sentences()]
        Xp3 = pad_sequences(maxlen=self.maxlen, sequences=Xp3, padding="post", value=self.pref3_index['PAD'])
        # encode and pad pref4
        Xp4 = [[self.pref4_index[w['lc_form'][:4]] if w['lc_form'][:4] in self.pref4_index else self.pref4_index['UNK'] for w in s] for s in data.sentences()]
        Xp4 = pad_sequences(maxlen=self.maxlen, sequences=Xp4, padding="post", value=self.pref4_index['PAD'])
        #encode and pad mtoc
        Xmcap = [[self.mtoc_index[w['form']] if w['form'] in self.mtoc_index else self.mtoc_index['UNK'] for w in s] for s in data.sentences()]
        Xmcap = pad_sequences(maxlen=self.maxlen, sequences=Xmcap, padding="post", value=self.mtoc_index['PAD'])
        #encode and pad oc
        Xscap = [[self.oc_index[w['form']] if w['form'] in self.oc_index else self.oc_index['UNK'] for w in s] for s in data.sentences()]
        Xscap = pad_sequences(maxlen=self.maxlen, sequences=Xscap, padding="post", value=self.oc_index['PAD'])
        #encode and pad punctuations
        Xpunc = [[self.punctuations_index[w['form']] if w['form'] in self.punctuations_index else self.punctuations_index['UNK'] for w in s] for s in data.sentences()]
        Xpunc = pad_sequences(maxlen=self.maxlen, sequences=Xpunc, padding="post", value=self.punctuations_index['PAD'])
        #encode and pad digits
        Xdig = [[self.digit_index[w['form']] if w['form'] in self.digit_index else self.digit_index['UNK'] for w in s] for s in data.sentences()]
        Xdig = pad_sequences(maxlen=self.maxlen, sequences=Xdig, padding="post", value=self.digit_index['PAD'])
        #encode and pad postags
        Xpost = []
        for s in data.sentences():
            sentence_tags = []
            for w in s:
                pos_tag = nltk.pos_tag([w['form']])[0][1]
                if pos_tag.startswith('J'):
                    wn_pos = wordnet.ADJ
                elif pos_tag.startswith('V'):
                    wn_pos = wordnet.VERB
                elif pos_tag.startswith('N'):
                    wn_pos = wordnet.NOUN
                elif pos_tag.startswith('R'):
                    wn_pos = wordnet.ADV
                else:
                    wn_pos = wordnet.NOUN

                if wn_pos in self.postag_index:
                    sentence_tags.append(self.postag_index[wn_pos])
                else:
                    sentence_tags.append(self.postag_index['UNK'])
            Xpost.append(sentence_tags)
        Xpost = pad_sequences(maxlen=self.maxlen, sequences=Xpost, padding="post", value=self.postag_index['PAD'])
        # return encoded sequences
        return [Xw,Xs,Xlc,Xs3,Xs4,Xp3,Xp4,Xmcap,Xscap,Xpunc,Xdig,Xpost]
    
    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    
    def get_n_suf3(self) :
        return len(self.suf3_index)

    def get_n_suf4(self) :
        return len(self.suf4_index)
    ## -------- get pref index size --------
    def get_n_pref3(self) :
        return len(self.pref3_index)
    
    def get_n_pref4(self) :
        return len(self.pref4_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)
    ## -------- get lc word index size ---------
    def get_lc_n_words(self):
        return len(self.lc_index)
    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError
    ## -------- get index for multiple capital letters --------
    def get_n_mtoc(self) :
        return len(self.mtoc_index)
    ## -------- get index for single capital letters --------
    def get_n_oc(self) :
        return len(self.oc_index)
    ## -------- get index for punctuations --------
    def get_n_punctuations(self):
        return len(self.punctuations_index)
    ## -------- get index for digits --------
    def get_n_digits(self):
        return len(self.digit_index)
    ## -------- get index for postags --------
    def get_n_postags(self):
        return len(self.postag_index)