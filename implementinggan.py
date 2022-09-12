



import pandas as pd
clean_data=pd.read_csv('../TweetvsRetweet/Data_Cleaned.csv')

clean_data=clean_data.drop('Unnamed: 0',axis=1)
comment_clean=clean_data.loc[clean_data['check']==1]
tweet_clean=clean_data.loc[clean_data['check']==0]
comment=comment_clean.reset_index()
tweet=tweet_clean.reset_index()

comment=comment[0:len(tweet)]

'''
!pip install nlpaug 

import nlpaug.augmenter.word as naw
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
'''

'''
a=len(tweet)
for i in range(1144):
  text=tweet['tweet'][i]
  augmented_text = aug.augment(text)
  tweet['tweet'][a]=augmented_text
  a=a+1
print(augmented_text)
'''




import torch
import torch
import torch, gc


import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import *
torch.cuda.empty_cache()
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from transformers import BertModel, BertTokenizer
from transformers import AdamW

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

"""Load the Transformer model

"""

import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
model_name = "bert-base-cased"
#model_name = "bert-base-uncased"
transformer = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""Input Parameters"""

max_seq_length=50

comment

from torch.functional import Tensor
def data_loader(df,batch_size,do_shuffle=False):
  tweets=df.tweet
  tweets = [tweet + " [SEP] [CLS]" for tweet in tweets]
  labels=df.check
  sen_labels=df['class']
  labels=labels.to_numpy()
  sen_labels=sen_labels.to_numpy()
 
 #Tokenizing the tweets
  tokenized_tweets=[tokenizer.tokenize(sent) for sent in tweets]
  tokenized_inputs=[tokenizer.convert_tokens_to_ids(x) for x in tokenized_tweets]
  inputs = pad_sequences(tokenized_inputs, maxlen=max_seq_length, dtype="long", truncating="post", padding="post") #Padding the sentences to be the size of length 60

  # Create a mask of 1s for each token followed by 0s for padding
  masks = []
  for seq in inputs:
    seq_mask = [float(i>0) for i in seq]
    masks.append(seq_mask)
  
  #Convert the inputs to tensor
  tensor_inputs=torch.tensor(inputs)
  tensor_labels=torch.tensor(labels)
  tensor_masks=torch.tensor(masks)
  tensor_sen_labels=torch.tensor(sen_labels)
  #Create a TensorDataset
  dataset=TensorDataset(tensor_inputs,tensor_masks,tensor_labels)
  if do_shuffle:
    sampler = RandomSampler
  else:
    sampler = SequentialSampler
  
  return DataLoader(dataset,sampler=sampler(dataset),batch_size=batch_size)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

input_tweet=data_loader(tweet,8,do_shuffle=False)
input_comment=data_loader(comment,8,do_shuffle=False)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

config = AutoConfig.from_pretrained(model_name)
hidden_size = int(config.hidden_size)
discriminator=Discriminator()
if torch.cuda.is_available():    
  discriminator.cuda()
  transformer.cuda()

training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

#models parameters
transformer_vars = [i for i in transformer.parameters()]
d_vars = transformer_vars + [v for v in discriminator.parameters()]

"""#Checking on comment and tweets"""

class Generator(nn.Module):

    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1);
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))

generator=Generator(768)
discriminator=Discriminator(768)
generator=generator.to(device)
discriminator=discriminator.to(device)

# Optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.00001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

# loss
loss = nn.BCELoss()

n_epochs=3
comment_id=[]
for ep_i in range(0,n_epochs):
  print('Epoch: ',ep_i)
  for i in range(len(input_tweet)):
    #Generating vector representation for the original tweets
    for step1,batch in enumerate(input_tweet):
      b_input_ids = batch[0].to(device)
      b_masks=batch[1].to(device)
      labels_1=batch[2].to(device)
      labels_1=torch.tensor(labels_1,dtype=torch.float32)
      #sen_labels1=batch[3]
    model_outputs1 = transformer(b_input_ids, attention_mask=b_masks)
    hidden_states1=model_outputs1[-1] # This is representataion for the original tweets
    
    #Generating vector representation for the comments
    for step2,batch2 in enumerate(input_comment):
      b_input_ids2 = batch2[0].to(device)
      b_masks2=batch2[1].to(device)
      labels_2=batch2[2].to(device)
      labels_2=torch.tensor(labels_2,dtype=torch.float32)
      #sen_labels2=batch[3].to
    model_outputs2 = transformer(b_input_ids2, attention_mask=b_masks2)
    hidden_states2=model_outputs2[-1] # This is representation for the comments, these are meant to be input for the generator


    generated_data = generator(hidden_states2) #Generating new samples from the comments
    
    
    generator_discriminator_out=discriminator(generated_data)
    generator_discriminator_out=torch.flatten(generator_discriminator_out)
    generator_discriminator_out=torch.tensor(generator_discriminator_out,dtype=torch.float32)
    generator_loss=loss(generator_discriminator_out,labels_1)
    generator_loss=Variable(generator_loss,requires_grad=True)
    generator_loss.backward()
    generator_optimizer.step()
    

    discriminator_optimizer.zero_grad()
    true_discriminator_out = discriminator(hidden_states1)
    true_discriminator_out=torch.flatten(true_discriminator_out)
    true_discriminator_out=torch.tensor(true_discriminator_out,dtype=torch.float32)
    true_discriminator_loss = loss(true_discriminator_out, labels_1)


    generator_discriminator_out = discriminator(generated_data.detach())
    generator_discriminator_out=torch.flatten(generator_discriminator_out)
    generator_discriminator_loss = loss(generator_discriminator_out, labels_2)
    discriminator_loss = (true_discriminator_loss + generator_discriminator_loss)/2

    discriminator_loss.backward()
    discriminator_optimizer.step()
    
    comment_id.append(generated_data)

  print('Generator loss',generator_loss.item())
  print('Discriminator loss',discriminator_loss.item())
  #noise = torch.zeros(32, 768, device=device).uniform_(0, 1)
  #feature_loss=torch.mean(torch.pow(torch.mean(hidden_states1, dim=0) - torch.mean(noise, dim=0), 2))
 # print(torch.mean(hidden_states1, dim=0))
  #print(torch.mean(hidden_states2, dim=0))

