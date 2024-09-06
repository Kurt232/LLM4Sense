import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LIMUBertEncoder
from torch.utils.data import Dataset
import time
import copy
import json
import os

class ClassifierCNN2D(nn.Module):
    def __init__(self, num_cnn=3, conv_io=[[8, 16, 3, 0], [16, 8, 3, 0], [8, 4, 3, 0]], pool=[3, [1, 2], 0], num_linear=2, flat_num=192, linear_io=[[0, 12], [12, 0]], output=None):
      super().__init__()
      for i in range(num_cnn):
          if i == 0:
              self.__setattr__('cnn' + str(i), nn.Conv2d(1, conv_io[i][1], conv_io[i][2], padding=conv_io[i][3]))
          else:
              self.__setattr__('cnn' + str(i), nn.Conv2d(conv_io[i][0], conv_io[i][1], conv_io[i][2], padding=conv_io[i][3]))
          self.__setattr__('bn' + str(i), nn.BatchNorm2d(conv_io[i][1]))
      self.pool = nn.MaxPool2d(pool[0], stride=pool[1], padding=pool[2])
      self.flatten = nn.Flatten()
      for i in range(num_linear):
          if i == 0:
              self.__setattr__('lin' + str(i), nn.Linear(flat_num, linear_io[i][1]))
          elif output is not None and i == num_linear - 1:
              self.__setattr__('lin' + str(i), nn.Linear(linear_io[i][0], output))
          else:
              self.__setattr__('lin' + str(i), nn.Linear(linear_io[i][0], linear_io[i][1]))
      self.activ = False
      self.dropout = True
      self.num_cnn = num_cnn
      self.num_linear = num_linear

    def forward(self, input_seqs, training=False):
        h = input_seqs.unsqueeze(1)
        for i in range(self.num_cnn):
            cnn = self.__getattr__('cnn' + str(i))
            bn = self.__getattr__('bn' + str(i))
            h = cnn(h)
            if self.activ:
                h = F.relu(h)
            h = bn(self.pool(h))
            # h = self.pool(h)
        h = self.flatten(h)
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h

class BERTClassifier(nn.Module):

    def __init__(self, frozen_bert=True):
        super().__init__()
        self.transformer = LIMUBertEncoder().transformer
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = ClassifierCNN2D()

    def forward(self, input_seqs, training=False): #, training
        h = self.transformer(input_seqs)
        h = self.classifier(h, training)
        return h

import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score

import random

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def get_device(gpu):
  if gpu is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(
        "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  print("%s (%d GPUs)" % (device, n_gpu))
  return device

batch_size = 32
lr = 0.001
epochs = 500
finetune_rate = 0.1

def stat_acc_f1(label, label_estimated):
  # label = np.concatenate(label, 0)
  # results_estimated = np.concatenate(results_estimated, 0)
  f1 = f1_score(label, label_estimated, average='macro')
  acc = np.sum(label == label_estimated) / label.size
  return acc, f1

def evaluate(model, dataloader, criterion, device):
  model.eval()  # Set the model to evaluation mode
  
  results = []  # prediction results
  labels = []
  total_loss = 0
  with torch.no_grad():  # No need to track gradients
    for data, target in dataloader:
      data = data.to(device)
      target = target.to(device)
      logits, outputs = model(data)
      loss = criterion(logits, target)
      total_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      results.append(predicted)
      labels.append(target)
  
  average_loss = total_loss / len(dataloader)  
  acc, f1 = stat_acc_f1(torch.cat(labels, 0).cpu().numpy(), torch.cat(results, 0).cpu().numpy())

  return average_loss, acc, f1

def load_data(data: list):
    data = np.array(data, dtype=np.float32)
    acc_norm = 9.8
    data[:, :3] = data[:, :3] / acc_norm
    return torch.from_numpy(data)

def split_vali_data(data, label, train_rate=0.8):
    arr = np.arange(len(data))
    np.random.shuffle(arr)
    train_index = int(len(data) * train_rate)
    train_data_index = arr[:train_index].tolist()
    dev_data_index = arr[train_index:].tolist()
    
    train_data = [data[i] for i in train_data_index]
    train_label = [label[i] for i in train_data_index]
    dev_data = [data[i] for i in dev_data_index]
    dev_label = [label[i] for i in dev_data_index]
    return train_data, train_label, dev_data, dev_label

class IMUDataset(Dataset):

    def __init__(self, data, label):
        """
        Args:
            data (numpy.array): The IMU data of shape [num_samples, window_size, num_features]
        """
        self.data = data
        self.label = label

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        return instance, torch.from_numpy(
            np.array(self.label[idx])).long()

if __name__ == "__main__":
  # Load the dataset
  label_index = {
        "climbing stairs": 0,
        "sitting": 1,
        "biking": 2,
        "standing": 3,
        "walking": 4,
        "descending stairs": 5,
    }
  train_data = []
  train_label = []
  raw_train = json.load(open('../../openaqa/data/hhar/train_toy.json', 'r'))
  for d in raw_train:
    if d['task'] != 'close-ended question':
      continue
    train_data.append(d['imu_input'])
    label = d['output'].split('\n')[0][11:]
    train_label.append(label_index[label])
    
  test_data = []
  test_label = []
  raw_test = json.load(open('../../openaqa/data/hhar/test_toy.json', 'r'))
  for d in raw_test:
    test_data.append(d['imu_input'])
    label = d['output'].split('\n')[0][11:]
    test_label.append(label_index[label])
  
  # splitting the dataset
  set_seeds(40)  # must be the same as the one in train.py
  train_data, train_label, vali_data, vali_label = split_vali_data(
      train_data, train_label, train_rate=0.9)
  
  train_data = load_data(train_data)
  vali_data = load_data(vali_data)
  test_data = load_data(test_data)
  
  dataset_train = IMUDataset(train_data, train_label)
  dataset_vali = IMUDataset(vali_data, vali_label)
  dataset_test = IMUDataset(test_data, test_label)
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_vali = DataLoader(dataset_vali, batch_size=batch_size, shuffle=False)
  dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
  
  # Load the pre-trained encoder weights
  model = BERTClassifier(frozen_bert=True)

  # Setup for classifier training
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  device = get_device("0")

  eval_mdl_path = '/data/wenhao/wjdu/output/imu_cla_10/'
  encoder_weights = torch.load(os.path.join(eval_mdl_path, 'encoder_model.bin'), map_location='cpu')
    
  for key in encoder_weights.keys():
    if 'transformer.embed.pos_embed.weight' in key:
      print(encoder_weights[key])
  exit()
  # Training loop for the classifier
  def train_classifier(model, dataloader_train, dataloader_vali, dataloader_test, optimizer, criterion, device, model_file, epochs=10):
    model.to(device)
    
    vali_acc_best = 0.0
    best_stat = None
    for epoch in range(epochs):
      model.train()
      loss_sum = 0.0  # the sum of iteration losses to get average loss in every epoch
      time_sum = 0.0
      for data, target in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{epochs}"):
        data = data.to(device)
        target = target.to(device)
        start_time = time.time()
        
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits, target)
        
        loss.backward()
        optimizer.step()
        time_sum += time.time() - start_time
        loss_sum += loss.item()
      
      train_loss, train_acc, train_f1 = evaluate(model, dataloader_train, criterion, device)
      vali_loss, vali_acc, vali_f1 = evaluate(model, dataloader_vali, criterion, device)
      test_loss, test_acc, test_f1 = evaluate(model, dataloader_test, criterion, device)
      
      # print
      print(f"Epoch {epoch+1}/{epochs} ({time_sum:.2f}s):")
      print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f}")
      print(f"  Vali loss: {vali_loss:.4f}, acc: {vali_acc:.4f}, f1: {vali_f1:.4f}")
      print(f"  Test loss: {test_loss:.4f}, acc: {test_acc:.4f}, f1: {test_f1:.4f}")
      
      if vali_acc > vali_acc_best:
        vali_acc_best = vali_acc
        best_stat = (train_loss, vali_loss, test_loss, train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
    
    print('The Total Epoch have been reached.')
    print('Loss:  %0.3f/%0.3f/%0.3f\nBest Accuracy: %0.3f/%0.3f/%0.3f\nF1: %0.3f/%0.3f/%0.3f' % best_stat)

  # Example training call
  train_classifier(model, dataloader_train, dataloader_vali, dataloader_test, optimizer, criterion, device, model_file, epochs)