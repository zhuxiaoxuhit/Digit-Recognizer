import torch.optim as optim
from model import Model as Net 
from dataProvider import DataProvider
import torch.nn as nn
import torch.nn.functional as F
import torch
from datetime import datetime
import pandas as pd
import numpy as np


def load_model():
    device = torch.device("cuda:2")
    model = Net()
    checkpoint = torch.load('saved_models/3000.pt')
    #checkpoint.eval()
    model.load_state_dict(checkpoint['state_dic'])
    model.eval()
    model.to(device)
    return device,model

def load_data():
    test_path = 'datasets/test.csv'
    test_data = pd.read_csv(test_path).values.reshape(-1,1,28,28).astype(np.float32)
    return test_data

def inference(test_data,device,model):
    inputs = torch.from_numpy(test_data).to(device)
    outputs = model.forward(inputs)
    _,outputs = torch.max(outputs,1)
    outputs = outputs.cpu().detach().numpy()
    print(outputs[1])


    df = pd.read_csv('datasets/sample_submission.csv')
    df = df.set_index('ImageId')
    df['Label'] = outputs
    df.to_csv('sample_submission.csv')
    



def main():
    device,model = load_model()
    data = load_data()
    inference(data,device,model)


if __name__ == '__main__':
    main()

