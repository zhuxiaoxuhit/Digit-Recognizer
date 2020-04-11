import torch.optim as optim
from model import Model as Net 
from dataProvider import DataProvider
import torch.nn as nn
import torch.nn.functional as F
import torch
from datetime import datetime 
import pandas as pd
import numpy as np
import os
def main():
    save_every = 1000 
    save_path = 'saved_models'
    not os.path.exists(save_path) and os.mkdir(save_path)   
    
    device = torch.device("cuda:2")
    net = Net()
    #if torch.cuda.device_count() > 1:
    #   print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   net = nn.DataParallel(net)
    #net =nn.DataParallel(net,device_ids=[2,3]) # multi-GPU
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
    train_path = 'datasets/train.csv'
    full_train_data = pd.read_csv(train_path).values
    valid_data  = torch.from_numpy(full_train_data[1:int(0.01*full_train_data.shape[0]),1:].reshape(-1,1,28,28).astype(np.float32))
    train_data  = torch.from_numpy(full_train_data[int(0.01*full_train_data.shape[0]):,1:].reshape(-1,1,28,28).astype(np.float32))
    valid_label = torch.from_numpy(full_train_data[1:int(0.01*full_train_data.shape[0]),0].reshape(-1).astype(np.int64))
    train_label = torch.from_numpy(full_train_data[int(0.01*full_train_data.shape[0]):,0].reshape(-1).astype(np.int64))

    for epoch in range(10000):  # loop over the dataset multiple times
        #trainloader = DataProvider(train_path)
        #testloader = DataProvider(test_path)
            
        running_loss = 0.0 
        # get the inputs
        inputs, labels = train_data,train_label
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
            
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        print('epoch: ',epoch,'train_loss: ',train_loss)    
        with torch.no_grad():
            inputs, labels = valid_data,valid_label
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            val_loss = criterion(outputs, labels)
            print('val_loss: ',val_loss)
                
        if epoch % save_every == 0 and epoch != 0 : 
            torch.save({'epoch':epoch+1,'state_dic':net.state_dict(),'optimizer':optimizer.state_dict(),'train_loss':train_loss,'val_loss':val_loss,},save_path+'/'+str(epoch)+'.pt')
            #torch.save(net,save_path+'/'+str(epoch)+'.pt')

if __name__ == "__main__":
    main()

