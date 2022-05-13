#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
import ray
import sys
sys.path.append("../")
from consistent_hashing import ConsistentHash

import math 
import json
import os
import time
import argparse
parser = argparse.ArgumentParser(description='Model Parallel')
parser.add_argument('-ns', '--num_servers',type=int, help='an integer for the number of servers to use')
parser.add_argument('-o', '--output_dir',type=str, help='an integer for the number of serve')
args = parser.parse_args()

def get_data_loader():
    
    """Safely downloads data. Returns training/validation set dataloader."""
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    
    class MNISTEvenOddDataset(torch.utils.data.Dataset):
        def __init__(self, ready_data):
            self.img_data = ready_data.data
            self.labels = ready_data.targets % 2
        
        def __len__(self):
            return len(self.labels)
    
        def __getitem__(self, ind):
            return torch.true_divide(self.img_data[ind].view(-1, 28 * 28).squeeze(), 255), torch.tensor([self.labels[ind]])


    
    with FileLock(os.path.expanduser("~/data.lock")):
        
        train_dataset = datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            )
        
        test_dataset = datasets.MNIST("~/data", train=False, transform=mnist_transforms)
        
        train_loader = torch.utils.data.DataLoader(
            MNISTEvenOddDataset(train_dataset),
            batch_size=128,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
             MNISTEvenOddDataset(test_dataset),
            batch_size=128,
            shuffle=False,
        )
    return train_loader, test_loader


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            outputs = nn.Sigmoid()(model(data))
            #_, predicted = torch.max(outputs.data, 1)
            predicted = outputs > 0.5
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


# In[2]:


class LinearNet(nn.Module):
    """Small Linear Network for MNIST."""

    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc_weights = nn.ParameterList([nn.Parameter(torch.empty(1)) for weight in range(784)])
        init_fc = [nn.init.uniform_(x) for x in self.fc_weights]
        
        self.fc_bias = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.fc_bias)
        
    #def __init__(self):
    #    super(LinearNet, self).__init__()
    #    self.fc = nn.Linear(28*28, 1)
    #    nn.init.normal(self.fc.weight)

    #def forward(self, x):
    #    x = self.fc(x)
    #    return x
    
    def forward(self, x):
        #fc_layer = torch.cat(tuple(self.fc_weights)).unsqueeze(0)
        #x = x @ fc_layer.T + self.fc_bias
        for i, param in enumerate(self.fc_weights):
            if i==0:
                p=x[:,i]*param
            else:
                p += x[:,i]*param
        x = p.unsqueeze(1) + self.fc_bias
        return x
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, keys, weights):  
        self.load_state_dict({keys[i]:weights[i] for i in range(len(keys))})
        
    def get_gradients(self, keys):
        grads = {}
        #grads = []
        for name, p in self.named_parameters():
            if name in keys:
                grad = None if p.grad is None else p.grad.data.cpu().numpy()
                grads[name] = grad
                #grads.append(grad)
        return [grads[key] for key in keys]
        #return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


# In[3]:


@ray.remote  
class ParameterServer(object):
    def __init__(self, keys, values):
        self.weights = dict(zip(keys, values))

    def apply_gradients(self, keys, lr, *values):

        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*values)
        ]
    
        idx = 0
        for key, value in zip(keys, summed_gradients):
            self.weights[key] -= lr * torch.from_numpy(summed_gradients[idx])
            idx+=1
            
      #  return [self.weights[key] for key in keys]

    def get_weights(self, keys):
        return [self.weights[key] for key in keys]


# In[4]:


@ray.remote
class DataWorker(object):
    def __init__(self, keys):
        self.model = LinearNet()
        self.data_iterator = iter(get_data_loader()[0])
        self.keys = keys
        self.key_set = set(self.keys)
        for key, value in dict(self.model.named_parameters()).items():
            if key not in self.key_set:
                value.requires_grad=False

        
    def update_weights(self, keys, weights):
        self.model.set_weights(keys, weights)

    def compute_gradients(self):
        #self.model.set_weights(keys, weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        
        self.model.zero_grad()
        output = self.model(data)
        loss = nn.BCEWithLogitsLoss()(output, target.float())
        loss.backward()
        
        return self.model.get_gradients(self.keys)


iterations = 500
num_workers = 1 # number of workers per server
num_servers = args.num_servers # number of servers
hashes_per_server = 50

def Scheduler(num_servers, hashes_per_server=50):
    
    model = LinearNet()
    key_values = model.get_weights()
    keys = np.array(list(key_values.keys()))
    #print(keys)
    #print(key_values)
    values = [key_values[key] for key in keys]
    #values = [key_values[key] for key in keys]
    
    key_indices = {key: x for x, key in enumerate(keys)}
   
    # distributing weights across servers - do this using consistency hashing
    server_ids = ["server" + str(ind) for ind in range(num_servers)]
    hasher = ConsistentHash(keys, server_ids, hashes_per_server)
    servers = [ParameterServer.remote(keys[[key_indices[key] for key in hasher.get_keys_per_node()[serv]]], 
                                      [values[key_indices[key]] for key in hasher.get_keys_per_node()[serv]]) for serv in server_ids]
    # servers = [ParameterServer.remote(keys[0:1], values[0:1]), ParameterServer.remote(keys[1:2], values[1:2])]
    
    return servers, keys, model, hasher.get_keys_per_node()

servers, keys, model, weight_assignments =  Scheduler(num_servers, hashes_per_server)
ray.init(ignore_reinit_error=True)

# creating equal workers per server

workers = [[DataWorker.remote(weight_assignments["server" + str(j)]) for i in range(num_workers)] for j in range(num_servers)]



len(weight_assignments["server0"])


test_loader = get_data_loader()[1]

print("Running synchronous parameter server training.")
lr=0.1

# we need to get a new keys order because we are not assuming a ordering in keys
current_weights = []
keys_order = []
accuracy_per_iteration=[]

for j in range(num_servers):
    keys_order.extend(weight_assignments["server" + str(j)])
    current_weights.extend(ray.get(servers[j].get_weights.remote(weight_assignments["server" + str(j)]))) 

start_time=time.time()
for i in range(iterations):
    
    # sync all weights on workers
    if i % 1 == 0:
        current_weights = []
        keys_order = []

        # get weights from server
        for j in range(num_servers):
            keys_order.extend(weight_assignments["server" + str(j)])
            current_weights.extend(ray.get(servers[j].get_weights.remote(weight_assignments["server" + str(j)]))) 
   
        # update weights on all workers
        for j in range(num_servers):
            for  idx  in range(num_workers):
                workers[j][idx].update_weights.remote(keys_order, current_weights)
    
    
    # use local cache of weights and get gradients from workers
    gradients = [[workers[j][idx].compute_gradients.remote() for  idx  in range(num_workers)] for j in range(num_servers)]
    
    # Updates gradients to specfic parameter servers
    [servers[j].apply_gradients.remote(weight_assignments["server" + str(j)], lr, *gradients[j]) for j in range(num_servers)]
           
    if i % 10 == 0:
        # Evaluate the current model.

        for j in range(num_servers):
            keys_order.extend(weight_assignments["server" + str(j)])
            current_weights.extend(ray.get(servers[j].get_weights.remote(weight_assignments["server" + str(j)]))) 
   
        # we are once again using the server to key mapping to set the weight back
        model.set_weights(keys_order, current_weights)
        accuracy = evaluate(model, test_loader)
        
        print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
        accuracy_per_iteration.append([i, accuracy])

print("Final accuracy is {:.1f}.".format(accuracy))
# Clean up Ray resources and processes before the next example.
ray.shutdown()
end_time=time.time()                       

time_taken=end_time-start_time
d = {"accuracy":accuracy_per_iteration, "time":time_taken}
out_file = open(os.path.join(args.output_dir,"num_servers_"+str(num_servers)+".json"),"w")
json.dump(d,out_file) 


