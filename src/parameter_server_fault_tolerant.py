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
from time import time
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
        flatten_weights =  [item for sublist in weights for item in sublist]
        self.load_state_dict({keys[i]:flatten_weights[i] for i in range(len(keys))})
        
    def get_gradients(self, keys):
        grads = {}

        for name, p in self.named_parameters():
            if name in keys:
                grad = None if p.grad is None else p.grad.data.cpu().numpy()
                grads[name] = grad

        return [grads[key] for key in keys]

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

        return [self.weights[key] for key in keys]
    
    def add_weight(self, key, value):
        self.weights[key] = value
    
    def get_weights(self, keys):
        return [self.weights[key] for key in keys]


# In[28]:


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

        
    def update_weights(self, keys, *weights):
        self.model.set_weights(keys, weights)
        
    def update_trainable(self, keys):
        self.keys = keys
        self.key_set = set(self.keys)
        for key, value in dict(self.model.named_parameters()).items():
            if key in self.key_set:
                value.requires_grad = True
            else:
                value.requires_grad = False
       

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


# In[29]:


iterations = 500
num_workers = 1 # number of workers per server
num_servers = 5 # number of servers
hashes_per_server = 100

def Scheduler(num_servers, hashes_per_server=50):
    
    model = LinearNet()
    key_values = model.get_weights()
    keys = np.array(list(key_values.keys()))
    #print(keys)
    #print(key_values) z
    values = [key_values[key] for key in keys]
    #values = [key_values[key] for key in keys]
    
    key_indices = {key: x for x, key in enumerate(keys)}
   
    # distributing weights across servers - do this using consistency hashing
    server_ids = ["server" + str(ind) for ind in range(num_servers)]
    hasher = ConsistentHash(keys, server_ids, hashes_per_server)
    servers = [ParameterServer.remote(keys[[key_indices[key] for key in hasher.get_keys_per_node()[serv]]], 
                                      [values[key_indices[key]] for key in hasher.get_keys_per_node()[serv]]) for serv in server_ids]
    # servers = [ParameterServer.remote(keys[0:1], values[0:1]), ParameterServer.remote(keys[1:2], values[1:2])]
    
    return hasher, servers, keys, model, hasher.get_keys_per_node(), server_ids.copy()

hasher, servers, keys, model, weight_assignments, server_ids =  Scheduler(num_servers, hashes_per_server)
ray.init(ignore_reinit_error=True)

# creating equal workers per server

workers = [[DataWorker.remote(weight_assignments["server" + str(j)]) for i in range(num_workers)] for j in range(num_servers)]




# In[30]:


len(weight_assignments["server0"])


# In[ ]:





# In[31]:


test_loader = get_data_loader()[1]


# In[32]:


print("Running synchronous parameter server training.")
lr=0.1
failure_iter=60
failure_server="server4"

# we need to get a new keys order because we are not assuming a ordering in keys
current_weights = []
keys_order = []

for j in range(num_servers):
    keys_order.extend(weight_assignments["server" + str(j)])
    current_weights.extend(ray.get(servers[j].get_weights.remote(weight_assignments["server" + str(j)]))) 
curr_weights_ckpt = current_weights.copy()

time_per_iteration = []
for i in range(iterations):
 
    #start = time()
    
    if i == failure_iter:
        #Define parameters that will need to be moved
        failure_params = weight_assignments[failure_server]
        #Delete server from hash ring and reassign params
        hasher.delete_node_and_reassign_to_others(failure_server)
        weight_assignments = hasher.get_keys_per_node()
        #Update servers and workers
        num_servers -= 1
        server_ind = server_ids.index(failure_server)
        server_ids = server_ids[0 : server_ind] + server_ids[server_ind + 1 : ]
        servers = servers[0 : server_ind] + servers[server_ind + 1 : ]
        workers = workers[0 : server_ind] + workers[server_ind + 1 : ]
        #Add each relevant parameter to its new server
        server_dict = {server_ids[x]:servers[x] for x in range(len(server_ids))}
        for ind, param in enumerate(failure_params):
            server_dict[hasher.get_key_to_node_map()[param]].add_weight.remote(param, curr_weights_ckpt[server_ind][ind])
        #Update these parameters for each worker to make them trainable
        [workers[j][idx].update_trainable.remote(weight_assignments["server" + str(j)]) for  idx  in range(num_workers) for j in range(num_servers)]
        keys_order = []
        for j in range(num_servers):
            keys_order.extend(weight_assignments["server" + str(j)])

    
    # sync all weights on workers
    if i % 20 == 0:
        curr_weights_ckpt = current_weights.copy()
        # get weights from server
        #current_weights = [servers[j].get_weights.remote(weight_assignments["server" + str(j)]) for j in range(num_servers)] 

        # update weights on all workers
        [workers[j][idx].update_weights.remote(keys_order, *current_weights) for  idx  in range(num_workers) for j in range(num_servers)]
    
        
    # use local cache of weights and get gradients from workers
    gradients = [[workers[j][idx].compute_gradients.remote() for  idx  in range(num_workers)] for j in range(num_servers)]

    start = time()
    # Updates gradients to specfic parameter servers
    current_weights_t = [servers[j].apply_gradients.remote(weight_assignments["server" + str(j)], lr, *gradients[j]) for j in range(num_servers)]
    current_weights = ray.get(current_weights_t)
    
    end = time()
    time_per_iteration.append(end-start)

    if i % 5 == 0:
        # Evaluate the current model.
        # current_weights = [servers[j].get_weights.remote(weight_assignments["server" + str(j)]) for j in range(num_servers)] 
      
        # we are once again using the server to key mapping to set the weight back
        model.set_weights(keys_order, current_weights)
        accuracy = evaluate(model, test_loader)
        print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
    #rint("\n")

#print("Final accuracy is {:.1f}.".format(accuracy))
# Clean up Ray resources and processes before the next example.
ray.shutdown()


# In[33]:


np.mean(time_per_iteration[1:])


# In[27]:


np.std(time_per_iteration[1:])


# In[20]:


time_per_iteration


# In[12]:


server_ids


# In[13]:


server_ids.index("server4")


# In[ ]:


hasher.get_key_to_node_map()['fc_bias']


# In[ ]:


testmodel=LinearNet()


# In[ ]:


testmodel.get_weights()


# In[ ]:


len(current_weights[0])


# In[ ]:


servers[0].weights


# In[ ]:


sum([184, 198, 191, 212])


# In[ ]:




