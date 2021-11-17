import torch
import numpy as np
import time
import network
import dataset
import cost_functions
import evaluations
from early_stopping import EarlyStopping, LRScheduler
from tqdm import tqdm
import umap_functions
from torch.utils.data import SubsetRandomSampler, random_split
start_time = time.time()
debugging = True


torch.manual_seed(2021)
np.random.seed(2021)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


epochs = 50
batch_size = 64
latent_dim = 32

dataset_type = 2

numworkers = 32 if torch.cuda.is_available() else 0
# 0 - 8

# Fairface
# gender predicts age, gender is sensitive
# race predicts gender, race is sensitive
datasets = ["CelebA_gender", "CelebA_race", "EyePACS", "Adult", 'Mh_age', 'Mh_gender', 'fairface_gender',
            'fairface_race', 'mnist']
'''
0 - CelebA_gender
1 - CelebA_race
2 - EyePACS
'''

methods = ["IB", "Skoglund", "Combined", "baseline"]
'''
0 - IB
1 - Skoglund
2 - Combined
'''
description = ""


#dataset - special function and dataloader
train_set, test_set = dataset.get_testaa_eyepacs()

dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True, num_workers=numworkers)

test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=numworkers)

#get model

baseline = False

if baseline:
    model = network.Baseline().to(device)
else:
    model = network.VAE(latent_dim).to(device)

name = 'EyePACS_Skoglund_0.8_30.07_50.0_30_'

if device == 'cuda':
    model.load_state_dict(torch.load(f'../results/{name}.pt'))
else:
    model.load_state_dict(torch.load(f'../results/{name}.pt',map_location=torch.device('cpu')))
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if baseline:
    frame = evaluations.evaluate_aa(model,test_dataloader,3,debugging,device,2,-1,-1,-1,-1)
else:
    frame = evaluations.evaluate_logistic_regression_eyepacs_aa(model, train_set, test_set, device, debugging, numworkers)

frame.to_csv(f'../results/{name}.csv',index=False)

