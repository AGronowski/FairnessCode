import torch
import torch.nn as nn
import numpy as np
import sklearn.ensemble, sklearn.linear_model
import sklearn.ensemble, sklearn.linear_model, sklearn.dummy, sklearn.metrics
import os
import time
import network
import dataset
import metrics
import cost_functions
import evaluations
from tqdm import tqdm
from progressbar import progressbar
import matplotlib.pyplot as plt
import torchvision.utils as vutils
start_time = time.time()

torch.manual_seed(2021)
np.random.seed(2021)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    numworkers = 0 #SET TO 0 if running on CPU. Setting to 1 causes uncaught exception with debugger and python crashes
    debugging = True
else:
    numworkers = 32
    debugging = False


#@profile
def main():
    # print(device)

    epochs = 10
    batch_size = 128
    latent_dim = 256

    dataset_type = 2
    alpha =1
    datasets = ["CelebA_gender","CelebA_race","EyePACS","Adult",'Mh']
    '''
    0 - CelebA_gender
    1 - CelebA_race
    2 - EyePACS
    '''
    for method in [0,1,2]:
        methods = ["IB","Skoglund","Combined"]
        '''
        0 - IB
        1 - Skoglund
        2 - Combined
        '''
        description = "reduced decoder beta1"

        if dataset_type == 0 or dataset_type == 1:
            train_set, test_set = dataset.get_celeba(debugging,dataset_type)
        elif dataset_type == 2:
            train_set, test_set = dataset.get_eyepacs(debugging)
        elif dataset_type ==3:
            train_set, test_set = dataset.get_adult()
        elif dataset_type == 4:
            train_set, test_set = dataset.get_mh()

        else:
            print("error")


        dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,
                                             shuffle=True,num_workers=numworkers)


        # print('eyepacs')
        # model = main_network.Main_Network().to(device)
        # #initialize weights
        # model.apply(main_network.weights_init)

        if dataset_type <= 2:
            model = network.VAE(latent_dim).to(device)
        elif dataset_type == 3: #adult
            model = network.VAETabular(latent_dim,13).to(device)
        elif dataset_type == 4: #mh
            model = network.VAETabular(latent_dim, 9).to(device)
        # print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # evaluations.evaluate_logistic_regression_baseline(model,train_set,test_set,device,debugging,numworkers)
        # return

        # baseline = network.Baseline().to(device)
        betas = [1]
        for beta in betas:

            loss_history =[]

            for epoch in range(epochs):

                train_loss = 0

                # Sets the module in training mode.
                model.train()
                #get minibatches of size batch_size
                for x, y, a in tqdm(dataloader,disable=not(debugging)):

                    # zero the parameter gradients
                    model.zero_grad(set_to_none=True)

                    x = x.to(device).float() #batch size x input_dim
                    y = y.to(device).float() #batch size x 1
                    a = a.to(device).float()

                    yhat, yhat_fair, mu, logvar = model(x,a)

                    #IB loss
                    if method == 0:
                        loss = cost_functions.get_IB_or_Skoglund_loss(yhat, y, mu, logvar, beta,alpha)
                    #Skoglund loss
                    elif method == 1:
                        loss = Skoglund_loss = cost_functions.get_IB_or_Skoglund_loss(yhat_fair,y,mu,logvar,beta,alpha)
                    #Combined loss
                    elif method == 2:
                        loss = Combined_loss = cost_functions.get_combined_loss(yhat,yhat_fair,y,mu,logvar,beta,alpha)

                    # output = baseline(x)
                    # loss = torch.nn.functional.binary_cross_entropy(output.view(-1), y,
                    #                                    reduction='sum')
                    # loss = utils.renyi_cross_entropy(output.view(-1),y,alpha=100)

                    #backward propagation
                    loss.backward()
                    train_loss += loss.item() #.item() changes tensor to float
                    #initiate gradient descent
                    optimizer.step()

                train_loss /= len(dataloader)
                # print(f'epoch = {epoch}')
                # print(f'train loss {train_loss}')
                loss_history.append(train_loss)

                # evaluations.evaluate(model,train_set,batch_size,numworkers,fair,beta,epoch,debugging,device,"Train")
                evaluations.evaluate(model,test_set,batch_size,numworkers,method,beta,epoch,debugging,device,"Test",alpha)
                evaluations.evaluate_logistic_regression(model, train_set, test_set,device,debugging,numworkers)

            # # Plot some training images
            # real_batch = next(iter(dataloader))
            # real_batch = next(iter(dataloader))
            # plt.figure(figsize=(8, 8))
            # plt.axis("off")
            # plt.title("Training Images")
            # plt.imshow(
            #     np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
            # plt.show()

            print("--- %s seconds ---" % (time.time() - start_time))
            print(f"beta  = {beta}")
            print(f"dataset = {datasets[dataset_type]}")
            print(f"method = {methods[method]}")
            print(f"alpha = {alpha}")
            print(f"numworkers = {numworkers}")
            print(f"representation_dim = {latent_dim}")
            print(description)
            # print(loss_history)
            # print(testloss_history)
            print('\n')

if __name__ == '__main__':
    main()

