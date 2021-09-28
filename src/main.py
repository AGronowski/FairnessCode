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
from torch.utils.data.sampler import SubsetRandomSampler
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

    umap = False

    epochs = 50
    batch_size = 64
    latent_dim = 32

    dataset_type = 0
    # 0 - 7
    datasets = ["CelebA_gender","CelebA_race","EyePACS","Adult",'Mh_age','Mh_gender','fairface_gender','fairface_race']
    '''
    0 - CelebA_gender
    1 - CelebA_race
    2 - EyePACS
    '''

    method = 2
    # for dataset_type in [7]:

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
            # train_set, test_set = dataset.get_mh()
            train_set, test_set, valset = dataset.get_mh_balanced(4) #A = age
        elif dataset_type == 5:
            # train_set, test_set = dataset.get_mh()
            train_set, test_set, valset = dataset.get_mh_balanced(5) #A = gender

        elif dataset_type == 6 or dataset_type == 7:
            train_set, test_set = dataset.get_fairface(debugging,dataset_type)
        else:
            print("error")


        stop_early = True
        lr_schedule =False

        if stop_early or lr_schedule:
            early_stopping = EarlyStopping()

            if dataset != 4 and dataset_type != 5:
                validation_split = .2

                # Creating data indices for training and validation splits:
                dataset_size = len(train_set)
                indices = list(range(dataset_size))
                split = int(np.floor(validation_split * dataset_size))
                np.random.shuffle(indices)
                train_indices, val_indices = indices[split:], indices[:split]

                # Creating PT data samplers and loaders:
                train_sampler = SubsetRandomSampler(train_indices)
                valid_sampler = SubsetRandomSampler(val_indices)

                dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,
                                                 sampler=train_sampler,num_workers=numworkers)

                val_dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,
                                                 sampler=valid_sampler,num_workers=numworkers)
            else:
                dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                         shuffle=True, num_workers=numworkers)

                val_dataloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                             shuffle=True, num_workers=numworkers)
        else:
            dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,
                                             shuffle=True,num_workers=numworkers)

        test_dataloader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,
                                             shuffle=True,num_workers=numworkers)


        # print('eyepacs')
        # model = main_network.Main_Network().to(device)
        # #initialize weights
        # model.apply(main_network.weights_init)

        beta =1
        alpha = 1
        alphas = []
        results = []

        baseline_only = False

        if baseline_only:
            evaluations.evaluate_logistic_regression_baseline(train_set,test_set,debugging,numworkers)
            result = evaluations.evaluate_logistic_regression(model, train_set, test_set, device, debugging, numworkers)
            print(f'dataset is {datasets[dataset_type]}')
            return



        for alpha in np.linspace(0,0.5,50):
        # for beta in np.linspace(0,50,50):
        # for alpha in np.linspace(0.5,1,25):
        # for alpha in [0]:

            if dataset_type <= 2 or dataset_type == 6 or dataset_type == 7: #celeba, eyepacs, fairface image datasets
                model = network.VAE(latent_dim).to(device)
            elif dataset_type == 3: #adult
                model = network.VAETabular(latent_dim,13).to(device)
            elif dataset_type == 4 or dataset_type == 5: #mh
                model = network.VAETabular(latent_dim, 9).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            lr_scheduler = LRScheduler(optimizer)
            early_stopping = EarlyStopping()

            loss_history =[]

            if umap:
                embedding, a_train, x_train, y_train = umap_functions.get_embedding(model,test_set,device,debugging,numworkers)
                umap_functions.plot(embedding,a_train,x_train,y_train,alpha,dataset_type,method)
            # return 0

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
                    # z = model.getz(x)

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

                if stop_early:
                    val_epoch_loss = evaluations.evaluate(model,val_dataloader,method,beta,epoch,debugging,device,"Val",alpha)
                    early_stopping(val_epoch_loss)
                    if early_stopping.early_stop:
                        break
                if lr_schedule:
                    val_epoch_loss = evaluations.evaluate(model,val_dataloader,method,beta,epoch,debugging,device,"Val",alpha)
                    print(val_epoch_loss)
                    lr_scheduler(val_epoch_loss)

                train_loss /= len(dataloader)
                # print(f'epoch = {epoch}')
                # print(f'train loss {train_loss}')
                loss_history.append(train_loss)

                # evaluations.evaluate(model,train_set,batch_size,numworkers,fair,beta,epoch,debugging,device,"Train")

                # evaluations.evaluate_logistic_regression_baseline(model,train_set,test_set,device,debugging,numworkers)

                evaluations.evaluate(model,test_dataloader,method,beta,epoch,debugging,device,"Test",alpha)
                result = evaluations.evaluate_logistic_regression(model, train_set, test_set,device,debugging,numworkers)

                if umap and epoch >= 5:
                    embedding, a_train, x_train, y_train = umap_functions.get_embedding(model, test_set, device, debugging, numworkers,representation=True)
                    umap_functions.plot(embedding, a_train, x_train, y_train,alpha,dataset_type,method,representation=True)

            alphas.append(alpha)
            results.append(result)
            np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_alphas', alphas)
            np.save(f'../results/{datasets[dataset_type]}_{methods[method]}_results', results)
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
            print(f"batch size = {batch_size}")
            print(description)
            # print(loss_history)
            # print(testloss_history)
            print('\n')

if __name__ == '__main__':
    main()

