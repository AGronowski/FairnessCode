import umap
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def plot(embedding, a_train,x_train,y_train,alpha,dataset_type,method,representation=False):

    datasets = ["CelebA_gender","CelebA_race","EyePACS","Adult",'Mh_age','Mh_gender']
    dataset = datasets[dataset_type]

    methods = ["IB","Skoglund","Combined"]
    method_name = methods[method]

    save = True
    show = False

    # plt.scatter(
    #     embedding[:, 0],
    #     embedding[:, 1],
    #     c=[sns.color_palette()[int(x)] for x in x_train[:,2]])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('X', fontsize=24)
    # plt.show()


    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[int(x)] for x in a_train])
    plt.gca().set_aspect('equal', 'datalim')
    if representation:
        plt.title(rf'{dataset} {method_name} $\alpha = ${alpha} A', fontsize=24)
    else:
        plt.title(rf'{dataset} original data A', fontsize=24)

    if save:
        if representation:
            plt.savefig(f"../results/umapplots/umap_{dataset}_{method_name}_{alpha}_A.png")
        else:
            plt.savefig(f"../results/umapplots/umap_{dataset}_original_data_A.png")
    if show:
        plt.show()

    plt.clf()

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[int(x)+2] for x in y_train])
    plt.gca().set_aspect('equal', 'datalim')
    if representation:
        plt.title(rf'{dataset} {method_name} $\alpha = ${alpha} Y', fontsize=24)
    else:
        plt.title(rf'{dataset} original data Y', fontsize=24)
    if save:
        if representation:
            plt.savefig(f"../results/umapplots/umap_{dataset}_{method_name}_{alpha}_Y.png")
        else:
            plt.savefig(f"../results/umapplots/umap_{dataset}_original_data_Y.png")
    if show:
        plt.show()
    plt.clf()

def get_embedding(model,dataset,device,debugging,numworkers,representation=False):
    # sets model in evalutation mode
    model.eval()
    reducer = umap.UMAP()

    with torch.no_grad():
        '''train '''
        # X_train,Y_train,A_train = trainset.images, trainset.targets, trainset.sensitives
        # X_train = X_train.to(device)
        # Y_train = Y_train.to(device)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                                     shuffle=True, num_workers=numworkers)

        x_list = []
        y_list = []
        z_list = []
        a_list = []

        for x, y, a in tqdm(dataloader, disable=not (debugging)):

            x = x.to(device).float()  # batch size x input_dim
            y = y.to(device).float()  # batch size x 1
            a = a.to(device).float()

            z, mu, logvar = model.getz(x)

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            a_list.append(a)

        X_train = torch.cat(x_list,dim=0)
        Z_train = torch.cat(z_list,dim=0)
        Y_train = torch.cat(y_list,dim=0)
        A_train = torch.cat(a_list,dim=0)

        X_train = X_train.cpu()
        Z_train = Z_train.cpu()
        Y_train = Y_train.cpu()
        A_train = A_train.cpu()



        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)


        if representation:
            embedding = reducer.fit_transform(Z_train)
            X_train = Z_train
        else:
            scaled_X_train = StandardScaler().fit_transform(X_train)
            embedding = reducer.fit_transform(scaled_X_train)

        return embedding, A_train, X_train, Y_train
