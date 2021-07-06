import torch
from tqdm import tqdm
import numpy as np
import metrics
import cost_functions
import sklearn.ensemble, sklearn.linear_model, sklearn.dummy

def evaluate_logistic_regression_baseline(model,trainset,testset,device,debugging,numworkers):
    # sets model in evalutation mode
    model.eval()
    predictor = sklearn.linear_model.LogisticRegression()
    with torch.no_grad():
        '''train '''
        # X_train,Y_train,A_train = trainset.images, trainset.targets, trainset.sensitives
        # X_train = X_train.to(device)
        # Y_train = Y_train.to(device)

        traindataloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                     shuffle=True, num_workers=numworkers)

        y_list = []
        x_list = []

        for x, y, a in tqdm(traindataloader, disable=not (debugging)):
            y_list.append(y)
            x_list.append(x)

        X_train = torch.cat(x_list,dim=0)
        Y_train = torch.cat(y_list,dim=0)


        X_train = X_train.cpu()
        Y_train = Y_train.cpu()
        predictor.fit(X_train.flatten(start_dim=1), Y_train)

        ''' test '''

        testdataloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                      shuffle=True, num_workers=numworkers)

        y_list = []
        x_list = []
        a_list = []

        for x, y, a in tqdm(testdataloader, disable=not (debugging)):


            y_list.append(y)
            x_list.append(x)
            a_list.append(a)

        X_test = torch.cat(x_list,dim=0)
        Y_test = torch.cat(y_list,dim=0)
        A_test = torch.cat(a_list,dim=0)

        X_test = X_test.cpu()

        predictions = predictor.predict_proba(X_test.flatten(start_dim=1))
        predictions = np.argmax(predictions,1)
        y = Y_test.cpu().detach().numpy()
        a = A_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions,y)
        accgap = metrics.get_acc_gap(predictions,y,a)
        dpgap = metrics.get_discrimination(predictions,a)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions,y,a)

        print(f"baseline logistic accuracy = {accuracy}")
        print(f"baseline logistic accgap = {accgap}")
        print(f"baseline logistic dpgap = {dpgap}")
        print(f"baseline logistic eqoddsgap = {eqoddsgap}")

def evaluate_logistic_regression(model,trainset,testset,device,debugging,numworkers):
    # sets model in evalutation mode
    model.eval()
    predictor = sklearn.linear_model.LogisticRegression()
    with torch.no_grad():
        '''train '''
        # X_train,Y_train,A_train = trainset.images, trainset.targets, trainset.sensitives
        # X_train = X_train.to(device)
        # Y_train = Y_train.to(device)

        traindataloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                     shuffle=True, num_workers=numworkers)

        y_list = []
        z_list = []

        for x, y, a in tqdm(traindataloader, disable=not (debugging)):

            x = x.to(device).float()  # batch size x input_dim
            y = y.to(device).float()  # batch size x 1

            z, mu, logvar = model.getz(x)

            y_list.append(y)
            z_list.append(z)

        Z_train = torch.cat(z_list,dim=0)
        Y_train = torch.cat(y_list,dim=0)


        Z_train = Z_train.cpu()
        Y_train = Y_train.cpu()
        predictor.fit(Z_train.flatten(start_dim=1), Y_train)

        ''' test '''

        testdataloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                      shuffle=True, num_workers=numworkers)

        y_list = []
        z_list = []
        a_list = []

        for x, y, a in tqdm(testdataloader, disable=not (debugging)):
            x = x.to(device).float()  # batch size x input_dim
            y = y.to(device).float()  # batch size x 1
            a = a.to(device).float()

            z, mu, logvar = model.getz(x)

            y_list.append(y)
            z_list.append(z)
            a_list.append(a)

        Z_test = torch.cat(z_list,dim=0)
        Y_test = torch.cat(y_list,dim=0)
        A_test = torch.cat(a_list,dim=0)

        Z_test = Z_test.cpu()

        predictions = predictor.predict_proba(Z_test.flatten(start_dim=1))
        predictions = np.argmax(predictions,1)
        y = Y_test.cpu().detach().numpy()
        a = A_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions,y)
        accgap = metrics.get_acc_gap(predictions,y,a)
        dpgap = metrics.get_discrimination(predictions,a)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions,y,a)

        print(f"logistic accuracy = {accuracy}")
        print(f"logistic accgap = {accgap}")
        print(f"logistic dpgap = {dpgap}")
        print(f"logistic eqoddsgap = {eqoddsgap}")




def evaluate(model,dataset,batch_size,numworkers,fair,beta,epoch,debugging,device,dataset_type):
    # sets model in evalutation mode
    model.eval()
    testloss_history = []
    testdataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=numworkers)
    accuracy = 0
    accgap = 0
    disc = 0
    eqodds = 0
    testloss = 0

    with torch.no_grad():
        for x, y, a in tqdm(testdataloader, disable=not (debugging)):

            x = x.to(device).float()  # batch size x input_dim
            y = y.to(device).float()  # batch size x 1
            a = a.to(device).float()

            yhat, yhat_fair, mu, logvar = model(x, a)

            if fair:
                output = yhat_fair
            else:
                output = yhat

            loss = cost_functions.get_loss(output, y, mu, logvar, beta)

            # output = baseline(x)
            # loss = torch.nn.functional.binary_cross_entropy(output.view(-1), y.view(-1),
            #                                                 reduction='sum')
            # loss = utils.renyi_cross_entropy(output.view(-1), y, alpha=100)

            testloss += loss.item()  # item gives tensors value as float

            # get predictions
            predictions = output.cpu().view(-1).detach().numpy()
            predictions[predictions < 0.5] = 0
            predictions[predictions > 0.5] = 1
            a = a.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            accuracy += metrics.get_accuracy(predictions, y)
            disc += metrics.get_discrimination(predictions, a)
            eqodds += metrics.get_equalized_odds_gap(predictions, y, a)
            accgap += metrics.get_acc_gap(predictions, y, a)

    testloss /= len(testdataloader)
    accuracy /= len(testdataloader)
    accgap /= len(testdataloader)
    disc /= len(testdataloader)
    eqodds /= len(testdataloader)

    testloss_history.append(testloss)

    print(f"epoch {epoch}")
    print(f"{dataset_type} loss {testloss}")
    print(f"{dataset_type} accuracy {accuracy}")
    print(f"{dataset_type} accgap {accgap}")
    print(f"{dataset_type} disc {disc}")
    print(f"{dataset_type} eqodds {eqodds}")

