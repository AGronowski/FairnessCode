import torch
from tqdm import tqdm
import numpy as np
import metrics
import cost_functions
import sklearn.ensemble, sklearn.linear_model, sklearn.dummy
from sklearn import preprocessing
import pandas as pd

def evaluate_logistic_regression_baseline(trainset,testset,debugging,numworkers):
    # model.eval()
    predictor = sklearn.linear_model.LogisticRegression()
    with torch.no_grad():
        '''train '''
        # X_train,Y_train,A_train = trainset.images, trainset.targets, trainset.sensitives
        # X_train = X_train.to(device)
        # Y_train = Y_train.to(device)

        traindataloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                     shuffle=True, num_workers=numworkers)

        y_list = []
        x_list = []

        debugging = not (debugging)

        for x, y, a in tqdm(traindataloader, disable=not (debugging)):
            y_list.append(y)
            x_list.append(x)

        X_train = torch.cat(x_list,dim=0)
        Y_train = torch.cat(y_list,dim=0)


        X_train = X_train.cpu()
        Y_train = Y_train.cpu()

        scaler = preprocessing.StandardScaler().fit(X_train.flatten(start_dim=1))
        X_scaled = scaler.transform(X_train.flatten(start_dim=1))
        predictor.fit(X_scaled, Y_train)


        ''' test '''

        testdataloader = torch.utils.data.DataLoader(testset, batch_size=64,
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
        accmin0, accmin1 = metrics.get_min_accuracy(predictions,y,a)


        print(f"baseline logistic accuracy = {accuracy}")
        print(f"baseline logistic accgap = {accgap}")
        print(f"baseline logistic dpgap = {dpgap}")
        print(f"baseline logistic eqoddsgap = {eqoddsgap}")
        print(f"baseline logistic acc_min_0 = {accmin0}")
        print(f"baseline logistic acc_min_1 = {accmin1}")

def evaluate_logistic_regression(model,trainset,testset,device,debugging,numworkers):
    # sets model in evalutation mode
    model.eval()
    predictor = sklearn.linear_model.LogisticRegression(solver='liblinear')
    # predictor = sklearn.ensemble.RandomForestClassifier()
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

        scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_scaled = scaler.transform(Z_train)
        #predictor.fit(Z_train.flatten(start_dim=1), Y_train)

        #try to prevent nan or inf error
        # torch.where: (condition, condition true, condition false)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        predictor.fit(Z_scaled, Y_train)


        ''' test '''

        testdataloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                      shuffle=False, num_workers=numworkers)

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
        Z_scaled = scaler.transform(Z_test)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)

        predictions = predictor.predict_proba(Z_scaled)
        # predictions = predictor.predict_proba(Z_scaled.flatten(start_dim=1))
        predictions = np.argmax(predictions,1)
        y = Y_test.cpu().detach().numpy()
        a = A_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions,y)
        accgap = metrics.get_acc_gap(predictions,y,a)
        dpgap = metrics.get_discrimination(predictions,a)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions,y,a)
        accmin0, accmin1 = metrics.get_min_accuracy(predictions,y,a)

        print(f"logistic accuracy = {accuracy}")
        print(f"logistic accgap = {accgap}")
        print(f"logistic dpgap = {dpgap}")
        print(f"logistic eqoddsgap = {eqoddsgap}")
        print(f"logistic acc_min_0 = {accmin0}")
        print(f"logistic acc_min_1 = {accmin1}")

        return np.array([accuracy,accgap,dpgap,eqoddsgap,accmin0,accmin1])


#only for testing eyepacs aa. regular train, modified test
def evaluate_logistic_regression_eyepacs_aa(model, trainset, testset, device, debugging, numworkers):
    # sets model in evalutation mode
    model.eval()
    predictor = sklearn.linear_model.LogisticRegression(solver='liblinear')
    # predictor = sklearn.ensemble.RandomForestClassifier()
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

        Z_train = torch.cat(z_list, dim=0)
        Y_train = torch.cat(y_list, dim=0)

        Z_train = Z_train.cpu()
        Y_train = Y_train.cpu()

        scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_scaled = scaler.transform(Z_train)
        # predictor.fit(Z_train.flatten(start_dim=1), Y_train)

        # try to prevent nan or inf error
        # torch.where: (condition, condition true, condition false)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        predictor.fit(Z_scaled, Y_train)

        ''' test '''

        testdataloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                     shuffle=False, num_workers=numworkers)

        y_list = []
        z_list = []
        image_list = []

        for x, y, image in tqdm(testdataloader, disable=not (debugging)):
           x = x.to(device).float()  # batch size x input_dim
           y = y.to(device).float()  # batch size x 1
           # image = image.to(device).float()

           z, mu, logvar = model.getz(x)

           y_list.append(y)
           z_list.append(z)
           image_list.append(list(image))

        Z_test = torch.cat(z_list,dim=0)
        Y_test = torch.cat(y_list,dim=0)
        images = np.concatenate(image_list)

        Z_test = Z_test.cpu()
        Z_scaled = scaler.transform(Z_test)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)

        predictions = predictor.predict_proba(Z_scaled)
        # predictions = predictor.predict_proba(Z_scaled.flatten(start_dim=1))
        predictions = np.argmax(predictions,1)
        y = Y_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions,y)

        combined_lists = [images,y,predictions]
        frame = pd.DataFrame(combined_lists)


        print(f"logistic accuracy = {accuracy}")


        return frame.transpose()







#gets loss
def evaluate(model,dataloader,method,epoch,debugging,device,dataset_type,description,beta=1,alpha=1,beta1=1,beta2=1):
    # sets model in evalutation mode
    model.eval()
    testloss_history = []
    # testdataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                              shuffle=True, num_workers=numworkers)
    # accuracy = 0
    # accgap = 0
    # disc = 0
    # eqodds = 0
    #
    # accmin0 = 0
    # accmin1 = 0

    testloss = 0

    # all_preds = np.array()
    # all_a = np.array()
    # all_y = np.array()

    y_list = []
    a_list = []
    yhat_list = []

    with torch.no_grad():
        for x, y, a in tqdm(dataloader, disable=not (debugging)):
            x = x.to(device).float()  # batch size x input_dim
            y = y.to(device).float()  # batch size x 1
            a = a.to(device).float()

            if method == 3:  #baseline
                yhat = model(x)
            else:
                yhat, yhat_fair, mu, logvar = model(x, a)


            y_list.append(y)
            a_list.append(a)
            yhat_list.append(yhat)


            mnist = False
            if dataset_type == 8:
                mnist = True

            # IB loss
            if method == 0:
                # loss = cost_functions.get_IB_or_Skoglund_loss(yhat, y, mu, logvar, beta,alpha)
                loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat, y, mu, logvar, beta,mnist)
            # Skoglund loss
            elif method == 1:
                # loss = Skoglund_loss = cost_functions.get_IB_or_Skoglund_loss(yhat_fair,y,mu,logvar,beta,alpha)
                loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat_fair, y, mu, logvar, beta,mnist)

            # Combined loss
            elif method == 2:
                loss = Combined_loss = cost_functions.get_combined_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1,
                                                                        beta2,mnist)
            #baseline loss
            elif method == 3:
                loss = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y,
                                                                            reduction='sum')

            #output = baseline(x)
            # loss = torch.nn.functional.binary_cross_entropy(output.view(-1), y.view(-1),
            #                                                 reduction='sum')
            # loss = utils.renyi_cross_entropy(output.view(-1), y, alpha=100)

            testloss += loss.item()  # item gives tensors value as float

            # get predictions


        Y_test = torch.cat(y_list,dim=0)
        A_test = torch.cat(a_list,dim=0)
        Yhat_test = torch.cat(yhat_list,dim=0)

        y = Y_test.cpu().detach().numpy()
        a = A_test.cpu().detach().numpy()
        predictions = Yhat_test.cpu().view(-1).detach().numpy()
        predictions[predictions < 0.5] = 0
        predictions[predictions > 0.5] = 1

        accuracy = metrics.get_accuracy(predictions,y)
        accgap = metrics.get_acc_gap(predictions,y,a)
        dpgap = metrics.get_discrimination(predictions,a)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions,y,a)
        accmin0, accmin1 = metrics.get_min_accuracy(predictions,y,a)

        if description != 'Val':
            print(f" accuracy = {accuracy}")
            print(f" accgap = {accgap}")
            print(f" dpgap = {dpgap}")
            print(f" eqoddsgap = {eqoddsgap}")
            print(f" acc_min_0 = {accmin0}")
            print(f" acc_min_1 = {accmin1}")

        print(f"{description}\n")
        # baseline and not used for validation
        if method == 3 and description != 'Val':
            return np.array([round(accuracy,6), round(accgap,6), round(dpgap,6), round(eqoddsgap,6), round(accmin0,6), round(accmin1,6)])
        else:
            return testloss


#gets loss
def evaluate_aa(model,dataloader,method,debugging,device,dataset_type,beta=1,alpha=1,beta1=1,beta2=1):
    # sets model in evalutation mode
    model.eval()

    testloss = 0

    y_list = []
    image_list = []
    yhat_list = []

    with torch.no_grad():
        for x, y, image in tqdm(dataloader, disable=not (debugging)):
            x = x.to(device).float()  # batch size x input_dim
            y = y.to(device).float()  # batch size x 1

            if method == 3:  #baseline
                yhat = model(x)
            else:
                yhat, yhat_fair, mu, logvar = model(x, a)


            y_list.append(y)
            yhat_list.append(yhat)
            image_list.append(list(image))

            mnist = False
            if dataset_type == 8:
                mnist = True

            # IB loss
            if method == 0:
                # loss = cost_functions.get_IB_or_Skoglund_loss(yhat, y, mu, logvar, beta,alpha)
                loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat, y, mu, logvar, beta,mnist)
            # Skoglund loss
            elif method == 1:
                # loss = Skoglund_loss = cost_functions.get_IB_or_Skoglund_loss(yhat_fair,y,mu,logvar,beta,alpha)
                loss = cost_functions.get_IB_or_Skoglund_original_loss(yhat_fair, y, mu, logvar, beta,mnist)

            # Combined loss
            elif method == 2:
                loss = Combined_loss = cost_functions.get_combined_loss(yhat, yhat_fair, y, mu, logvar, alpha, beta1,
                                                                        beta2,mnist)
            #baseline loss
            elif method == 3:
                loss = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y,
                                                                            reduction='sum')

            #output = baseline(x)
            # loss = torch.nn.functional.binary_cross_entropy(output.view(-1), y.view(-1),
            #                                                 reduction='sum')
            # loss = utils.renyi_cross_entropy(output.view(-1), y, alpha=100)

            testloss += loss.item()  # item gives tensors value as float

            # get predictions


        Y_test = torch.cat(y_list,dim=0)
        Yhat_test = torch.cat(yhat_list,dim=0)
        images = np.concatenate(image_list)

        y = Y_test.cpu().detach().numpy()
        predictions = Yhat_test.cpu().view(-1).detach().numpy()
        predictions[predictions < 0.5] = 0
        predictions[predictions > 0.5] = 1

        accuracy = metrics.get_accuracy(predictions,y)
        # accgap = metrics.get_acc_gap(predictions,y,a)
        # dpgap = metrics.get_discrimination(predictions,a)
        # eqoddsgap = metrics.get_equalized_odds_gap(predictions,y,a)
        # accmin0, accmin1 = metrics.get_min_accuracy(predictions,y,a)
        print(f" accuracy = {accuracy}")

        combined_lists = [images,y,predictions]
        frame = pd.DataFrame(combined_lists)

        return frame
