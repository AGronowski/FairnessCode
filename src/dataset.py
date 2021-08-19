import torch
import torchvision
from torchvision import transforms
from skimage import io, transform
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os

torch.manual_seed(2020)
np.random.seed(2020)

def getimages(csv_file,root_dir,image_size,transform):
    frame = pd.read_csv(csv_file)

    N = len(frame)
    images = torch.zeros(N, 3, image_size, image_size)
    for n in range(N):
        img_name = os.path.join(root_dir,
                                frame.iloc[n, 1])
        image = io.imread(img_name)  # numpy.ndarray
        image = torchvision.transforms.functional.to_pil_image(image)  # PIL image

        if transform:
            image = transform(image)

        images[n] = image  # shape (28,28)
    return images

class Celeba_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # self.targets = self.frame.iloc[:,2]
        # self.sensitives = self.frame.iloc[:,3]
        # self.images = images

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name) #numpy.ndarray
        image = torchvision.transforms.functional.to_pil_image(image) #PIL image

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index,2] #target is age
        sensitive = self.frame.iloc[index,3] #sensitive is gender

        return image, target, sensitive

class Eyepacs_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # self.targets = self.frame.iloc[:,2]
        # self.sensitives = self.frame.iloc[:, 4]
        # self.images = images


    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[index, 1])
        image = io.imread(img_name) #numpy.ndarray
        image = torchvision.transforms.functional.to_pil_image(image) #PIL image

        if self.transform:
            image = self.transform(image)

        target = self.frame.iloc[index,2] #target is diabetic_retinopathy
        sensitive = self.frame.iloc[index,4] #sensitive is ita_dark

        return image, target, sensitive



def get_eyepacs(debugging):
    root_dir = "../data/eyepacs_small"

    image_size = 256

    transform = transforms.Compose([transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]) #scales to [-1,1]


    if debugging:
        trainset = Eyepacs_dataset('../data/eyepacs_debugging.csv',root_dir,
                                  transform)
        testset = Eyepacs_dataset('../data/eyepacs_debugging.csv',root_dir,
                                  transform)
    else:
        csv_file = '../data/eyepacs_control_train_jpeg.csv'
        trainset = Eyepacs_dataset(csv_file,root_dir,
                                  transform)
        testset = Eyepacs_dataset('../data/eyepacs_test_dr_ita_jpeg.csv',root_dir,
                                      transform)

    return trainset, testset

def get_celeba(debugging,dataset):
    root_dir = '../data/celeba_small'
    image_size = 128
    transform = transforms.Compose([
        transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]) #scales to [-1,1]

    if debugging:

        csv_file = '../data/celeba_debugging.csv'
        trainset = Celeba_dataset(csv_file,root_dir,
                                  transform)
        testset = Celeba_dataset(csv_file,root_dir,transform)
    else:
        if dataset == 0: #gender
            trainset = Celeba_dataset('../data/celeba_gender_train_jpg.csv',root_dir,
                                      transform)
        elif dataset == 1 : #race
            trainset = Celeba_dataset('../data/celeba_skincolor_train_jpg.csv',root_dir,
                                      transform)
        else:
            print("error")
            return False
        testset = Celeba_dataset('../data/celeba_balanced_combo_test_jpg.csv',root_dir,
                                  transform)


    return trainset, testset


class Adult_dataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, data_hidden, transform=None, task='fairness'):
        self.data = data  # X
        self.targets = targets  # T
        self.hidden = data_hidden  # S
        self.transform = transform  # this is none
        self.target_vals = 2
        self.hidden_vals = 2
        self.task = task

    def __getitem__(self, index):
        datum, target, hidden = self.data[index], self.targets[index], self.hidden[index]
        if self.task == 'fairness':
            target, hidden = int(target), int(hidden)
        else:
            hidden = int(hidden)
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, target, hidden

    def __len__(self):
        return len(self.targets)


def get_adult(task='fairness'):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'salary']

    # these are the categorical variables that will be encoded
    dummy_variables = {
        'workclass': [
            'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked'],
        'education': ['Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, \
            12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool'],
        'education-num': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'],
        'marital-status': ['Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,\
            Married-AF-spouse'],
        'occupation': ['Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, \
            Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, \
            Protective-serv, Armed-Forces'],
        'relationship': ['Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'],
        'race': ['White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'],
        'sex': ['Female, Male'],
        'native-country': ['United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), \
            India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, \
            Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, \
            Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands']
    }

    # break apart each string into a list based on the comma separator then strip whitespace
    for k in dummy_variables:
        dummy_variables[k] = [v.strip() for v in dummy_variables[k][0].split(',')]

    # Load Adult dataset

    # This should be uncommented first time this is run to download the datasets
    # This code doesn't save the dataset, I downloaded it manually to data folder
    '''
    data_train = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        names=column_names,header=None
    )
    data_test = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
        names=column_names,skiprows=1,header=None
    ) 
    '''

    data_train = pd.read_csv(
        '../data/adult_us_cens/adult.data',
        names=column_names, header=None  # names creates header row
    )
    data_test = pd.read_csv(
        '../data/adult_us_cens/adult.test',
        names=column_names, skiprows=1, header=None
    )  # skip 1st row

    # apply function to each column
    # removes whitespace
    data_train = data_train.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    data_test = data_test.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)

    def get_variables(data, task='fairness'):

        # Encode target labels with value between 0 and n_classes-1
        le = preprocessing.LabelEncoder()
        dummy_columns = list(dummy_variables.keys())
        dummy_columns.remove('sex')
        # integer encodings to the dummy columns
        data[dummy_columns] = data[dummy_columns].apply(lambda col: le.fit_transform(col))
        X = data.drop('sex', axis=1).drop('salary', axis=1).to_numpy().astype(float)
        S = data['sex'].to_numpy()
        if task == 'fairness':
            T = data['salary'].to_numpy()
            # return 0 when T <=50K, 1 otherwis
            T = np.where(np.logical_or(T == '<=50K', T == '<=50K.'), 0, 1)
        else:
            T = data.drop('sex', axis=1).drop('salary', axis=1).to_numpy().astype(float)
        S = np.where(S == 'Male', 0, 1)

        return X, S, T

    X_train, S_train, T_train = get_variables(data_train, task)
    X_test, S_test, T_test = get_variables(data_test, task)
    # mean and std for each column (reduce rowns)
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    # normalize data
    X_train = (X_train - X_mean) / (X_std)
    X_test = (X_test - X_mean) / (X_std)
    if task == 'privacy':
        for i in range(len(T_train[1, :])):
            if len(np.unique(T_train[:, i])) > 42:
                t_mean, t_std = T_train[:, i].mean(), T_train[:, i].std()
                T_train[:, i] = (T_train[:, i] - t_mean) / t_std
                T_test[:, i] = (T_test[:, i] - t_mean) / t_std

    trainset = Adult_dataset(X_train, T_train, S_train, task=task)
    testset = Adult_dataset(X_test, T_test, S_test, task=task)

    return trainset, testset


def get_mh_balanced(dataset_type):
    if dataset_type == 4:
        age_train_data = pd.read_csv('../data/mental_health/AgeBinary_train_data.csv')
        age_train_labels = pd.read_csv('../data/mental_health/AgeBinary_train_labels.csv')

        age_test_data = pd.read_csv('../data/mental_health/AgeBinary_test_data.csv')
        age_test_labels = pd.read_csv('../data/mental_health/AgeBinary_test_labels.csv')

        age_val_data = pd.read_csv('../data/mental_health/AgeBinary_val_data.csv')
        age_val_labels = pd.read_csv('../data/mental_health/AgeBinary_val_labels.csv')


        # age_train_data = age_train_data.drop('mh_family_hist',axis=1).drop('mh_disorder_past',axis=1).drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        X_train = age_train_data.drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        S_train = age_train_labels['AgeBinary'].to_numpy()
        Y_hat_train = age_train_labels['treatment'].to_numpy()

        # age_test_data = age_test_data.drop('mh_family_hist',axis=1).drop('mh_disorder_past',axis=1).drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        X_test = age_test_data.drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        S_test = age_test_labels['AgeBinary'].to_numpy()
        Y_hat_test = age_test_labels['treatment'].to_numpy()

        # age_test_data = age_test_data.drop('mh_family_hist',axis=1).drop('mh_disorder_past',axis=1).drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        X_val = age_val_data.drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        S_val = age_val_labels['AgeBinary'].to_numpy()
        Y_hat_val = age_val_labels['treatment'].to_numpy()

        # # mean and std for each column (0 means column) (reduce rowns)
        X_mean, X_std = X_train.mean(0), X_train.std(0)
        # normalize data
        X_train = (X_train - X_mean) / (X_std)
        X_test = (X_test - X_mean) / (X_std)
        X_val = (X_val - X_mean) / X_std

        trainset = Adult_dataset(X_train, Y_hat_train, S_train, task='fairness')
        testset = Adult_dataset(X_test, Y_hat_test, S_test, task='fairness')
        valset = Adult_dataset(X_val, Y_hat_val, S_val, task='fairness')
    if dataset_type == 5:
        age_train_data = pd.read_csv('../data/mental_health/Gender_train_data.csv')
        age_train_labels = pd.read_csv('../data/mental_health/Gender_train_labels.csv')

        age_test_data = pd.read_csv('../data/mental_health/Gender_test_data.csv')
        age_test_labels = pd.read_csv('../data/mental_health/Gender_test_labels.csv')

        age_val_data = pd.read_csv('../data/mental_health/Gender_val_data.csv')
        age_val_labels = pd.read_csv('../data/mental_health/Gender_val_labels.csv')


        # age_train_data = age_train_data.drop('mh_family_hist',axis=1).drop('mh_disorder_past',axis=1).drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        X_train = age_train_data.drop('Age',axis=1).drop('Gender',axis=1).to_numpy()
        S_train = age_train_labels['Gender'].to_numpy()
        Y_hat_train = age_train_labels['treatment'].to_numpy()

        # age_test_data = age_test_data.drop('mh_family_hist',axis=1).drop('mh_disorder_past',axis=1).drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        X_test = age_test_data.drop('Age',axis=1).drop('Gender',axis=1).to_numpy()
        S_test = age_test_labels['AgeBinary'].to_numpy()
        Y_hat_test = age_test_labels['treatment'].to_numpy()

        # age_test_data = age_test_data.drop('mh_family_hist',axis=1).drop('mh_disorder_past',axis=1).drop('Age',axis=1).drop('AgeBinary',axis=1).to_numpy()
        X_val = age_val_data.drop('Age',axis=1).drop('Gender',axis=1).to_numpy()
        S_val = age_val_labels['AgeBinary'].to_numpy()
        Y_hat_val = age_val_labels['treatment'].to_numpy()

        # # mean and std for each column (0 means column) (reduce rowns)
        X_mean, X_std = X_train.mean(0), X_train.std(0)
        # normalize data
        X_train = (X_train - X_mean) / (X_std)
        X_test = (X_test - X_mean) / (X_std)
        X_val = (X_val - X_mean) / X_std

        trainset = Adult_dataset(X_train, Y_hat_train, S_train, task='fairness')
        testset = Adult_dataset(X_test, Y_hat_test, S_test, task='fairness')
        valset = Adult_dataset(X_val, Y_hat_val, S_val, task='fairness')
    return trainset, testset, valset


def get_mh(task='fairness'):
    # header is removed automatically (header=0)
    data = pd.read_csv('../data/mental_health/imputed_cleaned_mental_health.csv')

    # drop all rows where Gender is other. inplace=True means it's permanently removed from the original df
    # we now have 1402 examples instead of 1428
    data.drop(data[data['Gender'] == "other"].index, inplace=True)

    columns = ['mh_coverage_flag', 'mh_resources_provided', 'mh_anonimity_flag',
                    'mh_prod_impact', 'mh_medical_leave', 'mh_discussion_neg_impact',
                    'mh_family_hist', 'mh_disorder_past', 'AgeBinary', 'Gender','treatment']

    # drop all columns except for feature_cols
    data = data[columns]

    # replace the binary features with 0s and 1s
    data['Gender'] = data['Gender'].replace(to_replace=['male','female'],value=[1,0])
    data['AgeBinary'] = data['AgeBinary'].replace(to_replace=['< 40yo','>= 40yo'],value=[1,0])

    # these are categorical features that will be encoded
    dummy_variables = ['mh_coverage_flag', 'mh_resources_provided', 'mh_anonimity_flag',
                    'mh_prod_impact', 'mh_medical_leave', 'mh_discussion_neg_impact',
                    'mh_family_hist', 'mh_disorder_past']

    #train test split
    msk = np.zeros(len(data))
    #1st 70% of the list are 1s
    msk[:int(0.7*len(data))] = 1
    #permute the 1s and 0s at random
    msk = np.random.permutation(msk).astype('bool')
    data_train = data[msk]
    data_test = data[~msk]


    def get_variables(data, task='fairness'):

        #Encode target labels with value between 0 and n_classes-1
        le = preprocessing.LabelEncoder()

        #integer encodings to the dummy columns
        data = data.copy() #taking a coopy prevents warnings
        data[dummy_variables] = data[dummy_variables].apply(lambda col: le.fit_transform(col))
        X = data.drop('Gender',axis=1).drop('treatment',axis=1).to_numpy().astype(float)
        S = data['Gender'].to_numpy()
        if task=='fairness':
            T = data['treatment'].to_numpy()
        # else:
        #     T = data.drop('sex',axis=1).drop('salary',axis=1).to_numpy().astype(float)

        return X, S, T

    X_train, S_train, T_train = get_variables(data_train,task)
    X_test, S_test, T_test = get_variables(data_test,task)
    #mean and std for each column (0 means column) (reduce rowns)
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    #normalize data
    X_train = (X_train-X_mean) / (X_std)
    X_test = (X_test-X_mean) / (X_std)

    trainset = Adult_dataset(X_train, T_train, S_train, task=task)
    testset = Adult_dataset(X_test, T_test, S_test, task=task)

    return trainset, testset

# get_celeba()
# csv_file = '../data/celeba_gender_train_jpg.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../data/celeba'
# img_name =os.path.join(root_dir,
#              frame.iloc[1, 1])
# image = io.imread(img_name)

# from shutil import copyfile
# from sys import exit
# from progressbar import progressbar

'''
Change .png to .jpeg
'''

# csv_file = '../data/eyepacs_control_train.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../../eyepacs/train'
#
# dir = '../../eyepacs_small'
# from progressbar import progressbar
#
# for i in progressbar(range(len(frame))):
#     img_name = frame.iloc[i, 1]
#     name_beginning = img_name[:-3]
#     name = name_beginning + 'jpeg'
#     frame.iloc[i, 1] = name
#
# frame.to_csv("eyepacs_control_train_jpeg.csv",index=False)

'''
Copy image into another folder
'''

# csv_file = '../data/eyepacs_control_train_jpeg.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../../eyepacs/test'
# dir = '../../eyepacs_small'
#
# for i in progressbar(range(len(frame))):
#     img_path = os.path.join(root_dir,
#                             frame.iloc[i, 1])
#     output_path = os.path.join(dir,
#                           frame.iloc[i, 1])
#
#     try:
#         copyfile(img_path, output_path)
#         print(img_path)
#     except:
#         print("Unexpected error:")
