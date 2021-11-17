import torch
import torchvision
from torchvision import transforms
from skimage import io, transform
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from tqdm import tqdm

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

    def __init__(self, csv_file, root_dir, dataset, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = dataset

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

        if self.dataset == 0:
            sensitive = self.frame.iloc[index,3] #sensitive is gender
        elif self.dataset == 1:
            sensitive = self.frame.iloc[index, 4]  # sensitive is race

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

#for testing with the aa clinician labels testset
class Eyepacs_race_test_dataset(torch.utils.data.Dataset):

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

        return image, target, self.frame.iloc[index, 1] #name


class Fairface_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, dataset, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir # "../data/fairface/"
        self.transform = transform
        self.dataset = dataset

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

        if self.dataset == 6:
            target = self.frame.iloc[index,6] #target is in thirties
            sensitive = self.frame.iloc[index,7] #sensitive is is_male

        elif self.dataset == 7:
            target = self.frame.iloc[index,7] #target is male
            sensitive = self.frame.iloc[index,9] #sensitive is race_black

        return image, target, sensitive


def get_fairface(debugging,dataset_type):
    root_dir = "../data/fairface/"

    # image_size = 224

    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]) #scales to [-1,1]


    if debugging:
        trainset = Fairface_dataset('../data/fairface_test.csv',root_dir,dataset_type,
                                  transform)
        testset = Fairface_dataset('../data/fairface_test.csv',root_dir,dataset_type,
                                  transform)
    else:
        if dataset_type == 6: #predict age, gender sensitive
            trainset = Fairface_dataset('../data/fairface_train_nov2.csv',root_dir,dataset_type,
                                      transform)
            testset = Fairface_dataset('../data/fairface_val_nov2.csv',root_dir,dataset_type,
                                      transform)
        elif dataset_type == 7: #predict gender, race sensitive  black is minority
            trainset = Fairface_dataset('../data/fairface_train_nov2.csv', root_dir, dataset_type,
                                        transform)
            testset = Fairface_dataset('../data/fairface_val_nov2.csv',root_dir,dataset_type,
                                      transform)

    return trainset, testset


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

# returns ordinary test dataset, aa testset with special dataloader
def get_testaa_eyepacs():
    root_dir = "../data/eyepacs_aa"

    image_size = 256

    transform = transforms.Compose([transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]) #scales to [-1,1]

    #REMOVE THE DEBUGGING
    # csv_file = '../data/eyepacs_debugging.csv'
    csv_file = '../data/eyepacs_control_train_jpeg.csv'

    root_dir = "../data/eyepacs_small"
    trainset = Eyepacs_dataset(csv_file,root_dir,
                              transform)
    # testset =  Eyepacs_race_test_dataset('../data/eyepacs_debugging.csv',root_dir,
    #                               transform)


    root_dir = "../data/eyepacs_aa"
    testset = Eyepacs_race_test_dataset('../data/test_dr_aa_jpeg.csv',root_dir,
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
        trainset = Celeba_dataset(csv_file,root_dir,dataset,
                                  transform)
        testset = Celeba_dataset(csv_file,root_dir,dataset,transform)
    else:
        if dataset == 0: #gender
            trainset = Celeba_dataset('../data/celeba_gender_train_jpg.csv',root_dir,dataset,
                                      transform)
        elif dataset == 1 : #race
            trainset = Celeba_dataset('../data/celeba_skincolor_train_jpg.csv',root_dir,dataset,
                                      transform)
        else:
            print("error")
            return False
        testset = Celeba_dataset('../data/celeba_balanced_combo_test_jpg.csv',root_dir,dataset,
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


class MNIST_dataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, sensitive, transform=None):
        # data is (num_examples, 3, 28, 28) with 1 of the 3 rows containing the image data, rest 0s
        # data hidden is list of numbers 0,1,2
        self.data = data
        self.targets = targets
        self.sensitive = sensitive
        self.transform = transform
        self.target_vals = 10
        self.hidden_vals = 3

    def __getitem__(self, index):
        image, target, sensitive = self.data[index], self.targets[index], self.sensitive[index]
        # pil-python imaging library
        image, target, sensitive = torchvision.transforms.functional.to_pil_image(image), int(target), int(sensitive)
        # trainsform is trainset.transform / testset.transform
        if self.transform is not None:
            image = self.transform(image)

        return image, target, sensitive


    def __len__(self):
        return len(self.targets)

def get_mnist(debugging=False):
    # Load normal MNIST dataset
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
                                          transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.ToTensor(), ]))
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
                                         transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.ToTensor(), ]))

    # Add the color and normalize to 0..1
    N_tr = len(trainset)  # 60000
    data_n = torch.zeros(N_tr, 3, 28, 28)
    sensitive = torch.arange(len(trainset.targets)) % 3 % 2  # list of numbers from 0 to 2  1/3 2/3 split
    # mnist data gets added to either row 0, 1, or 2
    # for each n, 1 28x28 with data, 2 28x28 of 0s
    for n in range(N_tr):
        data_n[n, sensitive[n]] = trainset.data[n]  # shape (28,28)
    data_n /= 255.0
    trainset = MNIST_dataset(data_n, trainset.targets, sensitive, trainset.transform)

    N_tst = len(testset)
    data_n = torch.zeros(N_tst, 3, 28, 28)
    sensitive = torch.arange(len(testset.targets)) % 3 % 2
    for n in range(N_tst):
        data_n[n, sensitive[n]] = testset.data[n]
    data_n /= 255.0
    testset = MNIST_dataset(data_n, testset.targets, sensitive, testset.transform)

    if debugging:
        return testset, testset

    return trainset, testset


def process_fairface(debugging=True):
    root_dir = "../data"
    image_size = 255
    transform = transforms.Compose([
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]) #scales to [-1,1]

    csv_file = '../data/fairface/full sheets/fairface_label_train.csv'
    frame = pd.read_csv(csv_file)
    print(frame)

    frame['thirties_or_higher'] = np.nan
    frame['is_male'] = np.nan
    frame['is_white'] = np.nan
    frame['is_black'] = np.nan
    count_low = 0
    count_total = 0
    for i in range(len(frame)):
        try:
            a = int(frame['age'][i][0])
            if a <= 2 or frame['age'][i][1] == '-': #age is 1 digit number
                print ('t')

                frame['thirties_or_higher'][i] = 0
                count_low +=1
            else:
                print('f')
                frame['thirties_or_higher'][i] = 1
            count_total += 1
        except ValueError:
            pass
        if frame['gender'][i] == 'Male':
            frame['is_male'][i] = 1
            print('ismale')

        elif frame['gender'][i] == 'Female':
            frame['is_male'][i] = 0
            print('no')

        if frame['race'][i] == 'White':
            frame['is_white'][i] = 1
        else:
            frame['is_white'][i] = 0

        if frame['race'][i] == 'Black':
            frame['is_black'][i] = 1
        else:
            frame['is_black'][i] = 0



    frame['thirties_or_higher'] = frame['thirties_or_higher'].fillna(1)
    print(frame)
    print(count_low)
    print(count_total)
    print(count_low/count_total)

    frame.to_csv('../data/fairface_train_nov2.csv')

# process_fairface()

'''
Get stats on dataset

import pandas as pd
csv_file = '../data/celeba_gender_train_jpg.csv'

csv_file = '../data/celeba_balanced_combo_test_jpg.csv'
csv_file = '../data/fairface_val_nov2.csv'
frame = pd.read_csv(csv_file)
frame[['is_black','is_male']].sum()
frame[['Old','Female']].mean()


frame.loc[(frame['Old']==True) & (frame['Female']==True)].sum()
frame.loc[(frame['is_male']==True) & (frame['is_black']==True)].sum()



csv_file = '../data/fairface_val_good_oct27.csv'
frame = pd.read_csv(csv_file)
frame.loc[(frame['is_male']==True) & (frame['is_black']==True)].sum()

'''

def balance_dataset():

    csv_file = '../data/fairface_train_good_3.csv'
    frame = pd.read_csv(csv_file)
    newframe = pd.DataFrame()
    valframe = pd.DataFrame()

    count = 0
    i = 0
    while tqdm(count < 6000):
        row = frame.iloc[i]
        if row['is_male']==True and row['is_black']==True:
            if count <5500:
                newframe = newframe.append(row)
            else:
                valframe = valframe.append(row)
            count += 1
        i += 1

    count = 0
    i = 0
    while tqdm(count < 6000):
        row = frame.iloc[i]
        if row['is_male']==True and row['is_black']==False:
            if count <5500:
                newframe = newframe.append(row)
            else:
                valframe = valframe.append(row)
            count += 1
        i += 1

    count = 0
    i = 0
    while tqdm(count < 6000):
        row = frame.iloc[i]
        if row['is_male']==False and row['is_black']==False:
            if count <5500:
                newframe = newframe.append(row)
            else:
                valframe = valframe.append(row)
            count += 1
        i += 1

    newframe.to_csv('fairface_train_good_oct27.csv')
    valframe.to_csv('fairface_val_good_oct27.csv')

def balance_dataset_test():

    csv_file = '../data/fairface_val_good_4.csv'
    frame = pd.read_csv(csv_file)
    testframe = pd.DataFrame()

    count = 0
    i = 0
    while tqdm(count < 750):
        row = frame.iloc[i]
        if row['is_male']==True and row['is_black']==True:
            testframe = testframe.append(row)
            count += 1
        i += 1

    count = 0
    i = 0
    while tqdm(count < 750):
        row = frame.iloc[i]
        if row['is_male']==True and row['is_black']==False:
            testframe = testframe.append(row)
            count += 1
        i += 1

    count = 0
    i = 0
    while tqdm(count < 750):
        row = frame.iloc[i]
        if row['is_male']==False and row['is_black']==False:
            testframe = testframe.append(row)
            count += 1
        i += 1

    count = 0
    i = 0
    while tqdm(count < 750):
        row = frame.iloc[i]
        if row['is_male']==False and row['is_black']==True:
            testframe = testframe.append(row)
            count += 1
        i += 1

    testframe.to_csv('fairface_test_good_oct27.csv')

# balance_dataset_test()
# get_celeba()
# csv_file = '../data/celeba_gender_train_jpg.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../data/celeba'
# img_name =os.path.join(root_dir,
#              frame.iloc[1, 1])
# image = io.imread(img_name)

from shutil import copyfile
# from sys import exit
# from progressbar import progressbar

'''
Change .png to .jpeg
'''

# csv_file = '../data/test_dr_aa.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../../eyepacs/train'
# dir = '../../eyepacs_small'
# from progressbar import progressbar
#
# for i in progressbar(range(len(frame))):
#     img_name = frame.iloc[i, 1]
#     name_beginning = img_name[:-3]
#     name = name_beginning + 'jpeg'
#     frame.iloc[i, 1] = name
#
# frame.to_csv("../data/test_dr_aa_jpeg.csv",index=False)

'''
Copy image into another folder
'''

# csv_file = '../data/test_dr_aa_jpeg.csv'
# frame = pd.read_csv(csv_file)
# root_dir = '../../eyepacs/train'
# dir = '../data/eyepacs_aa'
#
# for i in progressbar(range(len(frame))):
#     img_path = os.path.join(root_dir,
#                             frame.iloc[i, 1])
#     output_path = os.path.join(dir,
#                           frame.iloc[i, 1])
#
#     try:
#         copyfile(img_path, output_path)
#         # print(img_path)
#     except:
#         print("fail")
#         print(img_path)
#
