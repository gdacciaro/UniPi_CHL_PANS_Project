import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import Constants
from Constants import ROOT
from dataset_managers.AbstractDatasetManager import AbstractDatasetManager


class DatasetManager(AbstractDatasetManager):

    def __init__(self):
        super().__init__()


    def __develop_test_split(self, develop_size = 0.7, shuffle = False):
        '''
        Performs splitting between develop and test set.
        :param develop_size: represent the proportion of the dataset to include in the develop split. 
        :param random_seed: random seed for shuffling.
        :param shuffle: whether or not to shuffle the data before splitting.
        '''
        
        assert self.pd_metadata is not None
        assert isinstance(develop_size, float) == True, 'Develop size must be float'
        assert develop_size > 0, 'Only positive numbers are allowed'
        assert self.is_dataset_already_splitted == False, 'Calling this function twice is not allowed: data between the test and development sets could get mixed up'

        self.is_dataset_already_splitted = True

        np.random.seed(42)

        unique_df = self.pd_metadata.drop_duplicates(subset=['Patient id'])
        unique_patients_id = np.array(unique_df.pop('Patient id'))

        if shuffle:
            np.random.shuffle(unique_patients_id)

        split_size = int(len(unique_patients_id)*develop_size)

        ids_development_set = unique_patients_id[:split_size]
        ids_test_set = unique_patients_id[split_size:]

        return ids_development_set, ids_test_set

    def read_dataset_with_feature_selection(self, file_name: str):
        root_folder = ROOT + '/data/' + Constants.folder

        df = pd.read_csv(root_folder + '/'+file_name, sep=',')

        df = df.sample(frac=1, random_state=42) # Shuffle
        development_set, test_set = train_test_split(df, test_size=Constants.DEVELOPMENT_SIZE, random_state=42)

        sample_ids_dev  = development_set.iloc[:, 0]
        targets_dev     = development_set.iloc[:, -1]
        sample_ids_test = test_set.iloc[:, 0]
        targets_test    = test_set.iloc[:, -1]

        development_set = development_set.iloc[:, 1:] #Remove first column
        development_set = development_set.iloc[: , :-1] #Remove last column
        test_set = test_set.iloc[:, 1:] #Remove first column
        test_set = test_set.iloc[: , :-1] #Remove last column

        #development_set = development_set.loc[:, (development_set != 0).any(axis=0)] #Remove any columns of zeros
        #test_set = test_set.loc[:, (test_set != 0).any(axis=0)] #Remove any columns of zeros

        features_dev = development_set.to_numpy()
        targets_dev = targets_dev.to_numpy()
        sample_ids_dev = sample_ids_dev.to_numpy()

        features_test = test_set.to_numpy()
        targets_test = targets_test.to_numpy()
        sample_ids_test = sample_ids_test.to_numpy()

        return features_dev, targets_dev, features_test, targets_test, sample_ids_dev, sample_ids_test




