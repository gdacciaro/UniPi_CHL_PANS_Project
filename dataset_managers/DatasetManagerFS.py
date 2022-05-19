import numpy as np
import numpy as np
import pandas as pd

import Constants
from dataset_managers.AbstractDatasetManager import AbstractDatasetManager
from utilities import clean_empty_data


class DatasetManager_FeatureSelection(AbstractDatasetManager):

    def __init__(self):
        super().__init__()

        self.patients_id_train1, self.patients_id_val1, self.patients_id_test1, \
        self.patients_id_train2, self.patients_id_val2, self.patients_id_test2, \
        self.patients_id_train3, self.patients_id_val3, self.patients_id_test3 = self.__test_develop_read()
        
        self.sample_ids_train1, self.sample_ids_val1, self.sample_ids_test1 = self.get_sample_ids(self.patients_id_train1), self.get_sample_ids(self.patients_id_val1), self.get_sample_ids(self.patients_id_test1)
        self.sample_ids_train2, self.sample_ids_val2, self.sample_ids_test2 = self.get_sample_ids(self.patients_id_train2), self.get_sample_ids(self.patients_id_val2), self.get_sample_ids(self.patients_id_test2)
        self.sample_ids_train3, self.sample_ids_val3, self.sample_ids_test3 = self.get_sample_ids(self.patients_id_train3), self.get_sample_ids(self.patients_id_val3), self.get_sample_ids(self.patients_id_test3)

    def __test_develop_read(self, training_size = 0.8):
        pd_metadata_first = pd.read_csv(Constants.ROOT + '/data/' + Constants.folder + '/metadata_testset_1.csv', sep=',')
        pd_metadata_second = pd.read_csv(Constants.ROOT + '/data/' + Constants.folder + '/metadata_testset_2.csv', sep=',')
        pd_metadata_third = pd.read_csv(Constants.ROOT + '/data/' + Constants.folder + '/metadata_testset_3.csv', sep=',')
                                                                   
        unique_patients_id1_test = pd_metadata_first.drop_duplicates(subset=['Patient id'])
        ids_test_set1 = np.array(unique_patients_id1_test.pop('Patient id'))

        unique_patients_id2_test = pd_metadata_second.drop_duplicates(subset=['Patient id'])
        ids_test_set2 = np.array(unique_patients_id2_test.pop('Patient id'))
        
        unique_patients_id3_test = pd_metadata_third.drop_duplicates(subset=['Patient id'])
        ids_test_set3 = np.array(unique_patients_id3_test.pop('Patient id'))
        
        df_copy = self.pd_metadata.copy()
        pd_development_first = df_copy.drop(df_copy[df_copy['Patient id'].isin(ids_test_set1)].index)
        
        df_copy = self.pd_metadata.copy()
        pd_development_second = df_copy.drop(df_copy[self.pd_metadata['Patient id'].isin(ids_test_set2)].index)
        
        df_copy = self.pd_metadata.copy()
        pd_development_third = df_copy.drop(df_copy[df_copy['Patient id'].isin(ids_test_set3)].index)
          
        unique_patients_id1_dev = pd_development_first.drop_duplicates(subset=['Patient id'])
        patients_id_first_development = np.array(unique_patients_id1_dev.pop('Patient id'))
        
        unique_patients_id2_dev = pd_development_second.drop_duplicates(subset=['Patient id'])
        patients_id_second_development = np.array(unique_patients_id2_dev.pop('Patient id'))
        
        unique_patients_id3_dev = pd_development_third.drop_duplicates(subset=['Patient id'])
        patients_id_third_development = np.array(unique_patients_id3_dev.pop('Patient id'))
        
        split_size1 = int(len(patients_id_first_development) * training_size)
        split_size2 = int(len(patients_id_second_development) * training_size)
        split_size3 = int(len(patients_id_third_development) * training_size)

        ids_training_set1 = patients_id_first_development[:split_size1]
        ids_val_set1 = patients_id_first_development[split_size1:]
        
        ids_training_set2 = patients_id_second_development[:split_size2]
        ids_val_set2 = patients_id_second_development[split_size2:]
        
        ids_training_set3 = patients_id_third_development[:split_size3]
        ids_val_set3 = patients_id_third_development[split_size3:]
        
        return ids_training_set1, ids_val_set1, ids_test_set1, \
               ids_training_set2, ids_val_set2, ids_test_set2, \
               ids_training_set3, ids_val_set3, ids_test_set3
        
    def __develop_test_split(self, training_size, development_size, shuffle=False):
        '''
        Performs splitting between training, validation and test set.
        :param training_size: represent the proportion of the dataset to include in the training split.
        :param development_size: represent the proportion of the dataset to include in the development split.
        :param shuffle: whether or not to shuffle the data before splitting.
        '''

        assert self.pd_metadata is not None
        assert isinstance(training_size, float) == True, 'Training size must be float'
        assert isinstance(development_size, float) == True, 'Development size must be float'
        assert development_size > 0, 'Only positive numbers are allowed'
        assert self.is_dataset_already_splitted == False, 'Calling this function twice is not allowed: data between the test and development sets could get mixed up'

        self.is_dataset_already_splitted = True

        np.random.seed(42)

        unique_df = self.pd_metadata.drop_duplicates(subset=['Patient id'])
        unique_patients_id = np.array(unique_df.pop('Patient id'))

        if shuffle:
            np.random.shuffle(unique_patients_id)

        split_size = int(len(unique_patients_id) * development_size)

        ids_development_set = unique_patients_id[:split_size]
        ids_test_set = unique_patients_id[split_size:]

        split_size = int(len(ids_development_set) * training_size)

        ids_training_set = unique_patients_id[:split_size]
        ids_validation_set = unique_patients_id[split_size:]

        return ids_training_set, ids_validation_set, ids_test_set

    def get_data2(self, dataset="meta"):    
        data_train1, target_train1 = self.get_features_related_to_ids(self.sample_ids_train1, dataset), self.get_targets(self.sample_ids_train1)
        data_val1, target_val1 = self.get_features_related_to_ids(self.sample_ids_val1, dataset), self.get_targets(self.sample_ids_val1)
        data_test1, target_test1 = self.get_features_related_to_ids(self.sample_ids_test1, dataset), self.get_targets(self.sample_ids_test1)

        data_train2, target_train2 = self.get_features_related_to_ids(self.sample_ids_train2, dataset), self.get_targets(self.sample_ids_train2)
        data_val2, target_val2 = self.get_features_related_to_ids(self.sample_ids_val2, dataset), self.get_targets(self.sample_ids_val2)
        data_test2, target_test2 = self.get_features_related_to_ids(self.sample_ids_test2, dataset), self.get_targets(self.sample_ids_test2)
        
        data_train3, target_train3 = self.get_features_related_to_ids(self.sample_ids_train3, dataset), self.get_targets(self.sample_ids_train3)
        data_val3, target_val3 = self.get_features_related_to_ids(self.sample_ids_val3, dataset), self.get_targets(self.sample_ids_val3)
        data_test3, target_test3 = self.get_features_related_to_ids(self.sample_ids_test3, dataset), self.get_targets(self.sample_ids_test3)
        
        data_train1, target_train1 = clean_empty_data(data_train1, target_train1)
        data_val1, target_val1 = clean_empty_data(data_val1, target_val1)
        data_test1, target_test1 = clean_empty_data(data_test1, target_test1)
        
        data_train2, target_train2 = clean_empty_data(data_train2, target_train2)
        data_val2, target_val2 = clean_empty_data(data_val2, target_val2)
        data_test2, target_test2 = clean_empty_data(data_test2, target_test2)
        
        data_train3, target_train3 = clean_empty_data(data_train3, target_train3)
        data_val3, target_val3 = clean_empty_data(data_val3, target_val3)
        data_test3, target_test3 = clean_empty_data(data_test3, target_test3)
        
        result = list()
        # 1 -> test, 1-> dev
        first_row = (([*self.sample_ids_train1],[*data_train1],[*target_train1]),
                     ([*self.sample_ids_val1],[*data_val1],[*target_val1]),
                     ([*self.sample_ids_test1],[*data_test1],[*target_test1]))
        result.append(first_row)

        # 2 -> test, 2-> dev
        second_row = (([*self.sample_ids_train2],[*data_train2],[*target_train2]),
                      ([*self.sample_ids_val2],[*data_val2],[*target_val2]),
                      ([*self.sample_ids_test2,],[*data_test2],[*target_test2]))
        result.append(second_row)

        # 3-> test, 3-> dev
        third_row = (([*self.sample_ids_train3], [*data_train3], [*target_train3]),
                     ([*self.sample_ids_val3],[*data_val3],[*target_val3]),
                     ([*self.sample_ids_test3], [*data_test3], [*target_test3]))
        result.append(third_row)

        return result
