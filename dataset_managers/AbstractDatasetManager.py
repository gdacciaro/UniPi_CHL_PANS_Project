from abc import ABC, abstractmethod

import pandas as pd
from sklearn import preprocessing
import numpy as np

import Constants
from Constants import ROOT


class AbstractDatasetManager(ABC):

    def __init__(self):
        self.pd_metadata = pd.read_csv(ROOT + '/data/' + Constants.folder + '/metadata.csv', sep=',')
        self.pd_metadata_first = pd.read_csv(ROOT + '/data/' + Constants.folder + '/metadata_testset_1.csv', sep=',')
        self.pd_metadata_second = pd.read_csv(ROOT + '/data/' + Constants.folder + '/metadata_testset_2.csv', sep=',')
        self.pd_metadata_third = pd.read_csv(ROOT + '/data/' + Constants.folder + '/metadata_testset_3.csv', sep=',')
        self.data_meta = pd.read_csv(ROOT+'/data/'  + Constants.folder + '/PANS_metabolomics.csv', sep=',')
        self.data_prote = pd.read_csv(ROOT+'/data/' + Constants.folder + '/PANS_proteomics.csv', sep=',')
        self.data_priori = pd.read_csv(ROOT+'/data/' + "a_priori" + '/a_priori_dataset.csv', sep=',')
        self.data_priori = self.data_priori.drop(columns=['Patient id', 'TIME POINT'])

        self.label_encoder = preprocessing.LabelEncoder()

        self.is_dataset_already_splitted = False

        self.data_meta.rename(columns={'Sample Identifier': 'Sample Id'}, inplace=True)
        self.data_prote.rename(columns={'SampleId': 'Sample Id'}, inplace=True)


    def get_sample_ids(self, patient_ids):
        result = list()

        for id in patient_ids:
            sample_ids = self.pd_metadata.loc[self.pd_metadata['Patient id'] == id]["Sample Id"].values
            for single_sample in sample_ids.tolist():
                result.append(single_sample)

        return result

    def get_features_related_to_ids(self, sample_ids, type):
        """
        Dato un id, restituisce una o più righe del dataset scelto (tramite type), associate a quell'id
        """
        assert type == "meta" or type == "prote" or type == "a_priori"
        assert sample_ids is not None

        result = list()

        column_id_to_delete = "Sample Id"

        if type == "meta":
            dataset = self.data_meta.copy()
        elif type == "prote":
            dataset = self.data_prote.copy()
        else:
            dataset = self.data_priori.copy()

        for sample_id in sample_ids:
            features = dataset.loc[dataset[column_id_to_delete] == sample_id]
            features.pop(column_id_to_delete)
            result.append(features.values.ravel())

        return result


    def get_targets(self, sample_ids):
        assert sample_ids is not None
        assert len(sample_ids) > 0

        result = list()
        target_column = "TIME POINT"

        targets = self.pd_metadata[[target_column]].to_numpy()

        target_values = list()  # Inserisco in questa lista tutti i valori di target
        for t in targets:
            target_values.append(t[0])

        target_values = np.unique(target_values)  # Rimuovo i duplicati

        self.label_encoder.fit(target_values)  # Applico la label encoder

        for sample_id in sample_ids:
            target = self.pd_metadata.loc[self.pd_metadata['Sample Id'] == sample_id][target_column].values[0]
            target_lable_encoded = self.label_encoder.transform([target])
            result.append(target_lable_encoded[0])

        return result
