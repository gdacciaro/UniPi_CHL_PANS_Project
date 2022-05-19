import Constants
from models.NaiveBayes import NaiveBayes
from models.DecisionTree import DecisionTree
from models.LRegression import LRegression
from models.RandomForest import RandomForest
from models.SVM import SVM
import json
from models.MLP import MLP
from models.KNN import KNN
from mysql.connector import (connection) #pip install mysql-connector-python

from utilities import get_all_models

def __open_connection():
    try:
        cnx = connection.MySQLConnection(user=Constants.MySQL_user, password=Constants.MySQL_password,
                                         host=Constants.MySQL_host, database=Constants.MySQL_db)
        cursor = cnx.cursor()
    except:
        print("Connection error")
        return

    return cnx, cursor

def __close_connection(connection, cursor):
    connection.commit()
    cursor.close()
    connection.close()


def writeTemp(table):
    all_models = get_all_models()
    connection, cursor = __open_connection()

    for model in all_models:
        serialized_model = model.serialize()
        query = "INSERT INTO "+table+" (reserved, model) VALUES (0, '"+serialized_model+"')"
        print(query)
        cursor.execute(query)

    __close_connection(connection, cursor)


def __parse_model(cursor):
    for (id, reserved, json_model, result, exec_time, who) in cursor:
        model = json.loads(json_model)
        model_type = model["model"]
        model_param = str(model["params"]).replace("\'", "\"")\
                                            .replace("[", "\"(").replace("]", ")\"")\
                                            .replace("True", "true")\
                                            .replace("False", "false")
      
        if model_type == "KNN":
            model = KNN.deserialize(model_param)
        elif model_type == "Naive Bayes":
            model = NaiveBayes.deserialize(model_param)
        elif model_type == "MLP":
            model = MLP.deserialize(model_param)
        elif model_type == "DecisionTree":
            model = DecisionTree.deserialize(model_param)
        elif model_type == "SVM":
            model = SVM.deserialize(model_param)
        elif model_type == "RandomForest":
            model = RandomForest.deserialize(model_param)
        elif model_type == "LRegression":
            model = LRegression.deserialize(model_param)
    return id,model,result

def how_many_models_are_remaining(table):
    connection, cursor = __open_connection()
    query = "select count(*) from " + table + " where result is null and reserved = 0"
    cursor.execute(query)

    count = 0
    for (c) in cursor:
        count = c

    __close_connection(connection, cursor)
    return count[0]


def getBestModel():
    connection, cursor = __open_connection()

    query = "select * from " + Constants.MySQL_name_table + " where result >-1 order by result desc limit 1"
    cursor.execute(query)

    id,model,result = __parse_model(cursor)
    if id is None:
        raise Exception("No model found")

    __close_connection(connection, cursor)

    assert id is not None
    assert model is not None
    return id,model,result

def get_trial_information(table_to_try, dataset_counter):
    table       = table_to_try[dataset_counter][0]

    count = how_many_models_are_remaining(table)

    if count == 0:
        dataset_counter += 1

    table       = table_to_try[dataset_counter][0]
    file_name   = table_to_try[dataset_counter][1]

    return dataset_counter,table,file_name,count

def create_table(table):
    connection, cursor = __open_connection()

    try:
        query = """create table """+table+"""
(
    id        int auto_increment,
    reserved  int default 0 not null,
    model     text          not null,
    result    mediumtext    null,
    exec_time mediumtext    null,
    who       text          null,
    constraint  """+table+"""_id_uindex
        unique (id)
);

alter table  """+table+"""
    add primary key (id);
    
    """
        print(query)
        cursor.execute(query)
        __close_connection(connection, cursor)
    except:
        pass

def getUnsolvedAttempt():
    connection, cursor = __open_connection()

    query = "select * from " + Constants.MySQL_name_table + " where reserved = 0 LIMIT 1"
    cursor.execute(query)

    id,model,_ = __parse_model(cursor)
    print(id, model)
    if id is None:
        raise Exception("No model found")
    assert id is not None
    assert model is not None

    try:
        worker_name = Constants.who
        query2 = "update " + Constants.MySQL_name_table + " set reserved=1, who=\"" + worker_name + "\" where id=" + str(id)
        cursor.execute(query2)

        __close_connection(connection, cursor)
    except:
        print("Connection error")
        return

    return id,model

def postResult(id, result, time):
    """ Once we calculated our attempt, we can post the result on the database"""
    connection, cursor = __open_connection()

    query2 = "update " + Constants.MySQL_name_table + " set result="+str(result)+\
                            ",exec_time=\"" + str(time) + "\", who=\"" + Constants.who + "\" where id=" + str(id)
    cursor.execute(query2)
    __close_connection(connection, cursor)

if __name__ == '__main__':
    lista_tabelle = [
        "a_posteriori_meta_isomap_min_max_scaling_0",
        "a_posteriori_meta_isomap_min_max_scaling_1",
        "a_posteriori_meta_isomap_min_max_scaling_2",
        "a_posteriori_meta_isomap_standard_scaling_0",
        "a_posteriori_meta_isomap_standard_scaling_1",
        "a_posteriori_meta_isomap_standard_scaling_2",
        "a_posteriori_meta_isomap_unscaled_0",
        "a_posteriori_meta_isomap_unscaled_1",
        "a_posteriori_meta_isomap_unscaled_2",
        "a_posteriori_meta_isomap_yeo_johnson_transformation_0",
        "a_posteriori_meta_isomap_yeo_johnson_transformation_1",
        "a_posteriori_meta_isomap_yeo_johnson_transformation_2",
        "a_posteriori_meta_pca_min_max_scaling_0",
        "a_posteriori_meta_pca_min_max_scaling_1",
        "a_posteriori_meta_pca_min_max_scaling_2",
        "a_posteriori_meta_pca_standard_scaling_0",
        "a_posteriori_meta_pca_standard_scaling_1",
        "a_posteriori_meta_pca_standard_scaling_2",
        "a_posteriori_meta_pca_unscaled_0",
        "a_posteriori_meta_pca_unscaled_1",
        "a_posteriori_meta_pca_unscaled_2",
        "a_posteriori_meta_pca_yeo_johnson_transformation_0",
        "a_posteriori_meta_pca_yeo_johnson_transformation_1",
        "a_posteriori_meta_pca_yeo_johnson_transformation_2",
        "a_posteriori_meta_rfe_min_max_scaling_0",
        "a_posteriori_meta_rfe_min_max_scaling_1",
        "a_posteriori_meta_rfe_min_max_scaling_2",
        "a_posteriori_meta_rfe_standard_scaling_0",
        "a_posteriori_meta_rfe_standard_scaling_1",
        "a_posteriori_meta_rfe_standard_scaling_2",
        "a_posteriori_meta_rfe_unscaled_0",
        "a_posteriori_meta_rfe_unscaled_1",
        "a_posteriori_meta_rfe_unscaled_2",
        "a_posteriori_meta_rfe_yeo_johnson_transformation_0",
        "a_posteriori_meta_rfe_yeo_johnson_transformation_1",
        "a_posteriori_meta_rfe_yeo_johnson_transformation_2",
        "a_posteriori_prote_isomap_min_max_scaling_0",
        "a_posteriori_prote_isomap_min_max_scaling_1",
        "a_posteriori_prote_isomap_min_max_scaling_2",
        "a_posteriori_prote_isomap_standard_scaling_0",
        "a_posteriori_prote_isomap_standard_scaling_1",
        "a_posteriori_prote_isomap_standard_scaling_2",
        "a_posteriori_prote_isomap_unscaled_0",
        "a_posteriori_prote_isomap_unscaled_1",
        "a_posteriori_prote_isomap_unscaled_2",
        "a_posteriori_prote_isomap_yeo_johnson_transformation_0",
        "a_posteriori_prote_isomap_yeo_johnson_transformation_1",
        "a_posteriori_prote_isomap_yeo_johnson_transformation_2",
        "a_posteriori_prote_pca_min_max_scaling_0",
        "a_posteriori_prote_pca_min_max_scaling_1",
        "a_posteriori_prote_pca_min_max_scaling_2",
        "a_posteriori_prote_pca_standard_scaling_0",
        "a_posteriori_prote_pca_standard_scaling_1",
        "a_posteriori_prote_pca_standard_scaling_2",
        "a_posteriori_prote_pca_unscaled_0",
        "a_posteriori_prote_pca_unscaled_1",
        "a_posteriori_prote_pca_unscaled_2",
        "a_posteriori_prote_pca_yeo_johnson_transformation_0",
        "a_posteriori_prote_pca_yeo_johnson_transformation_1",
        "a_posteriori_prote_pca_yeo_johnson_transformation_2",
        "a_posteriori_prote_rfe_min_max_scaling_0",
        "a_posteriori_prote_rfe_min_max_scaling_1",
        "a_posteriori_prote_rfe_min_max_scaling_2",
        "a_posteriori_prote_rfe_standard_scaling_0",
        "a_posteriori_prote_rfe_standard_scaling_1",
        "a_posteriori_prote_rfe_standard_scaling_2",
        "a_posteriori_prote_rfe_unscaled_0",
        "a_posteriori_prote_rfe_unscaled_1",
        "a_posteriori_prote_rfe_unscaled_2",
        "a_posteriori_prote_rfe_yeo_johnson_transformation_0",
        "a_posteriori_prote_rfe_yeo_johnson_transformation_1",
        "a_posteriori_prote_rfe_yeo_johnson_transformation_2",
        "a_priori_isomap_min_max_scaling_0",
        "a_priori_isomap_min_max_scaling_1",
        "a_priori_isomap_min_max_scaling_2",
        "a_priori_isomap_standard_scaling_0",
        "a_priori_isomap_standard_scaling_1",
        "a_priori_isomap_standard_scaling_2",
        "a_priori_isomap_unscaled_0",
        "a_priori_isomap_unscaled_1",
        "a_priori_isomap_unscaled_2",
        "a_priori_isomap_yeo_johnson_transformation_0",
        "a_priori_isomap_yeo_johnson_transformation_1",
        "a_priori_isomap_yeo_johnson_transformation_2",
        "a_priori_pca_min_max_scaling_0",
        "a_priori_pca_min_max_scaling_1",
        "a_priori_pca_min_max_scaling_2",
        "a_priori_pca_standard_scaling_0",
        "a_priori_pca_standard_scaling_1",
        "a_priori_pca_standard_scaling_2",
        "a_priori_pca_unscaled_0",
        "a_priori_pca_unscaled_1",
        "a_priori_pca_unscaled_2",
        "a_priori_pca_yeo_johnson_transformation_0",
        "a_priori_pca_yeo_johnson_transformation_1",
        "a_priori_pca_yeo_johnson_transformation_2",
        "a_priori_rfe_min_max_scaling_0",
        "a_priori_rfe_min_max_scaling_1",
        "a_priori_rfe_min_max_scaling_2",
        "a_priori_rfe_standard_scaling_0",
        "a_priori_rfe_standard_scaling_1",
        "a_priori_rfe_standard_scaling_2",
        "a_priori_rfe_unscaled_0",
        "a_priori_rfe_unscaled_1",
        "a_priori_rfe_unscaled_2",
        "a_priori_rfe_yeo_johnson_transformation_0",
        "a_priori_rfe_yeo_johnson_transformation_1",
        "a_priori_rfe_yeo_johnson_transformation_2",
    ]

    for t in lista_tabelle:
        create_table(t)

    for t in lista_tabelle:
        writeTemp(t)

