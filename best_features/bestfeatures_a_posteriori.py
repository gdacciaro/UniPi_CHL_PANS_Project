import matplotlib.pyplot as plot
import pandas as pd
from matplotlib_venn import venn3
import Constants
import warnings
warnings.simplefilter("ignore", UserWarning)

meta_best_models = [
    ("1", "a_posteriori/a_posteriori_meta_rfe_yeo_johnson_transformation_0.csv"),
    ("2", "a_posteriori/a_posteriori_meta_rfe_yeo_johnson_transformation_1.csv"),
    ("3", "a_posteriori/a_posteriori_meta_rfe_yeo_johnson_transformation_2.csv"),
]
prote_best_models = [
    ("1", "a_posteriori/a_posteriori_prote_rfe_yeo_johnson_transformation_0.csv"),
    ("2", "a_posteriori/a_posteriori_prote_rfe_yeo_johnson_transformation_1.csv"),
    ("3", "a_posteriori/a_posteriori_prote_rfe_yeo_johnson_transformation_2.csv"),
]
types = ["metabolomics","proteomics"]
for type in types:
    best_dataset = meta_best_models if type=="metabolomics" else prote_best_models
    for index,dictionary in enumerate([best_dataset]):
        print("=======",type,"=========")
        filenames = [filename for (title, filename) in dictionary]
        tmp_data = {}
        for table, filename in dictionary:
            data = pd.read_csv(Constants.ROOT + '/data/' + "" + filename, sep=',')
            data = data.iloc[: , 1:]
            data = data.iloc[:, :-1]
            tmp_data[filename] = (table, data.copy())
        size = len(dictionary)

        only_0 = 0
        only_1 = 0
        only_2 = 0
        in_0_and_1 = 0
        in_1_and_2 = 0
        in_0_and_2 = 0
        in_0_and_1_and_2 = 0

        for i in range(0, size):
            filename = filenames[i]
            table, df = tmp_data[filename]

            for col in df.columns:
                matched_table = list()

                for j in range(i+1, size):
                    next_filename = filenames[j]
                    next_table, next_df = tmp_data[next_filename]

                    for next_col in next_df.columns:
                        if next_col == col:
                            matched_table.append(j)
                            next_df = next_df.drop(columns=[next_col])
                            tmp_data[next_filename] = next_table, next_df

                if len(matched_table)==0:
                    if i == 0:
                        only_0 +=1
                    if i == 1:
                        only_1 +=1
                    if i == 2:
                        only_2 +=1

                if len(matched_table)==2:
                    in_0_and_1_and_2 +=1

                if len(matched_table)==1:
                    if i == 0 and matched_table[0]==1 or i == 1 and matched_table[0]==0:
                        in_0_and_1 +=1
                    if i == 1 and matched_table[0] == 2 or i == 2 and matched_table[0] == 1:
                        in_1_and_2 +=1
                    if i == 0 and matched_table[0] == 2 or i == 2 and matched_table[0] == 0:
                        in_0_and_2 +=1

        items=[only_0,  only_1,
               in_0_and_1,
               only_2,
               in_0_and_2,in_1_and_2,
                    in_0_and_1_and_2]

        labels=[title for (title, filenam) in best_dataset]

        venn3(subsets=items,set_labels=labels,alpha=0.5)
        plot.title(type)
        plot.show()


        scoring_features = {}
        tmp_data = {}
        for table, filename in dictionary:
            data = pd.read_csv(Constants.ROOT + '/data/' + "" + filename, sep=',')
            data = data.iloc[:, 1:]
            data = data.iloc[:, :-1]
            tmp_data[filename] = (table, data.copy())

        #Init the dictionary
        for i in range(0, size):
            filename = filenames[i]
            table, df = tmp_data[filename]
            for col in df.columns:
                scoring_features[col] = 0

        #Applying scoring
        for i in range(0, size):
            filename = filenames[i]
            table, df = tmp_data[filename]
            for col in df.columns:
                scoring_features[col] += 1

        for item in scoring_features:
            if scoring_features[item] == 3:
                print(3,"|",item)
        print("===")
        for item in scoring_features:
            if scoring_features[item] == 2:
                print(2,"|",item)

        print("\n\n\n")