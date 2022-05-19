from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

import warnings
warnings.filterwarnings("ignore")

def all_scaler(X_train, X_val, X_test):
    standard_scaler = StandardScaler().fit(X_train)
    minmax_scaler = MinMaxScaler().fit(X_train)
    log_scaler = PowerTransformer(method="yeo-johnson").fit(X_train)
    
    distributions_train = [
        ("unscaled", X_train),
        ("standard_scaling", standard_scaler.transform(X_train)),
        ("min_max_scaling", minmax_scaler.transform(X_train)),
        ("yeo_johnson_transformation", log_scaler.transform(X_train)),
    ]
    
    distributions_val = [
        ("unscaled", X_val),
        ("standard_scaling", standard_scaler.transform(X_val)),
        ("min_max_scaling", minmax_scaler.transform(X_val)),
        ("yeo_johnson_transformation", log_scaler.transform(X_val)),
    ]
    
    distributions_test = [
        ("unscaled", X_test),
        ("standard_scaling", standard_scaler.transform(X_test)),
        ("min_max_scaling", minmax_scaler.transform(X_test)),
        ("yeo_johnson_transformation", log_scaler.transform(X_test)),
    ]
    
    return distributions_train, distributions_val, distributions_test
