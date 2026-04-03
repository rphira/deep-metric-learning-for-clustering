from pycatch22 import catch22_all
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pickle
import pandas as pd
import numpy as np

with open('/home/rushang_phira/src/data/complete_feature_sets/all_blockades.pkl', 'rb') as f:
    currents_dict = pickle.load(f)
all_features = []

for key, currents in currents_dict.items():
    print("Extracting for ", key)
    feature_list = []
    feature_names = None

    for ts in currents:
        if len(ts) > 20:
            result = catch22_all(ts)
            feature_values = result['values']
            if feature_names is None:
                feature_names = result['names']
            feature_list.append(feature_values)
    
    features_df = pd.DataFrame(feature_list, columns=feature_names)

    features_df['label'] = key

    all_features.append(features_df)

catch_22_features = pd.concat(all_features, ignore_index=True)
catch_22_features.to_csv("/home/rushang_phira/src/data/complete_feature_sets/catch22_features.csv", index=False)