import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np
import pickle
import pyabf
from tqdm import tqdm
import pyabf
import os

# Load blockades from pickle
# with open('/home/rushang_phira/src/data/complete_feature_sets/all_blockades.pkl', 'rb') as f:
#     currents_dict = pickle.load(f)

# # Reconstruct blockades for tsfresh
# all_blockades = []
# meta_info = []

# series_id = 0

# # Iterate over each class label and its corresponding time series
# for label, currents in tqdm(currents_dict.items(), desc="Reformatting for tsfresh"):
#     for ts in currents:
#         if len(ts) > 20:
#             df = pd.DataFrame({
#                 "series_id": series_id,
#                 "time": np.arange(len(ts)),
#                 "value": ts
#             })
#             all_blockades.append(df)
#             meta_info.append({
#                 "series_id": series_id,
#                 "label": label,
#                 "length": len(ts)
#             })
#             series_id += 1

# # Combine into final DataFrames
# tsfresh_input = pd.concat(all_blockades, ignore_index=True)
# meta_df = pd.DataFrame(meta_info)

# # Extract TSFRESH features
# print("Beginning TSFRESH feature extraction")
# extracted_features = extract_features(
#     tsfresh_input,
#     column_id="series_id",
#     column_sort="time",
#     column_value="value",
#     disable_progressbar=False
# )

# # Reset index and merge with meta
# extracted_features = extracted_features.reset_index(names='series_id')
# final_df = pd.merge(meta_df, extracted_features, on='series_id', how='inner').drop(columns=['series_id'])

# # Save to disk
# final_df.to_csv("/home/rushang_phira/src/data/classified/feature_set/tsfresh/combined_filtered.csv", index=False)

# k8_k12_k16 = pd.read_csv('/home/rushang_phira/src/data/classified/2021-03-19_3.5_L_H4Ac_K8_K12-2.abf_currentlevels.10_classified.csv')
# raw_data = pyabf.ABF('/mnt/vnas/ml4nanopore_data/processed/2021-03-19 AeL R220S Acetylated Peptides ABF/2021-03-19 3.5 µL H4Ac K8 K12-2.abf')

# feature_list = []
# all_blockades = []

# k8_k12_k16 = k8_k12_k16.reset_index(drop=True)
# k8_k12_k16['series_id'] = k8_k12_k16.index

# for i, row in tqdm(k8_k12_k16.iterrows(), total=len(k8_k12_k16), desc="Processing blockades"):
#     s, e = int(row["idxstart"]), int(row["idxend"])
#     blockade = raw_data.data[0][s:e]

#     df = pd.DataFrame({
#         "series_id": i,
#         "time": np.arange(len(blockade)),
#         "value": blockade
#     })
#     all_blockades.append(df)


# tsfresh_input = pd.concat(all_blockades, ignore_index=False)
# print(len(tsfresh_input))
# print("Beginning feature extraction")
# extracted_features = extract_features(
#     tsfresh_input, 
#     column_id="series_id",
#     column_sort="time", 
#     column_value="value",
#     disable_progressbar=False
# )

# extracted_features = extracted_features.reset_index(names='series_id')

# meta_cols = ['EventId', 'series_id', 'idxstart', 'idxend', 'risetime', 'falltime', 'length',
#        'Irms', 'Io', 'I/Io', 'Imean', 'Isig', 'isBL?', 'log_length',
#        'final_label']
# meta = k8_k12_k16[meta_cols]

# k8_k12_k16 = pd.merge(
#     meta,
#     extracted_features,
#     on="series_id",
#     how='inner'
# )

# assert len(k8_k12_k16) == len(meta), "Merge lost/gained rows unexpectedly!"

# k8_k12_k16 = k8_k12_k16.drop(columns=['series_id'])

# k8_k12_k16.to_csv("/home/rushang_phira/src/data/complete_feature_sets/Mrunal_K8_K12_tsfresh.csv", index=False)  

'''this segment is for extracting tsfresh features for each single metadata and abf file combination.'''
# change paths to metadata and abf files as needed.
k8_k12_k16 = pd.read_csv('/home/rushang_phira/src/data/classified/2021-03-19_3.5_L_H4Ac_mono_di-2.abf_currentlevels_classified.csv')
raw_data = pyabf.ABF('/mnt/vnas/ml4nanopore_data/processed/2021-03-19 AeL R220S Acetylated Peptides ABF/2021-03-19 3.5 µL H4Ac mono di-2.abf')

feature_list = []
all_blockades = []

k8_k12_k16 = k8_k12_k16.reset_index(drop=True)
k8_k12_k16['series_id'] = k8_k12_k16.index

for i, row in tqdm(k8_k12_k16.iterrows(), total=len(k8_k12_k16), desc="Processing blockades"):
    s, e = int(row["idxstart"]), int(row["idxend"])
    blockade = raw_data.data[0][s:e]

    df = pd.DataFrame({
        "series_id": i,
        "time": np.arange(len(blockade)),
        "value": blockade
    })
    all_blockades.append(df)


tsfresh_input = pd.concat(all_blockades, ignore_index=False)
print(len(tsfresh_input))
print("Beginning feature extraction")
extracted_features = extract_features(
    tsfresh_input, 
    column_id="series_id",
    column_sort="time", 
    column_value="value",
    disable_progressbar=False
)

extracted_features = extracted_features.reset_index(names='series_id')

meta_cols = ['EventId', 'series_id', 'idxstart', 'idxend', 'risetime', 'falltime', 'length',
       'Irms', 'Io', 'I/Io', 'Imean', 'Isig', 'isBL?', 'log_length',
       'final_label']
meta = k8_k12_k16[meta_cols]

k8_k12_k16 = pd.merge(
    meta,
    extracted_features,
    on="series_id",
    how='inner'
)

assert len(k8_k12_k16) == len(meta), "Merge lost/gained rows unexpectedly!"

k8_k12_k16 = k8_k12_k16.drop(columns=['series_id'])

k8_k12_k16.to_csv("/home/rushang_phira/src/data/complete_feature_sets/Mrunal_mono_di_tsfresh.csv", index=False)  

'''this segment is for extracting tsfresh features for all blockades of all species combined together, and appropriately labelled'''
def find_abf_path(rel: str):
    rel = str(rel)
    rel = rel.replace('/home/mrunal', '/mnt/vnas')
    if "tobi_meth" in rel:
        tail = os.path.basename(rel)
        date = tail.split()[0]
        folder = f"{date} AeL R220S Trimethylated Peptides Biosynthan"
        return os.path.join("/home/mrunal/ml4nanopores/data/raw/Ensslen_JACS", folder, f"{tail}.abf")
    if "priyanka_new_outputs" in rel:
        mid = "aplysia_peptides" if "AP" in rel else "MDRprotein"
        tail = os.path.basename(rel)
        return os.path.join(f"/home/mrunal/ml4nanopores/data/raw/priyanka_data/{mid}", f"{tail}.abf")
    if "pseudo-labeled-PL" in rel:
        if "L1" in rel:
            mid = "/mnt/vnas/ml4nanopore_data/raw/PeptideLadders/Ladder1"
        elif "L2" in rel:
            mid = "/mnt/vnas/ml4nanopore_data/raw/PeptideLadders/Ladder2"
            tail = os.path.basename(rel).replace("_", " ").replace(".abf curren", "")
            abf_path = os.path.join(mid, f"{tail}.abf")
            p0, p1 = abf_path.split('L2')
            return p0.replace('1L','1µL') + 'L2' + p1.replace(' ','_')
        elif "L3" in rel:
            mid = "/mnt/vnas/ml4nanopore_data/raw/PeptideLadders/Ladder3"
        elif "L4" in rel:
            mid = "/mnt/vnas/ml4nanopore_data/raw/PeptideLadders/Ladder4"
        elif "L5" in rel:
            mid = "/mnt/vnas/ml4nanopore_data/raw/PeptideLadders/Ladder5"
        else:
            mid = "/mnt/vnas/ml4nanopore_data/raw/PeptideLadders/Ladder6"
        tail = os.path.basename(rel).replace("_", " ").replace(".abf curren", "")
        return os.path.join(mid, f"{tail}.abf")
    if "pseudo-labeled-H4" in rel:
        raw = os.path.basename(rel)
        tail = raw.replace("_"," ").replace(".abf curren","").replace(" L "," µL ")
        date = tail.split()[0]
        folder = f"{date} AeL R220S Acetylated Peptides ABF"
        return os.path.join("/home/mrunal/ml4nanopores/data/processed", folder, f"{tail}.abf")
    raise FileNotFoundError(f"No ABF mapping for {rel}")

# Load your CSV file
df = pd.read_csv('/home/rushang_phira/src/data/classified/feature_set/tsfresh/combined_filtered.csv')  # Replace with your actual CSV path
df = df[df['file'].str.contains('pseudo-labeled-PL', na=False)].copy()
# Remove rows with NaN labels if needed
df = df.dropna(subset=['final_label'])

# Group by file to process each ABF file efficiently
feature_list = []
all_blockades = []

# Add series_id for tsfresh
df = df.reset_index(drop=True)
df['series_id'] = df.index

# Process each unique ABF file
unique_files = df['file'].unique()

for file_path in tqdm(unique_files, desc="Processing ABF files"):
    try:
        # Find the actual ABF path using your function
        abf_path = find_abf_path(file_path)
        print(abf_path)
        # Load the ABF file
        raw_data = pyabf.ABF(abf_path)
        
        # Get all events for this file
        file_events = df[df['file'] == file_path]
        
        # Process each event in this file
        for i, row in tqdm(file_events.iterrows(), total=len(file_events), desc=f"Processing {os.path.basename(file_path)}", leave=False):
            s, e = int(row["idxstart"]), int(row["idxend"])
            blockade = raw_data.data[0][s:e]

            df_blockade = pd.DataFrame({
                "series_id": row['series_id'],
                "time": np.arange(len(blockade)),
                "value": blockade
            })
            all_blockades.append(df_blockade)
            
    except FileNotFoundError as e:
        print(f"Warning: Could not find ABF file for {file_path}: {e}")
        continue
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

# Combine all blockades for tsfresh
if all_blockades:
    tsfresh_input = pd.concat(all_blockades, ignore_index=True)
    print(f"Total time series points: {len(tsfresh_input)}")
    print(f"Number of unique series: {tsfresh_input['series_id'].nunique()}")
    
    print("Beginning feature extraction")
    extracted_features = extract_features(
        tsfresh_input, 
        column_id="series_id",
        column_sort="time", 
        column_value="value",
        disable_progressbar=False,
    )
    
    extracted_features = extracted_features.reset_index(names='series_id')
    
    # Define metadata columns to keep
    meta_cols = ['EventId', 'series_id', 'idxstart', 'idxend', 'risetime', 'falltime', 'length',
                 'Irms', 'Io', 'I/Io', 'Imean', 'Isig', 'isBL?', 'log_length',
                 'final_label', 'file']  # Added 'file' column
    
    meta = df[meta_cols]
    
    # Merge features with metadata
    result_df = pd.merge(
        meta,
        extracted_features,
        on="series_id",
        how='inner'
    )
    
    print(f"Original metadata rows: {len(meta)}")
    print(f"Final result rows: {len(result_df)}")
    
    # Verify no data loss
    if len(result_df) != len(meta):
        print(f"Warning: Merge resulted in row count mismatch. Expected {len(meta)}, got {len(result_df)}")
        # Find missing series_ids
        missing_ids = set(meta['series_id']) - set(extracted_features['series_id'])
        if missing_ids:
            print(f"Missing series_ids in features: {missing_ids}")
    
    # Drop series_id if not needed in final output
    result_df = result_df.drop(columns=['series_id'])
    
    # Save the result
    output_path = "/home/rushang_phira/src/data/complete_feature_sets/everything_tsfresh.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")
    
else:
    print("No blockades were successfully processed.")