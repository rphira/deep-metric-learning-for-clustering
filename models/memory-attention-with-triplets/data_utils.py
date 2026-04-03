import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """Handles data loading and preprocessing"""
    
    @staticmethod
    def load_and_preprocess(data_path):
        """preprocessing pipeline"""
        data = pd.read_csv(data_path)
        data = data.rename(columns={'label': 'final_label'})
        data = data[data['final_label'].notna()]
        
        # NaN filtering
        nan_percentage = data.isna().mean() * 100
        data = data.loc[:, nan_percentage <= 10]
        data = data.dropna()
        
        # Column filtering
        cleaned_data = data.loc[:, ~data.columns.str.contains(
            'c3|value__quantile|value__agg_linear_trend|large_standard_deviation|change_quantiles'
        )]
        
        # Drop specific columns
        to_drop = [
            'EventId', 'idxstart', 'idxend', 'risetime', 'value__standard_deviation', 
            'value__mean_second_derivative_central', 'value__linear_trend__attr_"intercept"', 
            'falltime', "I/Io", "Io", "Irms", "length", "value__maximum", "value__minimum", 
            "log_length", 'value__variance_larger_than_standard_deviation', "value__median",
            'value__root_mean_square', 'value__standard_deviation', 
            'value__variance', 'Imean', 'Isig', 'isBL?',
            'value__count_above_mean',
            'value__count_below_mean',
            'value__longest_strike_above_mean', 
            'value__longest_strike_below_mean',
        ]
        
        if 'value__mean' in cleaned_data.columns:
            print("Mean present")
            correlations = cleaned_data.drop(columns=['final_label']).corr()['value__mean'].abs()
            high_corr_features = correlations[correlations > 0.65].index.tolist()
            to_drop.extend(high_corr_features)
            print(f"Added {len(high_corr_features)} mean-correlated features to drop")

        to_drop_final = ['value__mean'] + to_drop
        cleaned_data = cleaned_data.drop(columns=[col for col in to_drop_final if col in cleaned_data.columns])
        
        return cleaned_data
    
    @staticmethod
    def prepare_dataset(cleaned_data, label_col='final_label'):
        """Convert DataFrame to features and labels"""
        X = cleaned_data.drop(columns=[label_col]).values
        y = cleaned_data[label_col].values
        feature_names = cleaned_data.drop(columns=label_col).columns.tolist()
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        return X, y_encoded, feature_names, le
    
    @staticmethod
    def align_features(X_unseen, unseen_feature_names, common_features):
        """Align unseen data features with training feature set"""
        if common_features is None:
            raise ValueError("Common features not set - run training first")
        
        unseen_df = pd.DataFrame(X_unseen, columns=unseen_feature_names)
        
        # Ensure we have all the common features
        aligned_df = pd.DataFrame()
        
        for feature in common_features:
            if feature in unseen_df.columns:
                aligned_df[feature] = unseen_df[feature]
            else:
                # Feature missing in unseen data fill with zeros
                aligned_df[feature] = 0.0
                print(f"    Warning: Feature '{feature}' missing in unseen data, filled with zeros")
        
        # get correct feature order
        aligned_df = aligned_df[common_features]
        
        X_aligned = aligned_df.values
        
        # checks
        print(f"Aligned unseen data: {X_unseen.shape[1]} to {X_aligned.shape[1]} features")
        
        return X_aligned
