import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the H2 dataset."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def split_features_targets(df: pd.DataFrame):
    """
    Separates the features from the targets and removes data leakage variables.
    Target A (Regression): ADR
    Target B (Classification): IsCanceled
    """
    # LEAKAGE AUDIT: 'ReservationStatus' and 'ReservationStatusDate' happen AFTER 
    # the booking lifecycle point where predictions are made. Keeping them causes massive leakage.
    leakage_cols = ['ReservationStatus', 'ReservationStatusDate']
    
    y_class = df['IsCanceled']
    y_reg = df['ADR']
    
    # Drop targets and leakage columns from matrix
    X = df.drop(columns=['IsCanceled', 'ADR'] + leakage_cols)
    return X, y_class, y_reg

def build_preprocessor() -> ColumnTransformer:
    """
    Constructs the preprocessing pipeline for numerical and categorical features.
    Handles missing values, scaling, and categorical encoding.
    """
   
    numeric_features = [
        'LeadTime', 'ArrivalDateYear', 'ArrivalDateWeekNumber', 'ArrivalDateDayOfMonth', 
        'StaysInWeekendNights', 'StaysInWeekNights', 'Adults', 'Children', 'Babies', 
        'PreviousCancellations', 'PreviousBookingsNotCanceled', 'BookingChanges', 
        'DaysInWaitingList', 'RequiredCarParkingSpaces', 'TotalOfSpecialRequests'
    ]
    
    categorical_features = [
        'ArrivalDateMonth', 'Meal', 'Country', 'MarketSegment', 'DistributionChannel', 
        'IsRepeatedGuest', 'ReservedRoomType', 'AssignedRoomType', 'DepositType', 
        'Agent', 'Company', 'CustomerType'
    ]

    #  Build Numeric Pipeline: Impute missing (median)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    #  Build Categorical Pipeline: Impute missing (constant) 
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # to drop any column not handled
    )
    
    return preprocessor

def run_preprocessing_pipeline(data_path: str, output_dir: str):
    """
    Executes the full preprocessing workflow: load, split, fit, transform, and export.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(data_path)
    
    # for handling invalid data in sheet
    df = df[df['ADR'] >= 0].reset_index(drop=True)
    
    # Splitting data
    X, y_class, y_reg = split_features_targets(df)
    
    # Train/Test Split (80% Train, 20% Test)
    # Use fixed  random state 
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )
    
    print("Building and fitting the preprocessing pipeline on training data...")
    preprocessor = build_preprocessor()
    
    # to prevent data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor 
    artifact_path = os.path.join(output_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, artifact_path)
    print(f"Preprocessing artifact saved to: {artifact_path}")
    
    # processed matrices to use it for training
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_processed)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_processed)
    
    # targets
    pd.DataFrame(y_class_train).to_csv(os.path.join(output_dir, 'y_class_train.csv'), index=False)
    pd.DataFrame(y_class_test).to_csv(os.path.join(output_dir, 'y_class_test.csv'), index=False)
    pd.DataFrame(y_reg_train).to_csv(os.path.join(output_dir, 'y_reg_train.csv'), index=False)
    pd.DataFrame(y_reg_test).to_csv(os.path.join(output_dir, 'y_reg_test.csv'), index=False)
    
    print("All processed data split and exported successfully. Preprocessing complete.")

if __name__ == "__main__":
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    DATA_FILE = os.path.join(script_dir, "../data/H2.csv")
    OUTPUT_DIRECTORY = os.path.join(script_dir, "../artifacts/")
    
    run_preprocessing_pipeline(DATA_FILE, OUTPUT_DIRECTORY)