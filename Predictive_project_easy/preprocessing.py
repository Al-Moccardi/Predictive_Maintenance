
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tqdm
from tqdm import tqdm  # noqa: F811
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # noqa: F401


def calculate_rul(dataset):
    """
    Calculate Remaining Useful Life (RUL) for each machineID in the dataset.
    RUL counts down to 0 anytime a failure is detected.

    Parameters:
    - dataset: pandas DataFrame with columns ['datetime', 'machineID', 'failure', ...]

    Returns:
    - dataset: pandas DataFrame with an additional 'RUL' column
    """
    # Step 1: Sort by machineID and datetime in descending order
    dataset_sorted = dataset.sort_values(by=['machineID', 'datetime'], ascending=[True, False])

    # Step 2: Indicate failure occurrence
    dataset_sorted['failure_occurred'] = dataset_sorted['failure'].notna().astype(int)

    # Step 3 & 4: Calculate RUL
    dataset_sorted['group'] = dataset_sorted.groupby('machineID')['failure_occurred'].cumsum()
    dataset_sorted['RUL'] = dataset_sorted.groupby(['machineID', 'group']).cumcount()

    # Optional: Drop the temporary columns
    dataset_sorted.drop(['failure_occurred', 'group'], axis=1, inplace=True)

    # Step 5: Sort back to original order
    dataset_final = dataset_sorted.sort_values(by=['machineID', 'datetime'], ascending=[True, True])
    
    # Reset index to maintain order
    dataset_final.reset_index(drop=True, inplace=True)
    
    return dataset_final


def add_cumulative_failures(dataset, failure_column, column_name):
    """
    Adds a column to the dataset indicating the cumulative number of failures for each machineID,
    based on the specified failure indicator column.

    Parameters:
    - dataset: pandas DataFrame with columns including 'datetime', 'machineID', and the specified failure_column
    - failure_column: string, the name of the column to use as the failure indicator

    Returns:
    - dataset: pandas DataFrame with an additional 'cumulative_failures' column
    """
    # Check if the specified failure_column exists in the dataset
    if failure_column in dataset.columns:
        # Convert to a binary indicator: 1 for failure, 0 otherwise
        dataset['failure_occurred'] = dataset[failure_column].notna().astype(int)
    else:
        raise ValueError(f"The dataset must contain a '{failure_column}' column.")

    # Calculate cumulative failures for each machineID
    dataset[column_name] = dataset.groupby('machineID')['failure_occurred'].cumsum()

    # Optional: Remove the temporary 'failure_occurred' column if you want to keep the dataset clean
    dataset.drop(['failure_occurred'], axis=1, inplace=True)

    return dataset



def fill_nulls_in_category_column(dataset, column_name, fill_value='missing'):
    """
    Fills null (NaN) values in a specified categorical column with a given string, ensuring the
    fill value is added to the column's categories if it's of type 'Categorical'.

    Parameters:
    - dataset: pandas DataFrame
    - column_name: string, the name of the column to fill null values in
    - fill_value: string, the value to use as the replacement for null values (default is 'missing')

    Returns:
    - dataset: pandas DataFrame with null values in the specified column filled
    """
    if column_name in dataset.columns:
        # Check if the column is categorical and the fill value is not a current category
        if pd.api.types.is_categorical_dtype(dataset[column_name]) and fill_value not in dataset[column_name].cat.categories:
            # Add the fill_value to the categories
            dataset[column_name] = dataset[column_name].cat.add_categories([fill_value])
        dataset[column_name] = dataset[column_name].fillna(fill_value)
    else:
        raise ValueError(f"The dataset must contain a '{column_name}' column.")

    return dataset


def add_cumulative_failures(dataset, failure_column, column_name):  # noqa: F811
    """
    Adds a column to the dataset indicating the cumulative number of failures for each machineID,
    based on the specified failure indicator column.

    Parameters:
    - dataset: pandas DataFrame with columns including 'datetime', 'machineID', and the specified failure_column
    - failure_column: string, the name of the column to use as the failure indicator

    Returns:
    - dataset: pandas DataFrame with an additional 'cumulative_failures' column
    """
    # Check if the specified failure_column exists in the dataset
    if failure_column in dataset.columns:
        # Convert to a binary indicator: 1 for failure, 0 otherwise
        dataset['failure_occurred'] = dataset[failure_column].notna().astype(int)
    else:
        raise ValueError(f"The dataset must contain a '{failure_column}' column.")

    # Calculate cumulative failures for each machineID
    dataset[column_name] = dataset.groupby('machineID')['failure_occurred'].cumsum()

    # Optional: Remove the temporary 'failure_occurred' column if you want to keep the dataset clean
    dataset.drop(['failure_occurred'], axis=1, inplace=True)

    return dataset



# def create_and_pad_sequences_with_augmentation(df, target_column, sequence_length, padding_value=0, step=1, min_rul=20, num_augmented_batches=3):
#     sequences = []  # To store sequences, including padded ones
#     labels = []  # To store labels for each sequence
    
#     # Ensure required columns are present
#     required_columns = [target_column, 'machineID', 'datetime']
#     if not all(column in df.columns for column in required_columns):
#         raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
#     # Iterate through each machineID with progress tracking
#     for _, machine_group in tqdm(df.groupby('machineID'), desc="Processing Machine IDs"):
#         # Filter segments by RUL < min_rul
#         machine_group = machine_group[machine_group[target_column] < min_rul]
        
#         # Track segments by splitting at points where RUL resets (if applicable)
#         rul_zero_indices = machine_group.index[machine_group[target_column] == 0].tolist()
#         segment_boundaries = [-1] + rul_zero_indices  # Start from -1 to include the first segment
        
#         for i in range(len(segment_boundaries) - 1):
#             start_index = segment_boundaries[i] + 1
#             end_index = segment_boundaries[i + 1] + 1
#             segment = machine_group.iloc[start_index:end_index]
            
#             # Sort segment by datetime
#             segment = segment.sort_values(by='datetime', ascending=True)
            
#             # Create and pad sequences within this segment with step-wise shifting for augmentation
#             aug_step = 0  # Initialize augmentation step
#             while aug_step < num_augmented_batches:
#                 for start_seq in range(aug_step, len(segment) - sequence_length + 1, step):
#                     end_seq = start_seq + sequence_length
#                     if end_seq <= len(segment):
#                         sequence = segment.iloc[start_seq:end_seq]
                        
#                         sequences.append(sequence.drop([target_column, 'machineID', 'datetime'], axis=1).values)
#                         label = sequence[target_column].iloc[-1]  # Last RUL value as label
#                         labels.append(label)
#                 aug_step += 1  # Increment to create next batch of augmented sequences

#     # Convert sequences and labels to numpy arrays
#     sequence_array = np.array(sequences, dtype=object)
#     label_array = np.array(labels)

#     return sequence_array, label_array



# def create_and_pad_sequences_with_progress_clipped(df, target_column, sequence_length, padding_value=0, step=1):
#     sequences = []  # To store sequences, including padded ones
#     labels = []  # To store labels for each sequence
    
#     # Ensure required columns are present
#     required_columns = [target_column, 'machineID', 'datetime']
#     if not all(column in df.columns for column in required_columns):
#         raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
#     # Iterate through each machineID with progress tracking
#     for _, machine_group in tqdm(df.groupby('machineID'), desc="Processing Machine IDs"):
#         # Filter segments by RUL < 50
#         machine_group = machine_group[machine_group[target_column] < 50]
        
#         # Track segments by splitting at points where RUL resets (if applicable)
#         rul_zero_indices = machine_group.index[machine_group[target_column] == 0].tolist()
#         segment_boundaries = [-1] + rul_zero_indices  # Start from -1 to include the first segment
        
#         for i in range(len(segment_boundaries) - 1):
#             start_index = segment_boundaries[i] + 1
#             end_index = segment_boundaries[i + 1] + 1
#             segment = machine_group.iloc[start_index:end_index]
            
#             # Sort segment by datetime
#             segment = segment.sort_values(by='datetime', ascending=True)
            
#             # Create and pad sequences within this segment with step-wise shifting for augmentation
#             for start_seq in range(0, len(segment) - sequence_length + 1, step):
#                 end_seq = start_seq + sequence_length
#                 sequence = segment.iloc[start_seq:end_seq]
                
#                 # Since we're filtering by RUL < 50, all sequences inherently meet this condition
#                 # No need to check length; it will always be sequence_length long due to the loop condition
#                 sequences.append(sequence.drop([target_column, 'machineID', 'datetime'], axis=1).values)
                
#                 # Determine label for the sequence
#                 label = sequence[target_column].iloc[-1]  # Always use the last RUL value as label
#                 labels.append(label)

#     # Convert sequences and labels to numpy arrays
#     sequence_array = np.array(sequences, dtype=object)  # Use dtype=object for arrays of arrays
#     label_array = np.array(labels)

#     return sequence_array, label_array


# def prepare_dataset_tensors(df, target_column='RUL', sequence_length=10, padding_value=0, test_size=0.2, valid_size=0.1, batch_size=150, shuffle=False):
#     """
#     Prepares dataset for ML modeling with PyTorch, suitable for LSTM models, using a sliding window approach and padding.
#     """
#     # Create sequences
#     X, y = create_and_pad_sequences_with_augmentation(df, target_column, sequence_length, padding_value)

#     # Split dataset into train, validation, and test sets
#     X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=42)
#     valid_test_split = valid_size / (1.0 - test_size)
#     X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_test_split, shuffle=shuffle, random_state=42)

#     # Standardize features (excluding padding)
#     scaler = StandardScaler()
#     # Fit on training data and transform all data respecting the learned parameters
#     num_sequences, sequence_len, num_features = X_train.shape
#     X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(num_sequences, sequence_len, num_features)
#     X_valid = scaler.transform(X_valid.reshape(-1, num_features)).reshape(X_valid.shape)
#     X_test = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

#     # Convert to PyTorch tensors
#     train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
#     valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float), torch.tensor(y_valid, dtype=torch.float))
#     test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

#     # Create DataLoader objects
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
#     valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

#     input_size = num_features  # The number of features in each sequence

#     return train_loader, valid_loader, test_loader, input_size


def create_and_pad_sequences_with_augmentation(df, target_column, sequence_length, padding_value=0, step=10, min_rul=100, num_augmented_batches=10):
    sequences = []  # To store sequences, including padded ones
    labels = []  # To store labels for each sequence
    
    # Ensure required columns are present
    required_columns = [target_column, 'machineID', 'datetime']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    # Iterate through each machineID with progress tracking
    for _, machine_group in tqdm(df.groupby('machineID'), desc="Processing Machine IDs"):
        # Filter segments by RUL < min_rul
        machine_group = machine_group[machine_group[target_column] < min_rul]
        
        # Track segments by splitting at points where RUL resets (if applicable)
        rul_zero_indices = machine_group.index[machine_group[target_column] == 0].tolist()
        segment_boundaries = [-1] + rul_zero_indices  # Start from -1 to include the first segment
        
        for i in range(len(segment_boundaries) - 1):
            start_index = segment_boundaries[i] + 1
            end_index = segment_boundaries[i + 1] + 1
            segment = machine_group.iloc[start_index:end_index]
            
            # Sort segment by datetime
            segment = segment.sort_values(by='datetime', ascending=True)
            
            # Create and pad sequences within this segment with step-wise shifting for augmentation
            aug_step = 0  # Initialize augmentation step
            while aug_step < num_augmented_batches:
                for start_seq in range(aug_step, len(segment) - sequence_length + 1, step):
                    end_seq = start_seq + sequence_length
                    if end_seq <= len(segment):
                        sequence = segment.iloc[start_seq:end_seq]
                        
                        sequences.append(sequence.drop([target_column, 'machineID', 'datetime'], axis=1).values)
                        label = sequence[target_column].iloc[-1]  # Last RUL value as label
                        labels.append(label)
                aug_step += 1  # Increment to create next batch of augmented sequences

    # Convert sequences and labels to numpy arrays
    sequence_array = np.array(sequences, dtype=object)
    label_array = np.array(labels)

    return sequence_array, label_array


def prepare_dataset_tensors(df, target_column='RUL', sequence_length=10, padding_value=0, test_size=0.2, valid_size=0.1, batch_size=150, shuffle=False , batches=30 , min_rul=50):
    """
    Prepares dataset for ML modeling with PyTorch, suitable for LSTM models, using a sliding window approach and padding.
    """
    # Create sequences
    X, y = create_and_pad_sequences_with_augmentation(df, target_column, sequence_length, padding_value, batches , min_rul)

    # Split dataset into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=42)
    valid_test_split = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_test_split, shuffle=shuffle, random_state=42)

    # Standardize features (excluding padding)
    scaler = StandardScaler()
    # Fit on training data and transform all data respecting the learned parameters
    num_sequences, sequence_len, num_features = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(num_sequences, sequence_len, num_features)
    X_valid = scaler.transform(X_valid.reshape(-1, num_features)).reshape(X_valid.shape)
    X_test = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float), torch.tensor(y_valid, dtype=torch.float))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

    # Create DataLoader objects
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    input_size = num_features  # The number of features in each sequence

    return train_loader, valid_loader, test_loader, input_size