import sys
from collections import deque
from itertools import chain
import pandas as pd
from pathlib import Path
import numpy as np 
import polars as pl 
import pyarrow as pa


def load_csv_directory_to_df(directory_path):
    """
    Loads all CSV files in the specified directory into a dictionary of DataFrames.
    The keys of the dictionary will be the file names without the '.csv' extension.

    Parameters:
    - directory_path: str, the path to the directory containing CSV files.

    Returns:
    - A dictionary with file names as keys (without '.csv') and DataFrames as values.
    """
    dataframes = {}
    directory = Path(directory_path)
    for file_path in directory.glob('*.csv'):
        df_name = file_path.stem  # 'stem' attribute provides the file name without extension
        dataframes[df_name] = pd.read_csv(file_path)
    
    return dataframes


def size(o, handlers={}):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and their subclasses: tuple, list, deque, dict, set and frozenset.
    """
    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof int if not available

    def sizeof(o):
        if id(o) in seen:  # Do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def optimize_dataframe(df):
    """
    Optimize a DataFrame by downcasting numeric columns to reduce memory usage.

    Parameters:
    - df (pd.DataFrame): The DataFrame to optimize.

    Returns:
    - pd.DataFrame: The optimized DataFrame with reduced memory usage.
    """
    # Downcast floating-point columns to float16
    for col in df.select_dtypes(include=['float32', 'float64']).columns:
        df[col] = df[col].astype('float16')

    
    # Downcast integer columns to int8 if they fit, else leave them as is.
    for col in df.select_dtypes(include=['integer']).columns:
        min_val, max_val = df[col].min(), df[col].max()
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            df[col] = df[col].astype('int8')
    
    # Convert object columns to categorical if they have a relatively low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df[col]) < 0.8:  # Arbitrary threshold for conversion
            df[col] = df[col].astype('category')
    
    df = pl.from_pandas(df)

    
    return df
