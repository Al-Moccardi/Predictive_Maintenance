import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
import seaborn as sns 


def scatter_plot_sample_machines(data, machine_id_column, variables_to_plot, time_column, sample_size=5):
    """
    Sample a specified number of machineIDs, group data by machineID, and perform scatter plots of selected variables along the time.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - machine_id_column (str): The name of the column containing the 'machineID'.
    - variables_to_plot (list of str): List of variable names to plot.
    - time_column (str): The name of the time column.
    - sample_size (int): The number of machineIDs to sample.

    Returns:
    - None (displays scatter plots).
    """
    # Sample machineIDs
    sampled_machine_ids = random.sample(data[machine_id_column].unique().tolist(), sample_size)
    
    # Group data by sampled 'machineID'
    grouped_data = data[data[machine_id_column].isin(sampled_machine_ids)].groupby(machine_id_column)
    
    # Create a scatter plot for each selected variable
    for variable in variables_to_plot:
        plt.figure(figsize=(10, 6))
        plt.title(f'Scatter Plot of {variable} vs. {time_column} for Sampled Machines')
        
        # Create a color map for each sampled machine
        color_cycle = plt.cm.viridis(np.linspace(0, 1, len(sampled_machine_ids)))
        
        # Iterate over sampled machines and corresponding colors
        for machine_id, color in zip(sampled_machine_ids, color_cycle):
            group = grouped_data.get_group(machine_id)
            plt.scatter(group[time_column], group[variable], label=f'Machine {machine_id}', color=color)
        
        plt.xlabel(time_column,rotation=90)
        plt.ylabel(variable)
        plt.legend()
        plt.xticks(rotation=90)  # Rotate x-axis labels vertically
        plt.grid(False)
        plt.show()



def plot_histogram_with_percentages(data, column_name):
    """
    Plot a histogram of a specified column in the DataFrame with percentages on top of each bar.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to plot.

    Returns:
    - None (displays the histogram).
    """
    # Convert the specified column to an ordered categorical column
    data[column_name] = data[column_name].astype('category').cat.as_ordered()

    # Calculate the histogram data and percentages
    hist_data = data[column_name].value_counts().sort_index()
    percentages = (hist_data / hist_data.sum()) * 100

    # Create a bar plot with percentages on top of each bar
    plt.figure(figsize=(8, 6))
    bars = plt.bar(hist_data.index.astype(str), hist_data)
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column_name} with Percentages (Sum to 100%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.show()



def plot_machine_data(dataset, column_name, sample_size, date_column='datetime', error_type_column='errorType'):
    """
    Generates line plots for a sample of machineIDs from a specified column in the dataset, all on the same figure,
    with different colors for each machineID. Marks each occurring error with a distinct point shape based on the specified error type column.

    Parameters:
    - dataset: pandas DataFrame with at least 'machineID', a date column, the specified column to plot, and an error type column
    - column_name: string, the name of the column to plot
    - sample_size: int, the number of machineIDs to sample for plotting
    - date_column: string, the name of the date column to use for the x-axis (default is 'datetime')
    - error_type_column: string, the name of the column indicating error types, allowing dynamic error type handling
    """
    if column_name not in dataset.columns or date_column not in dataset.columns or error_type_column not in dataset.columns:
        raise ValueError(f"The dataset must contain '{column_name}', '{date_column}', and '{error_type_column}' columns.")
    
    if 'machineID' not in dataset.columns:
        raise ValueError("The dataset must contain a 'machineID' column.")

    # Define a marker map for different error types. This may need to be adjusted based on the dataset's error types.
    marker_map = {
        'ErrorType1': 'o',  # Circle
        'ErrorType2': 's',  # Square
        'ErrorType3': '^',  # Triangle up
        'ErrorType4': 'D',  # Diamond
        'ErrorType5': '*',  # Star
        # Add or remove error types as needed
    }

    # Get a sample of unique machineIDs
    machine_ids = np.random.choice(dataset['machineID'].unique(), size=sample_size, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    # Create a colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(machine_ids)))
    
    # Plot data for each sampled machineID
    for idx, machine_id in enumerate(machine_ids):
        machine_data = dataset[dataset['machineID'] == machine_id].sort_values(by=date_column)
        
        # Plot the line for the machineID
        plt.plot(machine_data[date_column], machine_data[column_name], label=f'Machine {machine_id}', color=colors[idx], linestyle='-')
        
        # Plot points for each error type present in the dataset
        unique_error_types = machine_data[error_type_column].dropna().unique()
        for error_type in unique_error_types:
            if error_type in marker_map:
                error_data = machine_data[machine_data[error_type_column] == error_type]
                plt.scatter(error_data[date_column], error_data[column_name], color=colors[idx], marker=marker_map[error_type], s=100, edgecolors='black', label=f'{error_type} Machine {machine_id}')

    plt.title(f'Line Plot for {column_name}')
    plt.xlabel(date_column)
    plt.ylabel(column_name)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_scatter_sampled_machineID(df, x_column, y_column, machine_id_column='machineID', sample_size=3):
    """
    Creates a scatter plot for two columns from a DataFrame, for a sampled subset of machineIDs.

    Parameters:
    - df: pandas DataFrame containing the data, including a column for machineID and two columns for plotting.
    - x_column: string, the name of the first column to plot on the x-axis.
    - y_column: string, the name of the second column to plot on the y-axis.
    - machine_id_column: string, the name of the column containing machine IDs. Default is 'machineID'.
    - sample_size: int, the number of machineIDs to randomly sample for the scatter plot. Default is 3.
    """
    # Check if the DataFrame contains the specified columns
    if not all(column in df.columns for column in [x_column, y_column, machine_id_column]):
        raise ValueError("The DataFrame must contain the specified machine_id_column, x_column, and y_column.")
    
    # Sample a subset of unique machineIDs
    sampled_machine_ids = np.random.choice(df[machine_id_column].unique(), size=sample_size, replace=False)
    
    # Filter the DataFrame for only the sampled machineIDs
    sampled_df = df[df[machine_id_column].isin(sampled_machine_ids)]
    
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot for each sampled machineID
    for machine_id in sampled_machine_ids:
        # Filter the DataFrame for the current machineID
        machine_data = sampled_df[sampled_df[machine_id_column] == machine_id]
        
        # Generate the scatter plot
        plt.scatter(machine_data[x_column], machine_data[y_column], label=f'Machine {machine_id}', s=50)
    
    # Customize the plot
    plt.title(f'Scatter Plot of {x_column} vs. {y_column} for Sampled Machine IDs', fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_scatter_sampled_machineID_with_errors(df, x_column, y_column, error_type_column, machine_id_column='machineID', sample_size=3):
    """
    Creates a scatter plot for two columns from a DataFrame, for a sampled subset of machineIDs,
    with different shapes based on error type.

    Parameters:
    - df: pandas DataFrame containing the data, including a column for machineID, two columns for plotting, and an error type column.
    - x_column: string, the name of the first column to plot on the x-axis.
    - y_column: string, the name of the second column to plot on the y-axis.
    - error_type_column: string, the name of the column indicating the type of error.
    - machine_id_column: string, the name of the column containing machine IDs. Default is 'machineID'.
    - sample_size: int, the number of machineIDs to randomly sample for the scatter plot. Default is 3.
    """
    # Check if the DataFrame contains the specified columns
    if not all(column in df.columns for column in [x_column, y_column, error_type_column, machine_id_column]):
        raise ValueError("The DataFrame must contain the specified columns.")
    
    # Sample a subset of unique machineIDs
    sampled_machine_ids = np.random.choice(df[machine_id_column].unique(), size=sample_size, replace=False)
    
    # Filter the DataFrame for only the sampled machineIDs
    sampled_df = df[df[machine_id_column].isin(sampled_machine_ids)]
    
    # Define marker styles for different error types. Add or update as needed.
    markers = ['o', 's', '^', 'D', '*', 'X', 'P']
    error_types = sampled_df[error_type_column].unique()
    error_marker_map = {error_type: markers[i % len(markers)] for i, error_type in enumerate(error_types)}
    
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot for each sampled machineID and error type
    for machine_id in sampled_machine_ids:
        machine_data = sampled_df[sampled_df[machine_id_column] == machine_id]
        for error_type, marker in error_marker_map.items():
            error_data = machine_data[machine_data[error_type_column] == error_type]
            plt.scatter(error_data[x_column], error_data[y_column], label=f'Machine {machine_id} - {error_type}', s=50, marker=marker)
    
    # Customize the plot
    plt.title(f'Scatter Plot of {x_column} vs. {y_column} for Sampled Machine IDs by Error Type', fontsize=16)
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel(y_column, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_3d_with_datetime_sample(dataset, x_column, y_column, datetime_column, sample_size):
    """
    Creates a 3D plot with a sample of machineID on the x-axis, a specified numerical column on the y-axis,
    and datetime on the z-axis, for a sample of machine IDs, treating datetime as datetime.

    Parameters:
    - dataset: pandas DataFrame with at least 'machineID', a numerical column, and a datetime column
    - x_column: string, expected to be 'machineID' for selecting a sample
    - y_column: string, the name of the numerical column to plot on the y-axis
    - datetime_column: string, the name of the datetime column to plot on the z-axis
    - sample_size: int, the number of machineIDs to sample for plotting
    """
    # Sample machineIDs
    sampled_machine_ids = np.random.choice(dataset[x_column].unique(), size=sample_size, replace=False)
    sampled_dataset = dataset[dataset[x_column].isin(sampled_machine_ids)]
    
    # Convert datetime to Matplotlib's internal format for plotting
    sampled_dataset['datetime_num'] = mdates.date2num(pd.to_datetime(sampled_dataset[datetime_column]))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(sampled_machine_ids)))
    
    # Plotting for each sampled machineID
    for idx, machine_id in enumerate(sampled_machine_ids):
        machine_data = sampled_dataset[sampled_dataset[x_column] == machine_id]
        ax.scatter(np.full_like(machine_data['datetime_num'], fill_value=idx),  # X-axis: machineID index for differentiation
                   machine_data[y_column],  # Y-axis: specified numerical column
                   machine_data['datetime_num'],  # Z-axis: datetime
                   label=f'Machine {machine_id}', color=colors[idx])
    
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_zlabel(datetime_column)
    
    # Format z-axis (datetime) labels
    ax.zaxis.set_major_locator(mdates.AutoDateLocator())
    ax.zaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    
    plt.legend()
    plt.show()




def plot_aggregated_error_counts(error_counts):
    """
    Plots aggregated error counts near th
    e end of RUL.

    Parameters:
    - error_counts: DataFrame of aggregated error counts, as returned by analyze_errors_near_end_rul_global.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='errorID', y='count', data=error_counts, palette='viridis')
    plt.title('Aggregated Error Counts Near the End of RUL')
    plt.xlabel('Error ID')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()




def analyze_errors_near_end_rul_global(df, count_column, rul_threshold=30, exclude_value='unknown'):
    """
    Analyzes occurrences of specified categories occurring near the end of RUL within a specified threshold across all machines,
    completely ignoring rows with a specified exclude_value in the count_column.

    Parameters:
    - df: DataFrame containing 'RUL', a column to count (e.g., 'errorID'), and other columns.
    - count_column: string, the name of the column to count occurrences of.
    - rul_threshold: int, the RUL threshold to consider as "near the end" of life.
    - exclude_value: string or list, the value(s) to exclude in the count_column.
    """
    # Ensure exclude_value is a list to handle multiple exclude values
    if not isinstance(exclude_value, list):
        exclude_value = [exclude_value]
    
    # Filter out rows where the count_column has the exclude_value or is null
    df_filtered = df[~df[count_column].isin(exclude_value) & df[count_column].notnull()]

    # Further filter data for "near the end of RUL" based on the threshold
    end_rul_df = df_filtered[df_filtered['RUL'] <= rul_threshold]

    # Aggregate occurrences of each category in the specified column near the end of RUL
    category_counts = end_rul_df[count_column].value_counts().reset_index(name='count')
    category_counts.rename(columns={'index': count_column}, inplace=True)

    return category_counts