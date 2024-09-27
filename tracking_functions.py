import numpy as np
import pandas as pd
import os
import csv


def get_subfolders(directory):
    subfolders = [name for name in os.listdir(directory)
                  if os.path.isdir(os.path.join(directory, name))]
    return subfolders


def list_of_files(folder_path, std_path='.tiff'):
    tiff_files = [file for file in os.listdir(folder_path) if file.endswith(std_path)]
    return tiff_files


def save_to_csv(file_path, float_list):
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write each float as a separate row in the CSV file
        for value in float_list:
            writer.writerow([value])


def remove_short_tracks(df, min_points):
    return df.groupby('track_id').filter(lambda group: len(group) >= min_points)


def calculate_distance(dataframe, pixel_size):

    if dataframe.empty:
        return dataframe  # Return the unchanged dataframe if it's empty

    # Iterate through unique track_ids
    for track_id in dataframe['track_id'].unique():
        track_data = dataframe[dataframe['track_id'] == track_id]

        # Check if track_data is empty
        if not track_data.empty:
            # Sort the track_data by 'frame_y'
            track_data = track_data.sort_values(by='frame_y')

            if len(track_data) < 2:
                # If there are fewer than 2 data points, distance is 0
                dataframe.loc[dataframe['track_id'] == track_id, 'Distance (um)'] = 0.0
            else:
                x_values = track_data['x'].values
                y_values = track_data['y'].values

                # Calculate the squared differences in x and y coordinates
                squared_diffs = (np.diff(x_values) ** 2 + np.diff(y_values) ** 2) ** 0.5

                # Calculate the cumulative sum of squared differences (start with 0)
                cumulative_sum_squared_diffs = np.concatenate([[0], np.cumsum(squared_diffs)]) * pixel_size

                # Assign the cumulative sum to the 'Distance (um)' column in the original dataframe
                dataframe.loc[dataframe['track_id'] == track_id, 'Distance (um)'] = cumulative_sum_squared_diffs

    # Sort the dataframe by 'frame_y'
    dataframe = dataframe.sort_values(by='frame_y')

    return dataframe


def calculate_speed(df):
    if df.empty:
        return df
    if not df['Distance (um)'].empty:
        df['Speed (um/s)'] = df['Distance (um)'] / df['Time (s)']
        df['Speed (um/min)'] = df['Distance (um)'] / df['Time (s)'] * 60
    else:
        print("Calculate Distance first")

    return df


def calculate_speed_corr(df):

    df = df.sort_values(by=['track_id', 'Time (s)'], ascending=[True, True])

    if df.empty:
        return df
    if not df['Distance (um)'].empty:
        # df['Timedelta'] = df['Time (s)'].diff()
        df['Timedelta'] = df.groupby('track_id')['Time (s)'].diff()
        df['Distdelta'] = df.groupby('track_id')['Distance (um)'].diff()
        df['Speed (um/s)'] = df['Distdelta'] / df['Timedelta']
        df['Speed (um/min)'] = df['Distdelta'] / df['Timedelta'] * 60
    else:
        print("Calculate Distance first")

    return df


def calculate_displacement(dataframe, pixel_size):
    for track_id in dataframe['track_id'].unique():
        track_data = dataframe[dataframe['track_id'] == track_id]

        # Sort the track_data by 'frame_y'
        track_data = track_data.sort_values(by='frame_y')

        if len(track_data) < 2:
            dataframe.loc[dataframe['track_id'] == track_id, 'Displacement (um)'] = 0.0
        else:
            x_values = track_data['x'].values
            y_values = track_data['y'].values

            start_values = (x_values[0], y_values[0])
            end_values = (x_values[-1], y_values[-1])

            # Calculate the squared difference in x and y coordinates
            squared_diff_x = (end_values[0] - start_values[0]) ** 2
            squared_diff_y = (end_values[1] - start_values[1]) ** 2

            # Calculate the displacement
            displacement = np.sqrt(squared_diff_x + squared_diff_y) * pixel_size
            dataframe.loc[dataframe['track_id'] == track_id, 'Displacement (um)'] = displacement

        # Sort the dataframe by 'frame_y'
        dataframe = dataframe.sort_values(by='frame_y')

    return dataframe


def confinement_ratio(df):
    """
    Computes the confinement ratio for each unique track in the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing tracking data.

    Raises:
        ValueError: If any required column is missing from the input DataFrame.

    Returns:
        pd.DataFrame: The same DataFrame with an additional column:
            - 'confinement ratio': The computed confinement ratio for each track.
    """

    required_columns = ['track_id', 'frame_y', 'Distance (um)', 'Displacement (um)']
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id]

        # Sort the track_data by 'frame_y'
        track_data = track_data.sort_values(by='frame_y')

        if len(track_data) < 2:
            df.loc[df['track_id'] == track_id, 'confinement ratio'] = 0.0
        else:
            distance = track_data['Distance (um)'].values
            displacement = track_data['Displacement (um)'].values


            # Calculate confinement ratio
            cr = displacement[-1] / distance[-1] if distance[-1] else 0.0

            df.loc[df['track_id'] == track_id, 'confinement ratio'] = cr

        # Sort the dataframe by 'frame_y'
        df = df.sort_values(by='frame_y')

    return df


def mean_straight_line_speed(df):
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id]

        # Sort the track_data by 'frame_y'
        track_data = track_data.sort_values(by='frame_y')

        if len(track_data) < 2:
            df.loc[df['track_id'] == track_id, 'mean straight line speed (um/s)'] = 0.0
            df.loc[df['track_id'] == track_id, 'mean straight line speed (um/min)'] = 0.0
        else:
            time = track_data['Time (s)'].values
            time_h = time / 60
            displacement = track_data['Displacement (um)'].values

            # Calculate confinement ratio
            msls_min = displacement[-1] / time[-1]
            msls_h = displacement[-1] / time_h[-1]
            df.loc[df['track_id'] == track_id, 'mean straight line speed (um/s)'] = msls_min
            df.loc[df['track_id'] == track_id, 'mean straight line speed (um/min)'] = msls_h

        # Sort the dataframe by 'frame_y'
        df = df.sort_values(by='frame_y')

    return df


def calc_linearity_of_forward_progression_and_sort(df):
    """
    Calculates the linearity of forward progression for each unique track in the DataFrame.
    Compares the mean straight-line speed with the average speed for each track.
    Args:
        df (pd.DataFrame): A DataFrame containing tracking data. It must have the following columns:

    Raises:
        ValueError: If the required columns are missng or
        the mean speed is zero but the straight-line speed is not for a given track.

    Returns:
        pd.DataFrame: The same DataFrame with an additional column:
            - 'linearity of forward progression': The computed LFP for each track.
        Also sorted by frame_Y

    """

    required_columns = ['track_id', 'frame_y', 'Speed (um/s)', 'mean straight line speed (um/s)']

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id]

        # Sort the track_data by 'frame_y'
        track_data = track_data.sort_values(by='frame_y')

        if len(track_data) < 2:
            df.loc[df['track_id'] == track_id, 'linearity of forward progression'] = 0.0
        else:
            mean_speed_min = track_data['Speed (um/s)'].mean()
            msls_m = track_data['mean straight line speed (um/s)'].values

            if mean_speed_min == 0 and msls_m[-1] != 0:
                raise ValueError(
                    f"Mean speed is zero, but straight line speed is not for track_id {track_id}")

            lfp = msls_m[-1] / mean_speed_min if mean_speed_min else 0.0
            df.loc[df['track_id'] == track_id, 'linearity of forward progression'] = lfp

        # Sort the dataframe by 'frame_y'
        df = df.sort_values(by='frame_y')

    return df


def average_tracks(df, pixel_size):
    """
    Averages individual tracks within a DataFrame containing tracking data.
    Calculates displacement, migration time, speed, etc.

    Args:
        df (pd.DataFrame): A DataFrame with tracking data.
        pixel_size (float):
            The size of a pixel in micrometers,
            used to convert displacements from pixel units to real-world units.

    Returns:
        pd.DataFrame: A DataFrame containing the averaged metrics for each track.

    Raises:
        ValueError: If any required column is missing from the input DataFrame.
    """

    required_columns = ['track_id', 'frame_y', 'x', 'y', 'Time (s)', 'Distance (um)',
                        'Speed (um/s)', 'Speed (um/min)', 'confinement ratio',
                        'mean straight line speed (um/min)', 'linearity of forward progression']

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    track_means = []
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values(by='frame_y')

        first_point = track_data.iloc[0][['x', 'y']]
        last_point = track_data.iloc[-1][['x', 'y']]
        tot_displ = np.sqrt((last_point['x'] - first_point['x'])**2 +
                            (last_point['y'] - first_point['y'])**2) * pixel_size

        migration_time = track_data['Time (s)'].max() - track_data['Time (s)'].min()

        if migration_time == 0:
            print(f"Warning : track length is 1 for {track_id}")

        track_means.append(
            {
                'track_id': track_id,
                'time (s)': migration_time,
                'distance (um)': track_data['Distance (um)'].max(),
                'displ (um)': tot_displ,
                'mean straight line speed (um/s)':
                    tot_displ / migration_time if migration_time > 0 else 0.0,
                'speed (um/s)': track_data['Speed (um/s)'].mean(),
                'speed (um/min)': track_data['Speed (um/min)'].mean(),
                'confinement ratio': track_data['confinement ratio'].max(),
                'mean straight line speed (um/min)':
                    track_data['mean straight line speed (um/min)'].max(),
                'linearity of forward progression':
                    track_data['linearity of forward progression'].max(),
            }
        )
    track_means = pd.DataFrame(track_means)
    return track_means
