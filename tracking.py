import argparse
from pathlib import Path
from tracking_functions import *
import time
from skimage import io
from skimage.measure import label, regionprops
import napari
import trackpy as tp
import logging

# Suppress trackpy logging
logging.getLogger('trackpy').setLevel(logging.WARNING)
import tifffile as tiff
from enum import Enum

THROMBI_NUM_FOLDER = 'ThrombiNumbers'
THROMBI_AREAS_FOLDER = 'ThrombiAreas'
PLT_REMAPPING = [(0, 0), (1, 0), (2, 1)]
THROMBI_REMAPPING = [(0, 0), (1, 1), (2, 0)]
PIXEL_SIZE = 0.431
DT = 0.05

class ViewMode(Enum):
    """View mode for platelet tracks
    """
    ROLLING = 'rolling'
    BUMPED = 'bumped'
    MOTIONLESS = 'motionless'
    INVALID_DIRECTION = 'invalid_direction'
    INVALID_TURN_ANGLE = 'invalid_turn_angle'


def crop_video(vid):
    """Crops a 3D numpy array representing a video stack by removing empty rows along the y-axis.

    Args:
        vid (ndarray of dtype int):
            A 3D numpy array representing a video stack with dimensions (depth, height, width).
            The array should contain pixel values where non-zero values are considered for cropping.

    Raises:
        ValueError: Raises ValueError when all pixels are zero

    Returns:
        ndarray of dtype int:
            A cropped 3D numpy array with the same depth and width as the input,
            but with reduced height such that only rows containing
            non-zero values are retained.
    Notes:
        - The function assumes that the input video stack is a numpy array.
        - Non-zero values are defined as values strictly greater than 1.
        - The cropping is performed along the y-axis (height).
    """

    # Find the indices where any value in time (axis=0) is greater than 1
    y_indices = np.any(vid > 1, axis=0)

    # Find the rows (y-axis) where there are non-zero values
    non_zero_rows = np.any(y_indices, axis=1)

    if not np.any(non_zero_rows):
        raise ValueError("Video has no non-zero rows.")

    # Get the overall top and bottom y indices
    overall_low_y = np.argmax(non_zero_rows) # first non-zero column
    overall_top_y = len(non_zero_rows) - np.argmax(non_zero_rows[::-1]) - 1 # last non-zero col

    # Crop the stack
    cropped_stack = vid[:, overall_low_y:overall_top_y + 1, :]

    return cropped_stack


def remap_and_label_image(frame_image, mapping):
    """Remaps image and labels connected pixels 
    
    Args:
        frame_image (ndarray of dtype int):
            Image to remap and label.
        mapping (list of tuples):
            First value in tuple - remap from, second - remap to.

    Returns:
        labels (ndarray of dtype int):
            Remapped and labeled array.
    """
    remapped = np.zeros_like(frame_image)
    for from_value, to_value in mapping:
        remapped[frame_image == from_value] = to_value
    return label(remapped, background=0)

def write_to_file_with_directory(file_path, data):
    """
    Writes data to a CSV file, creating the directory if necessary.

    Args:
        file_path (str):
            The path to the output CSV file.
        data (list):
            The data to be written to the file.
    """
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    save_to_csv(file_path, data)

def preprocess_video(folder, file_name):
    """
    Processes a video file to extract thrombi information.
    Reads image as an array, crops video
    Calculate thrrombi number and areas and saves to the file
    Args:
        folder (str):
            The path to the folder containing the video file.
        file_name (str):
            The name of the video file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            1. Platelet labels.
            2. Cropped video array.
    """
    video_array = io.imread(folder + file_name)
    try:
        video_array = crop_video(video_array)
    except ValueError as e:
        print(f"Could not crop video {file_name} : {e}")
        raise

    duration, height, width = video_array.shape
    video_area = height * width

    thrombi_areas = np.zeros(duration)
    thrombi_number = np.zeros(duration)

    unique_labels_plt = np.zeros_like(video_array)

    for time_point, frame_image in enumerate(video_array):
        lbl_frame_plt = remap_and_label_image(frame_image, PLT_REMAPPING)
        unique_labels_plt[time_point] = lbl_frame_plt

        lbl_frame_thrombi = remap_and_label_image(frame_image, THROMBI_REMAPPING)
        tmp_thrombi_area = [region.area for region in regionprops(lbl_frame_thrombi)]
        thrombi_areas[time_point] = (np.sum(tmp_thrombi_area) / video_area) * 100
        thrombi_number[time_point] = len(tmp_thrombi_area)

    output_file_name = file_name.split('.ti')[0] + '.csv'

    # Save thrombi data to original folder
    write_to_file_with_directory(
        folder + THROMBI_NUM_FOLDER + '/' + output_file_name,
        thrombi_number.tolist()
    )
    write_to_file_with_directory(
        folder + THROMBI_AREAS_FOLDER + '/' + output_file_name,
        thrombi_areas.tolist()
    )

    return unique_labels_plt, video_array


def extract_centroids(labels_per_frame):
    """Extracts the centroids of labeled regions in a video.

    Args:
        labels_per_frame (np.ndarray):
        A 3D NumPy array containing the area labels for each frame of the video.

    Returns:
        pd.DataFrame:
        A Pandas DataFrame containing the x, y coordinates and frame numbers of the centroids.
    """
    data = []
    for frame_number, frame in enumerate(labels_per_frame):
        labeled_frame = label(frame)  # Probably we do no need this line
        for region in regionprops(labeled_frame):
            y, x = region.centroid
            data.append([x, y, frame_number])
    return pd.DataFrame(data, columns=['x', 'y', 'frame'])


def format_for_napari(tracks):
    """Formats a Pandas DataFrame containing object tracking data for Napari.

    Args:
        tracks (pd.DataFrame):
            A Pandas DataFrame containing the x, y coordinates and frame numbers of objects.
            Must have columns: ['particle', 'frame', 'y', 'x'].

    Raises:
        ValueError: If the input DataFrame is missing any required columns.

    Returns:
        np.ndarray:
            A NumPy array formatted for Napari, with columns: ['particle', 'frame', 'y', 'x'].
    """
    # Ensure the DataFrame has the required columns
    required_columns = ['particle', 'frame', 'y', 'x']
    missing_columns = set(required_columns) - set(tracks.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


    tracks.loc[:, 'particle'] = tracks['particle'].astype(int)

    # Extract the relevant columns and convert to NumPy array
    formatted_tracks = tracks[required_columns].to_numpy()
    return formatted_tracks


def calculate_turn_angles(track):
    """Calculates the turn angles between consecutive segments of a track.

    Args:
        tracks (pd.DataFrame):
            A Pandas DataFrame containing the x, y coordinates and frame numbers of objects.
            Must have columns: ['particle', 'frame', 'y', 'x'].

    Returns:
        np.ndarray:
            A NumPy array containing the turn angles (in degrees) between consecutive segments of the track.
            The length of the array will be one less than the number of rows in the input DataFrame.
    Raises:
        ValueError: If the input DataFrame contains data for more than one particle ID.
    """

    # Check if there's only one particle ID
    if len(track['particle'].unique()) > 1:
        raise ValueError(
            "This function expects data for a single particle ID. Please filter your data accordingly."
        )

    # Calculate the angle between consecutive segments
    angles = []
    for i in range(1, len(track) - 1):
        p1 = track.iloc[i - 1][['x', 'y']]
        p2 = track.iloc[i][['x', 'y']]
        p3 = track.iloc[i + 1][['x', 'y']]

        v1 = p2 - p1
        v2 = p3 - p2

        # Normalize vectors
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        v1 = v1 / v1_norm if v1_norm > 0 else v1
        v2 = v2 / v2_norm if v2_norm > 0 else v2

        # Calculate the angle in degrees
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        angles.append(angle)

    return angles

def calculate_direction_angle(first_point, last_point):
    """Calculates the direction angle of a motion vector relative to a downward direction.

    Args:
        first_point (pd.Series):
            A pandas series, initial x, y coordinates (float64)
        last_point (pd.Series):
            A pandas series, initial x, y coordinates (float64)

    Returns:
        float:
            The direction angle in degrees.
            0 degrees for downward, 90 degrees for rightward,
            180 degrees for upward, and 270 degrees for leftward.

    Raises:
        ValueError: If either `first_point` or `last_point` is not a 2D NumPy array.
    """
    if not isinstance(first_point, pd.Series) or not all(col in first_point for col in ['x', 'y']):
        raise ValueError("last_point must be a Pandas Series with 'x' and 'y' columns.")

    if not isinstance(last_point, pd.Series) or not all(col in last_point for col in ['x', 'y']):
        raise ValueError("last_point must be a Pandas Series with 'x' and 'y' columns.")

    motion_vector = last_point - first_point
    motion_vector_norm = np.linalg.norm(motion_vector)
    motion_vector = motion_vector / motion_vector_norm if motion_vector_norm else motion_vector

    vertical_vector = np.array([0, -1])  # Assuming the positive y-direction is down
    angle = np.degrees(np.arccos(np.clip(np.dot(motion_vector, vertical_vector), -1.0, 1.0)))
    return angle

import numpy as np

def calculate_total_displacement(particle_tracks):
    """Calculates displacement between the first and last points of a track."""
    first_point = particle_tracks.iloc[0][['x', 'y']]
    last_point = particle_tracks.iloc[-1][['x', 'y']]
    return np.sqrt((last_point['x'] - first_point['x'])**2 + (last_point['y'] - first_point['y'])**2)

def classify_track(particle_tracks, min_displacement, min_duration, max_turn_angle, max_direction_angle):
    """Classifies a track based on its displacement, duration, and angles."""
    displacement = calculate_total_displacement(particle_tracks)
    duration = len(particle_tracks)
    direction_angle = calculate_direction_angle(particle_tracks.iloc[0][['x', 'y']], particle_tracks.iloc[-1][['x', 'y']])
    turn_angles = calculate_turn_angles(particle_tracks)

    if displacement >= min_displacement and duration >= min_duration and direction_angle <= max_direction_angle and all(angle <= max_turn_angle for angle in turn_angles):
        return 'valid'
    elif displacement < min_displacement and duration < min_duration:
        return 'bumped'
    elif displacement < min_displacement * 1.5 and duration >= min_duration * 10:
        return 'motionless'
    elif displacement >= min_displacement and duration >= min_duration and direction_angle > max_direction_angle:
        return 'direction'
    elif displacement >= min_displacement and duration >= min_duration and not all(angle <= max_turn_angle for angle in turn_angles):
        return 'turn_angle'
    else:
        return 'other'

def filter_tracks_by_displacement_and_duration(
    tracks,
    min_displacement,
    min_duration=5,
    max_turn_angle=120,
    max_direction_angle=45
):
    """
    Filters tracks based on displacement, duration, turn angles, and direction angles.

    Args:
        tracks (pd.DataFrame): A Pandas DataFrame containing track information.
        min_displacement (float): The minimum displacement for a track to be considered valid.
        min_duration (int, optional): The minimum valid track duration (default: 5 frames).
        max_turn_angle (float, optional): The maximum allowed turn angle for a track (default: 120 degrees).
        max_direction_angle (float, optional): The maximum allowed direction angle for a track (default: 45 degrees).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            A tuple containing four DataFrames:
                - filtered_tracks: Tracks that meet all filtering criteria.
                - bumped_tracks: Tracks with displacement less than `min_displacement` and duration less than `min_duration`.
                - motionless_tracks: Tracks with displacement less than 1.5 times `min_displacement` and duration greater than 10 times `min_duration`.
                - other_tracks: Tracks that do not meet any of the other criteria.
    """
    status_dict = {'valid': [], 'bumped': [], 'motionless': [], 'direction' : [], 'turn_angle' : [], 'other': []}
    num_of_tracks = len(tracks['particle'].unique())

    print(f"Filtering {num_of_tracks} tracks")

    count = 0
    for particle in tracks['particle'].unique():
        if count % 500 == 0:
            print(f"Processed {count} tracks")
        count += 1
        particle_tracks = tracks[tracks['particle'] == particle]
        
        # Classify track based on criteria
        track_status = classify_track(particle_tracks, min_displacement, min_duration, max_turn_angle, max_direction_angle)
        
        # Store the particle ID based on its classification
        status_dict[track_status].append(particle)

    # Filter tracks DataFrame based on classified statuses
    filtered_tracks = tracks[tracks['particle'].isin(status_dict['valid'])]
    bumped_tracks = tracks[tracks['particle'].isin(status_dict['bumped'])]
    motionless_tracks = tracks[tracks['particle'].isin(status_dict['motionless'])]
    direction_tracks = tracks[tracks['particle'].isin(status_dict['direction'])]
    turn_angle_tracks = tracks[tracks['particle'].isin(status_dict['turn_angle'])]
    other_tracks = tracks[tracks['particle'].isin(status_dict['other'])]

    return filtered_tracks, bumped_tracks, motionless_tracks, direction_tracks, turn_angle_tracks, other_tracks


def track_trackpy(plt_labels, pixel_size, init_vid, view=True, view_modes=[ViewMode.ROLLING]):
    """
    Tracks objects in a video using TrackPy, shows tracks using napari (optional).

    Args:
        plt_labels (np.ndarray): A 3D NumPy array containing the platelet labels for each frame of the video.
        pixel_size (float): The size of a pixel in the video (in physical units).
        init_vid (np.ndarray): The original video data (used for visualization if view is True).
        view (bool, optional): Whether to display the tracks in Napari (default: True).
        view_modes (list[ViewMode], optional): A list of view modes to activate.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            A tuple containing four DataFrames:
                - filtered_tracks: Tracks that meet your filtering criteria (e.g., minimum displacement).
                - bumped_tracks: Tracks with significant displacement events.
                - motionless_tracks: Tracks with minimal displacement.
                - direction_tracks: Tracks with invalid direction
                - turn_angle_tracks: Tracks with invalid turn angles
                - other_tracks: Tracks that do not meet any of the other criteria.
    """
    centroid_data = extract_centroids(plt_labels)

    print("Running tracking")
    # Run tracking using trackpy
    tracks = tp.link(centroid_data, search_range=5 / pixel_size, memory=15)
    print("Tracking done")

    # Filter the tracks (optional, based on your criteria)
    filtered_tracks = tp.filter_stubs(tracks, threshold=3)  # Remove short tracks
    print("Short tracks filtered")

    # Updated to unpack all four returned DataFrames
    filtered_tracks, bumped_tracks, motionless_tracks, direction_tracks, turn_angle_tracks, other_tracks = filter_tracks_by_displacement_and_duration(
        filtered_tracks,
        min_displacement= 7.5 / pixel_size
    )

    print("Bumped, motionless, and other tracks filtered")

    if view:
        print("Napari view")
        napari_tracks = format_for_napari(filtered_tracks)
        napari_bump = format_for_napari(bumped_tracks)
        napari_motionless = format_for_napari(motionless_tracks)
        napari_direction = format_for_napari(direction_tracks)
        napari_turn_angle = format_for_napari(turn_angle_tracks)

        viewer = napari.Viewer()
        viewer.add_labels(init_vid, name='InitialVideo')
        viewer.add_labels(plt_labels, name='PlateletMasks')

        if ViewMode.ROLLING in view_modes:
            viewer.add_tracks(napari_tracks, name='Rolling')
        if ViewMode.BUMPED in view_modes:
            viewer.add_tracks(napari_bump, name='Bump')
        if ViewMode.MOTIONLESS in view_modes:
            viewer.add_tracks(napari_motionless, name='Motionless')
        if ViewMode.INVALID_DIRECTION in view_modes:
            viewer.add_tracks(napari_direction, name='Invalid Direction')
        if ViewMode.INVALID_TURN_ANGLE in view_modes:
            viewer.add_tracks(napari_turn_angle, name='Invalid Turn Angles')

        napari.run()
        #export_napari_layers(viewer, '')

    return filtered_tracks, bumped_tracks, motionless_tracks, direction_tracks, turn_angle_tracks, other_tracks


def export_napari_layers(viewer, output_path):
    # Initialize a list to store the layers data
    layers_data = []
    
    # Iterate over all layers in the viewer
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image):
            layers_data.append(layer.data)
        elif isinstance(layer, napari.layers.Labels):
            layers_data.append(layer.data)
        elif isinstance(layer, napari.layers.Tracks):
            # For tracks, we'll create a mask or image representation
            track_data = np.zeros_like(viewer.layers[0].data, dtype=np.uint8)  # Assuming same size as first image layer
            for track in layer.data:
                coords = track[1:].astype(int)  # Coordinates are in the second and third columns
                track_data[tuple(coords)] = 255  # Mark track locations
            layers_data.append(track_data)
    
    # Stack all layers along the first axis
    stacked_layers = np.stack(layers_data, axis=0)
    tiff.imwrite(output_path, stacked_layers, bigtiff=True, photometric='minisblack', compression='zlib')
    print(f"Exported layers saved to {output_path}")

def evaluate_tracks(tracks, folder, filename, pixel_size, postfix):


    tracks.rename(columns={'frame': 'frame_y', 'particle': 'track_id'}, inplace=True)
    tracks['Time (s)'] = tracks['frame_y'] * DT

    track_distance_df = calculate_distance(tracks, pixel_size=pixel_size)
    track_distance_df = calculate_speed_corr(track_distance_df)
    track_distance_df = calculate_displacement(track_distance_df, pixel_size=pixel_size)

    track_distance_df = confinement_ratio(track_distance_df)
    track_distance_df = mean_straight_line_speed(track_distance_df)
    track_distance_df = calc_linearity_of_forward_progression_and_sort(track_distance_df)

    track_means = average_tracks(track_distance_df, pixel_size=pixel_size)


    csv_file_dir = os.path.join(folder, "Averaged CSV")
    if not os.path.exists(csv_file_dir):
        os.makedirs(csv_file_dir, exist_ok=True)

    csv_file_tot = os.path.join(folder, "Tracked CSV")
    if not os.path.exists(csv_file_tot):
        os.makedirs(csv_file_tot, exist_ok=True)

    storage_dir = os.path.join(csv_file_dir, filename[:-4])
    total_dir = os.path.join(csv_file_tot, filename[:-4])
    storage_dir = storage_dir + postfix + ".csv"
    total_dir = total_dir + postfix + ".csv"

    track_means.to_csv(storage_dir)
    track_distance_df.to_csv(total_dir)


def analyze_stacks(data_folder, file_name):
    start_time_preprocess = time.time()

    plt_labels, vid = preprocess_video(folder=data_folder, file_name=file_name)

    time_preprocess = time.time() - start_time_preprocess
    print(f"Preprocessing is complete for {file_name}. Elapsed time: {time_preprocess}")

    # Updated to capture 'other_tracks' as well
    final_tracks, bumped_tracks, motionless_tracks, direction_tracks, turn_angles_tracks, other_tracks = track_trackpy(
        plt_labels=plt_labels, pixel_size=PIXEL_SIZE, init_vid=vid, view=False, view_modes=[ViewMode.ROLLING, ViewMode.MOTIONLESS, ViewMode.BUMPED, ViewMode.INVALID_DIRECTION, ViewMode.INVALID_TURN_ANGLE]
    )

    # Evaluate the generated tracks
    evaluate_tracks(tracks=final_tracks, pixel_size=PIXEL_SIZE, folder=data_folder,
                    filename=file_name, postfix='_rolling')
    evaluate_tracks(tracks=bumped_tracks, pixel_size=PIXEL_SIZE, folder=data_folder,
                    filename=file_name, postfix='_bumped')
    evaluate_tracks(tracks=motionless_tracks, pixel_size=PIXEL_SIZE, folder=data_folder,
                    filename=file_name, postfix='_motionless')
    evaluate_tracks(tracks=direction_tracks, pixel_size=PIXEL_SIZE, folder=data_folder,
                    filename=file_name, postfix='_direction_tracks')
    evaluate_tracks(tracks=turn_angles_tracks, pixel_size=PIXEL_SIZE, folder=data_folder,
                    filename=file_name, postfix='_turn_angles_tracks')
    evaluate_tracks(tracks=other_tracks, pixel_size=PIXEL_SIZE, folder=data_folder,
                    filename=file_name, postfix='_other')



def process_file(folder):
    """Function that runs processing for a file and reports that"""
    file_name = folder.split('/')[-1]
    folder = folder.replace(file_name, '')

    print('Processing the following file: ', file_name)
    analyze_stacks(folder, file_name)
    print('Completed processing for: ', file_name)


def list_folders(directory):
    """List all folders in a given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def track_ilastik(glob_path):
    """Function that runs processing for each .tif from a given folder."""

    l_o_files = [str(tif_file) for tif_file in Path(glob_path).rglob('*.tif')]
    for file in l_o_files:
        process_file(file)


def main():
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Process and track objects in .tif files from a given directory.')
    parser.add_argument('path', type=str, help='The path to the directory containing .tif files to process.')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Start processing files
    print(f"Processing files in directory: {args.path}")
    track_ilastik(glob_path=args.path)
    
    end_time = time.time()
    print(f'Completed. Elapsed time: {end_time - start_time} seconds')


if __name__ == '__main__':
    main()
