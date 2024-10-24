import argparse
import time
from skimage import io
from skimage.measure import label, regionprops
import napari
import trackpy as tp
import logging
import os
import csv
import numpy as np
import pandas as pd

# Suppress trackpy logging
logging.getLogger('trackpy').setLevel(logging.WARNING)
import tifffile as tiff

# Subfolder names for organizing output data
THROMBI_NUM_SUBFOLDER = 'ThrombiNumbers'
THROMBI_AREAS_SUBFOLDER = 'ThrombiAreas'
TRACKED_SUBFOLDER = 'Tracked CSV'
NAPARI_SUBFOLDER = 'Napari'

# Suffixes for output file names
THROMBI_AREAS_SUFFIX = '_thrombi_areas.csv'
THROMBI_NUMBER_SUFFIX = '_thrombi_number.csv'
TRACKED_SUFFIX = '_tracked.csv'

# Remapping rules for pixel values during labeling
PLT_REMAPPING = [(0, 0), (1, 0), (2, 1)]
THROMBI_REMAPPING = [(0, 0), (1, 1), (2, 0)]

# Pixel size for calculations (in micrometers)
PIXEL_SIZE = 0.431

class VideoProcessor:
    """
    Handles loading, preprocessing, and saving of video data.
    """
    def __init__(self, pixel_size=PIXEL_SIZE):
        self.pixel_size = pixel_size

    def load_file(self, file_path):
        """
        Loads a .tif image stack.

        Args:
            file_path (str): The path to the .tif file.

        Returns:
            ndarray: The loaded image stack as a NumPy array.
        """
        return io.imread(file_path)

    def crop_video(self, vid):
        """Crops a 3D numpy array representing a video stack.

        Removes empty rows along the y-axis.

        Args:
            vid (ndarray of dtype int):
                A 3D numpy array representing a video stack with dimensions (depth, height, width).
                The array should contain pixel values where non-zero values are considered for cropping.

        Raises:
            ValueError: If all pixels are zero.

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


    def _write_csv_data(self, file_path, data):
        """
        Writes data to a CSV file.

        Args:
            file_path (str): The path to the output CSV file.
            data (pd.DataFrame or list): The data to be written to the file. 
                                         If a list is provided, each item will be written on a separate line.

        Raises:
            IOError: If there is an error writing to the file.
            OSError: If there is an error accessing the file or directory.
        """
        try:
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                if isinstance(data, pd.DataFrame):
                    data.to_csv(csv_file, index=False)
                elif isinstance(data, list):
                    for item in data:
                        writer.writerow([item])  # Write each item on a separate line
                else:
                    print(f"Warning: Data type not supported for CSV writing: {type(data)}")
        except (IOError, OSError) as e:
            print(f"Error writing to file: {e}")

    def save_data(self, file_path, data):
        """
        Writes data to a CSV file, creating the directory if it doesn't exist.

        Args:
            file_path (str): The path to the output CSV file.
            data (list): The data to be written to the file.
        """
        directory_path = os.path.dirname(file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        self._write_csv_data(file_path, data)


class ObjectLabeler:
    """
    Responsible for labeling platelets and thrombi in image frames.
    """
    def __init__(self, plt_remapping=PLT_REMAPPING, thrombi_remapping=THROMBI_REMAPPING):
        self.plt_remapping = plt_remapping
        self.thrombi_remapping = thrombi_remapping

    def _label_objects(self, frame_image, mapping):
        """
        Remaps pixel values and labels connected components.

        Args:
            frame_image (np.ndarray): A 2D array representing a single frame.
            mapping (list of tuples): Remapping rules (from_value, to_value).

        Returns:
            np.ndarray: A 2D array with labeled connected components.
        """
        remapped = np.zeros_like(frame_image)
        for from_value, to_value in mapping:
            remapped[frame_image == from_value] = to_value
        return label(remapped, background=0)

    def _label_objects_in_video(self, video_array, mapping):
        """
        Labels objects in each frame of a video using the given mapping.

        Args:
            video_array (np.ndarray): A 3D array representing the video.
            mapping (list of tuples): Remapping rules (from_value, to_value).

        Returns:
            np.ndarray: A 3D array with labeled objects in each frame.
        """
        labeled_frames = np.zeros_like(video_array)
        for time_point, frame_image in enumerate(video_array):
            labeled_frames[time_point] = self._label_objects(frame_image, mapping)
        return labeled_frames

    def label_platelets(self, video_array):
        """
        Labels platelets in each frame of a video.

        Args:
            video_array (np.ndarray): A 3D array representing the video.

        Returns:
            np.ndarray: A 3D array with labeled platelets in each frame.
        """
        return self._label_objects_in_video(video_array, self.plt_remapping)

    def label_thrombi(self, video_array):
        """
        Labels thrombi in each frame of a video.

        Args:
            video_array (np.ndarray): A 3D array representing the video.

        Returns:
            np.ndarray: A 3D array with labeled thrombi in each frame.
        """
        return self._label_objects_in_video(video_array, self.thrombi_remapping)

class ThrombiAnalyzer:
    """
    Responsible for analyzing thrombi properties in image frames.
    """
    def characterize_thrombi(self, labeled_thrombi_frames):
        """
        Calculates thrombi area and number in each frame.

        Args:
            labeled_thrombi_frames (np.ndarray): A 3D array where each frame
                has already been labeled for thrombi.

        Returns:
            tuple: A tuple containing two lists:
                - thrombi_areas: A list of floats representing the percentage of area covered by thrombi in each frame.
                - thrombi_number: A list of integers representing the number of thrombi detected in each frame.
        """
        duration, height, width = labeled_thrombi_frames.shape
        video_area = height * width

        thrombi_areas = np.zeros(duration)
        thrombi_number = np.zeros(duration)

        for time_point, frame_image in enumerate(labeled_thrombi_frames):
            # No need to remap and label here, it's already done
            tmp_thrombi_area = [region.area for region in regionprops(frame_image)]
            thrombi_areas[time_point] = (np.sum(tmp_thrombi_area) / video_area) * 100
            thrombi_number[time_point] = len(tmp_thrombi_area)

        return thrombi_areas.tolist(), thrombi_number.tolist()

class PlateletTracker:
    """Tracks platelets in a video based on their centroids."""
    def __init__(self, pixel_size = PIXEL_SIZE,search_range=5, memory=15, filter_threshold=3):
        self.pixel_size = pixel_size
        self.search_range = search_range
        self.memory = memory
        self.filter_threshold = filter_threshold

    def extract_centroids(self, labels_per_frame):
        """Extracts the centroids of labeled regions in a video.

        Args:
            labels_per_frame (np.ndarray):
                A 3D NumPy array containing the area labels for each frame of the video.

        Returns:
            pd.DataFrame:
                A Pandas DataFrame containing the x, y coordinates and frame numbers of the centroids.
                The DataFrame has the following columns:
                    - 'x': The x-coordinate of the centroid.
                    - 'y': The y-coordinate of the centroid.
                    - 'frame': The frame number.
        """
        data = []
        for frame_number, frame in enumerate(labels_per_frame):
            for region in regionprops(frame):
                y, x = region.centroid
                data.append([x, y, frame_number])
        return pd.DataFrame(data, columns=['x', 'y', 'frame'])

    def track(self, centroid_data):
        """
        Performs object tracking using TrackPy.

        Args:
            centroid_data (pd.DataFrame): A Pandas DataFrame containing the centroid coordinates 
                                           and frame numbers of objects to track.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the filtered tracks, where each row 
                           represents a single object at a specific frame. If no tracks are found,
                           returns an empty DataFrame.
        """
        tracks = tp.link(centroid_data, 
                         search_range=self.search_range / self.pixel_size, 
                         memory=self.memory)
        if tracks.empty:
            print("Warning: No tracks found after linking.")
            return tracks  # Return the empty DataFrame
        filtered_tracks = tp.filter_stubs(tracks, threshold=self.filter_threshold)
        filtered_tracks.rename(columns={'frame': 'frame_y', 'particle': 'track_id'}, inplace=True)
        return filtered_tracks

class DataVisualizer:
    """
    Handles visualization of data using Napari (optional).
    """
    def _format_for_napari(self, tracks):
        """
        Formats a Pandas DataFrame containing object tracking data for Napari.

        Args:
            tracks (pd.DataFrame):
                A Pandas DataFrame containing the x, y coordinates and frame numbers of objects.
                Must have columns: ['track_id', 'frame_y', 'y', 'x'].

        Raises:
            ValueError: If the input DataFrame is missing any required columns.

        Returns:
            np.ndarray:
                A NumPy array formatted for Napari, with columns: ['track_id', 'frame_y', 'y', 'x'].
                If the input DataFrame is empty, returns an empty NumPy array.
        """
        if tracks.empty:
            print("Warning: No tracks to format for Napari.")
            return np.empty((0, 4))

        required_columns = ['track_id', 'frame_y', 'y', 'x']
        missing_columns = set(required_columns) - set(tracks.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


        tracks.loc[:, 'track_id'] = tracks['track_id'].astype(int)

        # Extract the relevant columns and convert to NumPy array
        formatted_tracks = tracks[required_columns].to_numpy()
        return formatted_tracks

    def _export_napari_layers(self, viewer, output_path):
        """
        Exports the layers from a Napari viewer to a multi-page TIFF file.

        Args:
            viewer (napari.Viewer): The Napari viewer containing the layers to export.
            output_path (str): The path to save the output TIFF file.
        """
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

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tiff.imwrite(output_path, stacked_layers, bigtiff=True, photometric='minisblack', compression='zlib')
        print(f"Exported layers saved to {output_path}")

    def visualize_data(self, tracks, plt_labels, init_vid, napari_file=None):
        """
        Visualizes data in Napari.

        Args:
            tracks (pd.DataFrame): A Pandas DataFrame containing the tracking data.
            plt_labels (np.ndarray): A NumPy array containing the platelet labels.
            init_vid (np.ndarray): A NumPy array representing the initial video data.
            napari_file (str, optional): The path to save the Napari layers as a multi-page TIFF file. 
                                        Defaults to None (no saving).
        """
        print("Napari view")
        napari_tracks = self._format_for_napari(tracks)

        viewer = napari.Viewer()
        viewer.add_labels(init_vid, name='InitialVideo')
        viewer.add_labels(plt_labels, name='PlateletMasks')

        viewer.add_tracks(napari_tracks, name='All')

        napari.run()
        if napari_file is not None:
            self._export_napari_layers(viewer, napari_file)

class AnalysisPipeline:
    """
    Orchestrates the entire analysis process.
    """
    def __init__(self, data_folder, pixel_size=PIXEL_SIZE):
        self.data_folder = data_folder
        self.video_processor = VideoProcessor(pixel_size=pixel_size)
        self.object_labeler = ObjectLabeler()
        self.thrombi_analyzer = ThrombiAnalyzer()
        self.platelet_tracker = PlateletTracker(pixel_size=pixel_size)
        self.data_visualizer = DataVisualizer()

    def process_file(self, file_name, visualize):
        """
        Processes a single .tif file.

        Args:
            file_name (str): The name of the .tif file to process.
        """
        file_start_time = time.time()
        print(f"Processing file: {file_name}...")
        
        video_array = self.video_processor.load_file(os.path.join(self.data_folder, file_name))
        video_array = self.video_processor.crop_video(video_array)

        plt_labels = self.object_labeler.label_platelets(video_array)
        thrombi_labels =self.object_labeler.label_thrombi(video_array)
        thrombi_areas, thrombi_number = self.thrombi_analyzer.characterize_thrombi(thrombi_labels)
        
        thrombi_areas_file_name = file_name.replace('.tif', THROMBI_AREAS_SUFFIX)
        thrombi_num_file_name = file_name.replace('.tif', THROMBI_NUMBER_SUFFIX)
        self.video_processor.save_data(os.path.join(self.data_folder, THROMBI_AREAS_SUBFOLDER, thrombi_areas_file_name), thrombi_areas)
        self.video_processor.save_data(os.path.join(self.data_folder, THROMBI_NUM_SUBFOLDER, thrombi_num_file_name), thrombi_number)

        centroid_data = self.platelet_tracker.extract_centroids(plt_labels)
        final_tracks = self.platelet_tracker.track(centroid_data)
        
        tracked_file_name = file_name.replace('.tif', TRACKED_SUFFIX)
        self.video_processor.save_data(os.path.join(self.data_folder, TRACKED_SUBFOLDER, tracked_file_name), final_tracks)

        if visualize:
            napari_file_path = os.path.join(self.data_folder, NAPARI_SUBFOLDER, file_name)
            self.data_visualizer.visualize_data(final_tracks, plt_labels, video_array, napari_file_path) 

        file_end_time = time.time()
        file_processing_time = file_end_time - file_start_time
        print(f"Finished processing {file_name} in {file_processing_time:.2f} seconds.")

    def run(self, visualize):
        """Processes all .tif files in the data folder."""
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith(".tif"):
                self.process_file(file_name, visualize)


def main():
    parser = argparse.ArgumentParser(description='Process and track objects in .tif files from a given directory.')
    parser.add_argument('path', type=str, help='The path to the directory containing .tif files to process.')
    parser.add_argument('--visualize', action='store_true', 
                    help='Visualize results using Napari (default: False)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    pipeline = AnalysisPipeline(data_folder=args.path)
    print(f"Processing files in directory: {args.path}")
    pipeline.run(visualize=args.visualize)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f'Completed. Elapsed time: {processing_time:.2f} seconds')


if __name__ == '__main__':
    main()

