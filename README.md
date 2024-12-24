# Platelet Tracking and Thrombi Analysis Tool

This Python tool provides a pipeline for analyzing microscopy videos of blood flow, enabling the tracking of individual platelets and the quantification of thrombi formation and platelet accumulation over time.

## Features

- **Automated Analysis:** Processes .tif image stacks (microscopy videos) to extract key information about platelet movement, thrombi formation, and platelet accumulation.
- **Platelet Tracking:** dentifies and tracks individual platelets across video frames.
- **Thrombi and Platelet Accumulation Characterization:**  Quantifies thrombi formation and platelet accumulation by calculating the area and count of both thrombi and individual platelets in each frame.
- **Data Output:** Generates CSV files containing thrombi data, platelet data, and tracked platelet information for further analysis. 
- **Optional Visualization:**  Enables interactive visualization of the original video, labeled platelets, and tracked platelet paths using Napari.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://gitlab.com/lerakaneva/tracking_tool.git
   cd tracking_tool

   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To process and analyze `.tif` mask files in a directory, run:

```bash
python tracking.py /path/to/tif/files --visualize
```

- Replace `/path/to/tif/files` with the directory containing your `.tif` files.

## Dependencies

- `numpy`
- `pandas`
- `scikit-image`
- `tifffile`
- `trackpy`
- `napari`

All dependencies are specified in `requirements.txt`.

## Output

The tool will process each `.tif` file in the specified directory and create corresponding output files organized into subdirectories within the same directory. For a video file named `example_video.tif`, the following outputs will be generated:

**- Thrombi Data:** (saved in the `ThrombiAreas` and `ThrombiNumbers` subdirectories, respectively)

    - `example_video_thrombi_areas.csv`: Contains the percentage of area covered by thrombi in each frame of the video.
    - `example_video_thrombi_number.csv`: Contains the number of thrombi detected in each frame of the video.

**- Platelet Data:** (saved in the `PlateletAreas` and `PlateletNumbers` subdirectories, respectively)

    - `example_video_platelet_areas.csv`: Contains the area covered by platelets in each frame.
    - `example_video_platelet_number.csv`: Contains the number of platelets detected in each frame.

**- Tracked Platelet Data:** (saved in the `Tracked CSV` subdirectory)

    - `example_video_tracked.csv`: Contains the tracked positions of individual platelets over time.  The file has the following columns:
        - `x`: x-coordinate of the platelet in the image.
        - `y`: y-coordinate of the platelet in the image.
        - `frame_y`: Frame number (time point) of the measurement.
        - `track_id`: Unique identifier for each tracked platelet.

**- Napari Visualization (if `--visualize` is used):**

    - A multi-page TIFF file named `example_video.tif` containing the original video, labeled platelets, and tracked platelet paths, viewable in Napari. This file will be saved in a subdirectory named `Napari`. 


