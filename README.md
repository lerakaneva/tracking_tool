# Platelet Tracking and Thrombi Analysis Tool

This Python tool provides a pipeline for analyzing microscopy videos of blood flow, enabling the tracking of individual platelets and the quantification of thrombi formation over time.

## Features

- **Automated Analysis:** Processes .tif image stacks (microscopy videos) to extract key information about platelet movement and thrombus development.
- **Platelet Tracking:** Identifies and tracks individual platelets across frames, providing insights into their trajectories and behavior.
- **Thrombi Characterization:**  Quantifies thrombi formation by calculating the area covered by thrombi and the number of thrombi in each frame.
- **Data Output:**  Generates CSV files containing thrombi data and tracked platelet information for further analysis.
- **Optional Visualization:**  Allows for interactive visualization of the original video, labeled platelets, and tracked platelet paths using the Napari visualization tool.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lerakaneva/tracking_tool.git
   cd tracking_tool

   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To process and analyze `.tif` video files in a directory, run:

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

The tool will process each `.tif` file in the specified directory and create a corresponding set of output files organized into subdirectories within the same directory. For a video file named `example_video.tif`, the following outputs will be generated:

**- Thrombi Data:** (saved in the `ThrombiAreas` and `ThrombiNumbers` subdirectories, respectively)

    - `example_video_thrombi_areas.csv`: Contains the percentage of area covered by thrombi in each frame of the video.
    - `example_video_thrombi_number.csv`: Contains the number of thrombi detected in each frame of the video.

**- Tracked Platelet Data:** (saved in the `Tracked CSV` subdirectory)

    - `example_video_tracked.csv`: Contains the tracked positions of individual platelets over time.

**- Napari Visualization (if `--visualize` is used):**

    - `example_video.tif`: A multi-page TIFF file containing the original video, labeled platelets, and tracked platelet paths, viewable in Napari. This file will be saved in a subdirectory named `Napari`. 

