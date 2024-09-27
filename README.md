Here's a concise `README.md` for your project:

---

# Track and Analyze Thrombi in Video Files

This project processes `.tif` video files to track and analyze thrombi and platelet movements.

## Features

- **Preprocess Video**: Crop, label, and map video frames.
- **Thrombi Analysis**: Calculate thrombi areas and count
- **Track Analysis**: Extract platelet centroids, track motion, and filter trajectories based on criteria.
- **Visualization**: Use `napari` for interactive visualization of tracks.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To process and analyze `.tif` video files in a directory, run:

```bash
python tracking.py /path/to/tif/files
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