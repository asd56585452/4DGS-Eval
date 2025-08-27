# 4DGS Reconstruction Evaluation Tool

This tool is designed to evaluate the quality of 4D Gaussian Splatting (4DGS) reconstruction results. It calculates the following metrics:

-   **PSNR** (Peak Signal-to-Noise Ratio)
-   **SSIM** (Structural Similarity Index)
-   **LPIPS** (Learned Perceptual Image Patch Similarity)

The evaluation is performed at the resolution of the predicted output.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the required dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

    Then, install the dependencies from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `evaluate.py` script is used to perform the evaluation. It takes two required arguments:

-   `gt_path`: The path to the ground truth data.
-   `pred_path`: The path to the prediction data.

Both `gt_path` and `pred_path` can be one of the following:

-   A path to a video file (e.g., `video.mp4`).
-   A path to a directory containing a sequence of images (e.g., `images/`).
-   A path with a C-style format string for image sequences (e.g., `images/frame_%d.png`). The tool will attempt to find images with and without zero-padding.

### Examples

#### Evaluating Video Files

```bash
python evaluate.py /path/to/ground_truth.mp4 /path/to/prediction.mp4
```

#### Evaluating Image Sequences (from a directory)

```bash
python evaluate.py /path/to/gt_images/ /path/to/pred_images/
```

#### Evaluating Image Sequences (with a format string)

This is useful when your images are named like `frame_001.png`, `frame_002.png`, etc.

```bash
python evaluate.py /path/to/gt_images/frame_%d.png /path/to/pred_images/frame_%03d.png
```

The tool will automatically detect the number of frames and handle potential zero-padding in the filenames.

### Example Output

```
PSNR: 25.1234
SSIM: 0.9123
LPIPS: 0.0456
```
