import argparse
import os
import imageio
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.transform import resize
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_data(path):
    if os.path.isfile(path):
        # It's a video file
        try:
            reader = imageio.get_reader(path)
            frames = [frame for frame in reader]
            reader.close()
            return frames
        except Exception as e:
            print(f"Error reading video file {path}: {e}")
            return None
    elif os.path.isdir(path):
        # It's a directory of images
        frames = []
        files = sorted(os.listdir(path), key=natural_sort_key)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    frames.append(imageio.imread(os.path.join(path, f)))
                except Exception as e:
                    print(f"Error reading image file {os.path.join(path, f)}: {e}")
        return frames
    else:
        # It might be a path with a format string, or a glob pattern
        import glob

        # Replace C-style format specifiers with a wildcard
        glob_path = re.sub(r'%[0-9]*d', '*', path)

        files = sorted(glob.glob(glob_path), key=natural_sort_key)

        if files:
            frames = []
            for f in files:
                try:
                    frames.append(imageio.imread(f))
                except Exception as e:
                    print(f"Error reading image file {f}: {e}")
            return frames

    return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate 4DGS reconstruction results.')
    parser.add_argument('gt_path', type=str, help='Path to the ground truth video or image sequence.')
    parser.add_argument('pred_path', type=str, help='Path to the predicted video or image sequence.')
    parser.add_argument('--start_frame', type=int, default=0, help='The starting frame number for evaluation.')
    parser.add_argument('--end_frame', type=int, default=None, help='The ending frame number for evaluation.')
    args = parser.parse_args()

    gt_frames = load_data(args.gt_path)
    pred_frames = load_data(args.pred_path)

    if gt_frames is None or not gt_frames:
        print(f"Could not load ground truth data from {args.gt_path}")
        return
    if pred_frames is None or not pred_frames:
        print(f"Could not load prediction data from {args.pred_path}")
        return

    # Slice frames based on start and end arguments
    gt_frames = gt_frames[args.start_frame:args.end_frame]
    pred_frames = pred_frames[args.start_frame:args.end_frame]

    if not gt_frames:
        print(f"GT frames are empty after applying start/end frame arguments.")
        return
    if not pred_frames:
        print(f"Prediction frames are empty after applying start/end frame arguments.")
        return

    if len(gt_frames) != len(pred_frames):
        print(f"Number of frames mismatch after slicing: GT has {len(gt_frames)}, prediction has {len(pred_frames)}")

    # Use the resolution of the prediction
    pred_res = pred_frames[0].shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    for i in range(min(len(gt_frames), len(pred_frames))):
        gt_frame = gt_frames[i]
        pred_frame = pred_frames[i]

        # Resize GT to match prediction resolution
        if gt_frame.shape[:2] != pred_res:
            gt_frame = resize(gt_frame, (pred_res[0], pred_res[1]), anti_aliasing=True)

        # Ensure pred_frame has the correct resolution (it should, but just in case)
        if pred_frame.shape[:2] != pred_res:
            pred_frame = resize(pred_frame, (pred_res[0], pred_res[1]), anti_aliasing=True)

        gt_tensor = torch.from_numpy(gt_frame).permute(2, 0, 1).float().to(device)
        pred_tensor = torch.from_numpy(pred_frame).permute(2, 0, 1).float().to(device)

        if gt_tensor.max() > 1.0:
            gt_tensor = gt_tensor / 255.0
        if pred_tensor.max() > 1.0:
            pred_tensor = pred_tensor / 255.0

        gt_tensor = gt_tensor.unsqueeze(0)
        pred_tensor = pred_tensor.unsqueeze(0)

        psnr.update(pred_tensor, gt_tensor)
        ssim.update(pred_tensor, gt_tensor)
        lpips.update(pred_tensor, gt_tensor)

    total_psnr = psnr.compute()
    total_ssim = ssim.compute()
    total_lpips = lpips.compute()

    print(f"PSNR: {total_psnr.item():.4f}")
    print(f"SSIM: {total_ssim.item():.4f}")
    print(f"LPIPS: {total_lpips.item():.4f}")

if __name__ == '__main__':
    main()
