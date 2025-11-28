import argparse
import os
import imageio
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.transform import resize
import re
import cv2
import shutil

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

def get_color(value, vmin, vmax, metric_type):
    # Normalize value to 0-1 range based on vmin and vmax
    if vmax == vmin:
        norm = 0.5
    else:
        norm = (value - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)

    # For LPIPS, lower is better, so we invert the logic for color mapping
    # We want Low LPIPS (Good) -> Green, High LPIPS (Bad) -> Red
    # Current norm: 0 (Low) -> 1 (High)
    if metric_type == 'lpips':
        # Invert norm so 1 is Good (Green) and 0 is Bad (Red)
        norm = 1.0 - norm
    
    # Red (0, 0, 255) -> Yellow (0, 255, 255) -> Green (0, 255, 0)
    if norm < 0.5:
        # Red to Yellow
        # B: 0, G: 0->255, R: 255
        ratio = norm * 2
        return (0, int(255 * ratio), 255)
    else:
        # Yellow to Green
        # B: 0, G: 255, R: 255->0
        ratio = (norm - 0.5) * 2
        return (0, 255, int(255 * (1 - ratio)))

def draw_patch_scores(image, scores, patch_size, vmin, vmax, metric_type):
    vis_img = image.copy()
    # Convert RGB to BGR for OpenCV
    if vis_img.shape[2] == 3:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    H, W = vis_img.shape[:2]
    
    # Create a semi-transparent overlay
    overlay = vis_img.copy()
    
    for r, row_scores in enumerate(scores):
        for c, score in enumerate(row_scores):
            y = r * patch_size
            x = c * patch_size
            
            color = get_color(score, vmin, vmax, metric_type)
            
            # Draw rectangle on overlay
            cv2.rectangle(overlay, (x, y), (min(x + patch_size, W), min(y + patch_size, H)), color, -1)
            
            # Draw text
            text = f"{score:.2f}"
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            text_x = x + (patch_size - text_size[0]) // 2
            text_y = y + (patch_size + text_size[1]) // 2
            
            # Draw text shadow/outline for visibility
            cv2.putText(vis_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(vis_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness - 1)

    # Blend overlay
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    return vis_img

def main():
    parser = argparse.ArgumentParser(description='Evaluate 4DGS reconstruction results with patch-based metrics.')
    parser.add_argument('gt_path', type=str, help='Path to the ground truth video or image sequence.')
    parser.add_argument('pred_path', type=str, help='Path to the predicted video or image sequence.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualization results.')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of the patches.')
    parser.add_argument('--gt_start_frame', type=int, default=0, help='The starting frame number for evaluation.')
    parser.add_argument('--pred_start_frame', type=int, default=0, help='The ending frame number for evaluation.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'psnr'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ssim'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'lpips'), exist_ok=True)

    gt_frames = load_data(args.gt_path)
    pred_frames = load_data(args.pred_path)

    if not gt_frames or not pred_frames:
        print("Could not load data.")
        return

    gt_frames = gt_frames[args.gt_start_frame:]
    pred_frames = pred_frames[args.pred_start_frame:]

    # Use the resolution of the prediction
    pred_res = pred_frames[0].shape[:2]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    patch_size = args.patch_size
    
    # Store results for second pass
    all_psnr_grids = []
    all_ssim_grids = []
    all_lpips_grids = []
    
    all_psnr_values = []
    all_ssim_values = []
    all_lpips_values = []

    print("Pass 1: Calculating metrics...")
    for i in range(min(len(gt_frames), len(pred_frames))):
        print(f"Processing frame {i}...")
        gt_frame = gt_frames[i]
        pred_frame = pred_frames[i]

        if gt_frame.shape[:2] != pred_res:
            gt_frame = resize(gt_frame, (pred_res[0], pred_res[1]), anti_aliasing=True)
            if gt_frame.max() <= 1.0:
                gt_frame = (gt_frame * 255).astype(np.uint8)

        if pred_frame.shape[:2] != pred_res:
            pred_frame = resize(pred_frame, (pred_res[0], pred_res[1]), anti_aliasing=True)
            if pred_frame.max() <= 1.0:
                pred_frame = (pred_frame * 255).astype(np.uint8)

        H, W = pred_res
        
        # Grid iteration
        rows = int(np.ceil(H / patch_size))
        cols = int(np.ceil(W / patch_size))

        psnr_grid = np.zeros((rows, cols))
        ssim_grid = np.zeros((rows, cols))
        lpips_grid = np.zeros((rows, cols))

        for r in range(rows):
            for c in range(cols):
                y_start = r * patch_size
                x_start = c * patch_size
                y_end = min(y_start + patch_size, H)
                x_end = min(x_start + patch_size, W)

                # Extract patches
                gt_patch = gt_frame[y_start:y_end, x_start:x_end]
                pred_patch = pred_frame[y_start:y_end, x_start:x_end]

                # Convert to tensor
                gt_tensor = torch.from_numpy(gt_patch).permute(2, 0, 1).float().to(device) / 255.0
                pred_tensor = torch.from_numpy(pred_patch).permute(2, 0, 1).float().to(device) / 255.0

                gt_tensor = gt_tensor.unsqueeze(0)
                pred_tensor = pred_tensor.unsqueeze(0)

                # Calculate metrics
                if gt_patch.shape[0] < 16 or gt_patch.shape[1] < 16:
                     try:
                         p = psnr_metric(pred_tensor, gt_tensor)
                         s = ssim_metric(pred_tensor, gt_tensor)
                         l = lpips_metric(pred_tensor, gt_tensor)
                     except Exception as e:
                         print(f"Error on patch {r},{c}: {e}")
                         p, s, l = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                else:
                    p = psnr_metric(pred_tensor, gt_tensor)
                    s = ssim_metric(pred_tensor, gt_tensor)
                    l = lpips_metric(pred_tensor, gt_tensor)

                psnr_grid[r, c] = p.item()
                ssim_grid[r, c] = s.item()
                lpips_grid[r, c] = l.item()
                
                all_psnr_values.append(p.item())
                all_ssim_values.append(s.item())
                all_lpips_values.append(l.item())

        all_psnr_grids.append(psnr_grid)
        all_ssim_grids.append(ssim_grid)
        all_lpips_grids.append(lpips_grid)

    # Calculate global stats
    psnr_min, psnr_max = min(all_psnr_values), max(all_psnr_values)
    ssim_min, ssim_max = min(all_ssim_values), max(all_ssim_values)
    lpips_min, lpips_max = min(all_lpips_values), max(all_lpips_values)

    print(f"PSNR Range: {psnr_min:.4f} - {psnr_max:.4f}")
    print(f"SSIM Range: {ssim_min:.4f} - {ssim_max:.4f}")
    print(f"LPIPS Range: {lpips_min:.4f} - {lpips_max:.4f}")

    print("Pass 2: Generating visualizations...")
    for i in range(min(len(gt_frames), len(pred_frames))):
        pred_frame = pred_frames[i]
        if pred_frame.shape[:2] != pred_res:
            pred_frame = resize(pred_frame, (pred_res[0], pred_res[1]), anti_aliasing=True)
            if pred_frame.max() <= 1.0:
                pred_frame = (pred_frame * 255).astype(np.uint8)

        # Generate visualizations
        vis_psnr = draw_patch_scores(pred_frame, all_psnr_grids[i], patch_size, psnr_min, psnr_max, 'psnr')
        vis_ssim = draw_patch_scores(pred_frame, all_ssim_grids[i], patch_size, ssim_min, ssim_max, 'ssim')
        vis_lpips = draw_patch_scores(pred_frame, all_lpips_grids[i], patch_size, lpips_min, lpips_max, 'lpips')

        # Save images
        cv2.imwrite(os.path.join(args.output_dir, 'psnr', f"frame_{i:03d}_psnr.png"), vis_psnr)
        cv2.imwrite(os.path.join(args.output_dir, 'ssim', f"frame_{i:03d}_ssim.png"), vis_ssim)
        cv2.imwrite(os.path.join(args.output_dir, 'lpips', f"frame_{i:03d}_lpips.png"), vis_lpips)

    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
