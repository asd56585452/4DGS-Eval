import os
import numpy as np
import imageio
import shutil
from PIL import Image

def create_pattern_image(width, height, pattern_type='checkerboard'):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if pattern_type == 'checkerboard':
        block_size = 32
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                if (x // block_size + y // block_size) % 2 == 0:
                    img[y:y+block_size, x:x+block_size] = (255, 255, 255)
                else:
                    img[y:y+block_size, x:x+block_size] = (0, 0, 0)
    elif pattern_type == 'noise':
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif pattern_type == 'gradient':
        for y in range(height):
            for x in range(width):
                img[y, x] = [int(x/width*255), int(y/height*255), 128]
    return img

def main():
    base_dir = 'test_data_complex'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    width, height = 256, 256
    
    # Structure: results/exp_name/Frame%d-*/renders/compress_step%d/rendered_testv0_fid%d.png
    
    # We want to test sorting by Frame number and then FID
    # Frame folders: Frame0-49, Frame50-99
    # Compress step: just a constant or variable, but we'll use one for simplicity in the pattern
    
    frame_folders = [
        ("Frame0-49", 0, 5),    # Start frame 0, 5 frames
        ("Frame50-99", 50, 5)   # Start frame 50, 5 frames
    ]
    
    exp_name = "sear_steak"
    compress_step = 10999
    
    print(f"Generating test data in {base_dir}...")
    
    for folder_name, start_fid, num_frames in frame_folders:
        dir_path = os.path.join(base_dir, "results", exp_name, folder_name, "renders", f"compress_step{compress_step}")
        os.makedirs(dir_path, exist_ok=True)
        
        for i in range(num_frames):
            fid = start_fid + i
            
            # GT: Checkerboard
            gt_img = create_pattern_image(width, height, 'gradient')
            
            # Pred: Gradient + Noise
            pred_img = create_pattern_image(width, height, 'gradient')
            noise = np.random.normal(0, 25, (height, width, 3)).astype(np.int16)
            pred_img = np.clip(pred_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Add a patch error in pred
            pred_img[50:100, 50:100] = (0, 0, 255)
            
            # Save Pred
            fname = f"rendered_testv0_fid{fid:04d}.png"
            imageio.imwrite(os.path.join(dir_path, fname), pred_img)
            
            # Save GT (in a separate simpler structure for comparison, or same if we want to test complex GT loading too)
            # Let's put GT in a simple folder for now to isolate the complex loading test to Pred
            gt_simple_dir = os.path.join(base_dir, "gt_simple")
            os.makedirs(gt_simple_dir, exist_ok=True)
            imageio.imwrite(os.path.join(gt_simple_dir, f"frame_{fid:04d}.png"), gt_img)

    print("Test data generated successfully.")

if __name__ == '__main__':
    main()
