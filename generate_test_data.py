import os
import numpy as np
import imageio
import cv2

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
    gt_dir = 'data/gt'
    pred_dir = 'data/pred'
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    width, height = 512, 512
    num_frames = 5

    for i in range(num_frames):
        # GT: Clean pattern
        if i % 2 == 0:
            gt_img = create_pattern_image(width, height, 'checkerboard')
        else:
            gt_img = create_pattern_image(width, height, 'gradient')
        
        # Pred: GT + Noise + slight blur
        noise = np.random.normal(0, 25, (height, width, 3)).astype(np.int16)
        pred_img = gt_img.astype(np.int16) + noise
        pred_img = np.clip(pred_img, 0, 255).astype(np.uint8)
        
        # Add a "bad patch" in prediction
        patch_size = 128
        start_x = np.random.randint(0, width - patch_size)
        start_y = np.random.randint(0, height - patch_size)
        pred_img[start_y:start_y+patch_size, start_x:start_x+patch_size] = (0, 0, 255) # Blue patch error

        imageio.imwrite(os.path.join(gt_dir, f'frame_{i:03d}.png'), gt_img)
        imageio.imwrite(os.path.join(pred_dir, f'frame_{i:03d}.png'), pred_img)

    print("Test data generated successfully.")

if __name__ == '__main__':
    main()
