import numpy as np
import imageio
import os

def generate_data():
    os.makedirs('data/gt', exist_ok=True)
    os.makedirs('data/pred', exist_ok=True)

    black_image = np.zeros((128, 128, 3), dtype=np.uint8)
    white_image = np.ones((128, 128, 3), dtype=np.uint8) * 255

    for i in range(5):
        imageio.imwrite(f'data/gt/frame_{i:03d}.png', black_image)
        imageio.imwrite(f'data/pred/frame_{i:03d}.png', white_image)

if __name__ == '__main__':
    generate_data()
    print("Test data generated successfully.")
