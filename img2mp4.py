import imageio
import os
import argparse
import glob

def make_video(image_folder, video_name, fps):
    images = []
    # Support common image extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    
    # Sort files to ensure correct order
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return

    print(f"Found {len(image_files)} images. Creating video...")
    
    # Use imageio.v2 to avoid deprecation warning
    # Force format='FFMPEG' to ensure video writing
    writer = imageio.get_writer(video_name, fps=fps, format='FFMPEG')

    for filename in image_files:
        writer.append_data(imageio.v2.imread(filename))
    
    writer.close()
    print(f"Video saved as {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of images to an mp4 video.")
    parser.add_argument("image_folder", help="Path to the folder containing images")
    parser.add_argument("--output", default="output.mp4", help="Output video filename")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")

    args = parser.parse_args()
    
    make_video(args.image_folder, args.output, args.fps)
