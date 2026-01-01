import os
import re
import imageio
import glob

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
        
        # Check if the path contains %d, indicating a pattern for sorting
        if '%d' in path:
            # Replace %d with * for globbing
            glob_path = re.sub(r'%[0-9]*d', '*', path)
            files = glob.glob(glob_path)
            
            # Create regex pattern for extraction and sorting
            # Split by %d to find parts
            parts = re.split(r'(%[0-9]*d)', path)
            regex_pattern = ""
            for part in parts:
                if re.match(r'%[0-9]*d', part):
                    regex_pattern += r'(\d+)'
                else:
                    # Escape other parts, but treat existing * as .*
                    escaped = re.escape(part)
                    escaped = escaped.replace(r'\*', '.*')
                    regex_pattern += escaped
            regex_pattern = '^' + regex_pattern + '$'
            
            matched_files = []
            for f in files:
                match = re.match(regex_pattern, f)
                if match:
                    # Extract numbers for sorting
                    numbers = tuple(int(g) for g in match.groups())
                    matched_files.append((numbers, f))
            
            # Sort based on extracted numbers
            matched_files.sort(key=lambda x: x[0])
            files = [f for _, f in matched_files]
            
            print(f"Found {len(files)} files matching pattern.")
            # Show the result as requested
            for i, f in enumerate(files):
                if i < 5 or i >= len(files) - 5:
                    print(f"  {f}")
                elif i == 5:
                    print("  ...")

        else:
            # Fallback to simple glob if no %d pattern
            # Replace C-style format specifiers with a wildcard just in case
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
