import os
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
import sys
import time


def rotate_and_save(file_path, output_dir, rotation_code):
    """Rotate image and save to output directory"""
    img = cv2.imread(file_path)
    if img is None:
        return False
    rotated = cv2.rotate(img, rotation_code)
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    cv2.imwrite(output_path, rotated)
    return True


def run_detection(input_dir, output_dir, model_path="../weights/yolov10-doclayout.pt", conf=0.35, imgsz=1024):
    """Run detection script on a directory"""
    # Get the full path to the input directory from current working directory
    full_input_path = os.path.abspath(input_dir)
    full_output_path = os.path.abspath(output_dir)
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        '../demo_keep_v2_crop.py',  # Script is in parent directory
        '--model', model_path,
        '--image-path', os.path.join(full_input_path, '*'),  # Use absolute path
        '--res-path', full_output_path,  # Use absolute path for output
        '--imgsz', str(imgsz),
        '--conf', str(conf),
        '--keep-class', 'abandon'  # You can change this as needed
    ]
    print(f"🚀 Running detection on {full_input_path} -> {full_output_path}")
    try:
        # Run from parent directory where model exists
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            cwd='..'  # Change working directory to parent
        )
        if result.returncode == 0:
            print(f"✅ Detection completed for {input_dir}")
            return True
        else:
            print(f"❌ Detection failed for {input_dir}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Detection failed for {input_dir} with exception: {e}")
        return False


def keep_smallest_and_delete_rest(directory):
    """
    For each group of files with the same base name (e.g., '0003' from '0003.JPG', '0003_res.JPG'),
    keep the one with the smallest file size, delete the rest, and rename the kept file to 'base.ext'.
    The base name is determined by the part of the filename before the first underscore.
    """
    print(f"🧹 Cleaning up directory: {directory}...")
    
    # Group files by their base name
    file_groups = {}
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Extract base name (part before first underscore)
            name_without_ext = os.path.splitext(filename)[0]
            base_name = name_without_ext.split('_', 1)[0]
            
            if base_name not in file_groups:
                file_groups[base_name] = []
            
            file_size = os.path.getsize(file_path)
            file_groups[base_name].append((filename, file_path, file_size))

    deleted_count = 0
    kept_count = 0
    
    # Process each group
    for base_name, files in file_groups.items():
        # Find the smallest file in the group
        smallest_file = min(files, key=lambda x: x[2])  # x[2] is the file size
        smallest_filename, smallest_filepath, smallest_size = smallest_file
        
        print(f"--- Processing group '{base_name}' ({len(files)} files) ---")
        
        # Delete all other files in the group
        for filename, file_path, file_size in files:
            if file_path != smallest_filepath:
                try:
                    os.remove(file_path)
                    print(f"   Deleted: {filename} ({file_size} bytes)")
                    deleted_count += 1
                except OSError as e:
                    print(f"   ⚠️ Could not delete {filename}: {e}")
        
        # Now, rename the smallest file to 'base.ext'
        original_extension = os.path.splitext(smallest_filename)[1]
        new_name = base_name + original_extension
        new_path = os.path.join(directory, new_name)

        # Check if a rename is necessary
        if smallest_filepath != new_path:
            try:
                os.rename(smallest_filepath, new_path)
                print(f"   Kept and renamed: {smallest_filename} -> {new_name}")
            except OSError as e:
                print(f"   ⚠️ Could not rename {smallest_filename}: {e}")
        else:
            # This case happens if the file was already named correctly (e.g., '0003.JPG')
            print(f"   Kept: {new_name} ({smallest_size} bytes)")
            
        kept_count += 1
            
    print(f"✅ Cleanup complete for {directory}. Kept {kept_count} unique files, deleted {deleted_count} duplicates.")
    return deleted_count


def main():
    start_time = time.time()  # Record start time
    
    # Create output directories
    dir_cw_1 = "cw_1"
    dir_ccw_2 = "ccw_2"
    dir_detect_1 = "detect_1"
    dir_detect_2 = "detect_2"

    os.makedirs(dir_cw_1, exist_ok=True)
    os.makedirs(dir_ccw_2, exist_ok=True)
    os.makedirs(dir_detect_1, exist_ok=True)
    os.makedirs(dir_detect_2, exist_ok=True)

    # Define rotation codes
    ROTATE_90_CW = cv2.ROTATE_90_CLOCKWISE      # 90 degrees clockwise
    ROTATE_90_CCW = cv2.ROTATE_90_COUNTERCLOCKWISE  # 270 degrees clockwise (90 CCW)

    # Get all image files from folder 1 (for 90 degree rotation)
    folder1_pattern = os.path.join("1", "*.*")
    folder1_files = [f for f in glob.glob(folder1_pattern) 
                     if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Get all image files from folder 2 (for 270 degree rotation)
    folder2_pattern = os.path.join("2", "*.*")
    folder2_files = [f for f in glob.glob(folder2_pattern) 
                     if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Process folder 1 images (90 degrees clockwise -> cw_1)
    if folder1_files:
        print(f"🔄 Processing {len(folder1_files)} images from folder '1' -> 'cw_1' (90° CW)")
        rotate_func_1 = partial(rotate_and_save, output_dir=dir_cw_1, rotation_code=ROTATE_90_CW)
        with ProcessPoolExecutor() as executor:
            results1 = list(executor.map(rotate_func_1, folder1_files))
        success_count1 = sum(1 for r in results1 if r)
        print(f"✅ Processed {success_count1}/{len(folder1_files)} images successfully")
    else:
        print("⚠️ No valid image files found in folder '1'")

    # Process folder 2 images (270 degrees clockwise -> ccw_2)
    if folder2_files:
        print(f"🔄 Processing {len(folder2_files)} images from folder '2' -> 'ccw_2' (270° CW)")
        rotate_func_2 = partial(rotate_and_save, output_dir=dir_ccw_2, rotation_code=ROTATE_90_CCW)
        with ProcessPoolExecutor() as executor:
            results2 = list(executor.map(rotate_func_2, folder2_files))
        success_count2 = sum(1 for r in results2 if r)
        print(f"✅ Processed {success_count2}/{len(folder2_files)} images successfully")
    else:
        print("⚠️ No valid image files found in folder '2'")

    # Run detection on cw_1 -> detect_1
    if folder1_files:
        run_detection(dir_cw_1, dir_detect_1, conf=0.35, imgsz=1024)

    # Run detection on ccw_2 -> detect_2
    if folder2_files:
        run_detection(dir_ccw_2, dir_detect_2, conf=0.35, imgsz=1024)

    # ✨ CLEANUP LOGIC: Keep smallest, delete rest, and rename
    print(f"\n🧹 Cleaning up detection output directories...")
    keep_smallest_and_delete_rest(dir_detect_1)
    keep_smallest_and_delete_rest(dir_detect_2)

    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    execution_minutes = execution_time / 60
    
    print(f"\n⏱️ Total execution time: {execution_minutes:.2f} minutes")
    print("\n🎉 All tasks completed!")


if __name__ == '__main__':
    main()