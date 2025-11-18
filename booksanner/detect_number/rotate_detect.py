import os
import cv2
import glob
import time
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import sys
import numpy as np

# Import detection function directly
sys.path.append("..")
from doclayout_yolo import YOLOv10

# Global model cache - load once, reuse everywhere
_LOADED_MODEL = None
_MODEL_DEVICE = None
_MODEL_CLASSES = None

def normalize(s: str) -> str:
    """Normalize class names for consistent matching"""
    return s.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_").replace("text", "_text")

def get_or_load_model(model_path, device=None):
    """Load model once and cache it for reuse"""
    global _LOADED_MODEL, _MODEL_DEVICE, _MODEL_CLASSES
    
    if _LOADED_MODEL is None:
        print(f"🧠 Loading model from {model_path}...")
        start_time = time.time()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        _LOADED_MODEL = YOLOv10(model_path)
        _MODEL_DEVICE = device
        
        # Enable half precision for CUDA
        if device == "cuda":
            try:
                _LOADED_MODEL.model.half()
                print("✅ Using FP16 (half precision) for faster inference")
            except Exception as e:
                print(f"⚠️ FP16 not available: {e}, using FP32")
        
        # Get class information
        _MODEL_CLASSES = _LOADED_MODEL.model.names
        
        load_time = time.time() - start_time
        print(f"⚡ Model loaded in {load_time:.2f} seconds")
        print(f"📦 Available classes: {sorted(_MODEL_CLASSES.values())}")
        
        # Find the "abandon" class ID
        abandon_class_id = None
        for cls_id, cls_name in _MODEL_CLASSES.items():
            if normalize(cls_name) == "abandon":
                abandon_class_id = cls_id
                break
        
        if abandon_class_id is not None:
            print(f"🎯 Found 'abandon' class ID: {abandon_class_id}")
        else:
            print(f"⚠️ WARNING: 'abandon' class not found in model classes!")
            print(f"   Available normalized classes: {[normalize(name) for name in _MODEL_CLASSES.values()]}")
            abandon_class_id = 0  # Default to first class if not found
        
    return _LOADED_MODEL, _MODEL_DEVICE, _MODEL_CLASSES, abandon_class_id

def rotate_and_save(file_path, output_dir, rotation_code):
    """Rotate image and save to output directory - optimized version"""
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Failed to read image: {file_path}")
            return False
        
        rotated = cv2.rotate(img, rotation_code)
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        cv2.imwrite(output_path, rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"❌ Rotation failed for {file_path}: {e}")
        return False

def run_detection_task(input_dir, output_dir, model, device, imgsz=1024, conf=0.35, 
                      abandon_class_id=0, batch_size=32):  # 👈 Use specific class ID
    """Run detection on a single directory - ONLY detect 'abandon' class"""
    try:
        print(f"🔍 Processing {input_dir} -> {output_dir} (ONLY 'abandon' class)")
        start_time = time.time()
        
        # Run prediction - ONLY detect 'abandon' class using classes parameter
        det_res = model.predict(
            os.path.join(input_dir, "*"),
            imgsz=imgsz,
            conf=conf,
            iou=0.7,
            device=device,
            batch=batch_size,
            half=(device == "cuda"),
            save=False,
            exist_ok=True,
            classes=[abandon_class_id],  # 👈 KEY FIX: Only detect this specific class
            verbose=True  # Show detection progress
        )
        
        if not det_res:
            print(f"⚠️ No images processed in {input_dir}")
            return False
        
        total_kept = 0
        os.makedirs(output_dir, exist_ok=True)
        total_images = len(det_res)
        
        print(f"📊 Processing {total_images} images with ONLY 'abandon' class detections")
        
        for idx, res in enumerate(det_res):
            if hasattr(res, 'path'):
                src_path = res.path
            else:
                src_path = os.path.join(input_dir, f"image_{idx}.jpg")
            
            base = os.path.splitext(os.path.basename(src_path))[0]
            ext = os.path.splitext(os.path.basename(src_path))[1] or ".jpg"
            
            # Skip if no detections (this should be rare now since we're only detecting one class)
            if res.boxes is None or len(res.boxes) == 0:
                print(f"[{idx+1}/{total_images}] [{base}] No 'abandon' detections found")
                # Create fallback crop
                img = cv2.imread(src_path)
                if img is not None:
                    h, w = img.shape[:2]
                    w_crop = min(30, w)
                    h_crop = min(30, h)
                    crop = img[0:h_crop, 0:w_crop]
                    fallback_name = f"{base}______crop_topleft_30x30_____{ext}"
                    fallback_path = os.path.join(output_dir, fallback_name)
                    cv2.imwrite(fallback_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                continue
            
            # Process detections - now ALL detections should be 'abandon' class
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf_scores = res.boxes.conf.cpu().numpy()
            
            img = cv2.imread(src_path)
            if img is None:
                print(f"[{idx+1}/{total_images}] [{base}] Failed to read image")
                continue
            
            # Create mask for kept regions (all detections are 'abandon')
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            kept = []
            per_image_counter = {}
            
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                class_id = int(cls[i])
                class_name = res.names[class_id]
                
                # Since we filtered at inference, this should always be 'abandon'
                x1g, y1g = max(0, x1), max(0, y1)
                x2g, y2g = min(img.shape[1], x2), min(img.shape[0], y2)
                if x2g <= x1g or y2g <= y1g:
                    continue
                
                class_norm = normalize(class_name)
                
                per_image_counter.setdefault(class_norm, 0)
                per_image_counter[class_norm] += 1
                num = per_image_counter[class_norm]
                
                # Save crop
                crop_name = f"{base}______crop_{class_norm}{num:02d}_____{ext}"
                crop_path = os.path.join(output_dir, crop_name)
                crop = img[y1g:y2g, x1g:x2g]
                cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                mask[y1g:y2g, x1g:x2g] = 255
                kept.append({
                    "cls_id": class_id,
                    "cls_name": class_name,
                    "conf": float(conf_scores[i]),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_path": crop_path.replace("\\", "/"),
                })
                total_kept += 1
            
            # Fallback if no crops kept (shouldn't happen often)
            if not kept:
                print(f"[{idx+1}/{total_images}] [{base}] No valid 'abandon' crops saved")
                h, w = img.shape[:2]
                w_crop = min(30, w)
                h_crop = min(30, h)
                crop = img[0:h_crop, 0:w_crop]
                fallback_name = f"{base}_____crop_topleft_30x30_____{ext}"
                fallback_path = os.path.join(output_dir, fallback_name)
                cv2.imwrite(fallback_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Progress update every 10 images
            if (idx + 1) % 10 == 0:
                print(f"[{idx+1}/{total_images}] Processed {base} - Found {len(xyxy)} abandon objects")
        
        process_time = time.time() - start_time
        avg_time_per_image = process_time / total_images if total_images > 0 else 0
        print(f"✅ Completed {input_dir} -> {output_dir} in {process_time:.2f}s")
        print(f"   Total 'abandon' detections: {total_kept}")
        print(f"   Average time per image: {avg_time_per_image:.3f}s")
        return True
    
    except Exception as e:
        print(f"❌ Detection failed for {input_dir}: {e}")
        import traceback
        traceback.print_exc()
        return False

def _process_file_group(group_data):
    """Optimized worker function for parallel file cleanup"""
    base_name, files = group_data
    if not files:
        return {'base_name': base_name, 'deleted': 0, 'kept': 'N/A', 'original_count': 0}
    
    # Find smallest file by size
    smallest_file = min(files, key=lambda x: x[2])
    smallest_filename, smallest_filepath, smallest_size = smallest_file
    
    deleted_count = 0
    kept_info = ""
    
    # Delete other files
    for filename, file_path, file_size in files:
        if file_path != smallest_filepath and os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                print(f"⚠️ Could not delete {filename}: {e}")
    
    # Rename smallest file
    if os.path.exists(smallest_filepath):
        original_extension = os.path.splitext(smallest_filename)[1].lower()
        new_name = f"{base_name}{original_extension}"
        new_path = os.path.join(os.path.dirname(smallest_filepath), new_name)
        
        if smallest_filepath != new_path and not os.path.exists(new_path):
            try:
                os.rename(smallest_filepath, new_path)
                kept_info = f"{smallest_filename} -> {new_name}"
                smallest_filepath = new_path
            except OSError as e:
                kept_info = f"{smallest_filename} (rename failed: {e})"
        else:
            kept_info = smallest_filename
    
    return {
        'base_name': base_name,
        'deleted': deleted_count,
        'kept': kept_info,
        'original_count': len(files)
    }

def keep_smallest_and_delete_rest_parallel(directory):
    """Parallel version of cleanup with significant speed improvements"""
    print(f"🧹 Cleaning up directory: {directory}...")
    start_time = time.time()
    
    # Fast file scanning using os.scandir
    file_groups = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                name_without_ext = os.path.splitext(entry.name)[0]
                base_name = name_without_ext.split('_', 1)[0]
                
                if base_name not in file_groups:
                    file_groups[base_name] = []
                
                try:
                    file_size = entry.stat().st_size
                    file_groups[base_name].append((entry.name, entry.path, file_size))
                except Exception as e:
                    print(f"⚠️ Error processing {entry.name}: {e}")
    
    total_deleted = 0
    total_kept = 0
    total_groups = len(file_groups)
    
    if total_groups == 0:
        print("✅ No files to clean up.")
        return 0
    
    print(f"ParallelGroups to process: {total_groups}")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        results = list(executor.map(_process_file_group, file_groups.items()))
    
    # Process results
    for res in results:
        if res['original_count'] > 0:
            total_deleted += res['deleted']
            total_kept += 1
    
    cleanup_time = time.time() - start_time
    print(f"✅ Cleanup complete! Kept {total_kept} files, deleted {total_deleted} duplicates.")
    print(f"⏱ Cleanup time: {cleanup_time:.2f} seconds ({cleanup_time/60:.2f} minutes)")
    return total_deleted

def main():
    start_time = time.time()
    print("🚀 Starting optimized document processing pipeline (ONLY 'abandon' class)...")
    
    # Output folders
    dir_cw_1 = "cw_1"
    dir_ccw_2 = "ccw_2"
    dir_detect_1 = "detect_1"
    dir_detect_2 = "detect_2"
    
    # Create directories
    for dir_path in [dir_cw_1, dir_ccw_2, dir_detect_1, dir_detect_2]:
        os.makedirs(dir_path, exist_ok=True)
    
    ROTATE_90_CW = cv2.ROTATE_90_CLOCKWISE
    ROTATE_90_CCW = cv2.ROTATE_90_COUNTERCLOCKWISE
    
    # Get files - optimized scanning
    def get_valid_files(folder):
        if not os.path.exists(folder):
            print(f"⚠️ Folder '{folder}' does not exist. Creating empty list.")
            return []
        return [f for f in glob.glob(os.path.join(folder, "*.*")) 
                if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    folder1_files = get_valid_files("1")
    folder2_files = get_valid_files("2")
    
    print(f"📁 Found {len(folder1_files)} files in folder '1'")
    print(f"📁 Found {len(folder2_files)} files in folder '2'")
    
    # -------------------------
    # ROTATION (Parallel)
    # -------------------------
    rotation_time_1 = rotation_time_2 = 0
    
    if folder1_files:
        print(f"\n🔄 Processing folder '1' -> 'cw_1' (90° CW)...")
        rotation_start = time.time()
        
        rotate_func_1 = partial(rotate_and_save, output_dir=dir_cw_1, rotation_code=ROTATE_90_CW)
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            results1 = list(executor.map(rotate_func_1, folder1_files))
        
        rotation_time_1 = time.time() - rotation_start
        success_count1 = sum(1 for r in results1 if r)
        print(f"✅ Rotated {success_count1}/{len(folder1_files)} images in {rotation_time_1:.2f}s")
    
    if folder2_files:
        print(f"\n🔄 Processing folder '2' -> 'ccw_2' (270° CW)...")
        rotation_start = time.time()
        
        rotate_func_2 = partial(rotate_and_save, output_dir=dir_ccw_2, rotation_code=ROTATE_90_CCW)
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            results2 = list(executor.map(rotate_func_2, folder2_files))
        
        rotation_time_2 = time.time() - rotation_start
        success_count2 = sum(1 for r in results2 if r)
        print(f"✅ Rotated {success_count2}/{len(folder2_files)} images in {rotation_time_2:.2f}s")
    
    # -------------------------
    # DETECTION (Parallel)
    # -------------------------
    detection_time_1 = detection_time_2 = 0
    
    # Get model and find abandon class ID
    model, device, classes, abandon_class_id = get_or_load_model("../weights/yolov10-doclayout.pt")
    
    print(f"\n🎯 Configuration: Only detecting 'abandon' class (ID: {abandon_class_id})")
    
    if folder1_files:
        print(f"\n🔍 Starting detection for folder '1' (ONLY 'abandon' class)...")
        detection_start_1 = time.time()
        run_detection_task(
            input_dir=dir_cw_1,
            output_dir=dir_detect_1,
            model=model,
            device=device,
            imgsz=1024,
            conf=0.35,
            abandon_class_id=abandon_class_id,  # 👈 Only detect this class
            batch_size=32
        )
        detection_time_1 = time.time() - detection_start_1
        print(f"⏱ Detection time '1': {detection_time_1:.2f} sec ({detection_time_1/60:.2f} min)")
    
    if folder2_files:
        print(f"\n🔍 Starting detection for folder '2' (ONLY 'abandon' class)...")
        detection_start_2 = time.time()
        run_detection_task(
            input_dir=dir_ccw_2,
            output_dir=dir_detect_2,
            model=model,
            device=device,
            imgsz=1024,
            conf=0.35,
            abandon_class_id=abandon_class_id,  # 👈 Only detect this class
            batch_size=32
        )
        detection_time_2 = time.time() - detection_start_2
        print(f"⏱ Detection time '2': {detection_time_2:.2f} sec ({detection_time_2/60:.2f} min)")
    
    # -------------------------
    # CLEANUP (Parallel)
    # -------------------------
    cleanup_time = 0
    if folder1_files or folder2_files:
        print(f"\n🧹 Starting cleanup...")
        cleanup_start = time.time()
        
        if folder1_files:
            keep_smallest_and_delete_rest_parallel(dir_detect_1)
        if folder2_files:
            keep_smallest_and_delete_rest_parallel(dir_detect_2)
        
        cleanup_time = time.time() - cleanup_start
        print(f"✅ Cleanup completed in {cleanup_time:.2f}s")
    
    # -------------------------
    # SUMMARY
    # -------------------------
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🚀 FINAL TIMING SUMMARY (ONLY 'abandon' class detected)")
    print("=" * 70)
    print(f"Rotation folder '1':    {rotation_time_1:8.2f}s  ({rotation_time_1/60:6.2f} min)")
    print(f"Rotation folder '2':    {rotation_time_2:8.2f}s  ({rotation_time_2/60:6.2f} min)")
    print(f"Total rotation time:    {rotation_time_1 + rotation_time_2:8.2f}s")
    print("-" * 70)
    print(f"Detection folder '1':   {detection_time_1:8.2f}s ({detection_time_1/60:6.2f} min)")
    print(f"Detection folder '2':   {detection_time_2:8.2f}s ({detection_time_2/60:6.2f} min)")
    print(f"Total detection time:   {detection_time_1 + detection_time_2:8.2f}s")
    print("-" * 70)
    print(f"Cleanup time:           {cleanup_time:8.2f}s")
    print("-" * 70)
    print(f"TOTAL execution time:   {total_time:8.2f}s ({total_time/60:6.2f} min)")
    print("=" * 70)
    
    # Performance metrics
    total_files = len(folder1_files) + len(folder2_files)
    if total_files > 0:
        avg_time_per_file = total_time / total_files
        print(f"\n📊 Performance: {avg_time_per_file:.2f} seconds per file")
        print(f"   {total_files} files processed in {total_time/60:.2f} minutes")
    
    print("\n🎉 All tasks completed successfully! Only 'abandon' class objects were detected and cropped.")


if __name__ == '__main__':
    main()
