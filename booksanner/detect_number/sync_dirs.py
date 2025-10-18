import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SyncHandler(FileSystemEventHandler):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        # Define directory pairs
        self.dir_pairs = {
            "detect_1": "cw_1",
            "cw_1": "detect_1",
            "detect_2": "ccw_2",
            "ccw_2": "detect_2"
        }
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        # Get the directory name and file name
        dir_name = os.path.basename(os.path.dirname(event.src_path))
        file_name = os.path.basename(event.src_path)
        
        # Check if this directory is in our pairs
        if dir_name in self.dir_pairs:
            # Get the paired directory
            paired_dir = self.dir_pairs[dir_name]
            # Construct the destination path
            dest_path = os.path.join(self.base_dir, paired_dir, file_name)
            
            # Wait a bit to ensure the file is fully written
            time.sleep(0.5)
            
            # Copy the file to the paired directory
            try:
                shutil.copy2(event.src_path, dest_path)
                print(f"Copied {file_name} from {dir_name} to {paired_dir}")
            except Exception as e:
                print(f"Error copying {file_name}: {e}")
    
    def on_deleted(self, event):
        if event.is_directory:
            return
        
        # Get the directory name and file name
        dir_name = os.path.basename(os.path.dirname(event.src_path))
        file_name = os.path.basename(event.src_path)
        
        # Check if this directory is in our pairs
        if dir_name in self.dir_pairs:
            # Get the paired directory
            paired_dir = self.dir_pairs[dir_name]
            # Construct the path to the file in the paired directory
            paired_file_path = os.path.join(self.base_dir, paired_dir, file_name)
            
            # Delete the file from the paired directory if it exists
            if os.path.exists(paired_file_path):
                try:
                    os.remove(paired_file_path)
                    print(f"Deleted {file_name} from {paired_dir}")
                except Exception as e:
                    print(f"Error deleting {file_name}: {e}")

def sync_directories(base_dir):
    # Create an event handler
    event_handler = SyncHandler(base_dir)
    
    # Create observers for each directory
    observers = []
    directories_to_watch = ["detect_1", "cw_1", "detect_2", "ccw_2"]
    
    for dir_name in directories_to_watch:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            observer = Observer()
            observer.schedule(event_handler, dir_path, recursive=False)
            observers.append(observer)
            print(f"Monitoring {dir_path}")
    
    # Start all observers
    for observer in observers:
        observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop all observers
        for observer in observers:
            observer.stop()
    
    # Wait for all observers to finish
    for observer in observers:
        observer.join()

if __name__ == "__main__":
    # Set the base directory
    base_dir = r"C:\Users\konos\DocLayout-YOLO\bookscanner\detect_number"
    
    print("Starting directory synchronization...")
    print("Press Ctrl+C to stop")
    
    # Start synchronizing
    sync_directories(base_dir)