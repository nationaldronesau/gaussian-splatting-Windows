import os
import logging
from argparse import ArgumentParser
import shutil
import math
import subprocess
import pycolmap
import gc
import psutil

# Configure argument parser
parser = ArgumentParser("Colmap converter with batch processing")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--input_folder", "-i", type=str, default="input", help="Folder containing images")
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--batch_size", type=int, default=100, help="Number of images per batch")
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if args.colmap_executable else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if args.magick_executable else "magick"
use_gpu = 1 if not args.no_gpu else 0

log_file_path = os.path.join(args.source_path, "conversion_log.txt")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(message)s")

def log_and_print(message):
    logging.info(message)
    print(message)

def run_colmap_command(command, timeout=1800, retries=1):
    attempt = 0
    while attempt <= retries:
        try:
            result = subprocess.run(command, shell=True, timeout=timeout)
            if result.returncode == 0:
                return
            log_and_print(f"Attempt {attempt+1}: Command failed with code {result.returncode}")
        except subprocess.TimeoutExpired:
            log_and_print(f"Attempt {attempt+1}: Command timed out: {command}")
        attempt += 1
    log_and_print("ERROR: All attempts failed. Exiting.")
    exit(1)

def create_batch_folders(images, batch_size, source_path, input_folder):
    total_images = len(images)
    num_batches = math.ceil(total_images / batch_size)
    batch_folders = []

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_images)
        batch_images = images[start_idx:end_idx]
        batch_folder = os.path.join(source_path, f"batch_{batch_num}")
        os.makedirs(batch_folder, exist_ok=True)
        log_and_print(f"Created batch folder: {batch_folder} for {len(batch_images)} images.")
        batch_folders.append(batch_folder)
        for img in batch_images:
            source_file = os.path.join(source_path, input_folder, img)
            destination_file = os.path.join(batch_folder, img)
            shutil.copy2(source_file, destination_file)
            log_and_print(f"Copied image {img} to {batch_folder}")

    log_and_print(f"Created {len(batch_folders)} batch folders.")
    return batch_folders

def batch_process_images(batch_folders, source_path):
    total_batches = len(batch_folders)
    all_sparse_folders = []

    for batch_num, batch_folder in enumerate(batch_folders):
        try:
            log_and_print(f"Processing batch {batch_num + 1}/{total_batches}...")
            feat_extraction_cmd = f"{colmap_command} feature_extractor --database_path {batch_folder}/database.db --image_path {batch_folder} --ImageReader.single_camera 1 --ImageReader.camera_model {args.camera} --SiftExtraction.use_gpu {use_gpu}"
            run_colmap_command(feat_extraction_cmd)
            feat_matching_cmd = f"{colmap_command} sequential_matcher --database_path {batch_folder}/database.db --SiftMatching.use_gpu {use_gpu}"
            run_colmap_command(feat_matching_cmd)
            output_dir = os.path.join(batch_folder, "sparse")
            os.makedirs(output_dir, exist_ok=True)
            mapper_cmd = (
                f"{colmap_command} mapper --database_path {batch_folder}/database.db --image_path {batch_folder} --output_path {output_dir} "
                "--Mapper.ba_global_function_tolerance=0.000001 --Mapper.ba_global_max_num_iterations=30 "
                "--Mapper.ba_local_max_num_iterations=15 --Mapper.abs_pose_min_num_inliers=5 "

                "--Mapper.tri_ignore_two_view_tracks=1 --Mapper.multiple_models=0 "
                "--Mapper.max_num_models=1 --Mapper.init_num_trials=200 "

                "--Mapper.abs_pose_min_inlier_ratio=0.10 --Mapper.min_focal_length_ratio=0.1 "
                "--Mapper.max_focal_length_ratio=10 --Mapper.filter_min_tri_angle=0.5 "
                "--Mapper.init_min_num_inliers=15 --Mapper.num_threads=1")
            run_colmap_command(mapper_cmd)
            sparse_folder = os.path.join(batch_folder, "sparse", "0")
            if os.path.exists(sparse_folder):
                log_and_print(f"Added sparse folder {sparse_folder} to merge list.")
                all_sparse_folders.append(sparse_folder)
            else:
                log_and_print(f"WARNING: Sparse folder not found in {batch_folder}. Skipping this batch for merging.")
        except Exception as e:
            log_and_print(f"Batch {batch_num} failed with error: {e}. Skipping.")
            continue
        del feat_extraction_cmd, feat_matching_cmd, mapper_cmd, sparse_folder
        gc.collect()

    if not all_sparse_folders:
        log_and_print("ERROR: No valid sparse folders found for merging.")
        exit(1)

    return all_sparse_folders

def merge_databases(batch_folders, source_path):
    log_and_print("Merging batch databases into one central database...")
    central_database_path = os.path.join(source_path, "distorted", "database.db")
    os.makedirs(os.path.dirname(central_database_path), exist_ok=True)

    first_batch_db = os.path.join(batch_folders[0], "database.db")
    if not os.path.exists(first_batch_db):
        raise FileNotFoundError(f"First batch database not found: {first_batch_db}")
    shutil.copy2(first_batch_db, central_database_path)
    log_and_print(f"Initialized central database with: {first_batch_db}")

    for batch_folder in batch_folders[1:]:
        batch_database = os.path.join(batch_folder, "database.db")
        if not os.path.exists(batch_database):
            log_and_print(f"Skipping missing database: {batch_database}")
            continue
        temp_merged_path = os.path.join(source_path, "distorted", "temp_merged.db")
        merge_cmd = f'{colmap_command} database_merger --database_path1 "{central_database_path}" --database_path2 "{batch_database}" --merged_database_path "{temp_merged_path}"'
        run_colmap_command(merge_cmd)
        shutil.move(temp_merged_path, central_database_path)

    log_and_print(f"Merged databases into {central_database_path}")

    log_and_print("Cleaning up temporary batch folders...")
    for batch_folder in batch_folders:
        try:
            shutil.rmtree(batch_folder)
            log_and_print(f"Deleted batch folder: {batch_folder}")
        except Exception as e:
            log_and_print(f"Failed to delete {batch_folder}: {str(e)}")

    return central_database_path

if not args.skip_matching:
    os.makedirs(os.path.join(args.source_path, "distorted", "sparse"), exist_ok=True)
    input_folder_path = os.path.join(args.source_path, args.input_folder)
    images = [img for img in os.listdir(input_folder_path) if img.lower().endswith((".jpg", ".png"))]
    log_and_print(f"Images found in {input_folder_path}: {images}")
    batch_folders = create_batch_folders(images, args.batch_size, args.source_path, args.input_folder)
    all_sparse_folders = batch_process_images(batch_folders, args.source_path)
    merge_databases(batch_folders, args.source_path)

log_and_print("=============================================================================")
log_and_print("The process Image undistortion")
log_and_print("=============================================================================")
img_undist_cmd = f"{colmap_command} image_undistorter --image_path {input_folder_path} --input_path {args.source_path}/sparse/0 --output_path {args.source_path} --output_type COLMAP"
run_colmap_command(img_undist_cmd)

sparse_dir = os.path.join(args.source_path, "sparse")
os.makedirs(os.path.join(sparse_dir, "0"), exist_ok=True)
for file in os.listdir(sparse_dir):
    if file != "0":
        shutil.move(os.path.join(sparse_dir, file), os.path.join(sparse_dir, "0", file))

if args.resize:
    log_and_print("Starting image resizing...")
    for scale, factor in zip(["images_2", "images_4", "images_8"], ["50%", "25%", "12.5%"]):
        os.makedirs(os.path.join(args.source_path, scale), exist_ok=True)
        for file in os.listdir(input_folder_path):
            if file.lower().endswith((".jpg", ".png")):
                log_and_print(f"Processing image: {file}")
                dest_file = os.path.join(args.source_path, scale, file)
                shutil.copy2(os.path.join(input_folder_path, file), dest_file)
                resize_cmd = f'{magick_command} mogrify -resize {factor} "{dest_file}"'
                result = subprocess.run(resize_cmd, shell=True)
                if result.returncode != 0:
                    log_and_print(f"ERROR: Resize failed for {file} at scale {factor} with code {result.returncode}. Exiting.")
                    exit(result.returncode)

log_and_print("Done.")
