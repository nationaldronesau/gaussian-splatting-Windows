#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# Configure logging to log to both file and terminal
log_file_path = os.path.join(args.source_path, "conversion_log.txt")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(message)s")

def log_and_print(message):
    """Log message to both the file and terminal."""
    logging.info(message)
    print(message)

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    log_and_print("Starting feature extraction...")
    feat_extraction_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db " \
        "--image_path " + args.source_path + "/input " \
        "--ImageReader.single_camera 1 " \
        "--ImageReader.camera_model " + args.camera + " " \
        "--SiftExtraction.use_gpu " + str(use_gpu) + " " \
        "--SiftExtraction.max_image_size 3000"
    exit_code = os.system(feat_extraction_cmd)
    if exit_code != 0:
        log_and_print(f"ERROR: Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    log_and_print("Starting feature matching...")
    feat_matching_cmd = colmap_command + " exhaustive_matcher " \
        "--database_path " + args.source_path + "/distorted/database.db " \
        "--SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        log_and_print(f"ERROR: Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Sparse reconstruction (Mapper)
    log_and_print("Starting mapping (bundle adjustment)...")
    mapper_cmd = (colmap_command + " mapper " \
        "--database_path " + args.source_path + "/distorted/database.db " \
        "--image_path "  + args.source_path + "/input " \
        "--output_path "  + args.source_path + "/distorted/sparse " \
        "--Mapper.ba_global_function_tolerance=0.000001 " \
        "--Mapper.ba_global_max_num_iterations=50 " \
        "--Mapper.ba_local_max_num_iterations=25 " \
        "--Mapper.tri_ignore_two_view_tracks=1 " \
        "--Mapper.multiple_models=0 " \
        "--Mapper.max_num_models=1 " \
        "--Mapper.init_num_trials=200 " \
        "--Mapper.abs_pose_min_num_inliers=30 " \
        "--Mapper.abs_pose_min_inlier_ratio=0.15 " \
        "--Mapper.min_focal_length_ratio=0.1 " \
        "--Mapper.max_focal_length_ratio=10 " \
        "--Mapper.num_threads=4")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
