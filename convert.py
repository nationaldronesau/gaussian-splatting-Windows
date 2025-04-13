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
import subprocess

# Configure argument parser
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
magick_command = f'"{args.magick_executable}"' if args.magick_executable else "magick"
use_gpu = 1 if not args.no_gpu else 0

# Configure logging to log to both file and terminal
log_file_path = os.path.join(args.source_path, "conversion_log.txt")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s - %(message)s")

def log_and_print(message):
    logging.info(message)
    print(message)

def run_command(command):
    log_and_print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_and_print(result.stdout)
    except subprocess.CalledProcessError as e:
        log_and_print(f"ERROR: Command failed with return code {e.returncode}\n{e.output}")
        exit(e.returncode)

if not args.skip_matching:
    os.makedirs(os.path.join(args.source_path, "distorted/sparse"), exist_ok=True)

    log_and_print("Starting feature extraction...")
    feat_extraction_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--image_path {args.source_path}/input "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_model {args.camera} "
        f"--SiftExtraction.use_gpu {use_gpu}"
    )
    run_command(feat_extraction_cmd)

    log_and_print("Starting feature matching...")
    feat_matching_cmd = (
        f"{colmap_command} sequential_matcher "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--SiftMatching.use_gpu {use_gpu}"
    )
    run_command(feat_matching_cmd)

    log_and_print("Starting mapping (bundle adjustment)...")
    mapper_cmd = (
        f"{colmap_command} mapper "
        f"--database_path {args.source_path}/distorted/database.db "
        f"--image_path {args.source_path}/input "
        f"--output_path {args.source_path}/distorted/sparse "
        f"--Mapper.ba_global_function_tolerance=0.000001 "
        f"--Mapper.ba_global_max_num_iterations=30 "
        f"--Mapper.ba_local_max_num_iterations=15 "
        f"--Mapper.abs_pose_min_num_inliers=5 "
        f"--Mapper.abs_pose_min_inlier_ratio=0.10 "
        f"--Mapper.min_focal_length_ratio=0.1 "
        f"--Mapper.max_focal_length_ratio=10 "
        f"--Mapper.filter_min_tri_angle=0.5 "
        f"--Mapper.init_min_num_inliers=15 "
        f"--Mapper.num_threads=1"
    )
    run_command(mapper_cmd)

log_and_print("Starting image undistortion...")
img_undist_cmd = (
    f"{colmap_command} image_undistorter "
    f"--image_path {args.source_path}/input "
    f"--input_path {args.source_path}/distorted/sparse/0 "
    f"--output_path {args.source_path} "
    f"--output_type COLMAP"
)
run_command(img_undist_cmd)

# Move sparse files
sparse_path = os.path.join(args.source_path, "sparse")
os.makedirs(os.path.join(sparse_path, "0"), exist_ok=True)
for file in os.listdir(sparse_path):
    if file == '0':
        continue
    source_file = os.path.join(sparse_path, file)
    destination_file = os.path.join(sparse_path, "0", file)
    shutil.move(source_file, destination_file)

if args.resize:
    log_and_print("Starting image resizing...")
    for scale, folder in [(50, "images_2"), (25, "images_4"), (12.5, "images_8")]:
        os.makedirs(os.path.join(args.source_path, folder), exist_ok=True)

    files = os.listdir(os.path.join(args.source_path, "images"))
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            log_and_print(f"Processing image: {file}")
            source_file = os.path.join(args.source_path, "images", file)

            for scale, folder in [(50, "images_2"), (25, "images_4"), (12.5, "images_8")]:
                destination_file = os.path.join(args.source_path, folder, file)
                shutil.copy2(source_file, destination_file)
                resize_cmd = f"{magick_command} mogrify -resize {scale}% {destination_file}"
                run_command(resize_cmd)

log_and_print("Done.")
