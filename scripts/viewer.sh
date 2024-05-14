#! /bin/bash


python local_viewer.py --model-path output/human/neuman/lab/hugs_wo_trimlp/20240303_2145_abl_trimlp-dataset.seq=citron/2024-05-14_10-29-35-7k --point-path point_cloud/iteration_7000/point_cloud.ply --orbit-path /home/mizeller/projects/experiments/ml-hugs/scripts/ellipse_orbit.npy

# python local_viewer.py --model-path $1 --point-path $2 --orbit_path ./scripts/ellipse_orbit.npy

# example: sh scripts/viewer.sh output/gaussian_avatars/twelve point_cloud/iteration_30000/point_cloud.ply
