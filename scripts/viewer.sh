#! /bin/bash
python local_viewer.py --model-path output/human/neuman/lab/hugs_wo_trimlp/w_gof_w_densification-dataset.seq=citron/2024-05-15_11-55-22 --point-path meshes/human_final_splat.ply --orbit-path ./scripts/ellipse_orbit.npy
# python local_viewer.py --model-path $1 --point-path $2 --orbit_path ./scripts/ellipse_orbit.npy

# example: sh scripts/viewer.sh output/gaussian_avatars/twelve point_cloud/iteration_30000/point_cloud.ply
