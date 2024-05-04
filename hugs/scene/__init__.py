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
import random
import json

from hugs.scene.dataset_readers import readNeumanSceneInfo
from hugs.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    """
    very stripped down version of the Scene class from the original
    GOF code to make it compatible w/ Neuman dataset
    """
    def __init__(self, cfg, train_dataset, gaussians):
        self.model_path = cfg.logdir 
        self.train_dataset = train_dataset
        self.gaussians = gaussians
        self.cfg = cfg
        self.train_cameras = {}
        self.test_cameras = {}

        data_path = os.path.join("data", cfg.dataset.name, "dataset", cfg.dataset.seq) # "data/neumann/dataset/parkinglot" 
        scene_info = readNeumanSceneInfo(path=data_path)

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        for id, cam in enumerate(scene_info.train_cameras):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        # shuffle = True
        random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras[1.0] = cameraList_from_camInfos(scene_info.train_cameras, 1.0, cfg)
        
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]