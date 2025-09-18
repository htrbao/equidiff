if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random

import escnn.nn as enn
from einops import rearrange

from equi_diffpo.workspace.base_workspace import BaseWorkspace
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.dataset.base_dataset import BaseImageDataset
from equi_diffpo.common.json_logger import JsonLogger

OmegaConf.register_new_resolver("eval", eval, replace=True)

def rotate_input(embedder, input, g_in):
    obs = input["obs"]["agentview_image"]
    ee_pos = input["obs"]["robot0_eef_pos"]
    ee_quat = input["obs"]["robot0_eef_quat"]
    ee_q = input["obs"]["robot0_gripper_qpos"]
    ih = input["obs"]["robot0_eye_in_hand_image"]
    # B, T, C, H, W
    batch_size = obs.shape[0]
    t = obs.shape[1]
    obs = rearrange(obs, "b t c h w -> (b t) c h w")
    ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
    ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
    ee_q = rearrange(ee_q, "b t d -> (b t) d")
    ee_rot = embedder.enc.get6DRotation(ee_quat)

    pos_xy = ee_pos[:, 0:2]
    pos_z = ee_pos[:, 2:3]
    features = torch.cat(
        [
            pos_xy,
            # ee_rot is the first two rows of the rotation matrix (i.e., the rotation 6D repr.)
            # each column vector in the first two rows of the rotation 6d forms a rho1 vector
            ee_rot[:, 0:1],
            ee_rot[:, 3:4],
            ee_rot[:, 1:2],
            ee_rot[:, 4:5],
            ee_rot[:, 2:3],
            ee_rot[:, 5:6],
            pos_z,
            ee_q,
        ],
        dim=1
    )
    features = enn.GeometricTensor(features, enn.FieldType(
                embedder.enc.group,
                + 4 * [embedder.enc.group.irrep(1)] # pos, rot
                + 3 * [embedder.enc.group.trivial_repr], # gripper (2), z zpos
            ),)
    if g_in is not None:
        features = features.transform(g_in)

    return dict(
        agentview_image=enn.GeometricTensor(obs, embedder.enc.enc_obs.conv[0].in_type).transform(g_in).tensor.reshape(batch_size, t, 3, 84, 84),
        robot0_eye_in_hand_image=ih,
        robot0_eef_pos=features.tensor[:, [0, 1, 8]].reshape(batch_size, t, 3),
        robot0_eef_quat=embedder.enc.quaternion_to_sixd.inverse(torch.cat([
            features.tensor[:, [2, 4, 6]],
            features.tensor[:, [3, 5, 7]],
        ], dim=1))[:, [1, 2, 3, 0]].reshape(batch_size, t, 4),
        robot0_gripper_qpos=features.tensor[:, 9:].reshape(batch_size, t, 2),
    )

class TestEquiWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        self.model.set_normalizer(normalizer)

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            step_log = dict()

            input = next(iter(dataloader))
            y = self.model.predict_action(input["obs"])["action_pred"]

            with torch.no_grad():
                for g in self.model.enc.group.testing_elements:
                    x_transformed = rotate_input(self.model, input, g)
                    y_new = rearrange(self.model.predict_action(x_transformed)["action_pred"], "b t d -> (b t) d")
                    y_new = y_new[:, [0, 1, 3, 4, 5, 6, 7, 8, 2, 9]]

                    y_transformed = rearrange(y, "b t d -> (b t) d")[:, [0, 1, 3, 4, 5, 6, 7, 8, 2, 9]] # posxy, rot, z, gripper
                    y_transformed = enn.GeometricTensor(y_transformed, self.model.diff.out_layer.out_type).transform(g).tensor
                        
                    err = (y_transformed - y_new).abs().mean()
                    print(torch.allclose(y_new, y_transformed, atol=1e-4), g, err)

            rotate_input(self.model, input, g_in=g)
            
            json_logger.log(step_log)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TestEquiWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
