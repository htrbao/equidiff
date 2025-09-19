import torch
import torchvision.transforms.functional as TF
from torch.nn import functional as F
import numpy as np

from einops import rearrange

from escnn import gspaces
from escnn import nn as enn

from equi_diffpo.model.equi.equi_obs_encoder import EquivariantObsEnc

def create_random_input():
    input = dict()
    input['agentview_image'] = torch.zeros(batch_size, seq_len, 3, 84, 84).cuda(1); input['agentview_image'][:, :, :, 0:10, 0:10] = 1
    input['robot0_eye_in_hand_image'] = torch.randn(batch_size, seq_len, 3, 84, 84).cuda(1)
    input['robot0_eef_pos'] = torch.randn(batch_size, seq_len, 3).cuda(1)
    input['robot0_eef_quat'] = torch.randn(batch_size, seq_len, 4).cuda(1)
    input['robot0_gripper_qpos'] = torch.randn(batch_size, seq_len, 2).cuda(1)
    return input

def rotate_input(embedder, input, g_in):
    obs = input["agentview_image"]
    ee_pos = input["robot0_eef_pos"]
    ee_quat = input["robot0_eef_quat"]
    ee_q = input["robot0_gripper_qpos"]
    ih = input["robot0_eye_in_hand_image"]
    # B, T, C, H, W
    batch_size = obs.shape[0]
    t = obs.shape[1]
    obs = rearrange(obs, "b t c h w -> (b t) c h w")
    ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
    ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
    ee_q = rearrange(ee_q, "b t d -> (b t) d")
    ee_rot = embedder.get6DRotation(ee_quat)

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
                embedder.group,
                + 4 * [embedder.group.irrep(1)] # pos, rot
                + 3 * [embedder.group.trivial_repr], # gripper (2), z zpos
            ),)
    if g_in is not None:
        features = features.transform(g_in)

    return dict(
        agentview_image=enn.GeometricTensor(obs, embedder.enc_obs.conv[0].in_type).transform(g_in).tensor.reshape(batch_size, t, 3, 84, 84),
        robot0_eye_in_hand_image=ih,
        robot0_eef_pos=features.tensor[:, [0, 1, 8]].reshape(batch_size, t, 3),
        robot0_eef_quat=embedder.quaternion_to_sixd.inverse(torch.cat([
            features.tensor[:, [2, 4, 6]],
            features.tensor[:, [3, 5, 7]],
        ], dim=1))[:, [1, 2, 3, 0]].reshape(batch_size, t, 4),
        robot0_gripper_qpos=features.tensor[:, 9:].reshape(batch_size, t, 2),
    )

num_rotations = 8
num_run = 14

seq_len = 8
batch_size = 32

# For actions only
embedding_size = 32

# create 1 equi embedder 
embedder = EquivariantObsEnc(obs_shape=(3, 84, 84)).cuda(1)

x = create_random_input()
y = embedder(x)

x_transformed = x
# for each group element
with torch.no_grad():
    for g in embedder.group.testing_elements:
        x_transformed = rotate_input(embedder, x, g)
        y_new = embedder(x_transformed).reshape(batch_size * seq_len, -1)

        y_transformed = enn.GeometricTensor(embedder(x).reshape(batch_size * seq_len, -1), embedder.enc_out.out_type).transform(g).tensor
        err = (y_transformed - y_new).abs().mean()
        print(torch.allclose(y_new, y_transformed, atol=1e-4), g, err)
