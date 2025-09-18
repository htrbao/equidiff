import torch
import torchvision.transforms.functional as TF
from torch.nn import functional as F
import numpy as np

from escnn import gspaces
from escnn import nn as enn

from equi_diffpo.model.equi.equi_encoder import EquivariantResEncoder76Cyclic

num_rotations = 8
num_run = 14

seq_len = 50
batch_size = 32

# For actions only
embedding_size = 32

# create 1 equi embedder 
embedder = EquivariantResEncoder76Cyclic(obs_channel=3, n_out=embedding_size, initialize=True).cuda(1)

x = torch.randn(batch_size, 3, 512, 512).cuda(1)
or_x = x
x = enn.GeometricTensor(x, enn.FieldType(embedder.group, 3*[embedder.group.trivial_repr]))


y = embedder.conv[0:num_run](x)

x_transformed = x
# for each group element
with torch.no_grad():
    for g in embedder.group.testing_elements:
        x_transformed = x.transform(g)
        x_transformed = x_transformed  # center crop to 84x84
        y_new = embedder.conv[0:num_run](x_transformed).tensor

        y_transformed = y.transform(g).tensor
        err = (y_transformed - y_new).abs().mean()
        print(torch.allclose(y_new, y_transformed, atol=1e-4), g, err)
