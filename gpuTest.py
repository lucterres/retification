import os

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything
from torchvision.io import read_image
from torchvision.utils import save_image

from masactrl.diffuser_utils_inversion_kv import MasaCtrlPipeline
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl_inversion import MutualSelfAttentionControlInversion
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

torch.cuda.set_device(0)  # set the GPU device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Device: " + str(device))

