import os
import time

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

#torch.cuda.set_device(0)  # set the GPU device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:" + str(device))


def load_image(image_path, res, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (res, res))
    image = image.to(device)
    return image


# ================================== change config here! ========================================
P1 = 20
P2 = 5
S1 = 20
S2 = 5

start_time = time.time()

#    load_image('./images/aug/204-1.jpg', 512, device),

ref_images = torch.cat([
    #load_image('./images/seismic/refs/1.jpg', 512, device),
    #load_image('./images/seismic/refs/2.jpg', 512, device),
    #load_image('./images/seismic/refs/3.jpg', 512, device),
    load_image('./images/seismic/refs/4.jpg', 512, device)
])
target_image = load_image('./images/seismic/target/m2.jpg', 512, device)

out_dir = "./workdir/exp/"
# ================================== config end ========================================

print(f"# 0. prepare - Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()

model_path = "CompVis/stable-diffusion-v1-4"
#model_path = "/nethome/atena_projetos/cym7/models/stable-diffusion-v1-4"

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"Sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

ref_num = ref_images.shape[0]
save_image(ref_images, os.path.join(out_dir, f"refs.jpg"), normalize=True)
save_image(target_image, os.path.join(out_dir, f"target.jpg"), normalize=True)

print(f"# 1. invert references - Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()

start_code_ref, latents_list_ref = model.invert(ref_images,
                                                [""] * ref_num,
                                                num_inference_steps=50,
                                                return_intermediates=True)

print(f"# 2. invert target to get IR - Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()

_, latents_list_target_self = model.invert(target_image,
                                           "",
                                           num_inference_steps=50,
                                           return_intermediates=True)
editor = MutualSelfAttentionControlInversion(P1, 10, ref_num=1)  # inversion target with target itself.
regiter_attention_editor_diffusers(model, editor)


# 3. inversion target with IR
print(f"# 3. inversion target with IR- Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()

start_code_tgt, _ = model.invert(target_image,
                                 "",
                                 num_inference_steps=50,
                                 return_intermediates=True,
                                 ref_intermediate_latents=latents_list_target_self)

# Note: `start_code_ref` was not actually used in sampling. It will be replaced if you specify ref_intermediate_latents.
# So `start_code_ref` can be replaced by any tensor which has same shape with `start_code_ref` to get same result.
# eg: torch.zeros_like(start_code_ref)
start_code = torch.cat([start_code_ref, start_code_tgt])

editor = MutualSelfAttentionControl(S1, 10, ref_num=ref_num)
regiter_attention_editor_diffusers(model, editor)

# 4. sampling to get result_01
print(f"# 4. sampling to get result_01 - Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()

image_masactrl = model([""] * (ref_num) + [""],
                       latents=start_code,
                       ref_intermediate_latents=latents_list_ref)
save_image(image_masactrl[-1:], os.path.join(out_dir, f"result_01.jpg"))

# # # ================= iter 2 ==============================
print(f"# Iteration 2: Processing result_01 - Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()
TARGET_PATH = os.path.join(out_dir, f"result_01.jpg")
target_image = load_image(TARGET_PATH, res=512, device=device)

editor = AttentionBase()
regiter_attention_editor_diffusers(model, editor)

# 5. invert result_01 using IR
print(f"# 5. invert result_01 using IR - Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()
editor = MutualSelfAttentionControlInversion(P2, 10, 1)
regiter_attention_editor_diffusers(model, editor)
start_code_tgt, latents_list_tgt = model.invert(target_image,
                                                "",
                                                num_inference_steps=50,
                                                return_intermediates=True,
                                                ref_intermediate_latents=latents_list_target_self)

# Note: start_code_ref was not actually used in sampling. It will be replaced if you specify ref_intermediate_latents.
# It can be replaced by any tensor which has same shape with start_code_ref to get same result.
# eg: torch.torch.zeros_like(start_code_ref)
start_code = torch.cat([start_code_ref, start_code_tgt])

# sampling to get result_02
print(f"# 6. sampling to get result_02 - Time elapsed: {time.time() - start_time:.2f} seconds")
start_time = time.time()
editor = MutualSelfAttentionControl(S2, 10, ref_num=ref_num)
regiter_attention_editor_diffusers(model, editor)
image_masactrl = model([""] * (ref_num) + [""],
                       latents=start_code,
                       ref_intermediate_latents=latents_list_ref)
save_image(image_masactrl[-1:], os.path.join(out_dir, f"result_02.jpg"))

print(f"Syntheiszed images are saved in  - Time elapsed: {time.time() - start_time:.2f} seconds in ", out_dir)
print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
print("Done!")
