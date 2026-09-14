import torch

from nets_GAN import Unet


model = Unet(
    inshape=[80, 80, 80],
    nb_features=[[48, 96, 192, 384], [384, 192, 96, 48, 24, 1]],
).cuda().eval()
with torch.inference_mode():
    input_tensor = torch.zeros((1, 1, 80, 80, 80), device="cuda")
    output_tensor = model(input_tensor)
print(
    "GPU_FORWARD_OK",
    tuple(input_tensor.shape),
    "->",
    tuple(output_tensor.shape),
    "finite",
    bool(torch.isfinite(output_tensor).all()),
    "allocated_MiB",
    round(torch.cuda.max_memory_allocated() / 1048576, 1),
)
