import launch
import os

use_gpu = False

try:
    import torch

    use_gpu = torch.cuda.is_available()
except:
    pass

if not launch.is_installed("rembg"):
    launch.run_pip(f"install rembg{'[gpu]' if use_gpu else ' --no-deps'}",
                   f"rembg{'[gpu]' if use_gpu else ''}")

for dep in ['onnxruntime' if not use_gpu else 'onnxruntime-gpu', 'pymatting', 'pooch']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for REMBG extension")
