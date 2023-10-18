from fastapi import FastAPI, Body
import traceback
from modules.api.models import *
from modules.api import api
import gradio as gr
from PIL import Image
import numpy as np
import rembg
from modules.shared import state
from modules.call_queue import queue_lock


def is_black_and_white(image):
    arr = np.array(image)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r, g, b = arr[i, j, :3]
            if r != g or g != b:
                return False
    return True


def white_to_transparent(mask: Image) -> Image:
    # 将白色像素全部转成透明
    arr = np.array(mask)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r, g, b, a = arr[i, j, :]
            if r == 255 and g == 255 and b == 255:
                arr[i, j, :] = [255, 255, 255, 0]
    return Image.fromarray(arr)


def crop_img(origin: Image, mask: Image):
    return origin.crop(mask.getbbox())


def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/rembg")
    async def rembg_remove(
            input_image: str = Body("", title='rembg input image'),
            mask: str = Body("", title="mask to crop input image"),
            model: str = Body("u2net", title='rembg model'),
            return_mask: bool = Body(False, title='return mask'),
            alpha_matting: bool = Body(False, title='alpha matting'),
            alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'),
            alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'),
            alpha_matting_erode_size: int = Body(10, title='alpha matting erode size')
    ):
        if not model or model == "None":
            model = 'u2net'

        try:
            state.begin()
            state.job_count = 1

            input_image = api.decode_base64_to_image(input_image)
            use_mask = mask is not None and mask != ""
            mask = api.decode_base64_to_image(mask) if use_mask else None
            if use_mask and not is_black_and_white(mask):
                return {"code": 400, "message": "mask is not black and white", "image": ""}
            if use_mask:
                input_image = crop_img(input_image, white_to_transparent(mask))

            with queue_lock:
                image = rembg.remove(
                    input_image,
                    session=rembg.new_session(model),
                    only_mask=return_mask,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                )

            state.end()
            return {"code": 200, "message": "success", "image": api.encode_pil_to_base64(image).decode("utf-8")}
        except Exception as e:
            state.end()
            traceback.print_exc()
            return {"code": 500, "message": f'Exception in rembg: {e}', "image": ""}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass
