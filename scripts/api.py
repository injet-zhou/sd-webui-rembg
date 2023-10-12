from fastapi import FastAPI, Body

from modules.api.models import *
from modules.api import api
import gradio as gr

import rembg

# models = [
#     "None",
#     "u2net",
#     "u2netp",
#     "u2net_human_seg",
#     "u2net_cloth_seg",
#     "silueta",
# ]

from modules.shared import state
from modules.call_queue import queue_lock


def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/rembg")
    async def rembg_remove(
        input_image: str = Body("", title='rembg input image'),
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
            return {"code": 500, "message": str(e), "image": ""}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass
