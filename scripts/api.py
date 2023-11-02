from fastapi import FastAPI, Body
import traceback
from modules.api.models import *
from modules.api import api
import gradio as gr
from PIL import Image
import rembg
from modules.shared import state
from modules.call_queue import queue_lock

from imgutils.validate import anime_classify_score

sessions = {}


def session_factory(model: str):
    if model not in sessions:
        sessions[model] = rembg.new_session(model)
    return sessions[model]


def box_validate(box: list, image: Image):
    w, h = image.size
    if len(box) != 4:
        return f'box length must be 4, but got {len(box)}'
    for i in box:
        if not isinstance(i, int):
            return f'box must be int, but got {type(i)}'
    x, y, x1, y1 = box
    if x < 0 or x1 > w or y < 0 or y1 > h:
        return f'box must be in image range, but got {box}'
    if x > x1 or y > y1:
        return f'box invalid'
    return None


def detect_model(img: Image.Image)-> str:
    scores = anime_classify_score(image=img, model_name='caformer_s36_plus')
    threeD = scores['3d']
    comic = scores['comic']
    bangumi = scores['bangumi']
    print(f"3D: {threeD}, comic: {comic}, bangumi: {bangumi}")
    if threeD == 0:
        return "isnet-anime"
    final_score = (comic + bangumi) / threeD
    if final_score > 1:
        return "isnet-anime"
    return 'silueta'


def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/rembg")
    async def rembg_remove(
            input_image: str = Body("", title='rembg input image'),
            box: list = Body([], title="crop box"),
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

            input_image = api.decode_base64_to_image(input_image)
            if box and type(box) == list and len(box) != 0:
                validate_msg = box_validate(box, input_image)
                if validate_msg:
                    return {"code": 400, "message": validate_msg, "image": ""}
                input_image = input_image.crop((box[0], box[1], box[2], box[3]))

            with queue_lock:
                detect_model_name = detect_model(input_image)
                print(f'use model: {detect_model_name}')
                if detect_model_name == 'isnet-anime':
                    alpha_matting = False
                image = rembg.remove(
                    input_image,
                    session=session_factory(detect_model_name),
                    only_mask=False,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                )

            # state.end()
            return {"code": 200, "message": "success", "image": api.encode_pil_to_base64(image).decode("utf-8")}
        except Exception as e:
            # state.end()
            traceback.print_exc()
            return {"code": 500, "message": f'Exception in rembg: {e}', "image": ""}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass
