from fastapi import FastAPI, Body
import traceback
from modules.api.models import *
from modules.api import api
import gradio as gr
from PIL import Image
import rembg
import numpy as np
from modules.shared import state
from modules.call_queue import queue_lock
from typing import Any, Optional

from imgutils.validate import anime_classify_score

sessions = {}

general_model = 'isnet-general-use'
anime_model = 'isnet-anime'


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


def detect_model(img: Image.Image) -> str:
    try:
        scores = anime_classify_score(
            image=img, model_name='caformer_s36_plus')
        threeD = scores['3d']
        comic = scores['comic']
        bangumi = scores['bangumi']
        # print(f"3D: {threeD}, comic: {comic}, bangumi: {bangumi}")
        if threeD == 0:
            return anime_model
        final_score = (comic + bangumi) / threeD
        if final_score > 1:
            return anime_model
    except Exception as e:
        print(f"Exception in anime_classify_score: {e}")
        traceback.print_exc()
    return general_model


def validate_points(points: list, image: Image.Image, type: str) -> str | None:
    w,h = image.size
    if points and len(points) != 0:
        for point in points:
            if len(point) != 2:
               return f'{type}_points length must be 2, but got {len(point)}'
            for i in point:
                if not isinstance(i, int):
                    return f'{type}_points must be int, but got {type(i)}'
            x, y = point
            if x < 0 or x > w or y < 0 or y > h:
                return f'{type}_points must be in image range, but got {point}'
    return None


def validate_input_points(positive_points: list, negative_points: list, image: Image.Image):
    msg = validate_points(positive_points, image, 'positive')
    if msg:
        return msg
    msg = validate_points(negative_points, image, 'negative')
    if msg:
        return msg
    return None


def rembg_batch(
        images: list,
        return_mask: bool = False,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        auto: bool = True,
) -> list:
    
    image_list = []
    for img in images:
        detect_model_name = detect_model(img)
        if detect_model_name == anime_model and auto:
            alpha_matting = False
        result = rembg.remove(
            img,
            session=session_factory(detect_model_name),
            only_mask=return_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        image_list.append(result)
    return image_list


def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/rembg")
    async def rembg_remove(
            input_image: str = Body("", title='rembg input image'),
            box: list = Body([], title="crop box"),
            model: str = Body("u2net", title='rembg model'),
            return_mask: bool = Body(False, title='return mask'),
            auto: bool = Body(True, title='auto detect anime'),
            alpha_matting: bool = Body(False, title='alpha matting'),
            alpha_matting_foreground_threshold: int = Body(
                240, title='alpha matting foreground threshold'),
            alpha_matting_background_threshold: int = Body(
                10, title='alpha matting background threshold'),
            alpha_matting_erode_size: int = Body(
                10, title='alpha matting erode size')
    ):
        if not model or model == "None":
            model = 'u2net'

        try:

            input_image = api.decode_base64_to_image(input_image)
            if box and type(box) == list and len(box) != 0:
                validate_msg = box_validate(box, input_image)
                if validate_msg:
                    return {"code": 400, "message": validate_msg, "image": ""}
                input_image = input_image.crop(
                    (box[0], box[1], box[2], box[3]))

            with queue_lock:
                # detect_model_name = detect_model(input_image)
                # print(f'use model: {detect_model_name}')
                # if detect_model_name == anime_model and auto:
                #     alpha_matting = False
                res = rembg_batch(
                    [input_image],
                    return_mask=return_mask,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                    auto=auto,
                )
            image = res[0]

            return {"code": 200, "message": "success", "image": api.encode_pil_to_base64(image).decode("utf-8")}
        except Exception as e:
            traceback.print_exc()
            return {"code": 500, "message": f'Exception in rembg: {e}', "image": ""}

    @app.post("/rembg/batch")
    def rembg_bath_api(
        input_images: list = Body([], title='rembg input images'),
        return_mask: bool = Body(False, title='return mask'),
        auto: bool = Body(True, title='auto detect anime'),
        alpha_matting: bool = Body(False, title='alpha matting'),
        alpha_matting_foreground_threshold: int = Body(
            240, title='alpha matting foreground threshold'),
        alpha_matting_background_threshold: int = Body(
            10, title='alpha matting background threshold'),
        alpha_matting_erode_size: int = Body(
            10, title='alpha matting erode size')
    ):
        if not input_images or type(input_images) != list or len(input_images) == 0:
            return {"code": 400, "message": "input_images must be list and not empty", "images": []}
        images = [api.decode_base64_to_image(i) for i in input_images]
        try:
            with queue_lock:
                res = rembg_batch(
                    images,
                    return_mask=return_mask,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                    auto=auto,
                )
            images = [api.encode_pil_to_base64(i).decode("utf-8") for i in res]
            return {"code": 200, "message": "success", "images": images}
        except Exception as e:
            traceback.print_exc()
            return {"code": 500, "message": f'Exception in rembg: {e}', "images": []}
        
    @app.post("/rembg/advanced")
    def rembg_advanced(
        input_image: str = Body("", title='rembg input image'),
        positive_points: list = Body([], title="positive points"),
        negative_points: list = Body([], title="negative points"),
        return_mask: bool = Body(False, title='return mask'),
        auto: bool = Body(True, title='auto detect anime'),
        alpha_matting: bool = Body(False, title='alpha matting'),
        alpha_matting_foreground_threshold: int = Body(
            240, title='alpha matting foreground threshold'),
        alpha_matting_background_threshold: int = Body(
            10, title='alpha matting background threshold'),
        alpha_matting_erode_size: int = Body(
            10, title='alpha matting erode size')
    ):
        
        if not positive_points and not negative_points:
            return {"code": 400, "message": "positive_points and negative_points must be both not empty", "image": ""}
        try:
            input_image = api.decode_base64_to_image(input_image)
            validate_msg = validate_input_points(positive_points, negative_points, input_image)
            if validate_msg:
                return {"code": 400, "message": validate_msg, "image": ""}
            
            session = session_factory('sam')
            input_points = []
            input_labels = []
            for point in positive_points:
                input_points.append(point)
                input_labels.append(1)
            for point in negative_points:
                input_points.append(point)
                input_labels.append(2)
            with queue_lock:
                detect_model_name = detect_model(input_image)
                if detect_model_name == anime_model and auto:
                    alpha_matting = False
                image = rembg.remove(
                    input_image,
                    session=session,
                    input_points=np.array(input_points),
                    input_labels=np.array(input_labels),
                    only_mask=return_mask,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                )
            return {"code": 200, "message": "success", "image": api.encode_pil_to_base64(image).decode("utf-8")}
        except Exception as e:
            traceback.print_exc()
            return {"code": 500, "message": f'Exception in rembg: {e}', "image": ""}

                


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass
