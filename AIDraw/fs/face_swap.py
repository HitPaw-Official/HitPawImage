import copy
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple
import tempfile
from ifnude import detect
import cv2
import numpy as np
from PIL import Image

import insightface
import onnxruntime

providers = ["CPUExecutionProvider"]


def convert_to_sd(img):
    shapes = []
    chunks = detect(img)
    for chunk in chunks:
        shapes.append(chunk["score"] > 0.7)
    return [any(shapes), tempfile.NamedTemporaryFile(delete=False, suffix=".png")]


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def swap_face(source_img, target_img, model, faces_index: Set[int] = {0}):
    output_path = target_img

    source_img = cv2.cvtColor(np.array(Image.open(source_img)), cv2.COLOR_RGB2BGR)
    target_img = cv2.cvtColor(np.array(Image.open(target_img)), cv2.COLOR_RGB2BGR)
    source_face = get_face_single(source_img, face_index=0)
    if source_face is not None:
        result = target_img
        # model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
        face_swapper = getFaceSwapModel(model)

        for face_num in faces_index:
            target_face = get_face_single(target_img, face_index=face_num)
            if target_face is not None:
                result = face_swapper.get(result, target_face, source_face)
            else:
                print(f"No target face found for {face_num}")

        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        result_image.save(output_path)
    else:
        print("No source face found")
        return target_img
    return output_path
