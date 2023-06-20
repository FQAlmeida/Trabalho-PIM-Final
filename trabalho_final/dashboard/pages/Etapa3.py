from os import PathLike
from typing import Union
import cv2
import numpy as np
import streamlit as st

def match_template(template: np.ndarray, image: np.ndarray, method_idx: int):
    img = image.copy()
    w, h = template.shape[::-1]

    # define o método a ser utilizado
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    if method_idx < 0 or method_idx >= len(methods):
        print("Método inválido.")
        return img

    method = methods[method_idx]

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    return img

#desmonta o video em frames
def split_video_frames(video_url: Union[PathLike, str]):
    cap: cv2.VideoCapture = cv2.VideoCapture(video_url)
    frames = []

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return frames

    while cap.isOpened():
        ret, frame = cap.read()
        
        if len(frames) > 90:#limite temporarario pra testar 3s
            break

        if not ret:
            break
        # print(frame) #apagar
        frames.append(frame)

    cap.release()
    return frames

#monta o video dnv
def create_video_from_frames(frames, output_url, fps=30.0):
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_url, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

video_url = "data/videoX.mp4"

template_path = "data/target.png"

template = cv2.imread(template_path, 0)

frames = split_video_frames(video_url)

method = 0

modified_frames = []

# Aplicar o template matching em cada frame
for frame in frames:

    print(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_frame = match_template(template, frame, method)
    modified_frames.append(result_frame)

output_url = "data/video_saida.mp4"  # Substitua pelo caminho de saída desejado
create_video_from_frames(modified_frames, output_url)

# Exibir o vetor de frames modificado
#for frame in modified_frames:
#    cv2.imshow('Modified Frame', frame)
#    cv2.waitKey(25)

cv2.destroyAllWindows()