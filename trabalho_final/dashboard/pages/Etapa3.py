from os import PathLike
from typing import Generator, Iterator, List, Union
import cv2
import numpy as np
import streamlit as st


def match_template(template: np.ndarray, image: np.ndarray, method_idx: int):
    img = image.copy()
    w, h = template.shape[::-1]

    # define o método a ser utilizado
    methods = [
        cv2.TM_CCOEFF,
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR,
        cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF,
        cv2.TM_SQDIFF_NORMED,
    ]

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
    cv2.rectangle(img, (*top_left, *bottom_right), 255, 2)

    return img


# desmonta o video em frames
def split_video_frames(
    video_url: Union[PathLike, str]
) -> Generator[np.ndarray, None, None]:
    cap: cv2.VideoCapture = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # print(frame) #apagar
        frame: np.ndarray = frame
        yield frame
        counter += 1
        if counter >= 270:  # limite temporarario pra testar 3s
            break
    cap.release()
    # return frames


# monta o video dnv
# @st.cache_data
def create_video_from_frames(
    frames: Iterator[np.ndarray], output_url: Union[PathLike, str], fps=30.0
):
    first_frame = next(frames)
    height, width = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(output_url, fourcc, fps, (width, height), isColor=False)
    out.write(first_frame)
    progress = st.progress(0, "Doing")
    done = 1.0
    for frame in frames:
        out.write(frame)
        done += 1
        progress.progress(done / 270.0, "Doing")
    out.release()
    st.write(f"done {done}")


video_url = "data/videoX.mp4"

template_path = "data/target.png"

template = cv2.imread(template_path, 0)

st.video(video_url)

frames = split_video_frames(video_url)

method = 0

modified_frames: List[np.ndarray] = []

# Aplicar o template matching em cada frame


def transform_frame(frame: np.ndarray):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_frame = match_template(template, frame, method)
    print(result_frame.dtype, frame.dtype)
    # print(result_frame)
    # result_frame = frame
    return result_frame


# for frame in frames:
# st.image(result_frame)
# modified_frames.append(result_frame)

output_url = "data/video_saida.webm"  # Substitua pelo caminho de saída desejado
create_video_from_frames(iter(map(transform_frame, frames)), output_url)
# st.write(len(modified_frames))
# del modified_frames[:]
# del modified_frames
st.video(output_url)

# Exibir o vetor de frames modificado
# for frame in modified_frames:
#    cv2.imshow('Modified Frame', frame)
#    cv2.waitKey(25)

cv2.destroyAllWindows()
