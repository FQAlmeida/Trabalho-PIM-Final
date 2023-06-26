from os import PathLike
from typing import Generator, Iterator, Union
import cv2
import ffmpy
import numpy as np
import streamlit as st

qtd_frames = 30.0 * 23


def match_template(template: np.ndarray, image: np.ndarray, method_idx: int):
    img = image.copy()
    w, h = template.shape[0:2]
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
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(image_gray, template_gray, method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)  # type: ignore

    return img


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
        frame: np.ndarray = frame
        yield frame
        counter += 1
        if counter >= qtd_frames:  # limite temporarario pra testar
            break
    cap.release()


# @st.cache_data
def create_video_from_frames(
    frames: Iterator[np.ndarray], output_url: Union[PathLike, str], fps=30.0
):
    first_frame = next(frames)
    height, width = first_frame.shape[0:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = 0x00000021
    out = cv2.VideoWriter(output_url, fourcc, fps, (width, height), isColor=True)
    out.write(first_frame)
    progress = st.progress(0, "Doing")
    done = 1.0
    for frame in frames:
        out.write(frame)
        done += 1
        progress.progress(done / qtd_frames, "Doing")
    out.release()
    st.write(f"done {done}")


video_url = "data/videoX.mp4"
template_path = "data/referencia_bom.png"

video_url_2 = "data/videoX_2.mp4"
template_path_2 = "data/referencia_ruim.png"

template = cv2.imread(template_path)
template_2 = cv2.imread(template_path_2)

col1, col2 = st.columns(2)
with col1:
    st.video(video_url)
with col2:
    st.video(video_url_2)

frames = split_video_frames(video_url)
frames_2 = split_video_frames(video_url_2)

method = 5


def transform_frame(frame: np.ndarray):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_frame = match_template(template, frame, method)
    # result_frame = frame
    return result_frame


output_url = "data/video_saida.mp4"
output_url_2 = "data/video_saida_2.mp4"
# create_video_from_frames(map(transform_frame, frames_2), output_url_2)

output_url_webm = "data/video_saida.webm"
output_url_webm_2 = "data/video_saida_2.webm"
ff = ffmpy.FFmpeg(
    global_options=("-y",),
    inputs={output_url_2: None},
    outputs={output_url_webm_2: None},
)
ff.run()

# st.write(len(modified_frames))
# del modified_frames[:]
col1, col2 = st.columns(2)
with col1:
    st.video(output_url_webm, format="video/webm")
with col2:
    st.video(output_url_webm_2, format="video/webm")
