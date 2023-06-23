import streamlit as st
from numpy import fft
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

TITLE = "Filtragem no Domínio da Frequência"

st.set_page_config(TITLE, layout="wide")

st.title(TITLE)

image_original = rgb2gray(cv2.imread("data/folhas1_Reticulada.jpg"))

st.markdown("### Imagem Original e Fourier")
_, col1, col2, _ = st.columns([1, 3, 3, 1])
with col1:
    st.image(image_original)

image_fourier = fft.fftshift(fft.fft2(image_original))
with col2:
    st.image(abs(image_fourier), clamp=True)

st.markdown("### Máscara")
mask = np.ones(image_fourier.shape)
mask_height, mask_width = mask.shape

_, col1, col2, _ = st.columns([1, 3, 3, 1])
with col2:
    pad_vert = st.slider("Padding Vertical", 0.0, mask_width / 2, 5.0, 1.0)
    pad_hori = st.slider("Padding Horizontal", 0.0, mask_height / 2, 5.0, 1.0)
    pad_center_start, pad_center_end = st.slider(
        "Padding Center",
        min_value=0,
        max_value=int(max(mask_width / 2, mask_height / 2)),
        # max_value=200,
        value=(35, 60),
        step=1,
    )

mask[:, int((mask_width / 2) - pad_vert) : int((mask_width / 2) + pad_vert)] = 0
mask[int((mask_height / 2) - pad_hori) : int((mask_height / 2) + pad_hori), :] = 0
y_meshgrid, x_meshgrid = np.ogrid[
    -int((mask_height / 2)) : int((mask_height / 2)),
    -int((mask_width / 2)) : int((mask_width / 2)),
]
mask_filter_allow = x_meshgrid**2 + y_meshgrid**2 <= pad_center_start**2
mask_filter_block = x_meshgrid**2 + y_meshgrid**2 <= pad_center_end**2
mask[mask_filter_block] = 0
mask[mask_filter_allow] = 1
image_fourier_masked = image_fourier.copy() * mask
image_fourier_result = abs(fft.ifft2(image_fourier_masked))

with col1:
    st.image(np.pad(255 * (mask), 1, "constant", constant_values=0), clamp=True)
    st.image(abs(image_fourier_masked), clamp=True)

with col2:
    st.image(image_fourier_result, clamp=True)

st.markdown("### Métricas")
image_original_target = rgb2gray(cv2.imread("data/folhas1.jpg"))
result_metric_ssim = ssim(image_original_target, image_fourier_result, data_range=1.0)
st.write(f"ssim: {result_metric_ssim}")
result_metric_mse = mean_squared_error(image_original_target, image_fourier_result)
st.write(f"mean_squared_error: {result_metric_mse}")
