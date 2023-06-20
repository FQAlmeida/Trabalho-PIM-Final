import streamlit as st
import cv2
import numpy as np
import skimage.morphology as morphology

TITLE = "Erosão e Dilatação"

st.set_page_config(TITLE, layout="wide")

st.title(TITLE)

w = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

b = np.array(
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)

st.markdown("### Imagem Original e Kernel")
_, col1, col2, _ = st.columns([1, 3, 3, 1])
with col1:
    st.image(
        cv2.resize(w.astype(np.float64), (800, 600), interpolation=cv2.INTER_NEAREST)
    )
with col2:
    st.image(
        cv2.resize(
            np.pad(b, 1, "constant", constant_values=0).astype(np.float64),
            (800, 600),
            interpolation=cv2.INTER_NEAREST,
        )
    )

x = morphology.erosion(w, b)

inters = st.slider("Quantidade Iterações", 0, 10, 0, 1)
for _ in range(inters):
    _, col1, col2, _ = st.columns([1, 3, 3, 1])
    a = morphology.dilation(x, b)
    with col1:
        st.image(
            cv2.resize(
                a.astype(np.float64),
                (800, 600),
                interpolation=cv2.INTER_NEAREST,
            )
        )
    x = a * w
    with col2:
        st.image(
            cv2.resize(
                x.astype(np.float64),
                (800, 600),
                interpolation=cv2.INTER_NEAREST,
            )
        )


# st.image(
#     cv2.resize(
#         x.astype(np.float64),
#         (800, 600),
#         interpolation=cv2.INTER_NEAREST,
#     )
# )
