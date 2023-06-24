import streamlit as st
import numpy as np
import skimage.morphology as morphology
from matplotlib import pyplot as plt

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
    fig, ax = plt.subplots()
    ax.imshow(w, cmap=plt.gray())
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    ax.imshow(b, cmap=plt.gray())
    st.pyplot(fig)

x = morphology.erosion(w, b)
fig, ax = plt.subplots()
ax.imshow(x, cmap=plt.gray())
st.pyplot(fig)
        # st.write(a)

inters = st.slider("Quantidade Iterações", 0, 10, 0, 1)
for _ in range(inters):
    _, col1, col2, _ = st.columns([1, 3, 3, 1])
    a = morphology.dilation(x, b)
    with col1:
        fig, ax = plt.subplots()
        ax.imshow(a, cmap=plt.gray())
        st.pyplot(fig)
        # st.write(a)
    x = np.logical_and(a, w)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(x, cmap=plt.gray())
        st.pyplot(fig)
        # st.write(x)


# st.image(
#     cv2.resize(
#         x.astype(np.float64),
#         (800, 600),
#         interpolation=cv2.INTER_NEAREST,
#     )
# )
