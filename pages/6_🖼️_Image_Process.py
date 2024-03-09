import cv2
import numpy as np
import streamlit as st
import libs.Chapter03 as ch3
import libs.Chapter04 as ch4
import libs.Chapter05 as ch5
import libs.Chapter09 as ch9

FILE_TYPE = ["jpg", "jpeg", "png","tif","bmp"]

uploaded_img = st.file_uploader("Choose an image...", type=FILE_TYPE)

def OpenNoneColor(uploaded_img):
    img_grey = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return img_grey

def OpenColor(uploaded_img):
    img_color = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), cv2.IMREAD_COLOR)
    imageRGB = cv2.cvtColor(img_color , cv2.COLOR_BGR2RGB)
    return imageRGB

# đảo ngược độ sáng và độ tối của một hình ảnh
def onNegative(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.Negative(img_grey)
                st.image(imgout, clamp=True)

# cân bằng độ sáng và độ tối của một hình ảnh
def onLogarit(uploaded_img):
    if uploaded_img is not None:
        col1, col2 = st.columns(2)
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.Logarit(img_grey)
                st.image(imgout, clamp=True)


def onPower(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.Power(img_grey)
                st.image(imgout, clamp=True)

def onPiecewiseLinear(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.PiecewiseLinear(img_grey)
                st.image(imgout, clamp=True)

def onHistogram(uploaded_img):
    if uploaded_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            img_grey = OpenNoneColor(uploaded_img)
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.Histogram(img_grey)
                st.image(imgout, clamp=True)

def onHistEqual(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = cv2.equalizeHist(img_grey)
                st.image(imgout, clamp=True)

def onHistEqualColor(uploaded_img):
    if uploaded_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            img_color = OpenColor(uploaded_img)
            st.image(img_color, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.HistEqualColor(img_color)
                st.image(imgout, clamp=True)

def onLocalHist(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.LocalHist(img_grey)
                st.image(imgout, clamp=True)

def onHistStat(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.HistStat(img_grey)
                st.image(imgout, clamp=True)

def onBoxFilter(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = cv2.blur(img_grey,(21,21))
                st.image(imgout, clamp=True)

def onGaussFilter(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = cv2.GaussianBlur(img_grey,(43,43),7.0)
                st.image(imgout, clamp=True)

def onThreshold(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.Threshold(img_grey)
                st.image(imgout, clamp=True)

def onMedianFilter(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = cv2.medianBlur(img_grey,7)
                st.image(imgout, clamp=True)

def onSharpen(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.Sharpen(img_grey)
                st.image(imgout, clamp=True)

def onGradient(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch3.Gradient(img_grey)
                st.image(imgout, clamp=True)

def onSpectrum(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch4.Spectrum(img_grey)
                st.image(imgout, clamp=True)

def onFrequencyFilter(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch4.FrequencyFilter(img_grey)
                st.image(imgout, clamp=True)
                
def onDrawNotchRejectFilter():
        draw_btn=st.button("Draw")
        if draw_btn:
            imgout = ch4.DrawNotchRejectFilter()
            st.image(imgout, clamp=True)

def onRemoveMoire(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch4.RemoveMoire(img_grey)
                st.image(imgout, clamp=True)

def onCreateMotionNoise(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch5.CreateMotionNoise(img_grey)
                st.image(imgout, clamp=True)

def onDenoiseMotion(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch5.DenoiseMotion(img_grey)
                st.image(imgout, clamp=True)

def onDenoisestMotion(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                temp = cv2.medianBlur(img_grey, 7)
                imgout = ch5.DenoiseMotion(temp)
                st.image(imgout, clamp=True)

def onConnectedComponent(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch9.ConnectedComponent(img_grey)
                st.image(imgout, clamp=True)

def onCountRice(uploaded_img):
    if uploaded_img is not None:
        img_grey = OpenNoneColor(uploaded_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_grey, clamp=True)
            apply_btn=st.button("Apply")
        if apply_btn:
            with col2:
                imgout = ch9.CountRice(img_grey)
                st.image(imgout, clamp=True)

st.subheader('Các chương xử lý ảnh')

page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://i.pinimg.com/originals/b3/56/7b/b3567bdb76c15bda852fad6c08a8d8f2.png");

        background-size: 100% 100%;
    }
    [data-testid="stHeader"]{
        background: rgba(0,0,0,0);
    }
    [data-testid="stToolbar"]{
        right:2rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-position: center;
        background-color: rgba(129, 110, 95, 0.8);  

    }
    </style>
    """
st.markdown(page_bg_img,unsafe_allow_html=True)
select = st.sidebar.selectbox("Select",["Chapter 3 - Negative", "Chapter 3 - Logarit", "Chapter 3 - Power", "Chapter 3 - PieceWiseLinear", "Chapter 3 - Histogram", "Chapter 3 - HistEqual", 
                                           "Chapter 3 - HistEqualColor", "Chapter 3 - LocalHist", "Chapter 3 - HistStat", "Chapter 3 - BoxFilter", "Chapter 3 - GaussFilter", 
                                           "Chapter 3 - Threshold", "Chapter 3 - MedianFilter", "Chapter 3 - Sharpen", "Chapter 3 - Gradient",
                                           "Chapter 4 - Spectrum", "Chapter 4 - FrequencyFilter", "Chapter 4 - DrawNotchRejectFilter", "Chapter 4 - RemoveMoire",
                                           "Chapter 5 - CreateMotionNoise", "Chapter 5 - DenoiseMotion", "Chapter 5 - DenoisestMotion",
                                           "Chapter 9 - ConnectedComponent", "Chapter 9 - CountRice"])

if select =="Chapter 3 - Negative":
    st.subheader('Chater 3 - Negative')
    onNegative(uploaded_img)
elif select =="Chapter 3 - Logarit":
    st.subheader('Chater 3 - Logarit')
    onLogarit(uploaded_img)
elif select =="Chapter 3 - Power":
    st.subheader('Chater 3 - Power')
    onPower(uploaded_img)
elif select =="Chapter 3 - PieceWiseLinear":
    st.subheader('Chater 3 - PiecewiseLinear')
    onPiecewiseLinear(uploaded_img)
elif select =="Chapter 3 - Histogram":
    st.subheader('Chater 3 - Histogram')
    onHistogram(uploaded_img)
elif select =="Chapter 3 - HistEqual":
    st.subheader('Chater 3 - Histogram Equalization')
    onHistEqual(uploaded_img)
elif select =="Chapter 3 - HistEqualColor":
    st.subheader('Chater 3 - Colored Histogram Equalization')
    onHistEqualColor(uploaded_img)
elif select =="Chapter 3 - LocalHist":
    st.subheader('Chater 3 - Histogram Localization')
    onLocalHist(uploaded_img)
elif select =="Chapter 3 - HistStat":
    st.subheader('Chater 3 - Histogram Statistics')
    onHistStat(uploaded_img)
elif select =="Chapter 3 - BoxFilter":
    st.subheader('Chater 3 - Box Filter')
    onBoxFilter(uploaded_img)
elif select =="Chapter 3 - GaussFilter":
    st.subheader('Chater 3 - Gaussian Filter')
    onGaussFilter(uploaded_img)
elif select =="Chapter 3 - Threshold":
    st.subheader('Chater 3 - Threshold Filter')
    onThreshold(uploaded_img)
elif select =="Chapter 3 - MedianFilter":
    st.subheader('Chater 3 - Median Filter')
    onMedianFilter(uploaded_img)
elif select =="Chapter 3 - Sharpen":
    st.subheader('Chater 3 - Sharpen')
    onSharpen(uploaded_img)
elif select =="Chapter 3 - Gradient":
    st.subheader('Chater 3 - Gradient')
    onGradient(uploaded_img)
    
elif select =="Chapter 4 - Spectrum":
    st.subheader('Chater 4 - Spectrum')
    onSpectrum(uploaded_img)
elif select =="Chapter 4 - FrequencyFilter":
    st.subheader('Chater 4 - Frequency Filter')
    onFrequencyFilter(uploaded_img)
elif select =="Chapter 4 - DrawNotchRejectFilter":
    st.subheader('Chater 4 - Draw Notch Reject Filter')
    onDrawNotchRejectFilter()
elif select =="Chapter 4 - RemoveMoire":
    st.subheader('Chater 4 - Remove Moire')
    onRemoveMoire(uploaded_img)

elif select =="Chapter 5 - CreateMotionNoise":
    st.subheader('Chater 5 - Create Motion Noise')
    onCreateMotionNoise(uploaded_img)
elif select =="Chapter 5 - DenoiseMotion":
    st.subheader('Chater 5 - Denoise Motion')
    onDenoiseMotion(uploaded_img)
elif select =="Chapter 5 - DenoisestMotion":
    st.subheader('Chater 5 - Denoisest Motion')
    onDenoisestMotion(uploaded_img)

elif select =="Chapter 9 - ConnectedComponent":
    st.subheader('Chater 9 - Connected Component')
    onConnectedComponent(uploaded_img)
elif select =="Chapter 9 - CountRice":
    st.subheader('Chater 9 - Count Rice')
    onCountRice(uploaded_img)
