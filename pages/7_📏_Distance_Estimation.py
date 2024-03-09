import streamlit as st
import cv2
import os

# Constants
KNOWN_DISTANCE = 76.2  # cm
KNOWN_WIDTH = 14.3  # cm
FACE_CASCADE_PATH = "/Users/vinguyen/Desktop/Project_CuoiKy_20133113_NguyenThiTuongVi/models/distance_est/haarcascade_frontalface_default.xml"
REF_IMAGE_PATH = "/Users/vinguyen/Desktop/Project_CuoiKy_20133113_NguyenThiTuongVi/models/distance_est/Ref_image.png"

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

#fonts
fonts = cv2.FONT_HERSHEY_SIMPLEX
# Focal length finder function
def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

def face_data(image, CallOut, Distance_level):

    face_width = 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        line_thickness = 2
        # print(len(faces))
        LLV = int(h * 0.12)
        # print(LLV)

        # cv2.rectangle(image, (x, y), (x+w, y+h), BLACK, 1)
        cv2.line(image, (x, y + LLV), (x + w, y + LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y + h), (x + w, y + h), (GREEN), line_thickness)
        cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), (GREEN), line_thickness)
        cv2.line(
            image, (x + w, y + LLV), (x + w, y + LLV + LLV), (GREEN), line_thickness
        )
        cv2.line(image, (x, y + h), (x, y + h - LLV), (GREEN), line_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), (GREEN), line_thickness)

        face_width = w
        face_center = []
        # Drwaing circle at the center of the face
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y
        if Distance_level < 10:
            Distance_level = 10

        if CallOut == True:
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (ORANGE), 28)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (YELLOW), 20)
            cv2.line(image, (x, y - 11), (x + Distance_level, y - 11), (GREEN), 18)

    return face_width, faces, face_center_x, face_center_y


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

st.subheader('Tính khoảng cách đến camera')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
Distance_level = 0

face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Read the reference image for distance estimation
ref_image = cv2.imread(REF_IMAGE_PATH)
ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)

# Stop button
stop_button = st.button('Stop')
save_button = st.button('Save Image')

while cap.isOpened() and not stop_button:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # calling face_data function
    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, Distance_level)
    # finding the distance by calling function Distance finder
    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:
            Distance = Distance_finder(
                Focal_length_found, KNOWN_WIDTH, face_width_in_frame
            )
            Distance = round(Distance, 2)
            Distance_level = int(Distance)
            cv2.putText(
                frame,
                f'Estimated Distance: {Distance_level} cm',
                (face_x - 30, face_y - 30),
                fonts,
                0.8,
                (RED),
                2,
                cv2.LINE_AA
            )

    FRAME_WINDOW.image(frame, channels='BGR')

    if save_button:
        saved_image_path = '/Users/vinguyen/Desktop/Project_CuoiKy_20133113_NguyenThiTuongVi/image/distance_est/saved_image_%04d.jpg' % Distance_level
        try:
            cv2.imwrite(saved_image_path, frame)
            st.success(f'Image saved: {saved_image_path}')
            
            # Display the saved image
            saved_image = cv2.imread(saved_image_path)
            cap.release()

        except Exception as e:
            st.error(f'Error saving image: {e}')

if stop_button:
    cap.release()
