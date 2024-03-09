import streamlit as st
import numpy as np
import cv2 as cv
import joblib

st.subheader('Nhận dạng khuôn mặt')

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
FRAME_WINDOW = st.image([])
cap = cv.VideoCapture(0)

if 'stop' not in st.session_state:
    st.session_state.stop = False
    stop = False

press = st.button('Stop')
if press:
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False

print('Trang thai nhan Stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('/Users/vinguyen/Desktop/Project_CuoiKy_20133113_NguyenThiTuongVi/models/face_recog/stop.jpg')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


svc = joblib.load('/Users/vinguyen/Desktop/Project_CuoiKy_20133113_NguyenThiTuongVi/models/face_recog/svc.pkl')
mydict = ['BanNghia','NhuY', 'PhucLam','ThayDuc',"TuongVi"]

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        '/Users/vinguyen/Desktop/Project_CuoiKy_20133113_NguyenThiTuongVi/models/face_recog/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(
    '/Users/vinguyen/Desktop/Project_CuoiKy_20133113_NguyenThiTuongVi/models/face_recog/face_recognition_sface_2021dec.onnx',"")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        frame = cv.flip(frame, 1) 
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        if faces[1] is not None:
            for idx, face_info in enumerate(faces[1]):
                face_align = recognizer.alignCrop(frame, face_info)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]

                # Điều chỉnh vị trí dọc của văn bản để tránh việc đè lên nhau
                text_position_y = 50 + idx * 30

                cv.putText(frame, result, (1, text_position_y),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()
