import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile


class TrafficMonitorPage:
    def __init__(self, pothole_model, traffic_classifier):
        self.pothole_model = pothole_model
        self.traffic_classifier = traffic_classifier

    def buildUI(self):
        st.title("🚧 Traffic Monitoring Using Computer Vision")
        st.markdown('''
                - Detect road potholes
                - Monitor congestion levels
                ''')

        media = st.radio('Choose type of media to upload: ', ['image', 'video'])
        model = st.selectbox('Choose AI model: ', ['Pothole Detection', 'Traffic Density Classifier'])

        if media == 'image':
            image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if image_file:
                image = Image.open(image_file)
                if model == 'Pothole Detection':
                    image = self.detect_potholes(np.array(image))
                else:
                    level = self.classify_traffic(image)
                    st.write(level)
                
                st.image(image)
        
        elif media == 'video':
            video_file = st.file_uploader("Upload a video", type=["mp4"])
            if video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(video_file.read())
                vid_cap = cv2.VideoCapture(tfile.name)
                
                label = st.empty()
                video = st.empty()
                
                while vid_cap.isOpened():
                    ret, frame = vid_cap.read()
                    if not ret:
                        break

                    if model == 'Pothole Detection':
                        frame = self.detect_potholes(frame)

                    else:
                        level = self.classify_traffic(frame)
                        label.write(level)
                    
                    video.image(frame, channels='BGR')
                
                vid_cap.release()

    def detect_potholes(self, frame):
        result = self.pothole_model.predict(frame, confidence=0.5).json()

        for prediction in result['predictions']:
            x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            conf = prediction['confidence']
            cls = prediction['class']

            if cls == 'pothole' and conf > 0.5:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + width), 
                                    int(y + height)), (0, 255, 0), 3)

        return frame
    
    def classify_traffic(self, frame):
        results = self.traffic_classifier(frame)
        idx = results[0].probs.top1
        return 'Congestion Level: ' + self.traffic_classifier.names[idx]
