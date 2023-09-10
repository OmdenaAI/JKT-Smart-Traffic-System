import cv2
from roboflow import Roboflow

class PotholeDetector:
    def __init__(self, roboflow_api_key, model_id, conf_threshold=0.5):
        self.rf = Roboflow(api_key=roboflow_api_key)
        self.project = self.rf.workspace().project(model_id)
        self.model = self.project.version(1).model
        self.conf_threshold = conf_threshold

    def detect_potholes(self, frame):
        result = self.model.predict(frame, confidence=self.conf_threshold).json()

        for prediction in result['predictions']:
            x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            conf = prediction['confidence']
            cls = prediction['class']

            if cls == 'pothole' and conf > self.conf_threshold:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), 3)
                cv2.putText(frame, f'{cls} {conf:.2f}', (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        return frame
