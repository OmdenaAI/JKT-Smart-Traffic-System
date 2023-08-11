import cv2
from roboflow import Roboflow
# Load the video
video_path = 'path'

class PotholeDetector:
    def __init__(self, roboflow_api_key, model_id, conf_threshold=0.5, width=1280, height=720):
        self.rf = Roboflow(api_key=roboflow_api_key)
        self.project = self.rf.workspace().project(model_id)
        self.model = self.project.version(1).model

        self.cap = cv2.VideoCapture()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.conf_threshold = conf_threshold

    def detect_potholes(self, video_source=0):
        self.cap.open(video_source)

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    break

                result = self.model.predict(frame, confidence=self.conf_threshold).json()

                for prediction in result['predictions']:
                    x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
                    conf = prediction['confidence']
                    cls = prediction['class']

                    if cls == 'pothole' and conf > self.conf_threshold:
                        cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 0, 255), 3)
                        cv2.putText(frame, f'{cls} {conf:.2f}', (int(x), int(y) - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                cv2.imshow('Pothole Detection', frame)

                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    break

        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    roboflow_api_key = "12j3lmSftmJ7Ga3dRc2E"
    model_id = "pothole-voxrl"
    pothole_detector = PotholeDetector(roboflow_api_key, model_id, conf_threshold=0.5, width=1280, height=720)

    try:

        pothole_detector.detect_potholes(video_source=video_path)
    except KeyboardInterrupt:
        print("")


if __name__ == "__main__":
    main()
