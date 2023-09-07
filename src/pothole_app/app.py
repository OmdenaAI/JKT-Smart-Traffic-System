import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import torch
import os
import torchvision
import torchvision.transforms.functional as tf
from utils import show_bbox, preprocess_bbox,get_model


st.title("Real Time Pothole Detection")
st.markdown("""**Single Shot Detector with vgg backbone is utilized for detecting potholes on the roads. The model \
            achieved a mean average precision of 0.536 at 0.50 intersection over union**""")
st.write("Default images and video are provided, you can just click detect for seeing detection, otherwise provide\
          your files." )

format=st.sidebar.radio("Image/Video",["Image","Video"],index=0)
model=get_model()
model.eval()

def detection_img(img,conf_threshold,iou_threshold):
    img=tf.to_tensor(img)
    predict={}
    with torch.no_grad():
        predict=model([img])
        predict=preprocess_bbox(predict[0],conf_threshold,iou_threshold)
        img=show_bbox(img,predict)
        img=np.clip(img,0,1)
        return img

if format=="Image":
    file=st.sidebar.file_uploader("Input Image File",type = ['jpg','png','jpeg'])
    button=st.button("Detect")
    conf_threshold=float(st.sidebar.slider("Confidence Threshold", min_value=0.0,max_value=1.0,value=0.2,step=0.02))
    iou_threshold=float(st.sidebar.slider("IOU Threshold", min_value=0.0,max_value=1.0,value=0.7,step=0.02))
    col1,col2=st.columns(2)
    if file is not None:
        title="Uploaded Image"
        img=Image.open(file)
        img=np.array(img)
        img=cv2.resize(img,(480,480))
        detect_img=detection_img(img,conf_threshold,iou_threshold)
    else:
        title="Default Image"
        dirname=os.path.dirname(os.path.abspath(__file__))
        img_dir=os.path.join(dirname,"images")
        idx=np.random.choice(range(4),1)[0]
        default_img_path=os.path.join(img_dir,f"{idx}.png")
        img=Image.open(default_img_path)
        img=np.array(img)
        img=cv2.resize(img,(480,480))
    with col1:
        st.write(title)
        st.image(img)
    if button:
        with col2:
            st.write("Pothole Detection")
            detect_img=detection_img(img,conf_threshold,iou_threshold)
            st.image(detect_img)
elif format=="Video":
    video_data=st.sidebar.file_uploader("Input Video File",type = ['mp4','mpeg'])
    button=st.button("Detect")
    conf_threshold=st.sidebar.slider("Confidence Threshold", min_value=0.0,max_value=1.0,value=0.2,step=0.05)
    iou_threshold=st.sidebar.slider("IOU Threshold", min_value=0.0,max_value=1.0,value=0.7,step=0.05)
    if video_data is not None:
        temp_file_1=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
        temp_file_1.write(video_data.getbuffer())
        video_file=temp_file_1.name
    else:
        dirname=os.path.dirname(os.path.abspath(__file__))
        video_dir=os.path.join(dirname,"videos")
        default_video_path=os.path.join(video_dir,"pothole_Trim.mp4")
        video_file=default_video_path
    with st.empty():
            if button:
                cap=cv2.VideoCapture(video_file)
                if (cap.isOpened() == False):
                    print('Error while trying to read video. Please check path again')
                while(cap.isOpened()):
                    ret,frame=cap.read()
                    if ret==True:
                        with torch.no_grad():
                            frame=cv2.resize(frame,(480,480))
                            detect_img=detection_img(frame,conf_threshold,iou_threshold)
                            st.image(detect_img)
                    else:
                        break
