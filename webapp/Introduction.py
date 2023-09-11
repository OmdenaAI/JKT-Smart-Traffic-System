import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np


root_dir=os.path.dirname(os.path.abspath(__file__))
homepage_img_dir=os.path.join(root_dir,"images","home_page")
st.title("Project Introduction")
col1,col2,col3=st.columns([1,3,1])
with col2:
    video_path=os.path.join(homepage_img_dir,"video.mp4")
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    
    st.video(video_bytes)
    st.success("Real Time Pothole Detection using trained ssdlite model")

st.write("**Traffic Management refers to the combination of measures that serve to preserve traffic capacity and improve the \
          security, safety and reliability of the overall road transport system. These measures make use of ITS systems,\
          services and projects in day-to-day operations that impact on road network performance. Proper traffic management\
          can ensure that**")
st.markdown("""
            - Traffic flows smoothly and efficiently.
            - Roads are well maintained and safe for all users, including pedestrians and cyclists.
            - Congestion, local pollution and accidents are minimized.
            - Vehicles are within speed limits and heading in correct direction along lane.""")
st.markdown("""**In this project we aim to provide smart solutions to address problems associated with Traffic Management**""")
st.markdown("""
            - Vehicle Category Classification and Detection.
            - Traffic Density Classification.
            - Pothole Object Detection.""")
st.write(" ")
st.subheader("Vehicle Category Classification and Detection")
col1,col2=st.columns(2)
with col1:
    img_path=os.path.join(homepage_img_dir,"Vehicle Categorization.png")
    img=Image.open(img_path)
    img=cv2.resize(np.array(img),(480,480))
    st.image(img)
with col2:
    st.markdown("""
                - Reducing vehicle speed can prevent road crash incidents as an average 1 km increase in vehicle speed can\
                 lead to 3 increase in road accidents.
                - Detecting and Categorizing different types of vehicle in traffic flow can help in determining vehicles \
                 speed using optical flow and enforcing speed limits.
                - Single Shot Detector architecture with mobileNet_v3 backone is utilized for categorizing different Vehicles \
                  type.
                - The model achieved a mean average precision of 0.645 on test dataset @ iou=0.50""")

st.write(" ")
st.subheader("Traffic Density Classification")
col1,col2=st.columns(2)
with col1:
    st.markdown("""
                - The primary goal of traffic management is to make the movement of goods and persons as efficient, orderly,\
                  and safe as possible.
                - Traffic Density Classification will be instrumental in redirecting traffic from major roads during peak \
                  hours and making the overall movement be safer and more efficient.
                - EfficientNet_b0 architecture is utilized for traffic density classification into Empty, Low, \
                  Medium, High, Traffic Jam categories.
                - The model achieved a mean average accuracy of 0.93 on test dataset.""")
with col2:
    img_path=os.path.join(homepage_img_dir,"Traffic Classification.png")
    img=Image.open(img_path)
    img=cv2.resize(np.array(img),(480,480))
    st.image(img)

st.write(" ")
st.subheader("Pothole Object Detection")
col1,col2=st.columns(2)
with col1:
    img_path=os.path.join(homepage_img_dir,"Pothole.png")
    img=Image.open(img_path)
    img=cv2.resize(np.array(img),(360,360))
    st.image(img)
with col2:
    st.markdown("""
                - The safety and efficiency of travel systems depend on how well roads are kept up.
                - Pothole Object Detection will be helpful in early identification of pothole and lowering long-term \
                  repair costs.
                -  It will be critical for ensuring the safety of drivers and the overall efficiency of transportation\
                   infrastructure.
                - Single Shot Detector with MobileNetv3 backbone is utilized for detecting potholes on the roads. 
                - The model achieved a mean average precision of 0.512 at 0.50 intersection over union.""")

st.write(" ")
st.warning("Multiple Default images are provided for each tasks, you can just click detect for seeing model predictions, otherwise provide\
            your files." )