import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.ssd import SSDHead,det_utils
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import streamlit as st
from urllib.request import urlopen
import tempfile
import os

def show_bbox(img,target,color=(0,255,0)):
    img=np.transpose(img.numpy(),(1,2,0))
    boxes=target["boxes"].numpy().astype("int")
    scores=target["scores"].numpy()
    img=img.copy()
    for i,box in enumerate(boxes):
        text=f"pothole {scores[i]:.2f}"
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color,2)
        y=box[1]-10 if box[1]-10>10 else box[1]+10
        cv2.putText(img,text,(box[0],y),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    return img

def preprocess_bbox(prediction,conf_threshold,iou_threshold):
    processed_bbox={}
    boxes=prediction["boxes"][prediction["scores"]>=conf_threshold]
    scores=prediction["scores"][prediction["scores"]>=conf_threshold]
    labels=prediction["labels"][prediction["scores"]>=conf_threshold]
    nms=torchvision.ops.nms(boxes,scores,iou_threshold=iou_threshold)
    processed_bbox["boxes"]=boxes[nms]
    processed_bbox["scores"]=scores[nms]
    processed_bbox["labels"]=labels[nms]
    return processed_bbox

@st.cache_resource
def get_model():
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None)
    in_channels=det_utils.retrieve_out_channels(model.backbone,(480,480))
    num_anchors=model.anchor_generator.num_anchors_per_location()
    model.head=SSDHead(in_channels=in_channels,num_anchors=num_anchors,
                       num_classes=2)
    
    st.write(os.path.dirname(os.path.abspath(__file__)))
    weights=torch.load("pothole_model_lite.pth",map_location="cpu")
    model.load_state_dict(weights)
    return model
