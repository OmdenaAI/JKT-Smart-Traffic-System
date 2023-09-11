import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssd import SSDHead,det_utils
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import efficientnet_b0
import torchvision.transforms.functional as tf
import streamlit as st
from urllib.request import urlopen
import tempfile

root_dir=os.path.dirname(os.path.abspath(__file__))
weights_dir=os.path.join(root_dir,"weights")

def classify_img(model,img):
    img=tf.to_tensor(img)
    img=img.unsqueeze(0)
    with torch.no_grad():
        predict=model(img)
        predict=nn.functional.softmax(predict,1)
        label=torch.argmax(predict)
        probability=torch.max(predict)
        return label,probability

def detection_img(model,img,classes,conf_threshold,iou_threshold):
    img=tf.to_tensor(img)
    predict={}
    with torch.no_grad():
        predict=model([img])
        predict=preprocess_bbox(predict[0],conf_threshold,iou_threshold)
        img=show_bbox(img,predict,classes)
        img=np.clip(img,0,1)
        return img

def show_bbox(img,target,classes,color=(0,255,0)):
    img=np.transpose(img.numpy(),(1,2,0))
    boxes=target["boxes"].numpy().astype("int")
    labels=target["labels"].numpy()
    scores=target["scores"].numpy()
    img=img.copy()
    for i,box in enumerate(boxes):
        text=f"{classes[labels[i]]} {scores[i]:.2f}"
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color,2)
        y=box[1]-10 if box[1]-10>10 else box[1]+10
        cv2.putText(img,text,(box[0],y),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    return img

def preprocess_bbox(prediction,conf_threshold,iou_threshold):
    index=[]
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
def get_density_model():
    model=efficientnet_b0(weights=None)
    in_features=model.classifier[1].in_features
    model.classifier[1]=nn.Linear(in_features=in_features,out_features=5)
    weights_path=os.path.join(weights_dir,"traffic_density.pth")
    weights=torch.load(weights_path,map_location="cpu")
    model.load_state_dict(weights)
    return model

@st.cache_resource
def get_pothole_model():
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None)
    in_channels=det_utils.retrieve_out_channels(model.backbone,(480,480))
    num_anchors=model.anchor_generator.num_anchors_per_location()
    model.head=SSDHead(in_channels=in_channels,num_anchors=num_anchors,
                       num_classes=2)
    weights_path=os.path.join(weights_dir,"pothole_model.pth")
    weights=torch.load(weights_path,map_location="cpu")
    model.load_state_dict(weights)
    return model

@st.cache_resource
def get_category_model():
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None)
    in_channels=det_utils.retrieve_out_channels(model.backbone,(320,320))
    num_anchors=model.anchor_generator.num_anchors_per_location()
    model.head=SSDHead(in_channels=in_channels,num_anchors=num_anchors,
                       num_classes=9)
    weights_path=os.path.join(weights_dir,"vehicle_categorization.pth")
    weights=torch.load(weights_path,map_location="cpu")
    model.load_state_dict(weights)
    return model