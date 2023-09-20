# Omdena Jakarta - Smart Traffic System

## Project Links: 
1. [Streamlit Webapp](https://omdena-jakarta-traffic-system.streamlit.app/)
2. [Pitch Deck Video](https://clipchamp.com/watch/R69z4mhQp3o)
3. [Omdena Chapter Link](https://omdena.com/chapter-challenges/predicting-road-defects-and-optimizing-traffic-light-countdown-to-reduce-congestion-in-indonesia/)

## Background
**Traffic Management refers to the combination of measures that serve to preserve traffic capacity and improve the security, safety and reliability of the overall road transport system. These measures make use of ITS systems, services and projects in day-to-day operations that impact on road network performance. Proper traffic management can ensure that**
- Traffic flows smoothly and efficiently.
- Roads are well maintained and safe for all users, including pedestrians and cyclists.
- Congestion, local pollution and accidents are minimized.
- Vehicles are within speed limits and heading in correct direction along lane.

## Solution
**In this project we aim to provide smart solutions to address problems associated with Traffic Management**

- Vehicle Category Classification and Detection.
- Traffic Density Classification.
- Pothole Object Detection.

| No | Description            | Dataset | Kaggle   | Weights|
|:---| :--------------------- | :-----:  | :-----:  | :-----:|
|1| Vehicle Category Classification and Detection | [Link](https://www.kaggle.com/datasets/sakshamjn/vehicle-detection-8-classes-object-detection)|  [Link](https://www.kaggle.com/code/sudhanshu2198/vehicle-category-object-detection-pytorch)        |   [Link](https://www.kaggle.com/datasets/sudhanshu2198/vehicle-categorization-detection-earned-weights)     |
|2| Traffic Density Classification | [Link](https://www.kaggle.com/datasets/rahat52/traffic-density-singapore)|  [Link](https://www.kaggle.com/code/sudhanshu2198/traffic-density-classification-using-efficientnet)        |   [Link](https://www.kaggle.com/datasets/sudhanshu2198/traffic-density-classification-learned-weights)     |
|3| Pothole Object Detection | [Link](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)|  [Link](https://www.kaggle.com/code/sudhanshu2198/real-time-pothole-detection-using-ssd)        |   [Link](https://www.kaggle.com/datasets/sudhanshu2198/pothole-detection-learned-weights)     |

## 🛠 Skills
Pytorch, Torchvision, Ultralytics, OpenCV, Numpy, Streamlit, Git

## Directory Tree
```bash

├── notebooks
│   │── real-time-pothole-detection-using-ssd.ipynb
|   |── traffic-density-classification-using-efficientnet.ipynb
│   └── vehicle-category-object-detection-pytorch.ipynb
├── webapp
│   ├── images
│   │   ├── category_img
│   │   ├── pothole_img
│   │   ├── density_img
│   │   └── home_page
│   ├── pages
│   │   ├── Pothole_Detection.py
│   │   ├── Traffic_Density_Classification.py
│   │   └── Vehicle_Category_Detection.py
│   ├── videos
│   │   ├── raw_clip.mp4
│   │   ├── annotated_clip.mp4
│   ├── weights
│   │   ├── pothole_model.pth
│   │   ├── traffic_density.pth
│   │   └── vehicle_categorization.pth
│   │── Introduction.py
│   │── utils.py
│   │── requirements.txt
│   │── packages.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Vehicle Category Classification and Detection
![](https://github.com/OmdenaAI/jakarta-indonesia-predicting-road-defects/blob/main/webapp/images/home_page/Vehicle%20Categorization.png)
- Reducing vehicle speed can prevent road crash incidents as an average 1 km increase in vehicle speed can lead to 3 percent increase in road accidents.
- Detecting and Categorizing different types of vehicle in traffic flow can help in determining vehicles speed using optical flow and enforcing speed limits.
- Single Shot Detector architecture with mobileNet_v3 backone is utilized for categorizing different Vehicles type.
- The model achieved a mean average precision of 0.645 on test dataset @ iou=0.50.

## Traffic Density Classification
![](https://github.com/OmdenaAI/jakarta-indonesia-predicting-road-defects/blob/main/webapp/images/home_page/Traffic%20Classification.png)
- The primary goal of traffic management is to make the movement of goods and persons as efficient, orderly, and safe as possible.
- Traffic Density Classification will be instrumental in redirecting traffic from major roads during peak hours and making the overall movement be safer and more efficient.
- EfficientNet_b0 architecture is utilized for traffic density classification into Empty, Low, Medium, High, Traffic Jam categories.
- The model achieved a mean average accuracy of 0.93 on test dataset.

## Pothole Object Detection
![](https://github.com/OmdenaAI/jakarta-indonesia-predicting-road-defects/blob/main/webapp/images/home_page/Pothole.png)
- The safety and efficiency of travel systems depend on how well roads are kept up.
- Pothole Object Detection will be helpful in early identification of pothole and lowering long-term repair costs.
- It will be critical for ensuring the safety of drivers and the overall efficiency of transportation infrastructure.
- Single Shot Detector with MobileNetv3 backbone is utilized for detecting potholes on the roads.
- The model achieved a mean average precision of 0.512 at 0.50 intersection over union.

## Future Development
- Road Lane Instance Segmentation
- Plate number recognition
- Vehicle tracking for speed determination

## Run Webapp Locally

Clone the project

```bash
  git clone https://github.com/OmdenaAI/jakarta-indonesia-predicting-road-defects
```

Change to project directory

```bash
  cd webapp
```
Create Virtaul Environment and install dependencies

```bash
  py -m venv venv
  venv/Scripts/activate
  pip install -r requirements.txt
```

Run Locally
```bash
  streamlit run Introduction.py
```

## Contributions
Chapter Lead: Louis Jefferson Zhang

Technical Lead: Sudhanshu Rastogi

Collaborators:
- Mohamed Chahed
- Mansoor Baig
- Nishtha Bhattacharjee

## Project Archives
Navigate to [archive branch](https://github.com/OmdenaAI/smart-traffic-system-JKT/tree/archive) to look into some project development notebooks.
