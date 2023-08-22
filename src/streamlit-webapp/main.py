import streamlit as st
from TrafficPages.traffic_monitor import TrafficMonitorPage
from TrafficPages.traffic_forecast import TrafficForecastPage
from roboflow import Roboflow
from ultralytics import YOLO

# set page header and initialize page state
st.set_page_config('Traffic Management System')
if 'page' not in st.session_state:
    st.session_state['page'] = 0

def update_page_idx(idx):
    st.session_state['page'] = idx

# create sidebar 
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Traffic Management System")
st.sidebar.image("assets/images/logo.jpg", width=200, use_column_width=True)

monitor_button = st.sidebar.button(" 🚧 Traffic Monitor using CV",use_container_width=True,on_click=lambda: update_page_idx(0))
forecast_button = st.sidebar.button(" 📈 Traffic Flow Forecasting",use_container_width=True,on_click=lambda: update_page_idx(1))

if st.session_state['page'] == 0:
    # Set up Roboflow Pothole Model
    roboflow_api_key = st.secrets['roboflow_api_key']
    model_id = "pothole-voxrl"
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace().project(model_id)
    pothole_model = project.version(1).model

    # Set up Ultralytics Traffic Density Classifier
    traffic_classifier = YOLO('assets/models/traffic_classifier.pt')
    monitor_page = TrafficMonitorPage(pothole_model, traffic_classifier)
    monitor_page.buildUI()

else:
    forecast_page = TrafficForecastPage()
    forecast_page.buildUI()
