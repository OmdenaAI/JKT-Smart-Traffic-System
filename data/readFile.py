import pandas as pd
import json

with open("jobs_3396405_results_test.json", 'r') as file:
    data = json.load(file)

df_segment_results = pd.json_normalize(data['network']['segmentResults'],
                                        record_path='segmentTimeResults',
                                        meta=['segmentId', 'newSegmentId', 'speedLimit', 'frc', 'streetName', 'distance'],errors='ignore')

print(df_segment_results)