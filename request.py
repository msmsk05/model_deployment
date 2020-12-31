import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={"Elevation":2434, "Vertical_Distance_To_Hydrology":80, "Horizontal_Distance_To_Roadways":234, "Hillshade_Noon":221, "Horizontal_Distance_To_Fire_Points":469})

print(r.json())