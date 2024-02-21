import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Year':2014, 'Kms_Driven':27000, 'Fuel_Type':1, 'Seller_Type':0, 'Transmission':0, 'Owner':0})


print(r.json())