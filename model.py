import warnings
import pickle
warnings.filterwarnings('ignore')
import pandas as pd
data=pd.read_csv('car data (1).csv')
data.head(5)
data.tail()
data.shape
print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])
data.info()
data.isnull().sum()
data.describe()
data.head(1)
import datetime
date_time = datetime.datetime.now()
data['Age']=date_time.year - data['Year']
data.head()
data.drop('Year',axis=1,inplace=True)
data.head()
sorted(data['Selling_Price'],reverse=True)
data = data[~(data['Selling_Price']>=33.0) & (data['Selling_Price']<=35.0)]
data.shape
data.head(1)
data['Fuel_Type'].unique()
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
data['Fuel_Type'].unique()
data['Seller_Type'].unique()
data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0,'Individual':1})
data['Seller_Type'].unique()
data['Transmission'].unique()
data['Transmission'] =data['Transmission'].map({'Manual':0,'Automatic':1})
data['Transmission'].unique()
data.head()
data_new = pd.DataFrame({'Present_Price':5.59,'Kms_Driven':27000,'Fuel_Type':0,'Seller_Type':0,'Transmission':0, 'Owner':0,'Age':8},index=[0])
import joblib
model = joblib.load('car_price_predictor')
model.predict(data_new)
X = data.drop(['Car_Name','Selling_Price'],axis=1)
y = data['Selling_Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

xg = XGBRegressor()
xg.fit(X,y)
pickle.dump(xg, open('modelbis.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modelbis.pkl','rb'))