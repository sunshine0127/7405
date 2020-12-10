# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:47:32 2020

@author: Lenovo
"""

#pre-processing steps
raw_data=pd.read_csv('obesity.csv',sep=',');
raw_data=raw_data.drop('Height',axis=1,inplace=False)
raw_data=raw_data.drop('Weight',axis=1,inplace=False)
#change the obesity level to number
raw_data['NObeyesdad'].replace('Insufficient_Weight',1.0,inplace=True)
raw_data['NObeyesdad'].replace('Normal_Weight',2.0,inplace=True)
raw_data['NObeyesdad'].replace('Overweight_Level_I',3.0,inplace=True)
raw_data['NObeyesdad'].replace('Overweight_Level_II',4.0,inplace=True)
raw_data['NObeyesdad'].replace('Obesity_Type_I',5.0,inplace=True)
raw_data['NObeyesdad'].replace('Obesity_Type_II',6.0,inplace=True)
raw_data['NObeyesdad'].replace('Obesity_Type_III',7.0,inplace=True)
#change the gender to number(male=1,famale=0)
raw_data['Gender'].replace('Male',1,inplace=True)
raw_data['Gender'].replace('Female',0,inplace=True)
#change the family history to number(Yes=1,No=0)
raw_data['family_history_with_overweight'].replace('yes',1,inplace=True)
raw_data['family_history_with_overweight'].replace('no',0,inplace=True)
#change the Frequent consumption of high caloric food(Yes=1,No=0)
raw_data['FAVC'].replace('yes',1,inplace=True)
raw_data['FAVC'].replace('no',0,inplace=True)
#change the  Consumption of food between meals to number(No=0,Sometimes=1,Frequently=2,Always=3)
raw_data['CAEC'].replace('no',0.0,inplace=True)
raw_data['CAEC'].replace('Sometimes',1.0,inplace=True)
raw_data['CAEC'].replace('Frequently',2.0,inplace=True)
raw_data['CAEC'].replace('Always',3.0,inplace=True)
#change the Smoke to number
raw_data['SMOKE'].replace('no',0,inplace=True)
raw_data['SMOKE'].replace('yes',1,inplace=True)
#change the  Calories consumption monitoring to number(No=0,Yes=1)
raw_data['SCC'].replace('no',0,inplace=True)
raw_data['SCC'].replace('yes',1,inplace=True)
raw_data['SMOKE'].replace('no',0,inplace=True)
raw_data['SMOKE'].replace('yes',1,inplace=True)
#change the Consumption of alcohol to number(No=0, Sometimes=1, Frequently=2)
raw_data['CALC'].replace('no',0,inplace=True)
raw_data['CALC'].replace('Sometimes',1.0,inplace=True)
raw_data['CALC'].replace('Frequently',2.0,inplace=True)
raw_data['CALC'].replace('Always',3.0,inplace=True)
#change the Transportation used to number()
raw_data['MTRANS'].replace('Automobile',0.0,inplace=True)
raw_data['MTRANS'].replace('Motorbike',1.0,inplace=True)
raw_data['MTRANS'].replace('Bike',2.0,inplace=True)
raw_data['MTRANS'].replace('Public_Transportation',3.0,inplace=True)
raw_data['MTRANS'].replace('Walking',4.0,inplace=True)

#Now we divide the data into training data and testing data
response=raw_data['NObeyesdad'];
raw_data=raw_data.drop('NObeyesdad',axis=1,inplace=False);
raw_data=raw_data.drop('SMOKE',axis=1,inplace=False);

training_data=raw_data[0:int(0.8*len(raw_data))];
testing_data=raw_data[int(0.8*len(raw_data)):len(raw_data)];
training_response=response[0:int(0.8*len(raw_data))];
testing_response=response[int(0.8*len(raw_data)):len(raw_data)];

#normalize the data 
mms = MinMaxScaler()
training_data = mms.fit_transform(training_data)
testing_data = mms.transform(testing_data)
stdsc = StandardScaler()
training_data = stdsc.fit_transform(training_data)
testing_data = stdsc.transform(testing_data)  

#doing PCA
pca = PCA(n_components=11);
pca.fit(training_data);       
sum(pca.explained_variance_ratio_)
train_white=pca.transform(training_data);   
test_white=pca.transform(testing_data)
