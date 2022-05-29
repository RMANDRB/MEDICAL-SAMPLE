import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


Roc = pd.read_excel("C:/Users/PRAGNA/Downloads/Medical.xlsx")
Roc.info()
Roc.head()

Roc.describe()

#### DPP
##Handling duplicates
Roc.duplicated().sum()

##Missing values
Roc.isna().sum()

##Zero variance
Roc.var()
#Drop zero variance column
Roc = Roc.drop("Mode_Of_Transport", axis = 1)
Roc = Roc.drop("Test_Booking_Date", axis = 1)
Roc = Roc.drop("Sample_Collection_Date", axis = 1)
Roc = Roc.drop("Patient_ID", axis = 1)
Roc = Roc.drop("Agent_ID", axis = 1)
Roc.info()

#Label encoding for categorical features
cat_col = ["Patient_Gender", "Test_Name", "Sample", "Way_Of_Storage_Of_Sample", "Cut-off Schedule", "Traffic_Conditions"]

lab = LabelEncoder()
mapping_dict ={}
for col in cat_col:
    Roc[col] = lab.fit_transform(Roc[col])
 
    le_name_mapping = dict(zip(lab.classes_,
                        lab.transform(lab.classes_))) #To find the mapping while encoding
 
    mapping_dict[col]= le_name_mapping
print(mapping_dict)

#Model Building
rf = RandomForestClassifier(n_estimators=5000, n_jobs=3, random_state=42, max_depth = 3)
rf.fit(Roc.iloc[:,:15], Roc.Reached_On_Time)

# saving the model
# importing pickle
import pickle
pickle.dump(rf, open('project.pkl', 'wb'))

# load the model from
project = pickle.load(open('project.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(Roc.iloc[:,:15])
list_value

print(project.predict(list_value))
