from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
import pandas as pd
import numpy as np


df_=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df_["Outcome"]
df=df_.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.25,random_state=200)

cart=DecisionTreeClassifier()
#karar ağaçlarında ve random forest ta max depth kullan
#karar ağaçlarında ve random forest ta min_sample_split kullan
cart_params={
    "max_depth":[1,3,5,8,10],
    "min_samples_split":[1,3,5,8,10,20,30]
}
cart_cv=GridSearchCV(cart,cart_params,cv=5,n_jobs=-1,verbose=2)
cart_cv.fit(x_train,y_train)
max_depth=cart_cv.best_params_["max_depth"]
min_sample_split=cart_cv.best_params_["min_samples_split"]

cart_tuned=DecisionTreeClassifier(min_samples_split=min_sample_split,max_depth=max_depth)
cart_tuned.fit(x_train,y_train)

predict=cart_tuned.predict(x_test)
acscore=accuracy_score(y_test,predict)
print(acscore)












