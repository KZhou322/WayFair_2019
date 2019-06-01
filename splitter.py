import csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

trainFileFeature = pd.read_csv('revenue.csv', usecols = [2,3,4,5,6,7,8,9,10,11])
onehot = OneHotEncoder(handle_unknown="ignore")



# with open("onehoted.csv", mode="w") as out:
#     writer = csv.writer(out, delimiter = ",")
# for x in range(2,12):
#     asdf=pd.read_csv("revenue.csv", usecols = [x])
#     print(asdf.shape)
#     onehot.fit(asdf)
#     print(onehot.transform(asdf).shape)
a=[]

for x in range (1,11):
    asdf=pd.read_csv("df_holdout_scholarjet.csv", usecols = [x])
    onehot.fit(asdf)
    a.append(onehot.transform(asdf))



pd.DataFrame(hstack(a).toarray()).to_csv("holdout.csv", header=None, index = None)

