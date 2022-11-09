import pickle

import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd


targets = np.array(["red", "green", "blue", "yellow", "pink",
                    "white"])
labelEnc = preprocessing.LabelEncoder()
new_target = labelEnc.fit_transform(targets)
onehotEnc = preprocessing.LabelEncoder()
xxx = new_target.reshape(-1, 1)
onehotEnc.fit(new_target.reshape(-1, 1))
targets_trans = onehotEnc.transform(new_target.reshape(-1, 1))
pickle.dump(labelEnc, open('testpkle.pkl', 'wb'))
print("The original data")
print(targets)
print("The transform data using OneHotEncoder")
print(targets_trans)
hh = pickle.load(open('testpkle.pkl', 'rb'))
targets1 = np.array(["fdfd", "fddfd", "bluexcx", "yellouyuyw", "picxcxnk",
                    "whidsste"])
new_target = hh.fit_transform(targets1)
qq = hh.inverse_transform(new_target)
print(qq)