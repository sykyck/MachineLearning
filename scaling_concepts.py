from sklearn.preprocessing import StandardScaler

import pandas as pd

data = pd.DataFrame({
    "Age": [20,30,40,50,60]
})

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print(scaled_data)