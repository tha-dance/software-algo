import pandas as pd 
import json
import os
from feature_extraction import extract

window_size = 250
step_size = 10

dir = os.listdir('Datasets_new/')
result = []
print(dir)
for csv_file in dir:
    dataframe = pd.read_csv(os.path.join('Datasets_new', csv_file))
    dataset = dataframe.values
    print(dataset)
    for row in range(int((len(dataset) - window_size) / step_size)):
        processed = extract(dataset[row*step_size:row*step_size+window_size])
        processed.append(dataset[row*step_size][-1])
        # print(processed)
        result.append(processed)

df = pd.DataFrame(result)
df.to_csv('processed.csv', header=0)