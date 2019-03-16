'''
This script is used to preprocess the data
The features that we would like to extract are 
1. max, min
2. Interquartile range
3. Mean and Median
4. SD
5. Root Mean Square
6. Skewness along the x, y, z axes
'''
import numpy as np 
# import pandas

# suppose there are a sequence of data fetched from sensor
input =  [[0, 0, 0, 0, 0, 0, 0, 0, 0,], 
          [0, 0, 0, 0, 0, 0, 0, 0, 0,], 
          [0, 0, 0, 0, 0, 0, 0, 0, 0,], 
          [0, 0, 0, 0, 0, 0, 0, 0, 0,], 
          [0, 0, 0, 0, 0, 0, 0, 0, 0,], 
          [0, 0, 0, 0, 0, 0, 0, 0, 0,], 
          [0, 0, 0, 0, 0, 0, 0, 0, 0,]]

# It will form a 1D array with all the features for all 9 accelerations 
res = np.max(input[:, 0])
print(res)






