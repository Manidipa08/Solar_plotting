# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:43:56 2022

@author: MANIDIPA BANERJEE
"""

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
#from astropy import modeling
import pandas as pd
from scipy.stats import skew
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


#a= np.loadtxt('C:/Users/MANIDIPA BANERJEE/Downloads/SN_d_tot_V2.0(1).txt').T
d = pd.read_csv("C:/Users/MANIDIPA BANERJEE/Downloads/sunspot-area.CSV",header=None)
final_df = d.sort_values(by=[4], ascending=True)




#folder_path = 'Data.txt'
#df_data = pd.read_csv('C:/Users/MANIDIPA BANERJEE/Downloads/SN_d_tot_V2.0(1).txt', sep="\t", header=None)
d_numpy=d.to_numpy(dtype ='float32')


# Generating the data
x_data=d_numpy[:,4]
y_data=d_numpy[:,2]

# Savitzky-Golay filter
y_filtered = savgol_filter(y_data, 100, 5)

sigma = 5
y = gaussian_filter1d(y_filtered, sigma)

# Plotting
fig = plt.figure()

ax = fig.subplots()
p = ax.plot(x_data, y_data, '-*',label='Raw data')
p, = ax.plot(x_data, y_filtered, '*y',label='Filtered data',linewidth = 5)
p_ = ax.plot(x_data, y, '-k', label='Gaussian fit')
plt.subplots_adjust(bottom=0.25)

plt.title("Time vs Sunspot Area",size='18')
plt.xlabel("Time(Year)",size='12')
plt.ylabel("Sunspot Area",size='12')
plt.legend()
plt.show()

print('\nSkewness for raw data : ', skew(y_data))
print('\nSkewness for data (savgol-filter) : ', skew(y_filtered))
print('\nSkewness for data (Gaussian-filter) : ', skew(y))

#For Raw Data
peaks = find_peaks(y_data[0], height = 1, threshold = None)  
height = peaks[1]['peak_heights']
peak_pos = x_data[peaks[0]]


#For Savgol Data
peaks1 = find_peaks(y_filtered[0], height = 1, threshold = 1)  
height1 = peaks1[1]['peak_heights']
peak_pos1 = x_data[peaks1[0]]


#For Gaussian Data
peaks2 = find_peaks(y[0], height = 1, threshold = 1)  
height2 = peaks2[1]['peak_heights']
peak_pos2 = x_data[peaks2[0]]

