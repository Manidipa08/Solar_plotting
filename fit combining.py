# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:15:25 2022

@author: MANIDIPA BANERJEE
"""

from astropy.io import fits
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib.image as img
import pylab as mplot
#from scipy.stats.distributions import  t
 
myfile1 = 'C:/Users/MANIDIPA BANERJEE/Desktop/Codes+tutorials for solar/Lecture_2/axis/BIR_20120704_164500_02.fit.gz'

hdu    = fits.open(myfile1)
data1  = hdu[0].data.astype(np.float32)
freqs1 = hdu[1].data['Frequency'][0] # extract frequency axis
time1  = hdu[1].data['Time'][0] # extract time axis
hdu.close()

myfile2 = 'C:/Users/MANIDIPA BANERJEE/Desktop/Codes+tutorials for solar/Lecture_2/axis/TRIEST_20120704_164501_59.fit.gz'

hdu    = fits.open(myfile2)
data2  = hdu[0].data.astype(np.float32)
freqs2 = hdu[1].data['Frequency'][0] # extract frequency axis
time2  = hdu[1].data['Time'][0] # extract time axis
hdu.close()

data = np.concatenate((data2[17:182,:], data1[10:178,:]))
#start_time = float(fits.open(myfile1[0])[0].header['Time'].split(":")[0])*60*60 + float(fits.open(myfile1[0])[0].header['Time'].split(":")[0])*60 + float(fits.open(myfile1[0])[0].header['Time'].split(":")[0])  
time = time1
freqs=np.hstack((freqs2[17:182],freqs1[10:178]))

#------------------------------------------------------------------------------
fig=plt.figure(figsize=(12,8))
extent = (time[1000], time[-1], freqs[-1], freqs[0])
bgs = data -  data.mean(axis=1, keepdims=True)  # subtract average
bgs = bgs[:,1000:-1]
plt.imshow(bgs, aspect = 'auto', extent = extent, cmap=cm.plasma, vmin=-1,vmax=20) 
plt.ylim(110,250)
yticks = plt.gca().get_yticks().tolist() # get list of ticks


plt.tick_params(labelsize=14)
plt.xlabel('Time [s] ',fontsize=15)
plt.ylabel('Plasma frequency [MHz]',fontsize=15)
plt.title('2012-07-04',fontsize=15)
#plt.savefig(myfile1 +'_'+myfile2 + ".png")
plt.show()

