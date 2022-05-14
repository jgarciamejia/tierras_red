import numpy as np
# Medsig function
def medsig(a):
  median = np.median(a)
  sigma = 1.482602218505601*np.median(np.absolute(a-median))
  return(median, sigma)
