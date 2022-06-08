import numpy

def medsig(a):
  median = numpy.median(a)
  sigma = 1.482602218505601*numpy.median(numpy.absolute(a-median))
  return(median, sigma)

