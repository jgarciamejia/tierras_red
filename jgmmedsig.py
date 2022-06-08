import numpy

def medsig(a):
  median = numpy.median(a)
  sigma = 1.482602218505601*numpy.median(numpy.absolute(a-median))
  return(median, sigma)

def nanmedsig(a):
  median = numpy.nanmedian(a)
  sigma = 1.482602218505601*numpy.nanmedian(numpy.absolute(a-median))
  return(median, sigma)


