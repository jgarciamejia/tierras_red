import glob
import argparse
import numpy as np

from ap_phot import main 

# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
ap.add_argument("-live_plot",required=False,default=True,help="Whether or not to plot the photometry as it is performed.",type=str)
#ap.add_argument("-regress_flux",required=False,default=False,help="Whether or not to perform a regression of relative target flux against ancillary variables (airmass, x/y position, FWHM, etc.).",type=str)
ap.add_argument("-dimness_limit",required=False,default=0.05,help="Minimum flux a reference star can have compared to the target to be considered as a reference star.",type=float)
ap.add_argument("-targ_distance_limit",required=False,default=2000,help="Maximum distance a source can be from the target in pixels to be considered as a reference star.",type=float)
args = ap.parse_args()

targetname = args.target
ffname = args.ffname
live_plot = args.live_plot
dimness_limit = args.dimness_limit
targ_distance_limit = args.targ_distance_limit

path = '/data/tierras/flattened'
folderlist = np.sort(glob.glob(path+"/**/"+targetname))
datelist =  [folderlist[ind].split("/")[4] for ind in range(len(folderlist))]
print ("{} has been observed AND reduced on {} nights. More nights may have been observed but remain unprocessed.".format(targetname,len(datelist)))

failed_dates = [ ] # future improvement: all exceptions are not necessarily photometry fails! Gotta make the loop better:
for i in range(len(datelist)):
	print ("Starting photometric extraction for night {}/{}, date:{}".format(str(i),str(len(datelist)),datelist[i]))
	arg_str = f'-target {targetname} -date {datelist[i]} -ffname {ffname} -live_plot {live_plot} -dimness_limit {dimness_limit} -targ_distance_limit {targ_distance_limit}'.split()
	try:
		main(arg_str)
	except Exception as e:
		print(f"Error in Iteration {i}: {e}")
		failed_dates.append(datelist[i])
		continue

print ("Photometric extraction failed on the following dates: {}".format(failed_dates))
