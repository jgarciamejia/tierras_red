import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt 

weeks = ['20220405','20220412']#,'20220419','20220503','20220510','20220517','20220524']

Mflats = []
for date in weeks:
	hdu = fits.open('/data/tierras/incoming/{}/MASTERFLAT_mednorm.fit'.format(date))
	Mflats.append(hdu)

#rows = np.linspace(100,900,9)

row = 100

for nth_ratio in np.arange(len(weeks)):

	f,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))
	flat_num = Mflats[nth_ratio+1]
	flat_denom = Mflats[nth_ratio]

	flat_num1, flat_num2 = flat_num[1].data,flat_num[2].data
	flat_denom1, flat_denom2 = flat_denom[1].data,flat_denom[2].data

	ax1.plot(flat_num1/flat_denom1)
	ax2.plot(flat_num2/flat_denom2)

	ax1.set_title('CCD Chip: Half 1')
	ax2.set_title('CCD Chip: Half 2')

	ax1.set_xlabel('X pixel Coord')
	ax2.set_xlabel('X pixel Coord')

	ax1.set_ylabel('Ratio {}:{}, Y (row) = {}'.format(weeks[nth_ratio+1],weeks[nth_ratio],str(row)))
	ax2.set_ylabel('Ratio {}:{}, Y (row) = {}'.format(weeks[nth_ratio+1],weeks[nth_ratio],str(row)))

	plt.show()
        plt.close






