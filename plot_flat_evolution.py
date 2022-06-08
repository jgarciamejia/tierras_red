import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt 
from medsig import *

weeks = ['20220405','20220412','20220419','20220503','20220510','20220517','20220524']

Mflats = []
for date in weeks:
	hdu = fits.open('/data/tierras/incoming/{}/MASTERFLAT_mednorm.fit'.format(date))
	Mflats.append(hdu)

#rows = np.linspace(100,900,9)

### For one Y row, plot contiguous week ratios separately
#row = 800
# # Emacs block comment/uncomment Alt + ;
# f,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))
# for nth_ratio in np.arange(len(weeks)-1):
#         f,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))

#         plt.subplots_adjust(wspace=0.3)
#         flat_num = Mflats[nth_ratio+1]
#         flat_denom = Mflats[nth_ratio]
        
#         flat_num1, flat_num2 = flat_num[1].data,flat_num[2].data
#         flat_denom1, flat_denom2 = flat_denom[1].data,flat_denom[2].data

#         ax1.scatter(np.arange(len(flat_num1[row])),flat_num1[row]/flat_denom1[row],s=2)
#         ax2.scatter(np.arange(len(flat_num2[row])),flat_num2[row]/flat_denom2[row],s=2)
        
#         f.suptitle('Y (row) = {}'.format(str(row)))
#         ax1.set_title('CCD Chip: Half 1')
#         ax2.set_title('CCD Chip: Half 2')

#         ax1.set_xlabel('X pixel Coord')
#         ax2.set_xlabel('X pixel Coord')

#         ax1.set_ylabel('Flux Ratio Week {}:{}'.format(str(nth_ratio+1),str(nth_ratio)))
#         ax2.set_ylabel('Flux Ratio Week {}:{}'.format(str(nth_ratio+1),str(nth_ratio)))
#         #ax1.set_ylim(.998,1.002)
#         #ax2.set_ylim(.998,1.002)
#         plt.show()
#         #f.savefig('weekly_flat_ratio_{}to{}_Yrow{}.png')

### For one Y row, plot flux values of two consecutive weeks for each half of the chip 
row = 800
# Emacs block comment/uncomment Alt + ;
f,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))
for nth_ratio in np.arange(len(weeks)-1):
        f,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))

        plt.subplots_adjust(wspace=0.3)
        flat_num = Mflats[nth_ratio+1]
        flat_denom = Mflats[nth_ratio]
        
        flat_num1, flat_num2 = flat_num[1].data,flat_num[2].data
        flat_denom1, flat_denom2 = flat_denom[1].data,flat_denom[2].data
        
        ax1.scatter(np.arange(len(flat_num1[row])),flat_num1[row],s=2,label = '{}'.format(weeks[nth_ratio]))
        ax1.scatter(np.arange(len(flat_denom1[row])),flat_denom1[row],s=2,label = '{}'.format(weeks[nth_ratio+1]))
        ax2.scatter(np.arange(len(flat_num2[row])),flat_num2[row],s=2)
        ax2.scatter(np.arange(len(flat_denom2[row])),flat_denom2[row],s=2)

        f.suptitle('Y (row) = {}'.format(str(row)))
        ax1.set_title('CCD Chip: Half 1')
        ax2.set_title('CCD Chip: Half 2')

        ax1.set_xlabel('X pixel Coord')
        ax2.set_xlabel('X pixel Coord')

        ax1.set_ylabel('Normalized Flux'.format(str(nth_ratio+1),str(nth_ratio)))
        ax2.set_ylabel('Normalized Flux'.format(str(nth_ratio+1),str(nth_ratio)))
        #ax1.set_ylim(.998,1.002)
        #ax2.set_ylim(.998,1.002)
        ax1.legend()
        plt.show()




# rows = np.linspace(100,900,9)
# f,ax1 = plt.subplots()
# for row in rows:
#         ratio_rmss = []
# for nth_ratio in np.arange(len(weeks)-1):
#         flat_num = Mflats[nth_ratio+1]
#         flat_denom = Mflats[nth_ratio]
        
#         flat_num1, flat_num2 = flat_num[1].data,flat_num[2].data
#         flat_denom1, flat_denom2 = flat_denom[1].data,flat_denom[2].data
        




