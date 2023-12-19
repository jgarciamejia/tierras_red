import pdb
import numpy as np
import pandas as pd
from astropy.stats import LombScargle
from glob import glob
from time import time
import matplotlib.pylab as plt
import pymc3 as pm  # installed as part of exoplanet
import pymc3_ext as pmx  # installed as part of exoplanet
from celerite2.theano import terms, GaussianProcess  # installed as part of exoplanet

''' This is my new-and-improved code inspired by how comparison stars are handled in the MEarth pipeline '''

# set period limits [d] for Lomb-Scargle search
min_period = 0.1
max_period = 100

# I've set up this toggle to identify the appropriate directories for the given object
#toggle = "2M3495"
toggle = "2M4890"

# start the timer (to see how long the code takes)
start = time()

# get the filenames for all the .xls files you want to analyze -- change these directories as needed
if toggle == "2M4890":
    fnames = glob("/data/tierras/lightcurves/*/2MASSJ03304890+/flat0000/*.xls")
elif toggle == "2M3495":
    fnames = glob("/data/tierras/lightcurves/*/2MASSJ13093495+/flat0000/*.xls")

# make sure the dates are in order (if the files aren't named by timestamp, this treatment won't be sufficient)
fnames.sort()


def mearth_style(bjds, flux, err, regressors):

    """ Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
    """ it's called "mearth_style" because it's inspired by the mearth pipeline """

    mask = np.ones_like(flux, dtype='bool')  # initialize a bad data mask
    mask[np.where(flux < 0)[0]] = 0  # if target counts are less than 0, this cadence is bad

    # if one of the reference stars has negative flux, this cadence is also bad
    for ii in range(regressors.shape[0]):
        mask[np.where(regressors[ii, :] < 0)[0]] = 0

    # apply mask
    regressors = regressors[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]

    tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point
    c0s = -2.5*np.log10(np.percentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points

    mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
    mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad

    # apply mask
    regressors = regressors[:, mask]
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]

    # repeat the cs estimate now that we've masked out the bad cadences
    phot_regressor = np.percentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
    cs = -2.5*np.log10(phot_regressor[:,None]/regressors)  # estimate c for each star
    c_noise = np.std(cs, axis=0)  # estimate the error in c
    c_unc = (np.percentile(cs, 84, axis=0) - np.percentile(cs, 16, axis=0)) / 2.  # error estimate that ignores outliers

    ''' c_unc will overestimate error introduced by zero-point offset because it is correlated. Attempt to correct
    for this by only considering the additional error compared to the cadence where c_unc is minimized '''
    c_unc_best = np.min(c_unc)
    c_unc = np.sqrt(c_unc**2 - c_unc_best**2)

    cs = np.median(cs, axis=0)  # take the median across all regressors. 

    # one more bad data mask: don't trust cadences where the regressors have big discrepancies
    mask = np.ones_like(flux, dtype='bool')
    mask[np.where(c_noise > 3*np.median(c_noise))[0]] = 0

    # apply mask
    flux = flux[mask]
    err = err[mask]
    bjds = bjds[mask]
    cs = cs[mask]
    c_unc = c_unc[mask]

    err = 10**(cs/(-2.5)) * np.sqrt(err**2 + (c_unc*flux*np.log(10)/(-2.5))**2)  # propagate error
    flux *= 10**(cs/(-2.5))  # adjust the flux based on the calculated zero points

    return bjds, flux, err


# arrays to hold the full dataset
full_bjd = []
full_flux = []
full_err = []
full_reg = None

# array to hold individual nights
bjd_save = []

# initialize the pplot
N = len(fnames)
fig, ax = plt.subplots(2, N, sharey='row', sharex=True, figsize=(14, 4))

# load the list of comparison stars to use

if toggle == "2M3495":
    complist = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15,16])
 
elif toggle == "2M4890":
    complist = np.arange(2,90)
    bad_comps = np.array([2,4,5,14,20,28,33,38,42,47,33,67,68,79,82,83,88])
    mask = ~np.isin(complist,bad_comps)
    complist = complist[mask]
#compname = (fnames[0]).split("-Tierras")[0] + "_comps_relflux.csv"
#complist = np.loadtxt(compname, skiprows=1)
#complist = complist.astype(int)

for ii,fname in enumerate(fnames):

    print("Reading", fname)

    # read the .xls file
    dat = pd.read_csv(fname,delimiter='\t')
    bjds = dat["BJD_TDB_MOBS"]
    flux = dat["Source-Sky_T1"]
    err = dat["Source_Error_T1"]
    expt = dat[" EXPTIME"]

    # get the comparison fluxes
    comps = {}
    for ii in complist:
        #if ii in [2, 5, 14, 38, 88] and toggle == "2M4890":  # there's an issue with some of the 2MA0330 comps
        #    continue
        try:
            comps[ii] = dat["Source-Sky_C"+str(ii)] / expt  # divide by exposure time since it can vary between nights
        except:
            print("Error with comp", str(ii))
            continue

    # make a list of all the comps
    regressors = []
    for key in comps.keys():
        regressors.append(comps[key])
    regressors = np.array(regressors)

    # add this night of data to the full data set
    full_bjd.extend(bjds)
    full_flux.extend(flux/expt)
    full_err.extend(err/expt)
    bjd_save.append(bjds)

    if full_reg is None:
        full_reg = regressors
    else:
        full_reg = np.concatenate((full_reg, regressors), axis=1)

# convert from lists to arrays
full_bjd = np.array(full_bjd)
full_flux = np.array(full_flux)
full_err = np.array(full_err)

# mask bad data and use comps to calculate frame-by-frame magnitude zero points
x, y, err = mearth_style(full_bjd, full_flux, full_err, full_reg)

#pdb.set_trace()

# plot the data night-by-night
for ii in range(N):
    # get the indices corresponding to a given night
    use_bjds = np.array(bjd_save[ii])
    inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]
    if len(inds) == 0:  # if the entire night was masked due to bad weather, don't plot anything
        continue
    else:
        # identify and plot the night of data
        bjd_plot = x[inds]
        flux_plot = y[inds]
        err_plot = err[inds]
        markers, caps, bars = ax[0, ii].errorbar((bjd_plot-np.min(bjd_plot))*24., flux_plot, yerr=err_plot, fmt='k.', alpha=0.2)
        [bar.set_alpha(0.05) for bar in bars]

# format the plot
fig.text(0.5, 0.01, 'hours since start of night', ha='center')
ax[0, 0].set_ylabel('corrected flux')
ax[0, 0].set_ylim(np.percentile(y, 1), np.percentile(y, 99))  # don't let outliers wreck the y-axis scale
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

# convert relative flux to ppt
mu = np.median(y)
y = (y / mu - 1) * 1e3
err = (err/mu) * 1e3

# do the Lomb Scargle
# ls = LombScargle(x, y, dy=err)
# freq, power = ls.autopower(minimum_frequency=1./max_period, maximum_frequency=1./min_period)
# peak = 1./freq[np.argmax(power)]

# # do the Lomb Scargle, w/ added Emily Pass hack to get better res.
freq = np.linspace(1./max_period, 1./min_period, 100000)
ls = LombScargle(x, y, dy=err)
power = ls.power(freq)
peak = 1./freq[np.argmax(power)]

# make the window function
ls_window = LombScargle(x, np.ones_like(x), fit_mean=False, center_data=False)
wf, wp = ls_window.autopower(minimum_frequency=1./max_period, maximum_frequency=1./min_period)

# report the periodogram peak
print("LS Period:", peak)

# make a pretty periodogram
fig_ls, ax_ls = plt.subplots(2,1, sharex=True)
ax_ls[0].plot(1./freq, power, color='orange', label='data', zorder=10)
ax_ls[1].plot(1./wf, wp, color='k', label='window')
ax_ls[0].set_xlim((1./freq).min(), (1./freq).max())
ax_ls[0].axvline(peak, color='k', linestyle='dashed')
ax_ls[1].axvline(peak, color='orange', linestyle='dashed')
ax_ls[0].set_ylabel("power")
ax_ls[1].set_ylabel("power")
ax_ls[1].set_xlabel("period [d]")
ax_ls[0].semilogx()

# this variable is called period1 because I've adapted this from another code I wrote that fits multiple periods
# to blended TESS light curves. I've removed the second period for simplicity here since Tierras doesn't have the same
# pixel scale issues, but that functionality can be reimplemented if needed
period1 = peak # if you want to override the LS period, you can set this to your desired value instead of "peak"


# a pymc3 model, inspired by the "exoplanet" package case studies
# note that this model does not fit for period -- it fixes period at the LS value
def build_model(mask):
    with pm.Model() as model:

        # an offset term
        mean_lc = pm.Normal("mean_lc", mu=0.0, sd=50.0)

        # I don't actually want to use a GP but I want to use "exoplanet"'s marginalization framework, so I've set the
        # GP kernel to have a zero amplitude hyperparameter (i.e., it can't do anything).
        kernel_lc = terms.SHOTerm(sigma=0., rho=1., Q=1.0 / 3)

        # How many sinusoids to use in the spot model? N=1 is fundamental mode only, N=2 adds the 2nd harmonic, etc.
        N = 3
        coeffs_a = []
        phases_a = []
        print ('Harmonics N = {}'.format(str(N)))

        for ii in range(N):
            # create appropriate fitting parameters based on N
            coeffs_a.append(
                pm.Uniform("coeffa_" + str(ii), lower=0, upper=1, testval=0.01))  # spot harmonic coefficients
            phases_a.append(pm.Uniform("phasea_" + str(ii), lower=0., upper=1, testval=0.2))  # spot harmonic phases

        # this spot model is defined in equation 1 of Hartman et al. 2018
        # it's called spota, because in the version of the code with multiple stars there'd also be a spotb
        def spota(t):
            total = coeffs_a[0] * (np.cos(2 * np.pi * (t / period1 + phases_a[0])))
            for ii in range(1, N):
                total += coeffs_a[0] * coeffs_a[ii] * np.cos(
                    2 * np.pi * (t * (ii + 1) / period1 + phases_a[ii] + (ii + 1) * phases_a[0]))
            return total

        # our model is the spot model plus a y-axis offset
        def model_lc(t):
            return mean_lc + 1e3 * spota(t)

        # Condition the light curve model on the data
        gp_lc = GaussianProcess(kernel_lc, t=x[mask], yerr=err[mask])
        gp_lc.marginal("obs_lc", observed=y[mask] - model_lc(x[mask]))

        # Optimize the logp
        map_soln = model.test_point
        map_soln = pmx.optimize(map_soln)

        # retain important variables for later
        extras = dict(x=x[mask], y=y[mask], yerr=err[mask], model_lc=model_lc, gp_lc_pred=gp_lc.predict(y[mask] - model_lc(x[mask])), spota=spota)

    return model, map_soln, extras


# this sigma clipping routine is from "exoplanet" and iteratively removes outliers from the fit
def sigma_clip():
    mask = np.ones(len(x), dtype=bool)
    num = len(mask)

    for i in range(10):
        model, map_soln, extras = build_model(mask)

        with model:
            mdl = pmx.eval_in_model(
                extras["model_lc"](extras["x"]) + extras["gp_lc_pred"],
                map_soln,
            )

        resid = y[mask] - mdl
        sigma = np.sqrt(np.median((resid - np.median(resid)) ** 2))
        mask[mask] = np.abs(resid - np.median(resid)) < 7 * sigma
        print("Sigma clipped {0} light curve points".format(num - mask.sum()))
        if num - mask.sum() < 10:
            break
        num = mask.sum()

    return model, map_soln, extras


model, map_soln, extras = sigma_clip()

# print the maximum a posteriori (MAP) results
for var in map_soln.keys():
    print(var, map_soln[var])

t_lc_pred = np.linspace(x.min(), x.max(), 10000)  # times at which we're going to plot

# get the light curve associated with the maximum a posteriori model
with model:
    gp_pred = (pmx.eval_in_model(extras["gp_lc_pred"], map_soln) + map_soln["mean_lc"])
    lc = (pmx.eval_in_model(extras["model_lc"](t_lc_pred), map_soln) - map_soln["mean_lc"])
    lc_obs = (pmx.eval_in_model(extras["model_lc"](x), map_soln) - map_soln["mean_lc"])
    spota = 1e3 * pmx.eval_in_model(extras["spota"](t_lc_pred), map_soln)


def bin(x, y, tbin):
    """ Basic binning routine, assumes tbin shares the units of x """
    bins = np.arange(np.min(x), np.max(x), tbin)
    binned = []
    binned_e = []
    if len(bins) < 2:
        return [np.nan], [np.nan], [np.nan]
    for ii in range(len(bins)-1):
        use_inds = np.where((x < bins[ii + 1]) & (x > bins[ii]))[0]
        if len(use_inds) < 1:
            binned.append(np.nan)
            binned_e.append(np.nan)
        else:
            binned.append(np.median(y[use_inds]))
            binned_e.append((np.percentile(y[use_inds], 84)-np.percentile(y[use_inds], 16))/2.)

    return bins[:-1] + (bins[1] - bins[0]) / 2., np.array(binned), np.array(binned_e)


# arrays to store residuals
all_res = []
all_res_bin = []

# overplot the fit on each night of data
for ii in range(N):
    bjds = bjd_save[ii]

    # get the indices corresponding to a given night
    use_bjds = np.array(bjd_save[ii])
    inds = np.where((x > np.min(use_bjds)) & (x < np.max(use_bjds)))[0]
    if len(inds) == 0:  # if the entire night was masked due to bad weather, don't plot anything
        continue
    else:
        # identify and plot the night of data
        bjd_plot = x[inds]
        flux_plot = y[inds]
        err_plot = err[inds]
        lc_plot = lc_obs[inds]

    xlim = ax[0, ii].get_xlim()  # remember the x-axis limits so we don't mess them up

    # plot the model fit
    ax[0, ii].plot((t_lc_pred - np.min(use_bjds))*24., (lc/1e3 + 1)*mu, color="C2", lw=1, zorder=10)

    # add bins
    tbin = 20  # bin size in minutes
    xs_b, binned, e_binned = bin((bjd_plot - np.min(bjd_plot))*24, (flux_plot/1e3+1)*mu, tbin/60.)
    _, binned_res, e_binned_res = bin((bjd_plot - np.min(bjd_plot))*24, flux_plot-lc_plot, tbin/60.)
    marker, caps, bars = ax[0, ii].errorbar(xs_b, binned, yerr=e_binned, color='purple', fmt='.', alpha=0.5, zorder=5)
    [bar.set_alpha(0.3) for bar in bars]

    # plot the residuals
    markers, caps, bars = ax[1, ii].errorbar((bjd_plot - np.min(bjd_plot))*24., flux_plot - lc_plot, yerr=err_plot, fmt='k.', alpha=0.2)
    [bar.set_alpha(0.05) for bar in bars]
    markers, caps, bars = ax[1, ii].errorbar(xs_b, binned_res, yerr=e_binned_res, color='purple', fmt='.', alpha=0.5, zorder=5)
    [bar.set_alpha(0.3) for bar in bars]
    ax[1, ii].axhline(0, linestyle='dashed', color='k')

    ax[0, ii].set_xlim(xlim)  # revert to original axis limits
    all_res.extend(flux_plot-lc_plot)  # keep track of residuals
    all_res_bin.extend(binned_res)  # keep track of binned residuals


# report the time it took to run the code
print("Elapsed time:", np.round((time()-start)/60.,2), "min")

all_res = np.array(all_res)
all_res_bin = np.array(all_res_bin)

ax[1, 0].set_ylabel("O-C [ppt]")
ax[1, 0].set_ylim(np.percentile(all_res, 1), np.percentile(all_res, 99))  # don't let outliers wreck the y-axis scale

print("RMS model:", np.round(np.sqrt(np.median(all_res**2))*1e3, 2), "ppm")
print("Binned RMS model:", np.round(np.sqrt(np.nanmedian(all_res_bin**2))*1e3, 2), "ppm in", tbin, "minute bins")

plt.show()

# subtract off first timestamp to make the x-axis less cluttered
min_t = np.min(extras["x"])
extras["x"] -= min_t
t_lc_pred -= min_t

# make plot of phased light curve
fig, ax1 = plt.subplots(1)
x_phased1 = (extras["x"] % period1) / period1
x_phased = (t_lc_pred % period1) / period1
y1 = extras['y']

# bin the data into 100 bins that are evenly spaced in phase
bins = np.linspace(0, 1, 101)
sort1 = np.argsort(x_phased1)
x_phased1 = x_phased1[sort1]
sort = np.argsort(x_phased)
x_phased = x_phased[sort]
y1 = y1[sort1]
spota = spota[sort]
binned1 = []
for ii in range(100):
    use_inds = np.where((x_phased1 < bins[ii + 1]) & (x_phased1 > bins[ii]))[0]
    binned1.append(np.median(y1[use_inds]))

# plot the phased data
ax1.plot(x_phased1, y1, "k.", alpha=0.2)
ax1.plot(x_phased, spota, 'r', lw=1, zorder=10)
ax1.plot(bins[:-1] + (bins[1] - bins[0]) / 2., binned1, 'go', alpha=0.5)

# I wanted this to adjust the y-axis to an appropriate scale automatically, but you may still need to play around with
# ylim_N depending on how noisy the data is.
ylim_N = 5
mean1 = np.median(spota)
max1 = np.max(spota)
diff1 = max1 - mean1
ax1.set_ylim(mean1 - ylim_N * diff1, mean1 + ylim_N * diff1)

# finish up the plot
ax1.set_ylabel("Tierras flux [ppt]")
ax1.set_xlabel("phase")
fig.tight_layout()
plt.show()
