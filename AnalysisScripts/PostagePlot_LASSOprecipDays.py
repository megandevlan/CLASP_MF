#!/usr/bin/env pyn_env2

# Load libraries
import datetime
import glob
import os
import warnings
from datetime import date, timedelta
import matplotlib as matplotlib
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
#import Ngl
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.dates import DateFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')

# - - - - - - - - - - - -
## Define some useful funcitons first... 
def preprocess_lasso(ds):
    keepVars = ['T','q','omega','divT','divq','Prec','lhflx','shflx','Ps','totcld','cldht']
    
    dsSel = ds[keepVars].load()
    
    ## Convert to local time
    localTimes  = dsSel['time'].values - np.timedelta64(5,'h')
    dsSel       = dsSel.assign_coords({"time": localTimes})

    
    return dsSel

def cesm_correct_time(ds):
    """Given a Dataset, check for time_bnds,
       and use avg(time_bnds) to replace the time coordinate.
       Purpose is to center the timestamp on the averaging inverval.   
       NOTE: ds should have been loaded using `decode_times=False`
    """
    assert 'time_bnds' in ds
    assert 'time' in ds
    correct_time_values = ds['time_bnds'].mean(dim='nbnd')
    # copy any metadata:
    correct_time_values.attrs = ds['time'].attrs
    ds = ds.assign_coords({"time": correct_time_values})
    ds = xr.decode_cf(ds)  # decode to datetime objects
    return ds

def preprocess_h0_getDay2(ds):

    keepVars = ['SWCF','LWCF','TS','CLOUD','FSNS','FLNS','PS','QREFHT',
                'U10','CLDHGH','CLDLIQ','CONCLD','TMQ','P0','hyam','hybm','hyai','hybi',
                'PHIS','USTAR','QT','GCLDLWP',
                'THETAL','CDNUMC','CLDBOT','CLDLOW',
                'CLDMED','CLDTOP','CLDTOT','THLP2_CLUBB','CLOUDCOVER_CLUBB','CLOUDFRAC_CLUBB',
                'RCM_CLUBB','RTP2_CLUBB','RTPTHLP_CLUBB','RVMTEND_CLUBB','STEND_CLUBB','UP2_CLUBB','UPWP_CLUBB',
                'VP2_CLUBB','T','Q','OMEGA','PBLH','U','V','WP2_CLUBB','WP3_CLUBB','WPRCP_CLUBB',
                'WPRTP_CLUBB',
                'WPTHLP_CLUBB','WPTHVP_CLUBB','Z3','PRECT','PRECC',
                # 'PRECZ',
                'TGCLDCWP','TGCLDLWP','GCLDLWP',
                'LHFLX','SHFLX','TREFHT','RHREFHT']
        
    ds         = cesm_correct_time(ds)
    ds['time'] = ds.indexes['time'].to_datetimeindex() 
        
    ## Select the second simulated day for analysis 
    iTimeStart_day2  = np.where( (ds.time.values >= (ds.time.values[0] + np.timedelta64(1,'D'))) & 
                                 (ds.time.values <= (ds.time.values[0] + np.timedelta64(2,'D'))))[0]
    dsSel      = ds.isel(time=np.sort(iTimeStart_day2))[keepVars]

    # Compute local time 
    localTimes = dsSel['time'].values - np.timedelta64(5,'h')
    dsSel      = dsSel.assign_coords({"time": localTimes})
    
    dsSel = dsSel.load()

    return dsSel

# - - - - - - - - - - - - 

## The LASSO stuff you probably won't need to change for anything... 
## For SCAM runs -- skip to below

## Open the LASSO forcing files
fileDir = '/project/amp/mdfowler/CLASP/inputData/LASSO_fromFinley/'
listFiles = np.sort(glob.glob(fileDir+'*LASSO*'))
lassoDS = xr.open_mfdataset(listFiles,  preprocess=preprocess_lasso, concat_dim='time', 
                                combine='nested', decode_times=True, 
                                data_vars='minimal')

# - - - - - - - - - - - - - -     
## Find days where the rain rate reaches the above threshold, 
##     and the time at which that max is reached/when rain starts
rain_mmhr = lassoDS.Prec.isel(lat=0,lon=0) * 3600  ## Convert from mm/s to mm/hr
DayMaxRain = rain_mmhr.resample(time='1D').max().dropna(dim='time')

cutoff     = 0.5 ## Threshold for where consider it a decent 'rain' event or not 
iRaining = np.where(DayMaxRain>=cutoff)[0] # Indices of raining days 
print('Fraction of days with >= %f mm/hr of rain: %f' % (cutoff, len(iRaining)/len(lassoDS.Prec.values)))
print('Actual number of days = %i' % (len(iRaining)))

iCount = 0 # Number of days saved 
for iDay in range(len(DayMaxRain.time.values)):
    
    if DayMaxRain[iDay]>=cutoff:
        # Get all the times in the full LASSSO DS (not the daily max) that cover those 'rainy' days 
        iTimes = np.where( (lassoDS['time.year'].values == DayMaxRain['time.year'].values[iDay]) & 
                           (lassoDS['time.month'].values == DayMaxRain['time.month'].values[iDay]) &  
                           (lassoDS['time.day'].values == DayMaxRain['time.day'].values[iDay]) )[0]
        
        dayDS = lassoDS.isel(time=iTimes,lat=0,lon=0)
        
        ## Save info about when the max prect occurs
        imax  = np.where((dayDS.Prec.values*3600)==DayMaxRain.values[iDay])[0]
        if iCount==0:
            maxRainDS = dayDS.isel(time=imax)
        else: 
            maxRainDS = xr.concat([maxRainDS, dayDS.isel(time=imax)], dim='time')
          
        iCount = iCount+1

## Select events with peak rain in afternoon/evening
iTargets = np.where( (maxRainDS['time.hour'].values>=15) & (maxRainDS['time.hour'].values<=21) & 
                     (maxRainDS['time.year'].values>=2015))[0]
# - - - - - - - - - - - - - -     

# - - - - - - - - - - - - - -     
# THIS IS THE SCAM SECTION 
# - - - - - - - - - - - - - -

# Set the directory that has all the SCAM output, the case names, and some short names to use as tags for 'case' 
#   This will be the primary section to modify with new runs
#   Note: My approach is to copy all the *cam.h* files into one directory so all the days are in one place 

testDir     = [
               '/scratch/cluster/mdfowler/clubbMF_lateRainDays/',
               '/scratch/cluster/mdfowler/clubbMF_lateRainDays/',
              ]

case_names  = [
               'Lopt6_day2',
               'DummySecondCase_duplicate first one',
              ]

caseStart   = 'FSCAM.T42_T42.lasso.CLUBBMF_usePatchDataFALSE_setSfcFlxFALSE_clmInit_PRECTfix_'

# These would normally be unique parts of the case name that indicagte how it's a different case
caseStrings = [
                'Lopt6_lateRainDay_asDay2.LASSO',
                'Lopt6_lateRainDay_asDay2.LASSO',
              ]

# Loop over however many cases you want to take a look at 
for iCase in range(len(case_names)):
    print('*** Starting on case %s ***' % (case_names[iCase]))
    ## Get list of files 
    listFiles    = np.sort(glob.glob(testDir[iCase]+caseStart+caseStrings[iCase]+'*cam.h0*'))
    # Open those files 
    case_h0      = xr.open_mfdataset(listFiles,  preprocess=preprocess_h0_getDay2, concat_dim='time', 
                            combine='nested', decode_times=False, 
                            data_vars='minimal')
    print('h0 files loaded')

    ## Combine all the cases into one DS 
    case_allDays      = case_h0.assign_coords({"case":  case_names[iCase]})

    if iCase==0:
        scamDS    = case_allDays
    else: 
        scamDS    = xr.concat([scamDS, case_allDays], "case") 
    print('Done with case %i of %i ' % (iCase+1, len(case_names)))

# - - - - - - - - - - - - - - 
# Now for the plot
# - - - - - - - - - - - - - - 

#Define the plot figure (subplots, etc)
fig,axs = plt.subplots(7,6,figsize=(18,18))
axs     = axs.ravel()

# Now handle plotting each day 
for iDay in range(len(iTargets)):
        iTimes = np.where( (lassoDS['time.year'].values == maxRainDS.isel(time=iTargets[iDay])['time.year'].values) & 
                           (lassoDS['time.month'].values == maxRainDS.isel(time=iTargets[iDay])['time.month'].values) &  
                           (lassoDS['time.day'].values == maxRainDS.isel(time=iTargets[iDay])['time.day'].values) & 
                           (lassoDS['time.hour'].values>=8) )[0]
        
        dayDS = lassoDS.isel(time=iTimes,lat=0,lon=0)
        
        ## Plot the LASSO rainfall 
        axs[iDay].plot(dayDS['time'].values, dayDS.Prec.values*3600,'k-')
        axs[iDay].set_title(maxRainDS.isel(time=iTargets[iDay]).time.values.astype(str)[0:10])
        
        # Plot the LASSO cloud fraction too 
        ax2 = axs[iDay].twinx()
        ax2.plot(dayDS['time'].values, dayDS.totcld.values, ':', color='grey')
        ax2.set_ylim([0, 101])

        ## Get just this day from SCAM
        iTimes_day  = np.where( (scamDS['time.year'].values == maxRainDS.isel(time=iTargets[iDay])['time.year'].values) & 
                                (scamDS['time.month'].values == maxRainDS.isel(time=iTargets[iDay])['time.month'].values) &  
                                (scamDS['time.day'].values == maxRainDS.isel(time=iTargets[iDay])['time.day'].values) & 
                                (scamDS['time.hour'].values>=8) )[0]

        ## THIS IS WHERE YOU SELECT WHICH CASE TO PLOT! 
        day2_ds = scamDS.sel(case='Lopt6_day2').isel(time=iTimes_day,lat=0,lon=0)

        ## Plot SCAM results 
        axs[iDay].plot(day2_ds['time'].values, day2_ds.PRECT.values*3600*1000, '-', color='darkviolet', label='Day2sim')
        axs[iDay].plot(day2_ds['time'].values, day2_ds.PRECC.values*3600*1000, '-', color='limegreen', label='Day2sim')
    
        axs[iDay].xaxis.set_major_formatter(mdates.DateFormatter('%H')) 
                
fig.subplots_adjust(hspace = 0.35, wspace=0.35)

figTitle = 'PostagePlot_'+str(day2_ds.case.values)+'.pdf'
plt.savefig(figTitle)

print('Done with generating plot!')

