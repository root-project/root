import sys, csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

datfiles = ["combined.dat","example3D.dat","fit2a.dat","fit2.dat","fit2dhist.dat","gauss2D_fit.dat","gauss_fit.dat","lorentz_fit.dat"]

#fig = plt.figure()
#ax = fig.add_subplot(211)
fig, axarr = plt.subplots(3,3)

i = 0
j = 0
N = 4
runs = -1
for f in datfiles:
    dat = loadtxt(f,dtype=float,comments='#')

    # best f-min
    bestfmin1 = dat[0,8]
    bestfmin2 = dat[1,8]

    # successes and failures
    cmaAvg = [dat[0,1],0.0,0.0,bestfmin1]
    cmaStd = [0.0,0.0,0.0,0.0]
    minAvg = [dat[1,1],0.0,0.0,bestfmin2]
    minStd = [0.0,0.0,0.0,0.0]

    # budget and cpu averages
    cmaAvg2 = [0.0,dat[0,4]*1000.0,dat[0,6],0.0]
    cmaStd2 = [0.0,dat[0,5]*1000.0,dat[0,7],0.0]
    minAvg2 = [0.0,dat[1,4]*1000.0,dat[1,6],0.0]
    minStd2 = [0.0,dat[1,5]*1000.0,dat[1,7],0.0]

    if runs == -1:
        runs = dat[0,2]+dat[0,3] # get the number of runs once.
    
    ## necessary variables
    #ind = np.arange(N)                # the x locations for the groups
    ind = array([2,1,0,3])
    width = 0.35                      # the width of the bars
    
    ## the bars
    ax2 = axarr[i,j].twinx()
    rects1 = ax2.bar(ind, cmaAvg, width,
                            color='Teal',
                            yerr=cmaStd,
                            error_kw=dict(elinewidth=2,ecolor='red'))
    
    rects2 = ax2.bar(ind+width, minAvg, width,
                            color='LightGreen',
                            yerr=minStd,
                            error_kw=dict(elinewidth=2,ecolor='black'))

    rects12 = axarr[i,j].bar(ind, cmaAvg2, width,
                      color='Teal',
                      yerr=cmaStd2,
                      error_kw=dict(elinewidth=2,ecolor='red'))
    
    rects22 = axarr[i,j].bar(ind+width, minAvg2, width,
                      color='LightGreen',
                      yerr=minStd2,
                      error_kw=dict(elinewidth=2,ecolor='black'))

    # axes and labels
    axarr[i,j].set_yscale("log",nonposy='clip')
    axarr[i,j].set_xlim(-width,len(ind)+width)
    axarr[i,j].set_ylim(0)
    ax2.set_xlim(-width,len(ind)+width)
    ax2.set_ylim(0)
    #axarr[i,j].set_ylabel('Scores')
    axarr[i,j].set_title(f,fontsize=12)
    xTickMarks = ["Found","CPU avg (ms)","Budget avg","wins"]
    axarr[i,j].set_xticks(ind+width)
    xtickNames = axarr[i,j].set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    ## add a legend
    if i == 0 and j == 0:    
        ax2.legend( (rects1[0], rects2[0]), ('aCMA-ES', 'Minuit2'), fontsize=10 )
    i = i + 1
    if i == 3:
        i = 0
        j = j + 1

plt.tight_layout()
plt.suptitle('aCMA-ES / Minuit2 Benchmark Suite / ' + str(len(datfiles)) + ' experiments / ' + str(int(runs)) + ' runs on each')
plt.show()
