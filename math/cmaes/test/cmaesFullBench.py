import sys, csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

datfiles = ["combined.dat","example3D.dat","fit2a.dat","fit2.dat","fit2dhist.dat","gauss2D_fit.dat","gauss_fit.dat","lorentz_fit.dat"]

fig, axarr = plt.subplots(8,2)

i = 0
j = 0
N = 2
runs = -1
for f in datfiles:
    dat = loadtxt(f,dtype=float,comments='#')
    nalgs = len(dat)

    if runs == -1:
        runs = dat[0,3]+dat[0,4] # get the number of runs once.

    hcolor = 'Teal'
    aa = 0
    oldrec = None
    for a in range(nalgs):
        if a == nalgs-1:
            hcolor = 'LightGreen'
        
        aavg1 = [dat[a,2],dat[a,9]]
        aastd1 = [0.0,0.0]
        aavg2 = [dat[a,5]*1000.0,dat[a,7]]
        aastd2 = [dat[a,6]*1000.0,dat[a,8]]
        
        ## necessary variables
        ind = np.arange(N)                # the x locations for the groups
        width = 0.05                      # the width of the bars
    
        ## the bars
        rects1 = axarr[i,j+1].bar(ind+aa*width, aavg1, width,
                                  color=hcolor,
                                  yerr=aastd1,
                                  error_kw=dict(elinewidth=2,ecolor='red'))
        if a == 0:
            oldrec = rects1
        
        #rects2 = ax2.bar(ind+width, minAvg, width,
        #                    color='LightGreen',
        #                    yerr=minStd,
        #                    error_kw=dict(elinewidth=2,ecolor='black'))

        rects12 = axarr[i,j].bar(ind+aa*width, aavg2, width,
                                 color=hcolor,
                                 yerr=aastd2,
                                 error_kw=dict(elinewidth=2,ecolor='red'))
    
        #rects22 = axarr[i,j].bar(ind+width, minAvg2, width,
         #                        color='LightGreen',
         #                       yerr=minStd2,
         #                        error_kw=dict(elinewidth=2,ecolor='black'))

         #add a legend
        if i == 6:
            axarr[6,1].legend((oldrec[0], rects12[0]), ('aCMA-ES', 'Minuit2'), fontsize=10 )
        aa = aa + 1

        
        
    # axes and labels
    axarr[i,j].set_yscale("log",nonposy='clip')
    #axarr[i,j].set_xlim(-width,len(ind)+width)
    axarr[i,j].set_ylim(0)
    #axarr[i,j+1].set_xlim(-width,len(ind)+width)
    axarr[i,j+1].set_ylim(top=runs)
    if i == 4 or i == 6:
        axarr[i,j+1].set_ylim(top=2*runs)
    if i == 7:
        axarr[i,j+1].set_ylim(top=100)    
    #axarr[i,j].set_ylabel('Scores')
    axarr[i,j].set_title(f + " / " + str(dat[0,0]) + "-D",fontsize=12)
    #axarr[i,j].set_xticks(ind+width)
    #axarr[i,j+1].set_xticks(ind+width)
    plt.setp(axarr[i,j].get_xticklabels(),visible=False)
    plt.setp(axarr[i,j+1].get_xticklabels(),visible=False)
    if i == 7:
        xTickMarks1 = ["CPU avg (ms)","","","","","Budget avg"]
        xTickMarks2 = ["Found","","","","","wins"]
        xtickNames = axarr[7,j].set_xticklabels(xTickMarks1)
        plt.setp(xtickNames, rotation=45, fontsize=10)
        xtickNames2 = axarr[7,j+1].set_xticklabels(xTickMarks2)
        plt.setp(xtickNames2, rotation=45, fontsize=10)
        plt.setp(axarr[i,j].get_xticklabels(),visible=True)
        plt.setp(axarr[i,j+1].get_xticklabels(),visible=True)
        
    #add a legend
#    if i == 6:
#        axarr[7,1].legend()# (rects1[0], rects12[0]), ('aCMA-ES', 'Minuit2'), fontsize=10 )
    i = i + 1
    j = 0
#    if i == 2:
#        i = 0
#        j = j + 1

#plt.tight_layout()
plt.suptitle('aCMA-ES / Minuit2 Benchmark Suite / ' + str(len(datfiles)) + ' experiments / ' + str(int(runs)) + ' runs on each\nlambda={auto, 50, 200, auto-aipop-4-restarts, auto-abipop-10-restarts}') #10, 20, 40, 80, 160, 320, 640, 1280}')
plt.show()
