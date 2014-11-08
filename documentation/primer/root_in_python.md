# ROOT in Python #

ROOT also offers an interface named PyRoot, see http://root.cern.ch/drupal/content/pyroot,
to the Python programming language. Python is used in a wide variety of application
areas and one of the most used scripting languages today. With its very high-level
data types with dynamic typing, its intuitive object orientation and the clear and
efficient syntax Python is very suited to control even complicated analysis work flows.
With the help of PyROOT it becomes possible to combine the power of a scripting
language with ROOT methods. Introductory material to Python is available from many
sources in the Internet, see e. g. http://docs.python.org/. There are additional
very powerful Python packages, like numpy, providing high-level mathematical functions
and handling of large multi-dimensional matrices, or matplotlib, providing plotting
tools for publication-quality graphics. PyROOT additionally adds to this access to
the vast capabilities of the ROOT universe. To use ROOT from Python, the environment
variable PYTHONPATH must include the path to the library path, `$ROOTSYS/lib`, of a
ROOT version with Python support. Then, PyROOT provides direct interactions with
ROOT classes from Python by importing ROOT.py into Python scrips via the command
import ROOT; it is also possible to import only selected classes from ROOT, e. g.
from ROOT import TF1.

## PyROOT ##

The access to ROOT classes and their methods in PyROOT is almost identical to C++
macros, except for the special language features of Python, most importantly dynamic
type declaration at the time of assignment.Coming back to our first example, simply
plotting a function in ROOT, the following C++ code:

``` {.cpp}
TF1 *f1 = new TF1("f2","[0]*sin([1]*x)/x",0.,10.);
f1->SetParameter(0,1);
f1->SetParameter(1,1);
f1->Draw();
```

in Python becomes:

``` {.cpp}
import ROOT
f1 = ROOT.TF1("f2","[0]*sin([1]*x)/x",0.,10.)
f1.SetParameter(0,1);
f1.SetParameter(1,1);
f1.Draw();
```

A slightly more advanced example hands over data defined in the macro to the ROOT
class `TGraphErrors`. Note that a Python array can be used to pass data between
Python and ROOT. The first line in the Python script allows it to be executed
directly from the operating system, without the need to start the script from
python or the highly recommended powerful interactive shell ipython. The last line
in the python script is there to allow you to have a look at the graphical output
in the ROOT canvas before it disappears upon termination of the script.

Here is the C++ version:

``` {.cpp}
void TGraphFit ( ) {
   //
   // Draw a graph with error bars and fit a function to it
   //
   gStyle->SetOptFit(111) ; //superimpose fit results
   // make nice Canvas
   TCanvas *c1 = new TCanvas("c1" ,"Daten" ,200 ,10 ,700 ,500) ;
   c 1 -> S e t G r i d ( ) ;
   //define some data points ...
   const Int_t n = 10;
   Float_t x[n] = {-0.22, 0.1, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 1.1};
   Float_t y[n] = {0.7, 2.9, 5.6, 7.4, 9., 9.6, 8.7, 6.3, 4.5, 1.1};
   Float_t ey[n] = {.8 ,.7 ,.6 ,.5 ,.4 ,.4 ,.5 ,.6 ,.7 ,.8};
   Float_t ex[n] = {.05 ,.1 ,.07 ,.07 ,.04 ,.05 ,.06 ,.07 ,.08 ,.05};
   // and hand over to TGraphErros object
   TGraphErrors *gr = new TGraphErrors(n,x,y,ex,ey);
   gr->SetTitle("TGraphErrors with Fit") ;
   gr->Draw("AP");
   // now perform a fit (with errors in x and y!)
   gr->Fit("gaus");
   c1->Update();
}

```

In Python it looks like this:

``` {.cpp}
#
# Draw a graph with error bars and fit a function to it
#
from ROOT import gStyle , TCanvas , TGraphErrors
from array import array
gStyle . SetOptFit (111) # superimpose fit results
c1=TCanvas("c1" ,"Data" ,200 ,10 ,700 ,500) #make nice
c1 . SetGrid ()
#define some data points . . .
x = array('f', (-0.22, 0.1, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 1.1) )
y = array('f', (0.7, 2.9, 5.6, 7.4, 9., 9.6, 8.7, 6.3, 4.5, 1.1) )
ey = array('f', (.8 ,.7 ,.6 ,.5 ,.4 ,.4 ,.5 ,.6 ,.7 ,.8) )
ex = array('f', (.05 ,.1 ,.07 ,.07 ,.04 ,.05 ,.06 ,.07 ,.08 ,.05) )
nPoints=len ( x )
# . . . and hand over to TGraphErros object
gr=TGraphErrors ( nPoints , x , y , ex , ey )
gr.SetTitle("TGraphErrors with Fit")
gr . Draw ( "AP" ) ;
gr.Fit("gaus")
c1 . Update ()
# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
```

Comparing the C++ and Python versions in these two examples, it now should be
clear how easy it is to convert any ROOT Macro in C++ to a Python version.

As another example, let us revisit macro3 from Chapter 4. A straight-forward
Python version relying on the ROOT class `TMath`:

``` {.cpp}
#!/ usr/bin/env python
# (the first line allows execution directly from the linux shell)
#
#------------------macro3 as python script------------
# Author: G. Quast Oct. 2013
# dependencies : PYTHON v2 .7 , pyroot
# last modified :
#-----------------------------------------------------
#
# *** Builds a polar graph in a square Canvas

from ROOT import TCanvas , TGraphPolar , TMath
from array import array

rmin=0.
rmax =6.\xE2\x88\x97 TMath . Pi ( )
npoints=300
r=array('d',npoints\xE2\x88\x97[0.])
theta=array('d',npoints\xE2\x88\x97[0.])
e=array('d',npoints\xE2\x88\x97[0.])
for ipt in range(0,npoints):
   r[ipt] = ipt*(rmax-rmin)/(npoints-1.)+rmin
   theta[ipt]=TMath.Sin(r[ipt])
c=TCanvas("myCanvas","myCanvas",600,600)
grP1=TGraphPolar(npoints,r,theta,e,e)
grP1.SetTitle("A Fan")
grP1.SetLineWidth(3)
grP1.SetLineColor(2)
grP1.Draw("AOL")

raw_input('Press <ret> to end -> ')
```

### More Python- less ROOT ###

You may have noticed already that there are some Python modules providing
functionality similar to ROOT classes, which fit more seamlessly into your
Python code.

A more “pythonic” version of the above macro3 would use a replacement of the
ROOT class TMath for the provisoining of data to TGraphPolar. With the math
package, the part of the code becomes

``` {.cpp}
import math
from array import array
from ROOT import TCanvas , TGraphPolar
...
ipt=range(0,npoints)
r=array('d',map(lambda x: x*(rmax-rmin)/(npoints-1.)+rmin,ipt))
theta=array('d',map(math.sin,r))
e=array('d',npoints*[0.])
...

```

Using the very powerful package numpy and the built-in functions to handle
numerical arrays makes the Python code more compact and readable:

``` {.cpp}
import numpy as np
from ROOT import TCanvas,TGraphPolar
...
r=np.linspace(rmin,rmax,npoints) theta=np.sin(r)
e=np.zeros(npoints)
...
```

#### Customised Binning ####
This example combines comfortable handling of arrays in Python to define
variable bin sizes of a ROOT his- togram. All we need to know is the interface
of the relevant ROOT class and its methods (from the ROOT documentation):

``` {.cpp}
TH1F(const char* name , const char* title , Int_t nbinsx , const Double_t* xbins)
```

Here is the Python code:

``` {.cpp}
import ROOT
from array import array
arrBins = array('d' ,(1 ,4 ,9 ,16) ) # array of bin edges
histo = ROOT.TH1F("hist", "hist", len(arrBins)-1, arrBins)
# fill it with equally spaced numbers
for i in range (1 ,16) :
   histo.Fill(i)
histo.Draw ()
```

#### A fit example in Python using TMinuit from ROOT ####

One may even wish to go one step further and do most of the implementation
directly in Python, while using only some ROOT classes. In the example below,
the ROOT class `TMinuit` is used as the minimizer in a $\chi^{2}$-fit. Data are provided
as Python arrays, the function to be fitted and the $\chi^{2}$-function are defined in
Python and iteratively called by Minuit. The results are extracted to Python
objects, and plotting is done via the very powerful and versatile python package
`matplotlib`.

``` {.cpp}
#!/ usr/bin/env python 2#
#---------python script---------------------------------------
# EXAMPLE showing how to set up a fit with MINUIT using pyroot
#-------------------------------------------------------------
# Author: G. Quast May 2013
# dependencies: PYTHON v2.7, pyroot, numpy, matplotlib, array
# last modified: Oct. 6, 2013
#-------------------------------------------------------------
#
from ROOT import TMinuit , Double , Long
import numpy as np
from array import array as arr
import matplotlib . pyplot as plt

# --> define some data
ax=arr('f',(
      0.05 ,0.36 ,0.68 ,0.80 ,1.09 ,1.46 ,1.71 ,1.83 ,2.44 ,2.09 ,3.72 ,4.36 ,4.60) )
ay=arr('f',(
      0.35 ,0.26 ,0.52 ,0.44 ,0.48 ,0.55 ,0.66 ,0.48 ,0.75 ,0.70 ,0.75 ,0.80 ,0.90) )
ey=arr('f',(
      0.06 ,0.07 ,0.05 ,0.05 ,0.07 ,0.07 ,0.09 ,0.10 ,0.11 ,0.10 ,0.11 ,0.12 ,0.10) )
nPoints = len(ax)

# --> Set parameters and function to f i t
# a list with convenient names,
name = ["a","m","b"]
# the initial values ,
vstart = arr( 'd', (1.0, 1.0, 1.0) )
# and the initial step size
step = arr( 'd' , (0.001 , 0.001 , 0.001) )
npar =len(name)
#
# this defines the function we want to fit:
def fitfunc(x, npar, apar): a = apar[0]
   m = apar[1]
   b = apar[2]
   f = Double(0) f=a*x*x + m*x + b
   return f
#

# --> this is the definition of the function to minimize , here a chi^2-function
def calcChi2 ( npar , apar ):
   chisq = 0.0
   for i in range(0,nPoints):
      x = ax[i]
      curFuncV = fitfunc ( x , npar , apar )
      curYV = ay[i]
      curYE = ey[i]
      chisq += ( ( curYV - curFuncV ) * ( curYV - curFuncV ) ) / ( curYE*curYE )
   return chisq
#--- the function fcn - called by MINUIT repeatedly with varying parameters
# NOTE: the function name is set via TMinuit.SetFCN
def fcn(npar, deriv, f, apar, iflag):
      """ meaning of parametrs:
            npar:  number of parameters
            deriv: aray of derivatives df/dp_i (x), optional
            f:     value of function to be minimised (typically chi2 or negLogL)
            apar:  the array of parameters
            iflag: internal flag: 1 at first call, 3 at the last, 4 during minimisation
      """
      f[0] = calcChi2(npar,apar)
#

# --> set up MINUIT
myMinuit = TMinuit ( npar ) # initialize TMinuit with maximum of npar parameters
myMinuit . SetFCN ( fcn )   # set function to minimize
arglist = arr('d', 2*[0.01]) # set error definition ierflg = Long(0)
arglist[0] = 1. # 1 sigma is Delta chi^2 = 1
myMinuit.mnexcm("SET ERR", arglist ,1,ierflg)

# --> Set starting values and step size for parameters
for i in range(0,npar): # Define the parameters for the fit
   myMinuit.mnparm(i, name[i] , vstart[i], step[i], 0, 0, ierflg)
arglist [0] = 6000 # Number of calls for FCN before gving up
arglist [1] = 0.3  # Tolerance
myMinuit.mnexcm("MIGRAD", arglist ,2,ierflg) # execute the minimisation

# --> check TMinuit status
amin , edm , errdef = Double (0.) , Double (0.) , Double (0.)
nvpar , nparx , icstat = Long (0) , Long (0) , Long (0)
myMinuit.mnstat ( amin , edm , errdef , nvpar , nparx , icstat )

# meaning of parameters:
#   amin:   value of fcn distance at minimum (=chi^2)
#   edm:    estimated distance to minimum
#   errdef: delta_fcn used to define 1 sigam errors
#   nvpar:  total number of parameters
#   icstat: status of error matrix:
#           3 = accurate
#           2 = forced pos. def
#           1 = approximative
#           0 = not calculated
#
myMinuit.mnprin(3,amin) # print-out by Minuit
# --> get results from MINUIT finalPar = []
finalParErr = [ ]
p, pe = Double(0) , Double(0)
for i in range(0,npar)
   myMinuit.GetParameter(i, p, pe) # retrieve parameters and errors
   finalPar.append(float(p))
   finalParErr.append(float(pe))
# get covariance matrix
buf = arr('d', npar*npar*[0.])
myMinuit . mnemat ( buf , npar ) # retrieve error matrix
emat=np . array ( buf ) . reshape ( npar , npar )

# --> provide formatted output of results
print "\n"
print "*==* MINUIT fit completed:"
print ' fcn@minimum = %.3g' %(amin)," error code =",ierflg," status =",icstat
print " Results: \t value error corr. mat."
for i in range(0,npar):
   print ' %s: \t%10.3e +/- %.1e '%(name[i] ,finalPar[i] ,finalParErr[i]) ,
   for j in range (0,i):
      print '%+.3g ' %(emat[i][j]/np.sqrt(emat[i][i])/np.sqrt(emat[j][j])),
   printf ' '

# --> plot result using matplotlib
plt.figure()
plt.errorbar(ax, ay, yerr=ey, fmt="o", label='data') # the data
x=np.arange(ax[0] ,ax[nPoints-1],abs((ax[nPoints-1]-ax[0]) /100.) )
y=fitfunc(x,npar,finalPar) # function at best fit-point
plt.title("Fit Result")
plt.grid()
plt.plot(x,y, label='fit function')
plt.legend(loc=0)
plt.show()
```

