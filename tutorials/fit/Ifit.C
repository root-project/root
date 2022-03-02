/// \file
/// \ingroup tutorial_fit
/// \notebook -nodraw
///   Example of a program to fit non-equidistant data points
///
///   The fitting function fcn is a simple chisquare function
///   The data consists of 5 data points (arrays x,y,z) + the errors in errorsz
///   More details on the various functions or parameters for these functions
///   can be obtained in an interactive ROOT session with:
///
/// ~~~{.cpp}
///    Root > TMinuit *minuit = new TMinuit(10);
/// ~~~
///
/// ~~~{.cpp}
///    Root > minuit->mnhelp("*")  to see the list of possible keywords
///    Root > minuit->mnhelp("SET") explains most parameters
/// ~~~
///
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

#include "TMinuit.h"

float z[5],x[5],y[5],errorz[5];

//______________________________________________________________________________
double func(float x,float y,double *par)
{
 double value=( (par[0]*par[0])/(x*x)-1)/ ( par[1]+par[2]*y-par[3]*y*y);
 return value;
}

//______________________________________________________________________________
void fcn(int &npar, double *gin, double &f, double *par, int iflag)
{
   const int nbins = 5;
   int i;

//calculate chisquare
   double chisq = 0;
   double delta;
   for (i=0;i<nbins; i++) {
     delta  = (z[i]-func(x[i],y[i],par))/errorz[i];
     chisq += delta*delta;
   }
   f = chisq;
}

//______________________________________________________________________________
void Ifit()
{
// The z values
   z[0]=1;
   z[1]=0.96;
   z[2]=0.89;
   z[3]=0.85;
   z[4]=0.78;
// The errors on z values
        float error = 0.01;
   errorz[0]=error;
   errorz[1]=error;
   errorz[2]=error;
   errorz[3]=error;
   errorz[4]=error;
// the x values
   x[0]=1.5751;
   x[1]=1.5825;
   x[2]=1.6069;
   x[3]=1.6339;
   x[4]=1.6706;
// the y values
   y[0]=1.0642;
   y[1]=0.97685;
   y[2]=1.13168;
   y[3]=1.128654;
   y[4]=1.44016;

   TMinuit *gMinuit = new TMinuit(5);  //initialize TMinuit with a maximum of 5 params
   gMinuit->SetFCN(fcn);

   double arglist[10];
   int ierflg = 0;

   arglist[0] = 1;
   gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);

// Set starting values and step sizes for parameters
   static double vstart[4] = {3, 1 , 0.1 , 0.01};
   static double step[4] = {0.1 , 0.1 , 0.01 , 0.001};
   gMinuit->mnparm(0, "a1", vstart[0], step[0], 0,0,ierflg);
   gMinuit->mnparm(1, "a2", vstart[1], step[1], 0,0,ierflg);
   gMinuit->mnparm(2, "a3", vstart[2], step[2], 0,0,ierflg);
   gMinuit->mnparm(3, "a4", vstart[3], step[3], 0,0,ierflg);

// Now ready for minimization step
   arglist[0] = 500;
   arglist[1] = 1.;
   gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);

// Print results
   double amin,edm,errdef;
   int nvpar,nparx,icstat;
   gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
   //gMinuit->mnprin(3,amin);

}

