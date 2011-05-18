//
//   Example of a program to fit non-equidistant data points
//   =======================================================
//
//   The fitting function fcn is a simple chisquare function
//   The data consists of 5 data points (arrays x,y,z) + the errors in errorsz
//   More details on the various functions or parameters for these functions
//   can be obtained in an interactive ROOT session with:
//    Root > TMinuit *minuit = new TMinuit(10);
//    Root > minuit->mnhelp("*")  to see the list of possible keywords
//    Root > minuit->mnhelp("SET") explains most parameters
//Author: Rene Brun

#include "TMinuit.h"

Float_t z[5],x[5],y[5],errorz[5];

//______________________________________________________________________________
Double_t func(float x,float y,Double_t *par)
{
 Double_t value=( (par[0]*par[0])/(x*x)-1)/ ( par[1]+par[2]*y-par[3]*y*y);
 return value;
}

//______________________________________________________________________________
void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
   const Int_t nbins = 5;
   Int_t i;

//calculate chisquare
   Double_t chisq = 0;
   Double_t delta;
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
        Float_t error = 0.01;
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

   Double_t arglist[10];
   Int_t ierflg = 0;

   arglist[0] = 1;
   gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);

// Set starting values and step sizes for parameters
   static Double_t vstart[4] = {3, 1 , 0.1 , 0.01};
   static Double_t step[4] = {0.1 , 0.1 , 0.01 , 0.001};
   gMinuit->mnparm(0, "a1", vstart[0], step[0], 0,0,ierflg);
   gMinuit->mnparm(1, "a2", vstart[1], step[1], 0,0,ierflg);
   gMinuit->mnparm(2, "a3", vstart[2], step[2], 0,0,ierflg);
   gMinuit->mnparm(3, "a4", vstart[3], step[3], 0,0,ierflg);

// Now ready for minimization step
   arglist[0] = 500;
   arglist[1] = 1.;
   gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);

// Print results
   Double_t amin,edm,errdef;
   Int_t nvpar,nparx,icstat;
   gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);
   //gMinuit->mnprin(3,amin);

}

