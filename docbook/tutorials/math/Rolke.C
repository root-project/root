// Example of the usage of the TRolke class 
#include "TROOT.h"
#include "TSystem.h"
#include "TRolke.h"
#include "Riostream.h"
      
void Rolke()
{
//////////////////////////////////////////////////////////
//
// The TRolke class computes the profile likelihood
// confidence limits for 7 different model assumptions
// on systematic/statistical uncertainties
//
// Author : Jan Conrad (CERN) <jan.conrad@cern.ch> 2004
//          Johan Lundberg (CERN) <johan.lundberg@cern.ch> 2009
//  
// Please read TRolke.cxx and TRolke.h for more docs.
//             ----------     --------
//
//////////////////////////////////////////////////////

   gSystem->Load("libPhysics.so");
   gSystem->Load("libTRolke.so");

   /* variables used throughout the example */
   Double_t bm;
   Double_t tau;
   Int_t mid;
   Int_t m;
   Int_t z;
   Int_t y;
   Int_t x;
   Double_t e;
   Double_t em;
   Double_t sde;
   Double_t sdb;
   Double_t b;

   Double_t alpha; //Confidence Level

   // make TRolke objects
   TRolke tr;   //

   Double_t ul ; // upper limit 
   Double_t ll ; // lower limit


/////////////////////////////////////////////////////////////
// Model 1 assumes:
//
// Poisson uncertainty in the background estimate
// Binomial uncertainty in the efficiency estimate
//
   cout << endl<<" ======================================================== " <<endl;
   mid =1;
   x = 5;     // events in the signal region
   y = 10;    // events observed in the background region
   tau = 2.5; // ratio between size of signal/background region
   m = 100;   // MC events have been produced  (signal)
   z = 50;    // MC events have been observed (signal)          

   alpha=0.9; //Confidence Level

   tr.SetCL(alpha);  

   tr.SetPoissonBkgBinomEff(x,y,z,tau,m); 
   tr.GetLimits(ll,ul);
 
   cout << "For model 1: Poisson / Binomial" << endl; 
   cout << "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;

 
/////////////////////////////////////////////////////////////
// Model 2 assumes:
//
// Poisson uncertainty in the background estimate
// Gaussian  uncertainty in the efficiency estimate
//
   cout << endl<<" ======================================================== " <<endl;
   mid =2;
   y = 3 ;      // events observed in the background region
   x = 10 ;     // events in the signal region
   tau = 2.5;   // ratio between size of signal/background region
   em = 0.9;    // measured efficiency
   sde = 0.05;  // standard deviation of efficiency
   alpha =0.95; // Confidence L evel

   tr.SetCL(alpha);

   tr.SetPoissonBkgGaussEff(x,y,em,tau,sde);
   tr.GetLimits(ll,ul);
 
   cout << "For model 2 : Poisson / Gaussian" << endl; 
   cout << "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;

  

/////////////////////////////////////////////////////////////
// Model 3 assumes:
//
// Gaussian uncertainty in the background estimate
// Gaussian  uncertainty in the efficiency estimate
//
   cout << endl<<" ======================================================== " <<endl;
   mid =3;
   bm = 5;      // expected background
   x = 10;      // events in the signal region
   sdb = 0.5;   // standard deviation in background estimate
   em = 0.9;    //  measured efficiency
   sde = 0.05;  // standard deviation of efficiency
   alpha =0.99; // Confidence Level

   tr.SetCL(alpha);

   tr.SetGaussBkgGaussEff(x,bm,em,sde,sdb); 
   tr.GetLimits(ll,ul);
   cout << "For model 3 : Gaussian / Gaussian" << endl; 
   cout << "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;


 
   cout << "***************************************" << endl;
   cout << "* some more example's for gauss/gauss *" << endl;
   cout << "*                                     *" << endl;
   Double_t slow,shigh;
   tr.GetSensitivity(slow,shigh);
   cout << "sensitivity:" << endl;
   cout << "[" << slow << "," << shigh << "]" << endl; 

   int outx;
   tr.GetLimitsQuantile(slow,shigh,outx,0.5);
   cout << "median limit:" << endl;
   cout << "[" << slow << "," << shigh << "] @ x =" << outx <<endl; 

   tr.GetLimitsML(slow,shigh,outx);
   cout << "ML limit:" << endl;
   cout << "[" << slow << "," << shigh << "] @ x =" << outx <<endl; 

   tr.GetSensitivity(slow,shigh);
   cout << "sensitivity:" << endl;
   cout << "[" << slow << "," << shigh << "]" << endl; 

   tr.GetLimits(ll,ul);
   cout << "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;

   Int_t ncrt;

   tr.GetCriticalNumber(ncrt);
   cout << "critical number: " << ncrt << endl;

   tr.SetCLSigmas(5);
   tr.GetCriticalNumber(ncrt);
   cout << "critical number for 5 sigma: " << ncrt << endl;

   cout << "***************************************" << endl;


/////////////////////////////////////////////////////////////
// Model 4 assumes:
//
// Poisson uncertainty in the background estimate
// known efficiency
//
   cout << endl<<" ======================================================== " <<endl;
   mid =4;
   y = 7;       // events observed in the background region
   x = 1;       // events in the signal region
   tau = 5;     // ratio between size of signal/background region
   e = 0.25;    // efficiency 

   alpha =0.68; // Confidence L evel

   tr.SetCL(alpha);

   tr.SetPoissonBkgKnownEff(x,y,tau,e);
   tr.GetLimits(ll,ul);
 
   cout << "For model 4 : Poissonian / Known" << endl; 
   cout <<  "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;

   
////////////////////////////////////////////////////////
// Model 5 assumes:
//
// Gaussian uncertainty in the background estimate
// Known efficiency
//
   cout << endl<<" ======================================================== " <<endl;
   mid =5;
   bm = 0;          // measured background expectation
   x = 1 ;          // events in the signal region
   e = 0.65;        // known eff
   sdb = 1.0;       // standard deviation of background estimate
   alpha =0.799999; // Confidence Level

   tr.SetCL(alpha);

   tr.SetGaussBkgKnownEff(x,bm,sdb,e);
   tr.GetLimits(ll,ul);
 
   cout << "For model 5 : Gaussian / Known" << endl; 
   cout <<  "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;

 

////////////////////////////////////////////////////////
// Model 6 assumes:
//
// Known background 
// Binomial uncertainty in the efficiency estimate
//
   cout << endl<<" ======================================================== " <<endl;
   mid =6;
   b = 10;      // known background
   x = 25;      // events in the signal region
   z = 500;     // Number of observed signal MC events
   m = 750;     // Number of produced MC signal events
   alpha =0.9;  // Confidence L evel

   tr.SetCL(alpha);

   tr.SetKnownBkgBinomEff(x, z,m,b);
   tr.GetLimits(ll,ul);
 
   cout << "For model 6 : Known / Binomial" << endl; 
   cout <<  "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;

  
////////////////////////////////////////////////////////
// Model 7 assumes:
//
// Known Background
// Gaussian  uncertainty in the efficiency estimate
//
   cout << endl<<" ======================================================== " <<endl;
   mid =7;
   x = 15;      // events in the signal region
   em = 0.77;   // measured efficiency
   sde = 0.15;  // standard deviation of efficiency estimate
   b = 10;      // known background
   alpha =0.95; // Confidence L evel

   y = 1;

   tr.SetCL(alpha);

   tr.SetKnownBkgGaussEff(x,em,sde,b);
   tr.GetLimits(ll,ul);
  
   cout << "For model 7 : Known / Gaussian " << endl; 
   cout <<  "the Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;


////////////////////////////////////////////////////////
// Example of bounded and unbounded likelihood
// Example for Model 1
///////////////////////////////////////////////////////

   bm = 0.0;
   tau = 5;
   mid = 1;
   m = 100;
   z = 90;
   y = 15;
   x = 0;
   alpha = 0.90;
   
   tr.SetCL(alpha);
   tr.SetPoissonBkgBinomEff(x,y,z,tau,m); 
   tr.SetBounding(true); //bounded
   tr.GetLimits(ll,ul);   
   
   cout << "Example of the effect of bounded vs unbounded, For model 1" << endl; 
   cout <<  "the BOUNDED Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;


   tr.SetBounding(false); //unbounded
   tr.GetLimits(ll,ul);   
   
   cout <<  "the UNBOUNDED Profile Likelihood interval is :" << endl;
   cout << "[" << ll << "," << ul << "]" << endl;
  
}

