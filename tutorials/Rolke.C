#include "TROOT.h"
#include "TSystem.h"
#include "TRolke.h"
#include "Riostream.h"
   
   
void Rolke()
{
//////////////////////////////////////////////////
// Routine computes the profile likelihood confidence
// limits for  7 different model assumptions
// on systematic/statistical uncertainties
//
// You must load libPhysics before executing this script.
//
// Author : Jan Conrad (CERN) <Jan.Conrad@cern.ch>
//////////////////////////////////////////////////



/////////////////////////////////////////////////////////////
// Model 1 assumes:
//
// Poisson uncertainty in the background estimate
// Binomial uncertainty in the efficiency estimate
//
// y = 10      events observed in the background region
// x = 5       events in the signal region
// tau = 2.5   ratio between size of signal/background region
// m = 100     MC events have been produced  (signal)
// z = 50      MC events have been observed (signal)
// alpha = 0.9 Confidence Level
//////////////////////////////////////////////////////////////

    //gSystem->Load("libPhysics");
 Double_t bm = 0.0;
 Double_t tau = 2.5;
 Int_t mid = 1;
 Int_t m = 100;
 Int_t z = 50;
 Int_t y = 10;
 Int_t x = 5;
 // Initialize parameters not used.
 Double_t e = 0.0;
 Double_t em = 0.0;
 Double_t sde=0.0;
 Double_t sdb=0.0;
 Double_t b = 0.0;


 TRolke g;
 
 g.SetCL(0.90);
 
 Double_t ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
 Double_t ll = g.GetLowerLimit();
 
 cout << "Assuming Model 1" << endl; 
 cout <<  "the Profile Likelihood interval is :" << endl;
 cout << "[" << ll << "," << ul << "]" << endl;

 
/////////////////////////////////////////////////////////////
// Model 2 assumes:
//
// Poisson uncertainty in the background estimate
// Gaussian  uncertainty in the efficiency estimate
//
// y = 3        events observed in the background region
// x = 10       events in the signal region
// tau = 2.5    ratio between size of signal/background region
// em = 0.9     measured efficiency
// sde = 0.05   standard deviation of efficiency
// alpha =0.95  Confidence L evel
//////////////////////////////////////////////////////////////


 tau = 2.5;
 mid = 2;
 y = 3;
 x = 10;
 em=0.9;
 sde=0.05;

 g.SetCL(0.95);

 ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
 ll = g.GetLowerLimit();
 
  cout << "Assuming MODEL 2" << endl; 
  cout <<  "the Profile Likelihood interval is :" << endl;
  cout << "[" << ll << "," << ul << "]" << endl;
  

/////////////////////////////////////////////////////////////
// Model 3 assumes:
//
// Gaussian uncertainty in the background estimate
// Gaussian  uncertainty in the efficiency estimate
//
// bm = 5       expected background
// x = 10       events in the signal region
// sdb = 0.5    standard deviation in background estimate
// em = 0.9     measured efficiency
// sde = 0.05   standard deviation of efficiency
// alpha =0.99 Confidence Level
//////////////////////////////////////////////////////////////



 mid = 3;
 bm = 5.0;
 x = 10;
 em = 0.9;
 sde=0.05;
 sdb=0.5;

 g.SetCL(0.99);


 ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
 ll = g.GetLowerLimit();
 
cout << "Assuming Model 3" << endl; 
cout <<  "the Profile Likelihood interval is :" << endl;
cout << "[" << ll << "," << ul << "]" << endl;





/////////////////////////////////////////////////////////////
// Model 4 assumes:
//
// Poisson uncertainty in the background estimate
// known efficiency
//
// y = 7       events observed in the background region
// x = 1       events in the signal region
// tau = 5     ratio between size of signal/background region
//
// alpha =0.68  Confidence L evel
//////////////////////////////////////////////////////////////


 tau = 5;
 mid = 4;
 y = 7;
 x = 1;
 e = 0.25;


 g.SetCL(0.68);

 ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
 ll = g.GetLowerLimit();
 
  cout << "Assuming Model 4" << endl; 
    cout <<  "the Profile Likelihood interval is :" << endl;
  cout << "[" << ll << "," << ul << "]" << endl;





////////////////////////////////////////////////////////
// Model 5 assumes:
//
// Gaussian uncertainty in the background estimate
// Known efficiency
//
// bm = 0           measured background expectation
// x = 1            events in the signal region
// e = 0.65
// sdb = 1.         standard deviation of background estimate
// alpha =0.799999  Confidence Level
///////////////////////////////////////////////////////


 mid = 5;
 bm = 0.0;
 x = 1;
 e = 0.65;
 sdb=1.0;

 g.SetCL(0.80);

 ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
 ll = g.GetLowerLimit();
 
  cout << "Assuming Model 5" << endl; 
  cout <<  "the Profile Likelihood interval is :" << endl;
  cout << "[" << ll << "," << ul << "]" << endl;
  

////////////////////////////////////////////////////////
// Model 6 assumes:
//
// Known background 
// Binomial uncertainty in the efficiency estimate
//
// b = 10       known background
// x = 25       events in the signal region
// z = 500      Number of observed signal MC events
// m = 750      Number of produced MC signal events
// alpha =0.9   Confidence L evel
///////////////////////////////////////////////////////

 y = 1;
 mid = 6;
 m = 750;
 z = 500;
 x = 25;
 b = 10.0;

 TRolke p; 
 p.SetCL(0.90);
 ul = p.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
 Double_t newll = p.GetLowerLimit();
  
  cout << "Assuming Model 6" << endl; 
  cout <<  "the Profile Likelihood interval is :" << endl;
  cout << "[" << newll << "," << ul << "]" << endl;
  


////////////////////////////////////////////////////////
// Model 7 assumes:
//
// Known Background
// Gaussian  uncertainty in the efficiency estimate
//
// x = 15       events in the signal region
// em = 0.77    measured efficiency
// sde = 0.15   standard deviation of efficiency estimate
// b = 10       known background
// alpha =0.95  Confidence L evel
///////////////////////////////////////////////////////


 mid = 7;
 x = 15;
 em = 0.77;
 sde=0.15;
 b = 10.0;

 g.SetCL(0.95);
 
  ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
  ll = g.GetLowerLimit();
  
  cout << "Assuming Model 7 " << endl; 
  cout <<  "the Profile Likelihood interval is :" << endl;
  cout << "[" << ll << "," << ul << "]" << endl;


////////////////////////////////////////////////////////
// Example of bounded and unbounded likelihood
//
// Example for Model 1
///////////////////////////////////////////////////////


 Double_t bm = 0.0;
 Double_t tau = 5;
 Int_t mid = 1;
 Int_t m = 100;
 Int_t z = 90;
 Int_t y = 15;
 Int_t x = 0;
 // Initialize parameters not used.
 Double_t e = 0.0;
 Double_t em = 0.0;
 Double_t sde=0.0;
 Double_t sdb=0.0;
 Double_t b = 0.0;


 TRolke g;
 
 g.SetCL(0.90);
 g.SetSwitch(1); //bounded
 
 Double_t ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
 Double_t ll = g.GetLowerLimit();

 g.SetSwitch(0); //unbounded

 cout << "Assuming Model 1" << endl; 
 cout <<  "the BOUNDED Profile Likelihood interval is :" << endl;
 cout << "[" << ll << "," << ul << "]" << endl;


  ul = g.CalculateInterval(x,y,z,bm,em,e,mid,sde,sdb,tau,b,m);
  ll = g.GetLowerLimit();

 cout << "Assuming Model 1" << endl; 
 cout <<  "the UNBOUNDED Profile Likelihood interval is :" << endl;
 cout << "[" << ll << "," << ul << "]" << endl;

}

