// @(#)root/test:$name:  $:$id: stressFit.cxx,v 1.15 2002/10/25 10:47:51 rdm exp $
// Authors: Rene Brun, Eddy Offermann  April 2006
   
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
//                                                                           //
// Function Minimization Examples, Fred James                                //
//                                                                           //
// from the                                                                  //
//   Proceedings of the 1972 CERN Computing and Data Processing School       // 
//   Pertisau, Austria, 10-24 September, 1972 (CERN 72-21)                   //
//                                                                           //
// Here a collection of test problems is assembled which were found to be    //
// useful in verifying and comparing minimization routines. Many of these    //
// are standard functions upon which it has become conventional to try all   //
// new methods, quoting the performance in the publication of the algorithm  //
//                                                                           //
// Each test will produce one line (Test OK or Test FAILED) . At the end of  //
// the test a table is printed showing the global results Real Time and      //
// Cpu Time. One single number (ROOTMARKS) is also calculated showing the    //
// relative performance of your machine compared to a reference machine      //
// a Pentium IV 2.4 Ghz) with 512 MBytes of memory and 120 GBytes IDE disk.  //
//                                                                           //
// In the main routine the fitter can be chosen through TVirtualFitter :     //
//   - Minuit                                                                //
//   - Minuit2                                                               //
//   - Fumili                                                                //
//
//  To run the test, do, eg
// root -b -q stressFit.cxx
// root -b -q "stressFit.cxx(\"Minuit2\")"
// root -b -q "stressFit.cxx+(\"Minuit2\")"
//                                                                           //
// The verbosity can be set through the global parameter gVerbose :          //
//   -1: off  1: on                                                          //
// The tolerance on the parameter deviation from the minimum can be set      //
// through gAbsTolerance .                                                   //
//                                                                           //
// An example of output when all the tests run OK is shown below:            //
// *******************************************************************       //
// *  Minimization - S T R E S S suite                               *       //
// *******************************************************************       //
// *******************************************************************       //
// *  Starting  S T R E S S                                          *       //
// *******************************************************************       //
// Test  1 : Wood.................................................. OK       //
// Test  2 : RosenBrock............................................ OK       //
// Test  3 : Powell................................................ OK       //
// Test  4 : Fletcher.............................................. OK       //
// Test  5 : GoldStein1............................................ OK       //
// Test  6 : GoldStein2............................................ OK       //
// Test  7 : TrigoFletcher......................................... OK       //
// *******************************************************************       //
//                                                                           //
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*_*//

#include <stdlib.h>
#include "TVirtualFitter.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TBenchmark.h"
#include "TMath.h"
#include "TStopwatch.h"
#include "Riostream.h"
#include "TVectorD.h"
#include "TMatrixD.h"

Int_t stressFit(const char *theFitter="Minuit", Int_t N=2000);
Int_t    gVerbose      = -1;
Double_t gAbsTolerance = 0.005;

//------------------------------------------------------------------------
void StatusPrint(Int_t id,const TString &title,Bool_t status)
{
  // Print test program number and its title
  const Int_t kMAX = 65;
  Char_t number[4];
  snprintf(number,4,"%2d",id);
  TString header = TString("Test ")+number+" : "+title;
  const Int_t nch = header.Length();
  for (Int_t i = nch; i < kMAX; i++) header += '.';
  cout << header << (status ? "OK" : "FAILED") << endl;
}

//______________________________________________________________________________
void RosenBrock(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
  const Double_t x = par[0];
  const Double_t y = par[1];
  const Double_t tmp1 = y-x*x;
  const Double_t tmp2 = 1-x;
  f = 100*tmp1*tmp1+tmp2*tmp2;
}

//______________________________________________________________________________
Bool_t RunRosenBrock()
{
//
// F(x,y) = 100 (y-x^2)^2 + (1-x)^2
//
//   start point: F(-1.2,1.0) = 24.20
//   minimum    : F(1.0,1.0)  = 0.
//
// This narrow, parabolic valley is probably the best known of all test cases. The floor
// of the valley follows approximately the parabola y = x^2+1/200 .
// There is a region where the covariance matrix is not positive-definite and even a path
// where it is singular . Stepping methods tend to perform at least as well as gradient
//  method for this function .
// [Reference: Comput. J. 3,175 (1960).]

  Bool_t ok = kTRUE;
  TVirtualFitter *min = TVirtualFitter::Fitter(0,2);
  min->SetFCN(RosenBrock);

  Double_t arglist[100];
  arglist[0] = gVerbose;
  min->ExecuteCommand("SET PRINT",arglist,1);

  min->SetParameter(0,"x",-1.2,0.01,0,0);
  min->SetParameter(1,"y", 1.0,0.01,0,0);

  arglist[0] = 0;
  min->ExecuteCommand("SET NOW",arglist,0);
  arglist[0] = 1000; // number of function calls 
  arglist[1] = 0.001; // tolerance 
  min->ExecuteCommand("MIGRAD",arglist,0);
  min->ExecuteCommand("MIGRAD",arglist,2);
  min->ExecuteCommand("MINOS",arglist,0);

  Double_t parx,pary;
  Double_t we,al,bl;
  Char_t parName[32];
  min->GetParameter(0,parName,parx,we,al,bl);
  min->GetParameter(1,parName,pary,we,al,bl);
  
  ok = ( TMath::Abs(parx-1.) < gAbsTolerance &&
         TMath::Abs(pary-1.) < gAbsTolerance );

  delete min;

  return ok;
}

//______________________________________________________________________________
void Wood4(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
  const Double_t w = par[0]; 
  const Double_t x = par[1]; 
  const Double_t y = par[2]; 
  const Double_t z = par[3]; 

  const Double_t w1 = w-1;
  const Double_t x1 = x-1;
  const Double_t y1 = y-1;
  const Double_t z1 = z-1;
  const Double_t tmp1 = x-w*w;
  const Double_t tmp2 = z-y*y;

  f = 100*tmp1*tmp1+w1*w1+90*tmp2*tmp2+y1*y1+10.1*(x1*x1+z1*z1)+19.8*x1*z1;
}

//______________________________________________________________________________
Bool_t RunWood4()
{
//
// F(w,x,y,z) = 100 (y-w^2)^2 + (w-1)^2 + 90 (z-y^2)^2
//              + (1-y)^2 + 10.1 [(x-1)^2 + (z-1)^2]
//              + 19.8 (x-1)(z-1)
//
//   start point: F(-3,-1,-3,-1) = 19192
//   minimum    : F(1,1,1,1)  =   0.
//
// This is a fourth-degree polynomial which is reasonably well-behaved near the minimum,
// but in order to get there one must cross a rather flat, four-dimensional "plateau"
// which often causes minimization algorithm to get "stuck" far from the minimum. As
// such it is a particularly good test of convergence criteria and simulates quite well a
// feature of many physical problems in many variables where no good starting
// approximation is known .
// [Reference: Unpublished. See IBM Technical Report No. 320-2949.]

  Bool_t ok = kTRUE;
  TVirtualFitter *min = TVirtualFitter::Fitter(0,4);
  min->SetFCN(Wood4);

  Double_t arglist[100];
  arglist[0] = gVerbose;
  min->ExecuteCommand("SET PRINT",arglist,1);

  min->SetParameter(0,"w",-3.0,0.01,0,0);
  min->SetParameter(1,"x",-1.0,0.01,0,0);
  min->SetParameter(2,"y",-3.0,0.01,0,0);
  min->SetParameter(3,"z",-1.0,0.01,0,0);

  arglist[0] = 0;
  min->ExecuteCommand("SET NOW",arglist,0);
  arglist[0] = 1000; // number of function calls 
  arglist[1] = 0.001; // tolerance 
  min->ExecuteCommand("MIGRAD",arglist,0);
  min->ExecuteCommand("MIGRAD",arglist,2);
  min->ExecuteCommand("MINOS",arglist,0);

  Double_t parw,parx,pary,parz;
  Double_t we,al,bl;
  Char_t parName[32];
  min->GetParameter(1,parName,parw,we,al,bl);
  min->GetParameter(0,parName,parx,we,al,bl);
  min->GetParameter(1,parName,pary,we,al,bl);
  min->GetParameter(1,parName,parz,we,al,bl);
  
  ok = ( TMath::Abs(parw-1.) < gAbsTolerance &&
         TMath::Abs(parx-1.) < gAbsTolerance &&
         TMath::Abs(pary-1.) < gAbsTolerance &&
         TMath::Abs(parz-1.) < gAbsTolerance );

  delete min;

  return ok;
}

//______________________________________________________________________________
void Powell(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
  const Double_t w = par[0]; 
  const Double_t x = par[1]; 
  const Double_t y = par[2]; 
  const Double_t z = par[3]; 

  const Double_t tmp1 = w+10*x;
  const Double_t tmp2 = y-z;
  const Double_t tmp3 = x-2*y;
  const Double_t tmp4 = w-z;

  f = tmp1*tmp1+5*tmp2*tmp2+tmp3*tmp3*tmp3*tmp3+10*tmp4*tmp4*tmp4*tmp4;
}

//______________________________________________________________________________
Bool_t RunPowell()
{
//
// F(w,x,y,z) = (w+10x)^2 + 5(y-z)^2 + (x-2y)^4+ 10 (w-z)^4
//
//   start point: F(-3,-1,0,1) = 215
//   minimum    : F(0,0,0,0)  =   0.
//
// This function is difficult because its matrix of second derivatives becomes singular
//  at the minimum. Near the minimum the function is given by (w + 10x)^2 + 5 (y-5)^2
// which does not determine the minimum uniquely.
// [Reference: Comput. J. 5, 147 (1962).]

  Bool_t ok = kTRUE;
  TVirtualFitter *min = TVirtualFitter::Fitter(0,4);
  min->SetFCN(Powell);
  
  Double_t arglist[100];
  arglist[0] = gVerbose;
  min->ExecuteCommand("SET PRINT",arglist,1);
  
  min->SetParameter(0,"w",+3.0,0.01,0,0);
  min->SetParameter(1,"x",-1.0,0.01,0,0);
  min->SetParameter(2,"y", 0.0,0.01,0,0);
  min->SetParameter(3,"z",+1.0,0.01,0,0);
    
  arglist[0] = 0;
  min->ExecuteCommand("SET NOW",arglist,0);
  arglist[0] = 1000; // number of function calls 
  arglist[1] = 0.001; // tolerance 
  min->ExecuteCommand("MIGRAD",arglist,0);
  min->ExecuteCommand("MIGRAD",arglist,2);
  min->ExecuteCommand("MINOS",arglist,0);
    
  Double_t parw,parx,pary,parz;
  Double_t we,al,bl;
  Char_t parName[32];
  min->GetParameter(1,parName,parw,we,al,bl);
  min->GetParameter(0,parName,parx,we,al,bl);
  min->GetParameter(1,parName,pary,we,al,bl);
  min->GetParameter(1,parName,parz,we,al,bl);
  
  ok = ( TMath::Abs(parw-0.) < gAbsTolerance &&
         TMath::Abs(parx-0.) < 10.*gAbsTolerance &&
         TMath::Abs(pary-0.) < gAbsTolerance &&
         TMath::Abs(parz-0.) < gAbsTolerance );

  delete min;

  return ok;
}

//______________________________________________________________________________
void Fletcher(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
  const Double_t x = par[0];
  const Double_t y = par[1];
  const Double_t z = par[2];

  Double_t psi;
  if (x > 0)
    psi = TMath::ATan(y/x)/2/TMath::Pi();
  else if (x < 0)
    psi = 0.5+TMath::ATan(y/x)/2/TMath::Pi();
  else
    psi = 0.0;

  const Double_t tmp1 = z-10*psi;
  const Double_t tmp2 = TMath::Sqrt(x*x+y*y)-1;

  f = 100*(tmp1*tmp1+tmp2*tmp2)+z*z;
}

//______________________________________________________________________________
Bool_t RunFletcher()
{
//
// F(x,y,z) = 100 {[z - 10 G(x,y)]^2 + ( (x^2+y^2)^1/2 - 1 )^2} + z^2
//
//                     | arctan(y/x)        for x > 0
// where 2 pi G(x,y) = |
//                     | pi + arctan(y/x)   for x < 0
//
//   start point: F(-1,0,0) = 2500
//   minimum    : F(1,0,0)  =   0.
//
// F is defined only for -0.25 < G(x,y) < 0.75
//
// This is a curved valley problem, similar to Rosenbrock's, but in three dimensions .
// [Reference: Comput. J. 6, 163 (1963).]

  Bool_t ok = kTRUE;
  TVirtualFitter *min = TVirtualFitter::Fitter(0,3);
  min->SetFCN(Fletcher);

  Double_t arglist[100];
  arglist[0] = gVerbose;
  min->ExecuteCommand("SET PRINT",arglist,1);

  min->SetParameter(0,"x",-1.0,0.01,0,0);
  min->SetParameter(1,"y", 0.0,0.01,0,0);
  min->SetParameter(2,"z", 0.0,0.01,0,0);

  arglist[0] = 0;
  min->ExecuteCommand("SET NOW",arglist,0);
  arglist[0] = 1000; // number of function calls 
  arglist[1] = 0.001; // tolerance 
  min->ExecuteCommand("MIGRAD",arglist,0);
  min->ExecuteCommand("MIGRAD",arglist,2);
  min->ExecuteCommand("MINOS",arglist,0);

  Double_t parx,pary,parz;
  Double_t we,al,bl;
  Char_t parName[32];
  min->GetParameter(0,parName,parx,we,al,bl);
  min->GetParameter(1,parName,pary,we,al,bl);
  min->GetParameter(1,parName,parz,we,al,bl);
  
  ok = ( TMath::Abs(parx-1.) < gAbsTolerance &&
         TMath::Abs(pary-0.) < gAbsTolerance &&
         TMath::Abs(parz-0.) < gAbsTolerance );

  delete min;

  return ok;
}

//______________________________________________________________________________
void GoldStein1(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
  const Double_t x = par[0];
  const Double_t y = par[1];

  const Double_t tmp1 = x+y+1;
  const Double_t tmp2 = 19-14*x+3*x*x-14*y+6*x*y+3*y*y;
  const Double_t tmp3 = 2*x-3*y;
  const Double_t tmp4 = 18-32*x+12*x*x+48*y-36*x*y+27*y*y;

  f = (1+tmp1*tmp1*tmp2)*(30+tmp3*tmp3*tmp4);
}

//______________________________________________________________________________
Bool_t RunGoldStein1()
{
//
// F(x,y) = (1 + (x+y+1)^2 * (19-14x+3x^2-14y+6xy+3y^2))
//           * (30 + (2x-3y)^2 * (18-32x+12x^2+48y-36xy+27y^2))
// 
//   start point     : F(-0.4,-0,6) = 35
//   local  minima   : F(1.2,0.8)   = 840
//                     F(1.8,0.2)   = 84
//                     F(-0.6,-0.4) = 30
//   global minimum  : F(0.0,-1.0)  = 3
//   
// This is an eighth-order polynomial in two variables which is well behaved near each
// minimum, but has four local minima and is of course non-positive-definite in many
// regions. The saddle point between the two lowest minima occurs at F(-0.4,-0.6)=35
// making this an interesting start point .
// [Reference: Math. Comp. 25, 571 (1971).]

  Bool_t ok = kTRUE;
  TVirtualFitter *min = TVirtualFitter::Fitter(0,2);
  min->SetFCN(GoldStein1);

  Double_t arglist[100];
  arglist[0] = gVerbose;
  min->ExecuteCommand("SET PRINT",arglist,1);
  
  min->SetParameter(0,"x",-0.3999,0.01,-2.0,+2.0);
  min->SetParameter(1,"y",-0.6,0.01,-2.0,+2.0);
  
  arglist[0] = 0;
  min->ExecuteCommand("SET NOW",arglist,0);
  arglist[0] = 1000; // number of function calls 
  arglist[1] = 0.001; // tolerance 
  min->ExecuteCommand("MIGRAD",arglist,0);
  min->ExecuteCommand("MIGRAD",arglist,2);
  min->ExecuteCommand("MIGRAD",arglist,2);
  min->ExecuteCommand("MINOS",arglist,0);

  Double_t parx,pary;
  Double_t we,al,bl;
  Char_t parName[32];
  min->GetParameter(0,parName,parx,we,al,bl);
  min->GetParameter(1,parName,pary,we,al,bl);

  ok = ( TMath::Abs(parx-0.) < gAbsTolerance &&
         TMath::Abs(pary+1.) < gAbsTolerance );
  
  delete min;

  return ok;
}

//______________________________________________________________________________
void GoldStein2(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
  const Double_t x = par[0];
  const Double_t y = par[1];

  const Double_t tmp1 = x*x+y*y-25;
  const Double_t tmp2 = TMath::Sin(4*x-3*y);
  const Double_t tmp3 = 2*x+y-10;

  f = TMath::Exp(0.5*tmp1*tmp1)+tmp2*tmp2*tmp2*tmp2+0.5*tmp3*tmp3;
}

//______________________________________________________________________________
Bool_t RunGoldStein2()
{
//
// F(x,y) = (1 + (x+y+1)^2 * (19-14x+3x^2-14y+6xy+3y^2))
//           * (30 + (2x-3y)^2 * (18-32x+12x^2+48y-36xy+27y^2))
// 
//   start point     : F(1.6,3.4) =
//   global minimum  : F(3,4)     = 1
//   
// This function has many local minima .
// [Reference: Math. Comp. 25, 571 (1971).]

  Bool_t ok = kTRUE;
  TVirtualFitter *min = TVirtualFitter::Fitter(0,2);
  min->SetFCN(GoldStein2);

  Double_t arglist[100];
  arglist[0] = gVerbose;
  min->ExecuteCommand("SET PRINT",arglist,1);

  min->SetParameter(0,"x",+1.0,0.01,-5.0,+5.0);
  min->SetParameter(1,"y",+3.2,0.01,-5.0,+5.0);

  arglist[0] = 0;
  min->ExecuteCommand("SET NOW",arglist,0);
  arglist[0] = 1000; // number of function calls 
  arglist[1] = 0.01; // tolerance 
  min->ExecuteCommand("MIGRAD",arglist,0);
  min->ExecuteCommand("MIGRAD",arglist,2);
  min->ExecuteCommand("MINOS",arglist,0);

  Double_t parx,pary;
  Double_t we,al,bl;
  Char_t parName[32];
  min->GetParameter(0,parName,parx,we,al,bl);
  min->GetParameter(1,parName,pary,we,al,bl);

  ok = ( TMath::Abs(parx-3.) < gAbsTolerance &&
         TMath::Abs(pary-4.) < gAbsTolerance );

  delete min;

  return ok;
}

Double_t seed = 3;
Int_t  nf;
TMatrixD A;
TMatrixD B;
TVectorD x0;
TVectorD sx0;
TVectorD cx0;
TVectorD sx;
TVectorD cx;
TVectorD v0;
TVectorD v;
TVectorD r;

//______________________________________________________________________________
void TrigoFletcher(Int_t &, Double_t *, Double_t &f, Double_t *par, Int_t /*iflag*/)
{
  Int_t i;
  for (i = 0; i < nf ; i++) {
    cx0[i] = TMath::Cos(x0[i]);
    sx0[i] = TMath::Sin(x0[i]);
    cx [i] = TMath::Cos(par[i]);
    sx [i] = TMath::Sin(par[i]);
  }

  v0 = A*sx0+B*cx0;
  v  = A*sx +B*cx;
  r  = v0-v;
 
  f = r * r;
}

//______________________________________________________________________________
Bool_t RunTrigoFletcher()
{
//
// F(\vec{x}) = \sum_{i=1}^n ( E_i - \sum_{j=1}^n (A_{ij} \sin x_j + B_{ij} \cos x_j) )^2
// 
//   where E_i = \sum_{j=1}^n ( A_{ij} \sin x_{0j} + B_{ij} \cos x_{0j} )
//
//   B_{ij} and A_{ij} are random matrices composed of integers between -100 and 100;
//   for j = 1,...,n: x_{0j} are any random numbers, -\pi < x_{0j} < \pi;
//
//   start point : x_j = x_{0j} + 0.1 \delta_j,  -\pi < \delta_j < \pi
//   minimum     : F(\vec{x} = \vec{x}_0) = 0
//   
// This is a set of functions of any number of variables n, where the minimum is always
// known in advance, but where the problem can be changed by choosing different
// (random) values of the constants A_{ij}, B_{ij}, and x_{0j} . The difficulty can be
// varied by choosing larger starting deviations \delta_j . In practice, most methods
// find the "right" minimum, corresponding to \vec{x} = \vec{x}_0, but there are usually
// many subsidiary minima.
// [Reference: Comput. J. 6 163 (1963).]

 
  const Double_t pi = TMath::Pi();
  Bool_t ok = kTRUE;
  Double_t delta = 0.1;
  
  for (nf = 5; nf<32;nf +=5) {
     TVirtualFitter *min = TVirtualFitter::Fitter(0,nf);
     min->SetFCN(TrigoFletcher);
     
     Double_t arglist[100];
     arglist[0] = gVerbose;
     min->ExecuteCommand("SET PRINT",arglist,1);
     A.ResizeTo(nf,nf);
     B.ResizeTo(nf,nf);
     x0.ResizeTo(nf);
     sx0.ResizeTo(nf);
     cx0.ResizeTo(nf);
     sx.ResizeTo(nf);
     cx.ResizeTo(nf);
     v0.ResizeTo(nf);
     v.ResizeTo(nf);
     r.ResizeTo(nf);
     A.Randomize(-100.,100,seed);
     B.Randomize(-100.,100,seed);
     for (Int_t i = 0; i < nf; i++) {
       for (Int_t j = 0; j < nf; j++) {
         A(i,j) = Int_t(A(i,j));
         B(i,j) = Int_t(B(i,j));
       }
     }

     x0.Randomize(-pi,pi,seed);
     TVectorD x1(nf); x1.Randomize(-delta*pi,delta*pi,seed);
     x1+= x0;

     for (Int_t i = 0; i < nf; i++)
       min->SetParameter(i,Form("x_%d",i),x1[i],0.01,-pi*(1+delta),+pi*(1+delta));

     arglist[0] = 0;
     min->ExecuteCommand("SET NOW",arglist,0);
     arglist[0] = 1000; // number of function calls 
     arglist[1] = 0.01; // tolerance 
     min->ExecuteCommand("MIGRAD",arglist,0);
     min->ExecuteCommand("MIGRAD",arglist,2);
     min->ExecuteCommand("MINOS",arglist,0);

     Double_t par,we,al,bl;
     Char_t parName[32];
     for (Int_t i = 0; i < nf; i++) {
       min->GetParameter(i,parName,par,we,al,bl);
       ok = ok && ( TMath::Abs(par) -TMath::Abs(x0[i]) < gAbsTolerance );
       if (!ok) printf("nf=%d, i=%d, par=%g, x0=%g\n",nf,i,par,x0[i]);
     }
     delete min;
  }  

  return ok;
}

//______________________________________________________________________________
Int_t stressFit(const char *theFitter, Int_t N)
{
  TVirtualFitter::SetDefaultFitter(theFitter);
  
  cout << "******************************************************************" <<endl;
  cout << "*  Minimization - S T R E S S suite                              *" <<endl;
  cout << "******************************************************************" <<endl;
  cout << "******************************************************************" <<endl;

   TStopwatch timer;
   timer.Start();

  cout << "*  Starting  S T R E S S  with fitter : "<<TVirtualFitter::GetDefaultFitter() <<endl;
  cout << "******************************************************************" <<endl;

  gBenchmark->Start("stressFit");

  Bool_t okRosenBrock    = kTRUE;
  Bool_t okWood          = kTRUE;
  Bool_t okPowell        = kTRUE;
  Bool_t okFletcher      = kTRUE;
  Bool_t okGoldStein1    = kTRUE;
  Bool_t okGoldStein2    = kTRUE;
  Bool_t okTrigoFletcher = kTRUE;
  Int_t i;
  for (i = 0; i < N; i++)  okWood          = RunWood4();
  StatusPrint(1,"Wood",okWood);
  for (i = 0; i < N; i++) okRosenBrock    = RunRosenBrock();
  StatusPrint(2,"RosenBrock",okRosenBrock);
  for (i = 0; i < N; i++) okPowell        = RunPowell();
  StatusPrint(3,"Powell",okPowell);
  for (i = 0; i < N; i++) okFletcher      = RunFletcher();
  StatusPrint(4,"Fletcher",okFletcher);
  for (i = 0; i < N; i++) okGoldStein1    = RunGoldStein1();
  StatusPrint(5,"GoldStein1",okGoldStein1);
  for (i = 0; i < N; i++) okGoldStein2    = RunGoldStein2();
  StatusPrint(6,"GoldStein2",okGoldStein2);
  okTrigoFletcher = RunTrigoFletcher();
  StatusPrint(7,"TrigoFletcher",okTrigoFletcher);

  gBenchmark->Stop("stressFit");


  //Print table with results
  Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
  printf("******************************************************************\n");
  if (UNIX) {
     TString sp = gSystem->GetFromPipe("uname -a");
     sp.Resize(60);
     printf("*  SYS: %s\n",sp.Data());
     if (strstr(gSystem->GetBuildNode(),"Linux")) {
        sp = gSystem->GetFromPipe("lsb_release -d -s");
        printf("*  SYS: %s\n",sp.Data());
     }
     if (strstr(gSystem->GetBuildNode(),"Darwin")) {
        sp  = gSystem->GetFromPipe("sw_vers -productVersion");
        sp += " Mac OS X ";
        printf("*  SYS: %s\n",sp.Data());
     }
  } else {
    const Char_t *os = gSystem->Getenv("OS");
    if (!os) printf("*  SYS: Windows 95\n");
    else     printf("*  SYS: %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
  }
  
  printf("******************************************************************\n");
  gBenchmark->Print("stressFit");
#ifdef __CINT__
  Double_t reftime = 86.34; //macbrun interpreted
#else
  Double_t reftime = 12.07; //macbrun compiled
#endif
  const Double_t rootmarks = 800.*reftime/gBenchmark->GetCpuTime("stressFit");
  
  printf("******************************************************************\n");
  printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),
         gROOT->GetVersionDate(),gROOT->GetVersionTime());
  printf("******************************************************************\n");

   return 0;
}

//_____________________________batch only_____________________
#ifndef __CINT__

int main(int argc,const char *argv[]) 
{
  gBenchmark = new TBenchmark();
  const char *fitter = "Minuit";
  if (argc > 1)  fitter = argv[1];
  if (strcmp(fitter,"Minuit") && strcmp(fitter,"Minuit2") && strcmp(fitter,"Fumili")) {
     printf("stressFit illegal option %s, using Minuit instead\n",fitter);
     fitter = "Minuit";
  }
  Int_t N = 2000;
  if (argc > 2) N = atoi(argv[2]);
  stressFit(fitter,N);  //default is Minuit
  return 0;
}

#endif
