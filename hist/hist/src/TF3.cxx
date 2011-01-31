// @(#)root/hist:$Id$
// Author: Rene Brun   27/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TF3.h"
#include "TMath.h"
#include "TH3.h"
#include "TVirtualPad.h"
#include "TRandom.h"
#include "TVectorD.h"
#include "Riostream.h"
#include "TColor.h"
#include "TVirtualFitter.h"
#include "TVirtualHistPainter.h"
#include "TClass.h"

ClassImp(TF3)

//______________________________________________________________________________
//
// a 3-Dim function with parameters
//

//______________________________________________________________________________
TF3::TF3(): TF2()
{
//*-*-*-*-*-*-*-*-*-*-*F3 default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

   fNpz  = 0;
   fZmin = 0;
   fZmax = 1;
}


//______________________________________________________________________________
TF3::TF3(const char *name,const char *formula, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax)
      :TF2(name,formula,xmin,xmax,ymax,ymin)
{
//*-*-*-*-*-*-*F3 constructor using a formula definition*-*-*-*-*-*-*-*-*-*-*
//*-*          =========================================
//*-*
//*-*  See TFormula constructor for explanation of the formula syntax.
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fZmin   = zmin;
   fZmax   = zmax;
   fNpz    = 30;
   if (fNdim != 3 && xmin < xmax) {
      Error("TF3","function: %s/%s has %d parameters instead of 3",name,formula,fNdim);
      MakeZombie();
   }
}


//______________________________________________________________________________
TF3::TF3(const char *name,void *fcn, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar)
      :TF2(name,fcn,xmin,xmax,ymin,ymax,npar)
{
//*-*-*-*-*-*-*F3 constructor using a pointer to an interpreted function*-*-*
//*-*          =========================================================
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-*  Creates a function of type C between xmin and xmax and ymin,ymax.
//*-*  The function is defined with npar parameters
//*-*  fcn must be a function of type:
//*-*     Double_t fcn(Double_t *x, Double_t *params)
//*-*
//*-*  This constructor is called for functions of type C by CINT.
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fZmin   = zmin;
   fZmax   = zmax;
   fNpz    = 30;
   fNdim   = 3;
}

//______________________________________________________________________________
TF3::TF3(const char *name,Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar)
      :TF2(name,fcn,xmin,xmax,ymin,ymax,npar)
{
//*-*-*-*-*-*-*F3 constructor using a pointer to real function*-*-*-*-*-*-*-*
//*-*          ===============================================
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-*  For example, for a 3-dim function with 3 parameters, the user function
//*-*      looks like:
//*-*    Double_t fun1(Double_t *x, Double_t *par)
//*-*        return par[0]*x[2] + par[1]*exp(par[2]*x[0]*x[1]);
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fZmin   = zmin;
   fZmax   = zmax;
   fNpz    = 30;
   fNdim   = 3;
}

//______________________________________________________________________________
TF3::TF3(const char *name,Double_t (*fcn)(const Double_t *, const Double_t *), Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar)
   : TF2(name,fcn,xmin,xmax,ymin,ymax,npar),
   fZmin(zmin), 
   fZmax(zmax), 
   fNpz(30) 
{
//*-*-*-*-*-*-*F3 constructor using a pointer to real function*-*-*-*-*-*-*-*
//*-*          ===============================================
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-*  For example, for a 3-dim function with 3 parameters, the user function
//*-*      looks like:
//*-*    Double_t fun1(Double_t *x, Double_t *par)
//*-*        return par[0]*x[2] + par[1]*exp(par[2]*x[0]*x[1]);
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fNdim   = 3;
}

//______________________________________________________________________________
TF3::TF3(const char *name, ROOT::Math::ParamFunctor f, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar)
   : TF2(name, f, xmin, xmax, ymin, ymax,  npar), 
   fZmin(zmin), 
   fZmax(zmax), 
   fNpz(30) 
{
//*-*-*-*-*-*-*F3 constructor using a ParamFunctor, 
//*-*          a functor class implementing operator() (double *, double *)  
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*

   fNdim   = 3;
}

//______________________________________________________________________________
TF3::TF3(const char *name, void * ptr, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar, const char *className)
   : TF2(name, ptr, xmin, xmax, ymin, ymax,  npar, className), 
   fZmin(zmin), 
   fZmax(zmax), 
   fNpz(30) 
{
//*-*-*-*-*-*-*F2 constructor used by CINT for interpreted function objects
//*-*          Used for having same syntax as the template constructor from callable C++ objects 
//*-*          which can be used only in compile C++ mode.   
//*-*          
//*-*   npar is the number of free parameters used by the function
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*

   fNdim   = 3;
}

//______________________________________________________________________________
TF3::TF3(const char *name, void * ptr, void *,Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar, const char *className, const char * methodName)
   : TF2(name, ptr, (void*)0,xmin, xmax, ymin, ymax, npar,className, methodName), 
   fZmin(zmin), 
   fZmax(zmax), 
   fNpz(30) 
{
//*-*-*-*-*-*-*F2 constructor used by CINT for member functions of interpreted classes
//*-*          Used for having same syntax as the template constructor from a generic 
//*-*          class pointer and a member function type 
//*-*          which can be used only in compile C++ mode  
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*

   fNdim   = 3;
}


//______________________________________________________________________________
TF3& TF3::operator=(const TF3 &rhs) 
{
   // Operator =

   if (this != &rhs) {
      rhs.Copy(*this);
   }
   return *this;
}

//______________________________________________________________________________
TF3::~TF3()
{
//*-*-*-*-*-*-*-*-*-*-*F3 default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================

}

//______________________________________________________________________________
TF3::TF3(const TF3 &f3) : TF2()
{
   // Copy constructor.

   ((TF3&)f3).Copy(*this);
}

//______________________________________________________________________________
void TF3::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*Copy this F3 to a new F3*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   TF2::Copy(obj);
   ((TF3&)obj).fZmin = fZmin;
   ((TF3&)obj).fZmax = fZmax;
   ((TF3&)obj).fNpz  = fNpz;
}

//______________________________________________________________________________
Int_t TF3::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a function*-*-*-*-*
//*-*                  ===============================================
//*-*  Compute the closest distance of approach from point px,py to this function.
//*-*  The distance is computed in pixels units.
//*-*
//*-*  Algorithm:
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   return TF1::DistancetoPrimitive(px, py);

}

//______________________________________________________________________________
void TF3::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this function with its current attributes*-*-*-*-*
//*-*                  ==============================================

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();

   AppendPad(option);

}

//______________________________________________________________________________
void TF3::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//*-*  This member function is called when a F3 is clicked with the locator
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TF1::ExecuteEvent(event, px, py);
}

//______________________________________________________________________________
void TF3::GetMinimumXYZ(Double_t &x, Double_t &y, Double_t &z)
{
// Return the X, Y and Z values corresponding to the minimum value of the function
// on its range. To find the minimum on a subrange, use the SetRange() function first.
// Method:
//   First, a grid search is performed to find the initial estimate of the 
//   minimum location. The range of the function is divided 
//   into fNpx,fNpy and fNpz sub-ranges. If the function is "good" (or "bad"), 
//   these values can be changed by SetNpx(), SetNpy() and SetNpz() functions.
//   Then, Minuit minimization is used with starting values found by the grid search


   //First do a grid search with step size fNpx adn fNpy
   Double_t xx, yy, zz, tt;
   Double_t dx = (fXmax - fXmin)/fNpx;
   Double_t dy = (fYmax - fYmin)/fNpy;
   Double_t dz = (fZmax - fZmin)/fNpz;

   Double_t xxmin = fXmin;
   Double_t yymin = fYmin;
   Double_t zzmin = fZmin;
   Double_t ttmin = Eval(xxmin, yymin, zzmin+dz);
   for (Int_t i=0; i<fNpx; i++){
      xx=fXmin + (i+0.5)*dx;
      for (Int_t j=0; j<fNpy; j++){
         yy=fYmin+(j+0.5)*dy;
         for (Int_t k=0; k<fNpz; k++){
            zz = fZmin+(k+0.5)*dz;
            tt = Eval(xx, yy, zz);
            if (tt<ttmin) {xxmin = xx, yymin = yy; zzmin = zz; ttmin=tt;}
         }
      }
   }

   x = TMath::Min(fXmax, xxmin);
   y = TMath::Min(fYmax, yymin);
   z = TMath::Min(fZmax, zzmin);

   //go to minuit for the final minimization
   char f[]="TFitter";

   Int_t strdiff = 0;
   if (TVirtualFitter::GetFitter()){
      //If the fitter is already set and it's not minuit, delete it and 
      //create a minuit fitter
      strdiff = strcmp(TVirtualFitter::GetFitter()->IsA()->GetName(), f);
      if (strdiff!=0) 
         delete TVirtualFitter::GetFitter();
   }

   TVirtualFitter *minuit = TVirtualFitter::Fitter(this, 3);
   minuit->Clear();
   minuit->SetFitMethod("F3Minimizer");
   Double_t arglist[10];
   arglist[0]=-1;
   minuit->ExecuteCommand("SET PRINT", arglist, 1);

   minuit->SetParameter(0, "x", x, 0.1, 0, 0);
   minuit->SetParameter(1, "y", y, 0.1, 0, 0);
   minuit->SetParameter(2, "z", z, 0.1, 0, 0);
   arglist[0] = 5;
   arglist[1] = 1e-5;
   // minuit->ExecuteCommand("CALL FCN", arglist, 1);

   Int_t fitResult = minuit->ExecuteCommand("MIGRAD", arglist, 0);
   if (fitResult!=0){
      //migrad might have not converged
      Warning("GetMinimumXYZ", "Abnormal termination of minimization");
   }
   Double_t xtemp = minuit->GetParameter(0);
   Double_t ytemp = minuit->GetParameter(1);
   Double_t ztemp = minuit->GetParameter(2);
   if (xtemp>fXmax || xtemp<fXmin || ytemp>fYmax || ytemp<fYmin || ztemp>fZmax || ztemp<fZmin){
      //converged to something outside limits, redo with bounds 
      minuit->SetParameter(0, "x", x, 0.1, fXmin, fXmax);
      minuit->SetParameter(1, "y", y, 0.1, fYmin, fYmax);
      minuit->SetParameter(2, "z", z, 0.1, fZmin, fZmax);
      fitResult = minuit->ExecuteCommand("MIGRAD", arglist, 0);
      if (fitResult!=0){
         //migrad might have not converged
         Warning("GetMinimumXYZ", "Abnormal termination of minimization");
      }
   }
   x = minuit->GetParameter(0);
   y = minuit->GetParameter(1);
   z = minuit->GetParameter(2);
}

//______________________________________________________________________________
void TF3::GetRandom3(Double_t &xrandom, Double_t &yrandom, Double_t &zrandom)
{
//*-*-*-*-*-*Return 3 random numbers following this function shape*-*-*-*-*-*
//*-*        =====================================================
//*-*
//*-*   The distribution contained in this TF3 function is integrated
//*-*   over the cell contents.
//*-*   It is normalized to 1.
//*-*   Getting the three random numbers implies:
//*-*     - Generating a random number between 0 and 1 (say r1)
//*-*     - Look in which cell in the normalized integral r1 corresponds to
//*-*     - make a linear interpolation in the returned cell
//*-*
//*-*
//*-*  IMPORTANT NOTE
//*-*  The integral of the function is computed at fNpx * fNpy * fNpz points. 
//*-*  If the function has sharp peaks, you should increase the number of 
//*-*  points (SetNpx, SetNpy, SetNpz) such that the peak is correctly tabulated 
//*-*  at several points.

   //  Check if integral array must be build
   Int_t i,j,k,cell;
   Double_t dx   = (fXmax-fXmin)/fNpx;
   Double_t dy   = (fYmax-fYmin)/fNpy;
   Double_t dz   = (fZmax-fZmin)/fNpz;
   Int_t ncells = fNpx*fNpy*fNpz;
   Double_t xx[3];
   InitArgs(xx,fParams);
   if (fIntegral == 0) {
      fIntegral = new Double_t[ncells+1];
      fIntegral[0] = 0;
      Double_t integ;
      Int_t intNegative = 0;
      cell = 0;
      for (k=0;k<fNpz;k++) {
         xx[2] = fZmin+(k+0.5)*dz;
         for (j=0;j<fNpy;j++) {
            xx[1] = fYmin+(j+0.5)*dy;
            for (i=0;i<fNpx;i++) {
               xx[0] = fXmin+(i+0.5)*dx;
               integ = EvalPar(xx,fParams);
               if (integ < 0) {intNegative++; integ = -integ;}
               fIntegral[cell+1] = fIntegral[cell] + integ;
               cell++;
            }
         }
      }
      if (intNegative > 0) {
         Warning("GetRandom3","function:%s has %d negative values: abs assumed",GetName(),intNegative);
      }
      if (fIntegral[ncells] == 0) {
         Error("GetRandom3","Integral of function is zero");
         return;
      }
      for (i=1;i<=ncells;i++) {  // normalize integral to 1
         fIntegral[i] /= fIntegral[ncells];
      }
   }

// return random numbers
   Double_t r;
   r    = gRandom->Rndm();
   cell = TMath::BinarySearch(ncells,fIntegral,r);
   k    = cell/(fNpx*fNpy);
   j    = (cell -k*fNpx*fNpy)/fNpx;
   i    = cell -fNpx*(j +fNpy*k);
   xrandom = fXmin +dx*i +dx*gRandom->Rndm();
   yrandom = fYmin +dy*j +dy*gRandom->Rndm();
   zrandom = fZmin +dz*k +dz*gRandom->Rndm();
}

//______________________________________________________________________________
void TF3::GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax) const
{
//*-*-*-*-*-*-*-*-*-*-*Return range of function*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   xmin = fXmin;
   xmax = fXmax;
   ymin = fYmin;
   ymax = fYmax;
   zmin = fZmin;
   zmax = fZmax;
}


//______________________________________________________________________________
Double_t TF3::GetSave(const Double_t *xx)
{
    // Get value corresponding to X in array of fSave values

   if (fNsave <= 0) return 0;
   if (fSave == 0) return 0;
   Int_t np = fNsave - 9;
   Double_t xmin = Double_t(fSave[np+0]);
   Double_t xmax = Double_t(fSave[np+1]);
   Double_t ymin = Double_t(fSave[np+2]);
   Double_t ymax = Double_t(fSave[np+3]);
   Double_t zmin = Double_t(fSave[np+4]);
   Double_t zmax = Double_t(fSave[np+5]);
   Int_t npx     = Int_t(fSave[np+6]);
   Int_t npy     = Int_t(fSave[np+7]);
   Int_t npz     = Int_t(fSave[np+8]);
   Double_t x    = Double_t(xx[0]);
   Double_t dx   = (xmax-xmin)/npx;
   if (x < xmin || x > xmax) return 0;
   if (dx <= 0) return 0;
   Double_t y    = Double_t(xx[1]);
   Double_t dy   = (ymax-ymin)/npy;
   if (y < ymin || y > ymax) return 0;
   if (dy <= 0) return 0;
   Double_t z    = Double_t(xx[2]);
   Double_t dz   = (zmax-zmin)/npz;
   if (z < zmin || z > zmax) return 0;
   if (dz <= 0) return 0;

   //we make a trilinear interpolation using the 8 points surrounding x,y,z
   Int_t ibin    = Int_t((x-xmin)/dx);
   Int_t jbin    = Int_t((y-ymin)/dy);
   Int_t kbin    = Int_t((z-zmin)/dz);
   Double_t xlow = xmin + ibin*dx;
   Double_t ylow = ymin + jbin*dy;
   Double_t zlow = zmin + kbin*dz;
   Double_t t    = (x-xlow)/dx;
   Double_t u    = (y-ylow)/dy;
   Double_t v    = (z-zlow)/dz;
   Int_t k1      = (ibin  ) + (npx+1)*((jbin  ) + (npy+1)*(kbin  ));
   Int_t k2      = (ibin+1) + (npx+1)*((jbin  ) + (npy+1)*(kbin  ));
   Int_t k3      = (ibin+1) + (npx+1)*((jbin+1) + (npy+1)*(kbin  ));
   Int_t k4      = (ibin  ) + (npx+1)*((jbin+1) + (npy+1)*(kbin  ));
   Int_t k5      = (ibin  ) + (npx+1)*((jbin  ) + (npy+1)*(kbin+1));
   Int_t k6      = (ibin+1) + (npx+1)*((jbin  ) + (npy+1)*(kbin+1));
   Int_t k7      = (ibin+1) + (npx+1)*((jbin+1) + (npy+1)*(kbin+1));
   Int_t k8      = (ibin  ) + (npx+1)*((jbin+1) + (npy+1)*(kbin+1));
   Double_t r    = (1-t)*(1-u)*(1-v)*fSave[k1] + t*(1-u)*(1-v)*fSave[k2] + t*u*(1-v)*fSave[k3] + (1-t)*u*(1-v)*fSave[k4] +
                   (1-t)*(1-u)*v*fSave[k5] + t*(1-u)*v*fSave[k6] + t*u*v*fSave[k7] + (1-t)*u*v*fSave[k8];
   return r;
}

//______________________________________________________________________________
Double_t TF3::Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t az, Double_t bz, Double_t epsilon)
{
// Return Integral of a 3d function in range [ax,bx],[ay,by],[az,bz]
//
   Double_t a[3], b[3];
   a[0] = ax;
   b[0] = bx;
   a[1] = ay;
   b[1] = by;
   a[2] = az;
   b[2] = bz;
   Double_t relerr  = 0;
   Int_t n = 3;
   Int_t minpts = 2*2*2+2*n*(n+1)+1;  // ie 33
   Int_t maxpts = 20*fNpx*fNpy*fNpz;
   Int_t nfnevl,ifail;
   Double_t result = IntegralMultiple(n,a,b,minpts,maxpts,epsilon,relerr,nfnevl,ifail);
   if (ifail > 0) {
      Warning("Integral","failed code=%d, minpts=%d, maxpts=%d, epsilon=%g, nfnevl=%d, relerr=%g ",ifail,minpts,maxpts,epsilon,nfnevl,relerr);
   }
   return result;
}

//______________________________________________________________________________
Bool_t TF3::IsInside(const Double_t *x) const
{
// Return kTRUE is the point is inside the function range
   
   if (x[0] < fXmin || x[0] > fXmax) return kFALSE;
   if (x[1] < fYmin || x[1] > fYmax) return kFALSE;
   if (x[2] < fZmin || x[2] > fZmax) return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
TH1* TF3::CreateHistogram()
{
// Create a histogram for axis range.

   TH1* h = new TH3F("R__TF3",(char*)GetTitle(),fNpx,fXmin,fXmax
                         ,fNpy,fYmin,fYmax
                         ,fNpz,fZmin,fZmax);
   h->SetDirectory(0);
   return h;
}

//______________________________________________________________________________
void TF3::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Paint this 3-D function with its current attributes*-*-*-*-*
//*-*              ===================================================


   TString opt = option;
   opt.ToLower();

//*-*-  Create a temporary histogram and fill each channel with the function value
   if (!fHistogram) {
      fHistogram = new TH3F("R__TF3",(char*)GetTitle(),fNpx,fXmin,fXmax
                                                      ,fNpy,fYmin,fYmax
                                                      ,fNpz,fZmin,fZmax);
      fHistogram->SetDirectory(0);
   }

   fHistogram->GetPainter(option)->ProcessMessage("SetF3",this);
   if (opt.Length() == 0 ) {
      fHistogram->Paint("tf3");
   } else {
      opt += "tf3";
      fHistogram->Paint(opt.Data());
   }
}

//______________________________________________________________________________
void TF3::SetClippingBoxOff()
{
   // Set the function clipping box (for drawing) "off".

   if (!fHistogram) {
      fHistogram = new TH3F("R__TF3",(char*)GetTitle(),fNpx,fXmin,fXmax
                                                    ,fNpy,fYmin,fYmax
                                                    ,fNpz,fZmin,fZmax);
      fHistogram->SetDirectory(0);
   }
   fHistogram->GetPainter()->ProcessMessage("SetF3ClippingBoxOff",0);
}

//______________________________________________________________________________
void TF3::Save(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax)
{
    // Save values of function in array fSave

   if (fSave != 0) {delete [] fSave; fSave = 0;}
   Int_t nsave = (fNpx+1)*(fNpy+1)*(fNpz+1);
   fNsave = nsave+9;
   if (fNsave <= 9) {fNsave=0; return;}
   fSave  = new Double_t[fNsave];
   Int_t i,j,k,l=0;
   Double_t dx = (xmax-xmin)/fNpx;
   Double_t dy = (ymax-ymin)/fNpy;
   Double_t dz = (zmax-zmin)/fNpz;
   if (dx <= 0) {
      dx = (fXmax-fXmin)/fNpx;
      xmin = fXmin +0.5*dx;
      xmax = fXmax -0.5*dx;
   }
   if (dy <= 0) {
      dy = (fYmax-fYmin)/fNpy;
      ymin = fYmin +0.5*dy;
      ymax = fYmax -0.5*dy;
   }
   if (dz <= 0) {
      dz = (fZmax-fZmin)/fNpz;
      zmin = fZmin +0.5*dz;
      zmax = fZmax -0.5*dz;
   }
   Double_t xv[3];
   InitArgs(xv,fParams);
   for (k=0;k<=fNpz;k++) {
      xv[2]    = zmin + dz*k;
      for (j=0;j<=fNpy;j++) {
         xv[1]    = ymin + dy*j;
         for (i=0;i<=fNpx;i++) {
            xv[0]    = xmin + dx*i;
            fSave[l] = EvalPar(xv,fParams);
            l++;
         }
      }
   }
   fSave[nsave+0] = xmin;
   fSave[nsave+1] = xmax;
   fSave[nsave+2] = ymin;
   fSave[nsave+3] = ymax;
   fSave[nsave+4] = zmin;
   fSave[nsave+5] = zmax;
   fSave[nsave+6] = fNpx;
   fSave[nsave+7] = fNpy;
   fSave[nsave+8] = fNpz;
}

//______________________________________________________________________________
void TF3::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TF3::Class())) {
      out<<"   ";
   } else {
      out<<"   TF3 *";
   }
   if (!fMethodCall) {
      out<<GetName()<<" = new TF3("<<quote<<GetName()<<quote<<","<<quote<<GetTitle()<<quote<<","<<fXmin<<","<<fXmax<<","<<fYmin<<","<<fYmax<<","<<fZmin<<","<<fZmax<<");"<<endl;
   } else {
      out<<GetName()<<" = new TF3("<<quote<<GetName()<<quote<<","<<GetTitle()<<","<<fXmin<<","<<fXmax<<","<<fYmin<<","<<fYmax<<","<<fZmin<<","<<fZmax<<","<<GetNpar()<<");"<<endl;
   }

   if (GetFillColor() != 0) {
      if (GetFillColor() > 228) {
         TColor::SaveColor(out, GetFillColor());
         out<<"   "<<GetName()<<"->SetFillColor(ci);" << endl;
      } else 
         out<<"   "<<GetName()<<"->SetFillColor("<<GetFillColor()<<");"<<endl;
   }
   if (GetLineColor() != 1) {
      if (GetLineColor() > 228) {
         TColor::SaveColor(out, GetLineColor());
         out<<"   "<<GetName()<<"->SetLineColor(ci);" << endl;
      } else 
         out<<"   "<<GetName()<<"->SetLineColor("<<GetLineColor()<<");"<<endl;
   }
   if (GetNpz() != 100) {
      out<<"   "<<GetName()<<"->SetNpz("<<GetNpz()<<");"<<endl;
   }
   if (GetChisquare() != 0) {
      out<<"   "<<GetName()<<"->SetChisquare("<<GetChisquare()<<");"<<endl;
   }
   Double_t parmin, parmax;
   for (Int_t i=0;i<fNpar;i++) {
      out<<"   "<<GetName()<<"->SetParameter("<<i<<","<<GetParameter(i)<<");"<<endl;
      out<<"   "<<GetName()<<"->SetParError("<<i<<","<<GetParError(i)<<");"<<endl;
      GetParLimits(i,parmin,parmax);
      out<<"   "<<GetName()<<"->SetParLimits("<<i<<","<<parmin<<","<<parmax<<");"<<endl;
   }
   out<<"   "<<GetName()<<"->Draw("
      <<quote<<option<<quote<<");"<<endl;
}

//______________________________________________________________________________
void TF3::SetClippingBoxOn(Double_t xclip, Double_t yclip, Double_t zclip)
{
   // Set the function clipping box (for drawing) "on" and define the clipping box.
   // xclip, yclip and zclip is a point within the function range. All the
   // function values having x<=xclip and y<=yclip and z>=zclip are clipped.
   
   if (!fHistogram) {
      fHistogram = new TH3F("R__TF3",(char*)GetTitle(),fNpx,fXmin,fXmax
                                                    ,fNpy,fYmin,fYmax
                                                    ,fNpz,fZmin,fZmax);
      fHistogram->SetDirectory(0);
   }
   
   TVectorD v(3);
   v(0) = xclip;
   v(1) = yclip;
   v(2) = zclip;
   fHistogram->GetPainter()->ProcessMessage("SetF3ClippingBoxOn",&v);
}

//______________________________________________________________________________
void TF3::SetNpz(Int_t npz)
{
// Set the number of points used to draw the function
//
// The default number of points along x is 30 for 2-d/3-d functions.
// You can increase this value to get a better resolution when drawing
// pictures with sharp peaks or to get a better result when using TF3::GetRandom2   
// the minimum number of points is 4, the maximum is 10000 for 2-d/3-d functions

   if (npz < 4) {
      Warning("SetNpz","Number of points must be >=4 && <= 10000, fNpz set to 4");
      fNpz = 4;
   } else if(npz > 10000) {
      Warning("SetNpz","Number of points must be >=4 && <= 10000, fNpz set to 10000");
      fNpz = 10000;
   } else {
      fNpz = npz;
   }
   Update();
}

//______________________________________________________________________________
void TF3::SetRange(Double_t xmin, Double_t ymin, Double_t zmin, Double_t xmax, Double_t ymax, Double_t zmax)
{
//*-*-*-*-*-*Initialize the upper and lower bounds to draw the function*-*-*-*
//*-*        ==========================================================

   fXmin = xmin;
   fXmax = xmax;
   fYmin = ymin;
   fYmax = ymax;
   fZmin = zmin;
   fZmax = zmax;
   Update();
}

//______________________________________________________________________________
void TF3::Streamer(TBuffer &R__b)
{
   // Stream an object of class TF3.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 0) {
         R__b.ReadClassBuffer(TF3::Class(), this, R__v, R__s, R__c);
         return;
      }

   } else {
      Int_t saved = 0;
      if (fType > 0 && fNsave <= 0) { saved = 1; Save(fXmin,fXmax,fYmin,fYmax,fZmin,fZmax);}

      R__b.WriteClassBuffer(TF3::Class(),this);

      if (saved) {delete [] fSave; fSave = 0; fNsave = 0;}
   }
}

//______________________________________________________________________________
Double_t TF3::Moment3(Double_t nx, Double_t ax, Double_t bx, Double_t ny, Double_t ay, Double_t by, Double_t nz, Double_t az, Double_t bz, Double_t epsilon)
{
// Return x^nx * y^ny * z^nz moment of a 3d function in range [ax,bx],[ay,by],[az,bz]
//   Author: Gene Van Buren <gene@bnl.gov>

   Double_t norm = Integral(ax,bx,ay,by,az,bz,epsilon);
   if (norm == 0) {
      Error("Moment3", "Integral zero over range");
      return 0;
   }

   TF3 fnc("TF3_ExpValHelper",Form("%s*pow(x,%f)*pow(y,%f)*pow(z,%f)",GetName(),nx,ny,nz));
   return fnc.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
}

//______________________________________________________________________________
Double_t TF3::CentralMoment3(Double_t nx, Double_t ax, Double_t bx, Double_t ny, Double_t ay, Double_t by, Double_t nz, Double_t az, Double_t bz, Double_t epsilon)
{
// Return x^nx * y^ny * z^nz central moment of a 3d function in range [ax,bx],[ay,by],[az,bz]
//   Author: Gene Van Buren <gene@bnl.gov>
   
   Double_t norm = Integral(ax,bx,ay,by,az,bz,epsilon);
   if (norm == 0) {
      Error("CentralMoment3", "Integral zero over range");
      return 0;
   }

   Double_t xbar = 0;
   Double_t ybar = 0;
   Double_t zbar = 0;
   if (nx!=0) {
      TF3 fncx("TF3_ExpValHelperx",Form("%s*x",GetName()));
      xbar = fncx.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
   }
   if (ny!=0) {
      TF3 fncy("TF3_ExpValHelpery",Form("%s*y",GetName()));
      ybar = fncy.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
   }
   if (nz!=0) {
      TF3 fncz("TF3_ExpValHelperz",Form("%s*z",GetName()));
      zbar = fncz.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
   }
   TF3 fnc("TF3_ExpValHelper",Form("%s*pow(x-%f,%f)*pow(y-%f,%f)*pow(z-%f,%f)",GetName(),xbar,nx,ybar,ny,zbar,nz));
   return fnc.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
}

