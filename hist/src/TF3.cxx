// @(#)root/hist:$Name:  $:$Id: TF3.cxx,v 1.1.1.1 2000/05/16 17:00:40 rdm Exp $
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

}


//______________________________________________________________________________
TF3::TF3(const char *name,const char *formula, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax)
      :TF2(name,formula,xmin,xmax,ymin,ymax)
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
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fZmin   = zmin;
   fZmax   = zmax;
   fNpz    = 30;
   fNdim   = 3;
}

//______________________________________________________________________________
TF3::~TF3()
{
//*-*-*-*-*-*-*-*-*-*-*F3 default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================

}

//______________________________________________________________________________
TF3::TF3(const TF3 &f3)
{
   ((TF3&)f3).Copy(*this);
}

//______________________________________________________________________________
void TF3::Copy(TObject &obj)
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
   if (!opt.Contains("s")) gPad->Clear();

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
void TF3::GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax)
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
   Double_t result = IntegralMultiple(3,a,b,epsilon,relerr);
   return result;
}

//______________________________________________________________________________
void TF3::Paint(Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint this 2-D function with its current attributes*-*-*-*-*
//*-*              ===================================================

}

//______________________________________________________________________________
void TF3::SetNpz(Int_t npz)
{
//*-*-*-*-*-*-*-*Set the number of points used to draw the function*-*-*-*-*-*
//*-*            ==================================================

   if(npz > 4 && npz < 1000) fNpz = npz;
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
