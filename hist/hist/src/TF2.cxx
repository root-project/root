// @(#)root/hist:$Id$
// Author: Rene Brun   23/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TF2.h"
#include "TMath.h"
#include "TRandom.h"
#include "TH2.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "Riostream.h"
#include "TColor.h"
#include "TVirtualFitter.h"
#include "TClass.h"

ClassImp(TF2)

//______________________________________________________________________________
//
// a 2-Dim function with parameters
// TF2 graphics function is via the TH1 drawing functions.
//
//      Example of a function
//
//   TF2 *f2 = new TF2("f2","sin(x)*sin(y)/(x*y)",0,5,0,5);
//   f2->Draw();
//Begin_Html
/*
<img src="gif/function2.gif">
*/
//End_Html
//
//      See TF1 class for the list of functions formats
//

//______________________________________________________________________________
TF2::TF2(): TF1(),fYmin(0),fYmax(0),fNpy(100)
{
//*-*-*-*-*-*-*-*-*-*-*F2 default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

}


//______________________________________________________________________________
TF2::TF2(const char *name,const char *formula, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax)
      :TF1(name,formula,xmax,xmin)
{
// F2 constructor using a formula definition
//
//  See TFormula constructor for explanation of the formula syntax.
//
//  if formula has the form "fffffff;xxxx;yyyy", it is assumed that
//  the formula string is "fffffff" and "xxxx" and "yyyy" are the
//  titles for the X and Y axis respectively.

   if (ymin < ymax) {
      fYmin   = ymin;
      fYmax   = ymax;
   } else {
      fYmin = ymax;
      fYmax = ymin;
   }
   fNpx    = 30;
   fNpy    = 30;
   fContour.Set(0);
   if (fNdim != 2 && xmin < xmax && ymin < ymax) {
      Error("TF2","function: %s/%s has %d parameters instead of 2",name,formula,fNdim);
      MakeZombie();
   }
}

//______________________________________________________________________________
TF2::TF2(const char *name, void *fcn, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Int_t npar)
      : TF1(name, fcn, xmin, xmax, npar)
{
//*-*-*-*-*-*-*F2 constructor using a pointer to an interpreted function*-*-*
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

   fYmin   = ymin;
   fYmax   = ymax;
   fNpx    = 30;
   fNpy    = 30;
   fNdim   = 2;
   fContour.Set(0);

}

//______________________________________________________________________________
TF2::TF2(const char *name, Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Int_t npar)
      : TF1(name, fcn, xmin, xmax, npar)
{
//*-*-*-*-*-*-*F2 constructor using a pointer to a compiled function*-*-*-*-*
//*-*          =====================================================
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-*   This constructor creates a function of type C when invoked
//*-*   with the normal C++ compiler.
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fYmin   = ymin;
   fYmax   = ymax;
   fNpx    = 30;
   fNpy    = 30;
   fNdim   = 2;
   fContour.Set(0);

}

//______________________________________________________________________________
TF2::TF2(const char *name, Double_t (*fcn)(const Double_t *, const Double_t *), Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Int_t npar)
      : TF1(name, fcn, xmin, xmax, npar)
{
//*-*-*-*-*-*-*F2 constructor using a pointer to a compiled function*-*-*-*-*
//*-*          =====================================================
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-*   This constructor creates a function of type C when invoked
//*-*   with the normal C++ compiler.
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fYmin   = ymin;
   fYmax   = ymax;
   fNpx    = 30;
   fNpy    = 30;
   fNdim   = 2;
   fContour.Set(0);

}

//______________________________________________________________________________
TF2::TF2(const char *name, ROOT::Math::ParamFunctor f, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Int_t npar)
      : TF1(name, f, xmin, xmax, npar)
{
//*-*-*-*-*-*-*F2 constructor using a ParamFunctor, 
//*-*          a functor class implementing operator() (double *, double *)  
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*

   fYmin   = ymin;
   fYmax   = ymax;
   fNpx    = 30;
   fNpy    = 30;
   fNdim   = 2;
   fContour.Set(0);

}

//______________________________________________________________________________
TF2::TF2(const char *name, void * ptr, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Int_t npar, const char *className)
      : TF1(name, ptr, xmin, xmax, npar,className)
{
//*-*-*-*-*-*-*F2 constructor used by CINT for interpreted function objects
//*-*          Used for having same syntax as the template constructor from callable C++ objects 
//*-*          which can be used only in compile C++ mode.   
//*-*          
//*-*   npar is the number of free parameters used by the function
//*-*
//*-* WARNING! A function created with this constructor cannot be Cloned.
//*-*

   fYmin   = ymin;
   fYmax   = ymax;
   fNpx    = 30;
   fNpy    = 30;
   fNdim   = 2;
   fContour.Set(0);

}

//______________________________________________________________________________
TF2::TF2(const char *name, void * ptr, void *,Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Int_t npar, const char *className, const char * methodName)
      : TF1(name, ptr, (void*)0,xmin, xmax, npar,className, methodName)
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

   fYmin   = ymin;
   fYmax   = ymax;
   fNpx    = 30;
   fNpy    = 30;
   fNdim   = 2;
   fContour.Set(0);

}


//______________________________________________________________________________
TF2& TF2::operator=(const TF2 &rhs)
{
   // Operator =

   if (this != &rhs) {
      rhs.Copy(*this);
   }
   return *this;
}

//______________________________________________________________________________
TF2::~TF2()
{
//*-*-*-*-*-*-*-*-*-*-*F2 default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================

}

//______________________________________________________________________________
TF2::TF2(const TF2 &f2) : TF1()
{
   // Copy constructor.

   ((TF2&)f2).Copy(*this);
}

//______________________________________________________________________________
void TF2::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*Copy this F2 to a new F2*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   TF1::Copy(obj);
   ((TF2&)obj).fYmin    = fYmin;
   ((TF2&)obj).fYmax    = fYmax;
   ((TF2&)obj).fNpy     = fNpy;
   fContour.Copy(((TF2&)obj).fContour);
}

//______________________________________________________________________________
Int_t TF2::DistancetoPrimitive(Int_t px, Int_t py)
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

   if (!fHistogram) return 9999;
   Int_t distance = fHistogram->DistancetoPrimitive(px,py);
   if (distance <= 1) return distance;

   Double_t x = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y = gPad->PadtoY(gPad->AbsPixeltoY(py));
   const char *drawOption = GetDrawOption();
   Double_t uxmin,uxmax;
   Double_t uymin,uymax;
   if (gPad->GetView() || strncmp(drawOption,"cont",4) == 0
                       || strncmp(drawOption,"CONT",4) == 0) {
      uxmin=gPad->GetUxmin();
      uxmax=gPad->GetUxmax();
      x = fXmin +(fXmax-fXmin)*(x-uxmin)/(uxmax-uxmin);
      uymin=gPad->GetUymin();
      uymax=gPad->GetUymax();
      y = fYmin +(fYmax-fYmin)*(y-uymin)/(uymax-uymin);
   }
   if (x < fXmin || x > fXmax) return distance;
   if (y < fYmin || y > fYmax) return distance;
   return 0;
}

//______________________________________________________________________________
void TF2::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this function with its current attributes*-*-*-*-*
//*-*                  ==============================================
//*-* NB. You must use DrawCopy if you want to draw several times the same
//*-*     function in the current canvas.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();

   AppendPad(option);
}

//______________________________________________________________________________
TF1 *TF2::DrawCopy(Option_t *option) const
{
//*-*-*-*-*-*-*-*Draw a copy of this function with its current attributes*-*-*
//*-*            ========================================================
//*-*
//*-*  This function MUST be used instead of Draw when you want to draw
//*-*  the same function with different parameters settings in the same canvas.
//*-*
//*-* Possible option values are:
//*-*   "SAME"  superimpose on top of existing picture
//*-*   "L"     connect all computed points with a straight line
//*-*   "C"     connect all computed points with a smooth curve.
//*-*
//*-* Note that the default value is "F". Therefore to draw on top
//*-* of an existing picture, specify option "SL"
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TF2 *newf2 = new TF2();
   Copy(*newf2);
   newf2->AppendPad(option);
   newf2->SetBit(kCanDelete);
   return newf2;
}

//______________________________________________________________________________
void TF2::DrawF2(const char *formula, Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Draw formula between xmin,ymin and xmax,ymax*-*-*-*-*-*-*-*
//*-*                ============================================
//*-*

   if (Compile((char*)formula)) return;

   SetRange(xmin, ymin, xmax, ymax);

   Draw(option);

}

//______________________________________________________________________________
void TF2::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//*-*  This member function is called when a F2 is clicked with the locator
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TF1::ExecuteEvent(event, px, py);
}

//______________________________________________________________________________
Int_t TF2::GetContour(Double_t *levels)
{
//*-*-*-*-*-*-*-*Return contour values into array levels*-*-*-*-*-*-*-*-*-*
//*-*            =======================================
//*-*
//*-*  The number of contour levels can be returned by getContourLevel
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t nlevels = fContour.fN;
   if (levels) {
      for (Int_t level=0; level<nlevels; level++) levels[level] = GetContourLevel(level);
   }
   return nlevels;
}

//______________________________________________________________________________
Double_t TF2::GetContourLevel(Int_t level) const
{
//*-*-*-*-*-*-*-*Return the number of contour levels*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ===================================
   if (level <0 || level >= fContour.fN) return 0;
   if (fContour.fArray[0] != -9999) return fContour.fArray[level];
   if (fHistogram == 0) return 0;
   return fHistogram->GetContourLevel(level);
}

//______________________________________________________________________________
void TF2::GetMinimumXY(Double_t &x, Double_t &y)
{
// return the X and Y values corresponding to the minimum value of the function
// To find the minimum on a range, first set this range via the SetRange function
// Method:
//   First, a grid search is performed to find the initial estimate of the 
//   minimum location. The range of the function is divided into fNpx and fNpy
//   sub-ranges. If the function is "good" (or "bad"), these values can be changed
//   by SetNpx and SetNpy functions
//   Then, Minuit minimization is used with starting values found by the grid search

   //First do a grid search with step size fNpx and fNpy
   Double_t xx, yy, zz;
   Double_t dx = (fXmax - fXmin)/fNpx;
   Double_t dy = (fYmax - fYmin)/fNpy;
   Double_t xxmin = fXmin;
   Double_t yymin = fYmin;
   Double_t zzmin = Eval(xxmin, yymin+dy);
   for (Int_t i=0; i<fNpx; i++){
      xx=fXmin + (i+0.5)*dx;
      for (Int_t j=0; j<fNpy; j++){
         yy=fYmin+(j+0.5)*dy;
         zz = Eval(xx, yy);
         if (zz<zzmin) {xxmin = xx, yymin = yy; zzmin = zz;}
      }
   }

   x = TMath::Min(fXmax, xxmin);
   y = TMath::Min(fYmax, yymin);

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

   TVirtualFitter *minuit = TVirtualFitter::Fitter(this, 2);
   if (!minuit) { 
      Error("GetMinimumXY", "Cannot create fitter");
      return;
   }
   minuit->Clear();
   minuit->SetFitMethod("F2Minimizer");
   Double_t arglist[10];
   arglist[0]=-1;
   minuit->ExecuteCommand("SET PRINT", arglist, 1);

   minuit->SetParameter(0, "x", x, 0.1, 0, 0);
   minuit->SetParameter(1, "y", y, 0.1, 0, 0);
   arglist[0] = 5;
   arglist[1] = 1e-5;
   // minuit->ExecuteCommand("CALL FCN", arglist, 1);

   Int_t fitResult = minuit->ExecuteCommand("MIGRAD", arglist, 0);
   if (fitResult!=0){
      //migrad might have not converged
      Warning("GetMinimumXY", "Abnormal termination of minimization");
   }
   Double_t xtemp=minuit->GetParameter(0);
   Double_t ytemp=minuit->GetParameter(1);
   if (xtemp>fXmax || xtemp<fXmin || ytemp>fYmax || ytemp<fYmin){
      //converged to something outside limits, redo with bounds 

      minuit->SetParameter(0, "x", x, 0.1, fXmin, fXmax);
      minuit->SetParameter(1, "y", y, 0.1, fYmin, fYmax);
      fitResult = minuit->ExecuteCommand("MIGRAD", arglist, 0);
      if (fitResult!=0){
         //migrad might have not converged
         Warning("GetMinimumXY", "Abnormal termination of minimization");
      }
   }
   x = minuit->GetParameter(0);
   y = minuit->GetParameter(1);

}

//______________________________________________________________________________
char *TF2::GetObjectInfo(Int_t px, Int_t py) const
{
//   Redefines TObject::GetObjectInfo.
//   Displays the function value
//   corresponding to cursor position px,py
//
   const char *snull = "";
   if (!gPad) return (char*)snull;
   static char info[64];
   Double_t x = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y = gPad->PadtoY(gPad->AbsPixeltoY(py));
   const char *drawOption = GetDrawOption();
   Double_t uxmin,uxmax;
   Double_t uymin,uymax;
   if (gPad->GetView() || strncmp(drawOption,"cont",4) == 0
                       || strncmp(drawOption,"CONT",4) == 0) {
      uxmin=gPad->GetUxmin();
      uxmax=gPad->GetUxmax();
      x = fXmin +(fXmax-fXmin)*(x-uxmin)/(uxmax-uxmin);
      uymin=gPad->GetUymin();
      uymax=gPad->GetUymax();
      y = fYmin +(fYmax-fYmin)*(y-uymin)/(uymax-uymin);
   }
   snprintf(info,64,"(x=%g, y=%g, f=%.18g)",x,y,((TF2*)this)->Eval(x,y));
   return info;
}

//______________________________________________________________________________
Double_t TF2::GetRandom()
{
//*-*-*-*-*-*Return a random number following this function shape*-*-*-*-*-*-*
//*-*        ====================================================
//*-*
   Error("GetRandom","cannot be called for TF2/3, use GetRandom2/3 instead");
   return 0;  // not yet implemented
}

//______________________________________________________________________________
Double_t TF2::GetRandom(Double_t, Double_t)
{
//*-*-*-*-*-*Return a random number following this function shape*-*-*-*-*-*-*
//*-*        ====================================================
//*-*
   Error("GetRandom","cannot be called for TF2/3, use GetRandom2/3 instead");
   return 0;  // not yet implemented
}

//______________________________________________________________________________
void TF2::GetRandom2(Double_t &xrandom, Double_t &yrandom)
{
//*-*-*-*-*-*Return 2 random numbers following this function shape*-*-*-*-*-*
//*-*        =====================================================
//*-*
//*-*   The distribution contained in this TF2 function is integrated
//*-*   over the cell contents.
//*-*   It is normalized to 1.
//*-*   Getting the two random numbers implies:
//*-*     - Generating a random number between 0 and 1 (say r1)
//*-*     - Look in which cell in the normalized integral r1 corresponds to
//*-*     - make a linear interpolation in the returned cell
//*-*
//*-*
//*-*  IMPORTANT NOTE
//*-*  The integral of the function is computed at fNpx * fNpy points. 
//*-*  If the function has sharp peaks, you should increase the number of 
//*-*  points (SetNpx, SetNpy) such that the peak is correctly tabulated 
//*-*  at several points.

   //  Check if integral array must be build
   Int_t i,j,cell;
   Double_t dx   = (fXmax-fXmin)/fNpx;
   Double_t dy   = (fYmax-fYmin)/fNpy;
   Int_t ncells = fNpx*fNpy;
   if (fIntegral == 0) {
      fIntegral = new Double_t[ncells+1];
      fIntegral[0] = 0;
      Double_t integ;
      Int_t intNegative = 0;
      cell = 0;
      for (j=0;j<fNpy;j++) {
         for (i=0;i<fNpx;i++) {
            integ = Integral(fXmin+i*dx,fXmin+i*dx+dx,fYmin+j*dy,fYmin+j*dy+dy);
            if (integ < 0) {intNegative++; integ = -integ;}
            fIntegral[cell+1] = fIntegral[cell] + integ;
            cell++;
         }
      }
      if (intNegative > 0) {
         Warning("GetRandom2","function:%s has %d negative values: abs assumed",GetName(),intNegative);
      }
      if (fIntegral[ncells] == 0) {
         Error("GetRandom2","Integral of function is zero");
         return;
      }
      for (i=1;i<=ncells;i++) {  // normalize integral to 1
         fIntegral[i] /= fIntegral[ncells];
      }
   }

// return random numbers
   Double_t r,ddx,ddy,dxint;
   r     = gRandom->Rndm();
   cell  = TMath::BinarySearch(ncells,fIntegral,r);
   dxint = fIntegral[cell+1] - fIntegral[cell];
   if (dxint > 0) ddx = dx*(r - fIntegral[cell])/dxint;
   else           ddx = 0;
   ddy = dy*gRandom->Rndm();
   j   = cell/fNpx;
   i   = cell%fNpx;
   xrandom = fXmin +dx*i +ddx;
   yrandom = fYmin +dy*j +ddy;
}

//______________________________________________________________________________
void TF2::GetRange(Double_t &xmin, Double_t &ymin,  Double_t &xmax, Double_t &ymax) const
{
//*-*-*-*-*-*-*-*-*-*-*Return range of a 2-D function*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================

   xmin = fXmin;
   xmax = fXmax;
   ymin = fYmin;
   ymax = fYmax;
}

//______________________________________________________________________________
void TF2::GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax) const
{
//*-*-*-*-*-*-*-*-*-*-*Return range of function*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   xmin = fXmin;
   xmax = fXmax;
   ymin = fYmin;
   ymax = fYmax;
   zmin = 0;
   zmax = 0;
}


//______________________________________________________________________________
Double_t TF2::GetSave(const Double_t *xx)
{
    // Get value corresponding to X in array of fSave values

   if (fNsave <= 0) return 0;
   if (fSave == 0) return 0;
   Int_t np = fNsave - 6;
   Double_t xmin = Double_t(fSave[np+0]);
   Double_t xmax = Double_t(fSave[np+1]);
   Double_t ymin = Double_t(fSave[np+2]);
   Double_t ymax = Double_t(fSave[np+3]);
   Int_t npx     = Int_t(fSave[np+4]);
   Int_t npy     = Int_t(fSave[np+5]);
   Double_t x    = Double_t(xx[0]);
   Double_t dx   = (xmax-xmin)/npx;
   if (x < xmin || x > xmax) return 0;
   if (dx <= 0) return 0;
   Double_t y    = Double_t(xx[1]);
   Double_t dy   = (ymax-ymin)/npy;
   if (y < ymin || y > ymax) return 0;
   if (dy <= 0) return 0;

   //we make a bilinear interpolation using the 4 points surrounding x,y
   Int_t ibin    = Int_t((x-xmin)/dx);
   Int_t jbin    = Int_t((y-ymin)/dy);
   Double_t xlow = xmin + ibin*dx;
   Double_t ylow = ymin + jbin*dy;
   Double_t t    = (x-xlow)/dx;
   Double_t u    = (y-ylow)/dy;
   Int_t k1      = jbin*(npx+1) + ibin;
   Int_t k2      = jbin*(npx+1) + ibin +1;
   Int_t k3      = (jbin+1)*(npx+1) + ibin +1;
   Int_t k4      = (jbin+1)*(npx+1) + ibin;
   Double_t z    = (1-t)*(1-u)*fSave[k1] +t*(1-u)*fSave[k2] +t*u*fSave[k3] + (1-t)*u*fSave[k4];
   return z;
}

//______________________________________________________________________________
Double_t TF2::Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t epsilon)
{
// Return Integral of a 2d function in range [ax,bx],[ay,by]
//
   Double_t a[2], b[2];
   a[0] = ax;
   b[0] = bx;
   a[1] = ay;
   b[1] = by;
   Double_t relerr  = 0;
   Int_t n = 2;
   Int_t minpts = 2*2+2*n*(n+1)+1; //ie 17
   Int_t maxpts = 20*fNpx*fNpy;
   Int_t nfnevl,ifail;
   Double_t result = IntegralMultiple(n,a,b,minpts,maxpts,epsilon,relerr,nfnevl,ifail);
   if (ifail > 0) {
      Warning("Integral","failed code=%d, minpts=%d, maxpts=%d, epsilon=%g, nfnevl=%d, relerr=%g ",ifail,minpts,maxpts,epsilon,nfnevl,relerr);
   }
   return result;
}

//______________________________________________________________________________
Bool_t TF2::IsInside(const Double_t *x) const
{
// Return kTRUE is the point is inside the function range

   if (x[0] < fXmin || x[0] > fXmax) return kFALSE;
   if (x[1] < fYmin || x[1] > fYmax) return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
TH1* TF2::CreateHistogram()
{
// Create a histogram from function.

   Int_t i,j,bin;
   Double_t dx, dy;
   Double_t xv[2];

   TH2F* h = new TH2F("Func",(char*)GetTitle(),fNpx,fXmin,fXmax,fNpy,fYmin,fYmax);
   h->SetDirectory(0);

   InitArgs(xv,fParams);
   dx = (fXmax - fXmin)/Double_t(fNpx);
   dy = (fYmax - fYmin)/Double_t(fNpy);
   for (i=1;i<=fNpx;i++) {
      xv[0] = fXmin + (Double_t(i) - 0.5)*dx;
      for (j=1;j<=fNpy;j++) {
         xv[1] = fYmin + (Double_t(j) - 0.5)*dy;
         bin   = j*(fNpx + 2) + i;
         h->SetBinContent(bin,EvalPar(xv,fParams));
      }
   }
   h->Fill(fXmin-1,fYmin-1,0);  //This call to force fNentries non zero

   Double_t *levels = fContour.GetArray();
   if (levels && levels[0] == -9999) levels = 0;
   h->SetMinimum(fMinimum);
   h->SetMaximum(fMaximum);
   h->SetContour(fContour.fN, levels);
   h->SetLineColor(GetLineColor());
   h->SetLineStyle(GetLineStyle());
   h->SetLineWidth(GetLineWidth());
   h->SetFillColor(GetFillColor());
   h->SetFillStyle(GetFillStyle());
   h->SetMarkerColor(GetMarkerColor());
   h->SetMarkerStyle(GetMarkerStyle());
   h->SetMarkerSize(GetMarkerSize());
   h->SetStats(0);

   return h;
}

//______________________________________________________________________________
void TF2::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Paint this 2-D function with its current attributes*-*-*-*-*
//*-*              ===================================================

   Int_t i,j,bin;
   Double_t dx, dy;
   Double_t xv[2];

   TString opt = option;
   opt.ToLower();

//*-*-  Create a temporary histogram and fill each channel with the function value
   if (!fHistogram) {
      fHistogram = new TH2F("Func",(char*)GetTitle(),fNpx,fXmin,fXmax,fNpy,fYmin,fYmax);
      if (!fHistogram) return;
      fHistogram->SetDirectory(0);
   }
   InitArgs(xv,fParams);
   dx = (fXmax - fXmin)/Double_t(fNpx);
   dy = (fYmax - fYmin)/Double_t(fNpy);
   for (i=1;i<=fNpx;i++) {
      xv[0] = fXmin + (Double_t(i) - 0.5)*dx;
      for (j=1;j<=fNpy;j++) {
         xv[1] = fYmin + (Double_t(j) - 0.5)*dy;
         bin   = j*(fNpx + 2) + i;
         fHistogram->SetBinContent(bin,EvalPar(xv,fParams));
      }
   }
   ((TH2F*)fHistogram)->Fill(fXmin-1,fYmin-1,0);  //This call to force fNentries non zero

//*-*- Copy Function attributes to histogram attributes
   Double_t *levels = fContour.GetArray();
   if (levels && levels[0] == -9999) levels = 0;
   fHistogram->SetMinimum(fMinimum);
   fHistogram->SetMaximum(fMaximum);
   fHistogram->SetContour(fContour.fN, levels);
   fHistogram->SetLineColor(GetLineColor());
   fHistogram->SetLineStyle(GetLineStyle());
   fHistogram->SetLineWidth(GetLineWidth());
   fHistogram->SetFillColor(GetFillColor());
   fHistogram->SetFillStyle(GetFillStyle());
   fHistogram->SetMarkerColor(GetMarkerColor());
   fHistogram->SetMarkerStyle(GetMarkerStyle());
   fHistogram->SetMarkerSize(GetMarkerSize());
   fHistogram->SetStats(0);

//*-*-  Draw the histogram
   if (!gPad) return;
   if (opt.Length() == 0)  fHistogram->Paint("cont3");
   else if (opt == "same") fHistogram->Paint("cont2same");
   else                    fHistogram->Paint(option);
}

//______________________________________________________________________________
void TF2::Save(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t, Double_t)
{
    // Save values of function in array fSave

   if (fSave != 0) {delete [] fSave; fSave = 0;}
   Int_t nsave = (fNpx+1)*(fNpy+1);
   fNsave = nsave+6;
   if (fNsave <= 6) {fNsave=0; return;}
   fSave  = new Double_t[fNsave];
   Int_t i,j,k=0;
   Double_t dx = (xmax-xmin)/fNpx;
   Double_t dy = (ymax-ymin)/fNpy;
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
   Double_t xv[2];
   InitArgs(xv,fParams);
   for (j=0;j<=fNpy;j++) {
      xv[1]    = ymin + dy*j;
      for (i=0;i<=fNpx;i++) {
         xv[0]    = xmin + dx*i;
         fSave[k] = EvalPar(xv,fParams);
         k++;
      }
   }
   fSave[nsave+0] = xmin;
   fSave[nsave+1] = xmax;
   fSave[nsave+2] = ymin;
   fSave[nsave+3] = ymax;
   fSave[nsave+4] = fNpx;
   fSave[nsave+5] = fNpy;
}

//______________________________________________________________________________
void TF2::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TF2::Class())) {
      out<<"   ";
   } else {
      out<<"   TF2 *";
   }
   if (!fMethodCall) {
      out<<GetName()<<" = new TF2("<<quote<<GetName()<<quote<<","<<quote<<GetTitle()<<quote<<","<<fXmin<<","<<fXmax<<","<<fYmin<<","<<fYmax<<");"<<endl;
   } else {
      out<<GetName()<<" = new TF2("<<quote<<GetName()<<quote<<","<<GetTitle()<<","<<fXmin<<","<<fXmax<<","<<fYmin<<","<<fYmax<<","<<GetNpar()<<");"<<endl;
   }

   if (GetFillColor() != 0) {
      if (GetFillColor() > 228) {
         TColor::SaveColor(out, GetFillColor());
         out<<"   "<<GetName()<<"->SetFillColor(ci);" << endl;
      } else 
         out<<"   "<<GetName()<<"->SetFillColor("<<GetFillColor()<<");"<<endl;
   }
   if (GetFillStyle() != 1001) {
      out<<"   "<<GetName()<<"->SetFillStyle("<<GetFillStyle()<<");"<<endl;
   }
   if (GetMarkerColor() != 1) {
      if (GetMarkerColor() > 228) {
         TColor::SaveColor(out, GetMarkerColor());
         out<<"   "<<GetName()<<"->SetMarkerColor(ci);" << endl;
      } else 
         out<<"   "<<GetName()<<"->SetMarkerColor("<<GetMarkerColor()<<");"<<endl;
   }
   if (GetMarkerStyle() != 1) {
      out<<"   "<<GetName()<<"->SetMarkerStyle("<<GetMarkerStyle()<<");"<<endl;
   }
   if (GetMarkerSize() != 1) {
      out<<"   "<<GetName()<<"->SetMarkerSize("<<GetMarkerSize()<<");"<<endl;
   }
   if (GetLineColor() != 1) {
      if (GetLineColor() > 228) {
         TColor::SaveColor(out, GetLineColor());
         out<<"   "<<GetName()<<"->SetLineColor(ci);" << endl;
      } else 
         out<<"   "<<GetName()<<"->SetLineColor("<<GetLineColor()<<");"<<endl;
   }
   if (GetLineWidth() != 4) {
      out<<"   "<<GetName()<<"->SetLineWidth("<<GetLineWidth()<<");"<<endl;
   }
   if (GetLineStyle() != 1) {
      out<<"   "<<GetName()<<"->SetLineStyle("<<GetLineStyle()<<");"<<endl;
   }
   if (GetNpx() != 100) {
      out<<"   "<<GetName()<<"->SetNpx("<<GetNpx()<<");"<<endl;
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
void TF2::SetContour(Int_t  nlevels, const Double_t *levels)
{
   //*-*-*-*-*-*-*-*Set the number and values of contour levels*-*-*-*-*-*-*-*-*
   //*-*            ===========================================
   //
   //  By default the number of contour levels is set to 20.
   //
   //  if argument levels = 0 or missing, equidistant contours are computed
   //

   Int_t level;
   if (nlevels <=0 ) {
      fContour.Set(0);
      return;
   }
   fContour.Set(nlevels);

   //*-*-  Contour levels are specified
   if (levels) {
      for (level=0; level<nlevels; level++) fContour.fArray[level] = levels[level];
   } else {
      fContour.fArray[0] = -9999; // means not defined at this point
   }
}


//______________________________________________________________________________
void TF2::SetContourLevel(Int_t level, Double_t value)
{
   //*-*-*-*-*-*-*-*-*-*-*Set value for one contour level*-*-*-*-*-*-*-*-*-*-*-*
   //*-*                  ===============================
   if (level <0 || level >= fContour.fN) return;
   fContour.fArray[level] = value;
}

//______________________________________________________________________________
void TF2::SetNpy(Int_t npy)
{
   // Set the number of points used to draw the function
   //
   // The default number of points along x is 30 for 2-d/3-d functions.
   // You can increase this value to get a better resolution when drawing
   // pictures with sharp peaks or to get a better result when using TF2::GetRandom2   
   // the minimum number of points is 4, the maximum is 10000 for 2-d/3-d functions

   if (npy < 4) {
      Warning("SetNpy","Number of points must be >=4 && <= 10000, fNpy set to 4");
      fNpy = 4;
   } else if(npy > 10000) {
      Warning("SetNpy","Number of points must be >=4 && <= 10000, fNpy set to 10000");
      fNpy = 10000;
   } else {
      fNpy = npy;
   }
   Update();
}

//______________________________________________________________________________
void TF2::SetRange(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
//*-*-*-*-*-*Initialize the upper and lower bounds to draw the function*-*-*-*
//*-*        ==========================================================

   fXmin = xmin;
   fXmax = xmax;
   fYmin = ymin;
   fYmax = ymax;
   Update();
}

//______________________________________________________________________________
void TF2::Streamer(TBuffer &R__b)
{
   // Stream an object of class TF2.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 3) {
         R__b.ReadClassBuffer(TF2::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      Int_t nlevels;
      TF1::Streamer(R__b);
      if (R__v < 3) {
         Float_t ymin,ymax;
         R__b >> ymin; fYmin = ymin;
         R__b >> ymax; fYmax = ymax;
      } else {
         R__b >> fYmin;
         R__b >> fYmax;
      }
      R__b >> fNpy;
      R__b >> nlevels;
      if (R__v < 3) {
         Float_t *contour = 0;
         Int_t n = R__b.ReadArray(contour);
         fContour.Set(n);
         for (Int_t i=0;i<n;i++) fContour.fArray[i] = contour[i];
         delete [] contour;
      } else {
         fContour.Streamer(R__b);
      }
      R__b.CheckByteCount(R__s, R__c, TF2::IsA());
      //====end of old versions

   } else {
      Int_t saved = 0;
      if (fType > 0 && fNsave <= 0) { saved = 1; Save(fXmin,fXmax,fYmin,fYmax,0,0);}

      R__b.WriteClassBuffer(TF2::Class(),this);

      if (saved) {delete [] fSave; fSave = 0; fNsave = 0;}
   }
}

//______________________________________________________________________________
Double_t TF2::Moment2(Double_t nx, Double_t ax, Double_t bx, Double_t ny, Double_t ay, Double_t by, Double_t epsilon)
{
// Return x^nx * y^ny moment of a 2d function in range [ax,bx],[ay,by]
//   Author: Gene Van Buren <gene@bnl.gov>

   Double_t norm = Integral(ax,bx,ay,by,epsilon);
   if (norm == 0) {
      Error("Moment2", "Integral zero over range");
      return 0;
   }

   TF2 fnc("TF2_ExpValHelper",Form("%s*pow(x,%f)*pow(y,%f)",GetName(),nx,ny));
   return fnc.Integral(ax,bx,ay,by,epsilon)/norm;
}

//______________________________________________________________________________
Double_t TF2::CentralMoment2(Double_t nx, Double_t ax, Double_t bx, Double_t ny, Double_t ay, Double_t by, Double_t epsilon)
{
// Return x^nx * y^ny central moment of a 2d function in range [ax,bx],[ay,by]
//   Author: Gene Van Buren <gene@bnl.gov>

   Double_t norm = Integral(ax,bx,ay,by,epsilon);
   if (norm == 0) {
      Error("CentralMoment2", "Integral zero over range");
      return 0;
   }

   Double_t xbar = 0;
   Double_t ybar = 0;
   if (nx!=0) {
      TF2 fncx("TF2_ExpValHelperx",Form("%s*x",GetName()));
      xbar = fncx.Integral(ax,bx,ay,by,epsilon)/norm;
   }
   if (ny!=0) {
      TF2 fncx("TF2_ExpValHelpery",Form("%s*y",GetName()));
      ybar = fncx.Integral(ax,bx,ay,by,epsilon)/norm;
   }
   TF2 fnc("TF2_ExpValHelper",Form("%s*pow(x-%f,%f)*pow(y-%f,%f)",GetName(),xbar,nx,ybar,ny));
   return fnc.Integral(ax,bx,ay,by,epsilon)/norm;
}

