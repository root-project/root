// @(#)root/hist:$Name:  $:$Id: TF3.cxx,v 1.15 2003/08/20 07:00:47 brun Exp $
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
#include "TPainter3dAlgorithms.h"

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
TF3& TF3::operator=(const TF3 &rhs) 
{
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
Bool_t TF3::IsInside(const Double_t *x) const
{
// Return kTRUE is the point is inside the function range
   
   if (x[0] < fXmin || x[0] > fXmax) return kFALSE;
   if (x[1] < fYmin || x[1] > fYmax) return kFALSE;
   if (x[2] < fZmin || x[2] > fZmax) return kFALSE;
   return kTRUE;
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

   fHistogram->GetPainter()->ProcessMessage("SetF3",this);
   fHistogram->Paint("tf3");
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
      Warning("SetNpz","Number of points must be >4 && < 10000, fNpz set to 4");
      fNpz = 4;
   } else if(npz > 100000) {
      Warning("SetNpz","Number of points must be >4 && < 10000, fNpz set to 10000");
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

