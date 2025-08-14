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
#include "TBuffer.h"
#include "TMath.h"
#include "TH3.h"
#include "TVirtualPad.h"
#include "TRandom.h"
#include "TVectorD.h"
#include "Riostream.h"
#include "TColor.h"
#include "TVirtualFitter.h"
#include "TVirtualHistPainter.h"
#include "Math/IntegratorOptions.h"
#include <cassert>

ClassImp(TF3);

/** \class TF3
    \ingroup Functions
A 3-Dim function with parameters
*/

////////////////////////////////////////////////////////////////////////////////
/// F3 default constructor

TF3::TF3()
{
   fNpz  = 0;
   fZmin = 0;
   fZmax = 1;
}


////////////////////////////////////////////////////////////////////////////////
/// F3 constructor using a formula definition
///
/// See TFormula constructor for explanation of the formula syntax.

TF3::TF3(const char *name,const char *formula, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Option_t * opt)
   :TF2(name,formula,xmin,xmax,ymax,ymin,opt)
{
   fZmin   = zmin;
   fZmax   = zmax;
   fNpz    = 30;
   Int_t ndim = GetNdim();
   // accept 1-d or 2-d formula
   if (ndim < 3) fNdim = 3;
   if (ndim > 3 && xmin < xmax && ymin < ymax && zmin < zmax) {
      Error("TF3","function: %s/%s has dimension %d instead of 3",name,formula,ndim);
      MakeZombie();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// F3 constructor using a pointer to real function
///
/// \param[in] name object name
/// \param[in] fcn pointer to real function
/// \param[in] xmin,xmax x axis limits
/// \param[in] ymin,ymax y axis limits
/// \param[in] zmin,zmax z axis limits
/// \param[in] npar is the number of free parameters used by the function
/// \param[in] ndim number of dimensions
///
/// For example, for a 3-dim function with 3 parameters, the user function
/// looks like:
///
///     Double_t fun1(Double_t *x, Double_t *par)
///     return par[0]*x[2] + par[1]*exp(par[2]*x[0]*x[1]);
///
/// \warning A function created with this constructor cannot be Cloned.

TF3::TF3(const char *name,Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar, Int_t ndim, EAddToList addToGlobList)
      :TF2(name,fcn,xmin,xmax,ymin,ymax,npar,ndim,addToGlobList)
{
   fZmin   = zmin;
   fZmax   = zmax;
   fNpz    = 30;
}

////////////////////////////////////////////////////////////////////////////////
/// F3 constructor using a pointer to real function---
///
/// \param[in] name object name
/// \param[in] fcn pointer to real function
/// \param[in] xmin,xmax x axis limits
/// \param[in] ymin,ymax y axis limits
/// \param[in] zmin,zmax z axis limits
/// \param[in] npar is the number of free parameters used by the function
/// \param[in] ndim number of dimensions
///
/// For example, for a 3-dim function with 3 parameters, the user function
/// looks like:
///
///     Double_t fun1(Double_t *x, Double_t *par)
///     return par[0]*x[2] + par[1]*exp(par[2]*x[0]*x[1]);
///
/// WARNING! A function created with this constructor cannot be Cloned.

TF3::TF3(const char *name,Double_t (*fcn)(const Double_t *, const Double_t *), Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar, Int_t ndim, EAddToList addToGlobList)
   : TF2(name,fcn,xmin,xmax,ymin,ymax,npar,ndim,addToGlobList),
   fZmin(zmin),
   fZmax(zmax),
   fNpz(30)
{
}

////////////////////////////////////////////////////////////////////////////////
/// F3 constructor using a ParamFunctor
///
/// a functor class implementing operator() (double *, double *)
///
/// \param[in] name object name
/// \param[in] f parameter functor
/// \param[in] xmin,xmax x axis limits
/// \param[in] ymin,ymax y axis limits
/// \param[in] zmin,zmax z axis limits
/// \param[in] npar is the number of free parameters used by the function
/// \param[in] ndim number of dimensions
///
/// \warning A function created with this constructor cannot be Cloned.

TF3::TF3(const char *name, ROOT::Math::ParamFunctor f, Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax, Int_t npar, Int_t ndim, EAddToList addToGlobList)
   : TF2(name, f, xmin, xmax, ymin, ymax,  npar, ndim, addToGlobList),
   fZmin(zmin),
   fZmax(zmax),
   fNpz(30)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Operator =

TF3& TF3::operator=(const TF3 &rhs)
{
   if (this != &rhs)
      rhs.TF3::Copy(*this);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// F3 default destructor

TF3::~TF3()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TF3::TF3(const TF3 &f3) : TF2()
{
   f3.TF3::Copy(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy this F3 to a new F3

void TF3::Copy(TObject &obj) const
{
   TF2::Copy(obj);
   ((TF3&)obj).fZmin = fZmin;
   ((TF3&)obj).fZmax = fZmax;
   ((TF3&)obj).fNpz  = fNpz;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to a function
///
///  Compute the closest distance of approach from point px,py to this function.
///  The distance is computed in pixels units.


Int_t TF3::DistancetoPrimitive(Int_t px, Int_t py)
{
   return TF1::DistancetoPrimitive(px, py);

}

////////////////////////////////////////////////////////////////////////////////
/// Draw this function with its current attributes

void TF3::Draw(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();

   AppendPad(option);

}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event
///
///  This member function is called when a F3 is clicked with the locator

void TF3::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   TF1::ExecuteEvent(event, px, py);
}

////////////////////////////////////////////////////////////////////////////////
/// Return minimum/maximum value of the function
///
/// To find the minimum on a range, first set this range via the SetRange function
/// If a vector x of coordinate is passed it will be used as starting point for the minimum.
/// In addition on exit x will contain the coordinate values at the minimuma
/// If x is NULL or x is inifinity or NaN, first, a grid search is performed to find the initial estimate of the
/// minimum location. The range of the function is divided into fNpx and fNpy
/// sub-ranges. If the function is "good" (or "bad"), these values can be changed
/// by SetNpx and SetNpy functions
///
/// Then, a minimization is used with starting values found by the grid search
/// The minimizer algorithm used (by default Minuit) can be changed by callinga
///  ROOT::Math::Minimizer::SetDefaultMinimizerType("..")
/// Other option for the minimizer can be set using the static method of the MinimizerOptions class

Double_t TF3::FindMinMax(Double_t *x, Bool_t findmax) const
{
   //First do a grid search with step size fNpx and fNpy

   Double_t xx[3];
   Double_t rsign = (findmax) ? -1. : 1.;
   TF3 & function = const_cast<TF3&>(*this); // needed since EvalPar is not const
   Double_t xxmin = 0, yymin = 0, zzmin = 0, ttmin = 0;
   if (x == nullptr || ( (x!= nullptr) && ( !TMath::Finite(x[0]) || !TMath::Finite(x[1]) || !TMath::Finite(x[2]) ) ) ){
      Double_t dx = (fXmax - fXmin)/fNpx;
      Double_t dy = (fYmax - fYmin)/fNpy;
      Double_t dz = (fZmax - fZmin)/fNpz;
      xxmin = fXmin;
      yymin = fYmin;
      zzmin = fZmin;
      ttmin = rsign * TMath::Infinity();
      for (Int_t i=0; i<fNpx; i++){
         xx[0]=fXmin + (i+0.5)*dx;
         for (Int_t j=0; j<fNpy; j++){
            xx[1]=fYmin+(j+0.5)*dy;
            for (Int_t k=0; k<fNpz; k++){
               xx[2] = fZmin+(k+0.5)*dz;
               Double_t tt = function(xx);
               if (rsign*tt < rsign*ttmin) {xxmin = xx[0], yymin = xx[1]; zzmin = xx[2]; ttmin=tt;}
            }
         }
      }

      xxmin = TMath::Min(fXmax, xxmin);
      yymin = TMath::Min(fYmax, yymin);
      zzmin = TMath::Min(fZmax, zzmin);
   }
   else {
      xxmin = x[0];
      yymin = x[1];
      zzmin = x[2];
      zzmin = function(x);
   }
   xx[0] = xxmin;
   xx[1] = yymin;
   xx[2] = zzmin;

   double fmin = GetMinMaxNDim(xx,findmax);
   if (rsign*fmin < rsign*zzmin) {
      if (x) {x[0] = xx[0]; x[1] = xx[1]; x[2] = xx[2];}
      return fmin;
   }
   // here if minimization failed
   if (x) { x[0] = xxmin; x[1] = yymin; x[2] = zzmin; }
   return ttmin;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the X, Y and Z values corresponding to the minimum value of the function
/// on its range.
///
/// Returns the function value at the minimum.
/// To find the minimum on a subrange, use the SetRange() function first.
///
/// Method:
///   First, a grid search is performed to find the initial estimate of the
///   minimum location. The range of the function is divided
///   into fNpx,fNpy and fNpz sub-ranges. If the function is "good" (or "bad"),
///   these values can be changed by SetNpx(), SetNpy() and SetNpz() functions.
///   Then, Minuit minimization is used with starting values found by the grid search
///
///   Note that this method will always do first a grid search in contrast to GetMinimum

Double_t TF3::GetMinimumXYZ(Double_t &x, Double_t &y, Double_t &z)
{
   double xx[3] = { 0,0,0 };
   xx[0] = TMath::QuietNaN();  // to force to do grid search in TF3::FindMinMax
   double fmin = FindMinMax(xx, false);
   x = xx[0]; y = xx[1]; z = xx[2];
   return fmin;

}

////////////////////////////////////////////////////////////////////////////////
/// Compute the X, Y and Z values corresponding to the maximum value of the function
/// on its range.
///
/// Return the function value at the maximum. See TF3::GetMinimumXYZ

Double_t TF3::GetMaximumXYZ(Double_t &x, Double_t &y, Double_t &z)
{
   double xx[3] = { 0,0,0 };
   xx[0] = TMath::QuietNaN();  // to force to do grid search in TF3::FindMinMax
   double fmax = FindMinMax(xx, true);
   x = xx[0]; y = xx[1]; z = xx[2];
   return fmax;

}

////////////////////////////////////////////////////////////////////////////////
/// Return 3 random numbers following this function shape
///
/// The distribution contained in this TF3 function is integrated
/// over the cell contents.
/// It is normalized to 1.
/// Getting the three random numbers implies:
///   - Generating a random number between 0 and 1 (say r1)
///   - Look in which cell in the normalized integral r1 corresponds to
///   - make a linear interpolation in the returned cell
///
///  IMPORTANT NOTE
///
///  The integral of the function is computed at fNpx * fNpy * fNpz points.
///  If the function has sharp peaks, you should increase the number of
///  points (SetNpx, SetNpy, SetNpz) such that the peak is correctly tabulated
///  at several points.

void TF3::GetRandom3(Double_t &xrandom, Double_t &yrandom, Double_t &zrandom, TRandom * rng)
{
   //  Check if integral array must be built
   Int_t i,j,k,cell;
   Double_t dx   = (fXmax-fXmin)/fNpx;
   Double_t dy   = (fYmax-fYmin)/fNpy;
   Double_t dz   = (fZmax-fZmin)/fNpz;
   Int_t ncells = fNpx*fNpy*fNpz;
   Double_t xx[3];
   Double_t *parameters = GetParameters();
   InitArgs(xx,parameters);
   if (fIntegral.empty() ) {
      fIntegral.resize(ncells+1);
      //fIntegral = new Double_t[ncells+1];
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
               integ = EvalPar(xx,parameters);
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
   if (!rng) rng = gRandom;
   r    = rng->Rndm();
   cell = TMath::BinarySearch(ncells,fIntegral.data(),r);
   k    = cell/(fNpx*fNpy);
   j    = (cell -k*fNpx*fNpy)/fNpx;
   i    = cell -fNpx*(j +fNpy*k);
   xrandom = fXmin +dx*i +dx*rng->Rndm();
   yrandom = fYmin +dy*j +dy*rng->Rndm();
   zrandom = fZmin +dz*k +dz*rng->Rndm();
}

////////////////////////////////////////////////////////////////////////////////
/// Return range of function

void TF3::GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax) const
{
   xmin = fXmin;
   xmax = fXmax;
   ymin = fYmin;
   ymax = fYmax;
   zmin = fZmin;
   zmax = fZmax;
}


////////////////////////////////////////////////////////////////////////////////
/// Get value corresponding to X in array of fSave values

Double_t TF3::GetSave(const Double_t *xx)
{
   if (fSave.size() < 9) return 0;
   Int_t nsave = fSave.size() - 9;
   Double_t xmin = fSave[nsave+0];
   Double_t xmax = fSave[nsave+1];
   Double_t ymin = fSave[nsave+2];
   Double_t ymax = fSave[nsave+3];
   Double_t zmin = fSave[nsave+4];
   Double_t zmax = fSave[nsave+5];
   Int_t npx     = Int_t(fSave[nsave+6]);
   Int_t npy     = Int_t(fSave[nsave+7]);
   Int_t npz     = Int_t(fSave[nsave+8]);
   Double_t x    = xx[0];
   Double_t dx   = (xmax-xmin)/npx;
   if (x < xmin || x > xmax) return 0;
   if (dx <= 0) return 0;
   Double_t y    = xx[1];
   Double_t dy   = (ymax-ymin)/npy;
   if (y < ymin || y > ymax) return 0;
   if (dy <= 0) return 0;
   Double_t z    = xx[2];
   Double_t dz   = (zmax-zmin)/npz;
   if (z < zmin || z > zmax) return 0;
   if (dz <= 0) return 0;

   //we make a trilinear interpolation using the 8 points surrounding x,y,z
   Int_t ibin    = TMath::Min(npx-1, Int_t((x-xmin)/dx));
   Int_t jbin    = TMath::Min(npy-1, Int_t((y-ymin)/dy));
   Int_t kbin    = TMath::Min(npz-1, Int_t((z-zmin)/dz));
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

////////////////////////////////////////////////////////////////////////////////
/// Return Integral of a 3d function in range [ax,bx],[ay,by],[az,bz]
/// with a desired relative accuracy.

Double_t TF3::Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t az, Double_t bz, Double_t epsrel)
{
   Double_t a[3], b[3];
   a[0] = ax;
   b[0] = bx;
   a[1] = ay;
   b[1] = by;
   a[2] = az;
   b[2] = bz;
   Double_t relerr  = 0;
   Int_t n = 3;
   Int_t maxpts = TMath::Max(UInt_t(fNpx * fNpy * fNpz), ROOT::Math::IntegratorMultiDimOptions::DefaultNCalls());
   Int_t nfnevl, ifail;
   Double_t result = IntegralMultiple(n,a,b,maxpts,epsrel,epsrel, relerr,nfnevl,ifail);
   if (ifail > 0) {
      Warning("Integral","failed for %s code=%d, maxpts=%d, epsrel=%g, nfnevl=%d, relerr=%g ",GetName(),ifail,maxpts,epsrel,nfnevl,relerr);
   }
   if (gDebug) {
      Info("Integral","Integral of %s using %d and tol=%f is %f , relerr=%f nfcn=%d",GetName(),maxpts,epsrel,result,relerr,nfnevl);
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Return kTRUE is the point is inside the function range

Bool_t TF3::IsInside(const Double_t *x) const
{
   if (x[0] < fXmin || x[0] > fXmax) return kFALSE;
   if (x[1] < fYmin || x[1] > fYmax) return kFALSE;
   if (x[2] < fZmin || x[2] > fZmax) return kFALSE;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a histogram for axis range.

TH1* TF3::CreateHistogram()
{
   TH1* h = new TH3F("R__TF3",(char*)GetTitle(),fNpx,fXmin,fXmax
                         ,fNpy,fYmin,fYmax
                         ,fNpz,fZmin,fZmax);
   h->SetDirectory(nullptr);
   return h;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this 3-D function with its current attributes

void TF3::Paint(Option_t *option)
{

   TString opt = option;
   opt.ToLower();

//-  Create a temporary histogram and fill each channel with the function value
   if (!fHistogram) {
      fHistogram = new TH3F("R__TF3",(char*)GetTitle(),fNpx,fXmin,fXmax
                                                      ,fNpy,fYmin,fYmax
                                                      ,fNpz,fZmin,fZmax);
      fHistogram->SetDirectory(nullptr);
   }

   fHistogram->GetPainter(option)->ProcessMessage("SetF3",this);

   if (opt.Index("tf3") == kNPOS)
      opt.Append("tf3");

   fHistogram->Paint(opt.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Set the function clipping box (for drawing) "off".

void TF3::SetClippingBoxOff()
{
   fClipBoxOn = kFALSE;
   fClipBox[0] = fClipBox[1] = fClipBox[2] = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Save values of function in array fSave

void TF3::Save(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax)
{
   if (!fSave.empty()) fSave.clear();
   Int_t npx = fNpx, npy = fNpy, npz = fNpz;
   if ((npx < 2) || (npy < 2) || (npz < 2))
      return;

   Double_t dx = (xmax-xmin)/fNpx;
   Double_t dy = (ymax-ymin)/fNpy;
   Double_t dz = (zmax-zmin)/fNpz;
   if (dx <= 0) {
      dx = (fXmax-fXmin)/fNpx;
      npx--;
      xmin = fXmin + 0.5*dx;
      xmax = fXmax - 0.5*dx;
   }
   if (dy <= 0) {
      dy = (fYmax-fYmin)/fNpy;
      npy--;
      ymin = fYmin + 0.5*dy;
      ymax = fYmax - 0.5*dy;
   }
   if (dz <= 0) {
      dz = (fZmax-fZmin)/fNpz;
      npz--;
      zmin = fZmin + 0.5*dz;
      zmax = fZmax - 0.5*dz;
   }
   Int_t nsave = (npx + 1)*(npy + 1)*(npz + 1);
   fSave.resize(nsave + 9);
   Double_t xv[3];
   Double_t *pp = GetParameters();
   InitArgs(xv,pp);
   for (Int_t k = 0, l = 0; k <= npz; k++) {
      xv[2]    = zmin + dz*k;
      for (Int_t j = 0; j <= npy; j++) {
         xv[1]    = ymin + dy*j;
         for (Int_t i = 0; i <= npx; i++) {
            xv[0]    = xmin + dx*i;
            fSave[l++] = EvalPar(xv, pp);
         }
      }
   }
   fSave[nsave+0] = xmin;
   fSave[nsave+1] = xmax;
   fSave[nsave+2] = ymin;
   fSave[nsave+3] = ymax;
   fSave[nsave+4] = zmin;
   fSave[nsave+5] = zmax;
   fSave[nsave+6] = npx;
   fSave[nsave+7] = npy;
   fSave[nsave+8] = npz;
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TF3::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   TString f3Name = ProvideSaveName(option);
   out << "   \n";
   out << "   TF3 *";

   if (!fMethodCall)
      out << f3Name << " = new TF3(\"" << GetName() << "\", \"" << TString(GetTitle()).ReplaceSpecialCppChars() << "\","
          << fXmin << "," << fXmax << "," << fYmin << "," << fYmax << "," << fZmin << "," << fZmax << ");\n";
   else
      out << f3Name << " = new TF3(\"" << GetName() << "\", " << GetTitle() << "," << fXmin << "," << fXmax << ","
          << fYmin << "," << fYmax << "," << fZmin << "," << fZmax << "," << GetNpar() << ");\n";

   SaveFillAttributes(out, f3Name, 0, 1001);
   SaveMarkerAttributes(out, f3Name, 1, 1, 1);
   SaveLineAttributes(out, f3Name, 1, 1, 4);

   if (GetNpx() != 30)
      out << "   " << f3Name << "->SetNpx(" << GetNpx() << ");\n";
   if (GetNpy() != 30)
      out << "   " << f3Name << "->SetNpy(" << GetNpy() << ");\n";
   if (GetNpz() != 30)
      out << "   " << f3Name << "->SetNpz(" << GetNpz() << ");\n";

   if (GetChisquare() != 0)
      out << "   " << f3Name << "->SetChisquare(" << GetChisquare() << ");\n";

   Double_t parmin, parmax;
   for (Int_t i = 0; i < GetNpar(); i++) {
      out << "   " << f3Name << "->SetParameter(" << i << "," << GetParameter(i) << ");\n";
      out << "   " << f3Name << "->SetParError(" << i << "," << GetParError(i) << ");\n";
      GetParLimits(i, parmin, parmax);
      out << "   " << f3Name << "->SetParLimits(" << i << "," << parmin << "," << parmax << ");\n";
   }

   if (GetXaxis())
      GetXaxis()->SaveAttributes(out, f3Name, "->GetXaxis()");
   if (GetYaxis())
      GetYaxis()->SaveAttributes(out, f3Name, "->GetYaxis()");
   if (GetZaxis())
      GetZaxis()->SaveAttributes(out, f3Name, "->GetZaxis()");

   SavePrimitiveDraw(out, f3Name, option);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the function clipping box (for drawing) "on" and define the clipping box.
/// xclip, yclip and zclip is a point within the function range. All the
/// function values having x<=xclip and y<=yclip and z>=zclip are clipped.

void TF3::SetClippingBoxOn(Double_t xclip, Double_t yclip, Double_t zclip)
{
   fClipBoxOn = kTRUE;
   fClipBox[0] = xclip;
   fClipBox[1] = yclip;
   fClipBox[2] = zclip;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the number of points used to draw the function
///
/// The default number of points along x is 30 for 2-d/3-d functions.
/// You can increase this value to get a better resolution when drawing
/// pictures with sharp peaks or to get a better result when using TF3::GetRandom2
/// the minimum number of points is 4, the maximum is 10000 for 2-d/3-d functions

void TF3::SetNpz(Int_t npz)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Initialize the upper and lower bounds to draw the function

void TF3::SetRange(Double_t xmin, Double_t ymin, Double_t zmin, Double_t xmax, Double_t ymax, Double_t zmax)
{
   fXmin = xmin;
   fXmax = xmax;
   fYmin = ymin;
   fYmax = ymax;
   fZmin = zmin;
   fZmax = zmax;
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TF3.

void TF3::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 0) {
         R__b.ReadClassBuffer(TF3::Class(), this, R__v, R__s, R__c);
         return;
      }

   } else {
      Int_t saved = 0;
      if (fType != EFType::kFormula && fSave.empty() ) { saved = 1; Save(fXmin,fXmax,fYmin,fYmax,fZmin,fZmax);}

      R__b.WriteClassBuffer(TF3::Class(),this);

      if (saved) { fSave.clear(); }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return x^nx * y^ny * z^nz moment of a 3d function in range [ax,bx],[ay,by],[az,bz]
/// \author Gene Van Buren <gene@bnl.gov>

Double_t TF3::Moment3(Double_t nx, Double_t ax, Double_t bx, Double_t ny, Double_t ay, Double_t by, Double_t nz, Double_t az, Double_t bz, Double_t epsilon)
{
   Double_t norm = Integral(ax,bx,ay,by,az,bz,epsilon);
   if (norm == 0) {
      Error("Moment3", "Integral zero over range");
      return 0;
   }

   // define  integrand function as a lambda : g(x,y,z)=  x^(nx) * y^(ny) * z^(nz) * f(x,y,z)
   auto integrand = [&](double *x, double *) {
      return std::pow(x[0], nx) * std::pow(x[1], ny) * std::pow(x[2], nz) * this->EvalPar(x, nullptr);
   };
   // compute integral of g(x,y,z)
   TF3 fnc("TF3_ExpValHelper", integrand, ax, bx, ay, by, az, bz, 0);
   // set same points as current function to get correct max points when computing the integral
   fnc.fNpx = fNpx;
   fnc.fNpy = fNpy;
   fnc.fNpz = fNpz;
   return fnc.Integral(ax, bx, ay, by, az, bz, epsilon) / norm;
}

////////////////////////////////////////////////////////////////////////////////
/// Return x^nx * y^ny * z^nz central moment of a 3d function in range [ax,bx],[ay,by],[az,bz]
/// \author Gene Van Buren <gene@bnl.gov>

Double_t TF3::CentralMoment3(Double_t nx, Double_t ax, Double_t bx, Double_t ny, Double_t ay, Double_t by, Double_t nz, Double_t az, Double_t bz, Double_t epsilon)
{
   Double_t norm = Integral(ax,bx,ay,by,az,bz,epsilon);
   if (norm == 0) {
      Error("CentralMoment3", "Integral zero over range");
      return 0;
   }

   Double_t xbar = 0;
   Double_t ybar = 0;
   Double_t zbar = 0;
   if (nx!=0) {
      // compute first momentum in x
      auto integrandX = [&](double *x, double *) { return x[0] * this->EvalPar(x, nullptr); };
      TF3 fncx("TF3_ExpValHelperx", integrandX, ax, bx, ay, by, az, bz, 0);
      fncx.fNpx = fNpx;
      fncx.fNpy = fNpy;
      fncx.fNpz = fNpz;
      xbar = fncx.Integral(ax, bx, ay, by, az, bz, epsilon) / norm;
   }
   if (ny!=0) {
      auto integrandY = [&](double *x, double *) { return x[1] * this->EvalPar(x, nullptr); };
      TF3 fncy("TF3_ExpValHelpery", integrandY, ax, bx, ay, by, az, bz, 0);
      fncy.fNpx = fNpx;
      fncy.fNpy = fNpy;
      fncy.fNpz = fNpz;
      ybar = fncy.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
   }
   if (nz!=0) {
      auto integrandZ = [&](double *x, double *) { return x[2] * this->EvalPar(x, nullptr); };
      TF3 fncz("TF3_ExpValHelperz", integrandZ, ax, bx, ay, by, az, bz, 0);
      fncz.fNpx = fNpx;
      fncz.fNpy = fNpy;
      fncz.fNpz = fNpz;
      zbar = fncz.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
   }
   // define  integrand function as a lambda : g(x,y)=  (x-xbar)^(nx) * (y-ybar)^(ny) * f(x,y)
   auto integrand = [&](double *x, double *) {
      double xxx = (nx != 0) ? std::pow(x[0] - xbar, nx) : 1.;
      double yyy = (ny != 0) ? std::pow(x[1] - ybar, ny) : 1.;
      double zzz = (nz != 0) ? std::pow(x[2] - zbar, nz) : 1.;
      return xxx * yyy * zzz * this->EvalPar(x, nullptr);
   };
   // compute integral of g(x,y, z)
   TF3 fnc("TF3_ExpValHelper",integrand,ax,bx,ay,by,az,bz,0) ;
   fnc.fNpx = fNpx;
   fnc.fNpy = fNpy;
   fnc.fNpz = fNpz;
   return fnc.Integral(ax,bx,ay,by,az,bz,epsilon)/norm;
}
