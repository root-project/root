// Author: Enrico Guiraud, Danilo Piparo CERN  09/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDFHISTOMODELS
#define ROOT_TDFHISTOMODELS

#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <memory>

namespace ROOT {
namespace Experimental {
namespace TDF {

class TH1DModel {
public:
   TString fName;
   TString fTitle;
   int fNbinsX;
   double fXLow;
   double fXUp;

   TH1DModel() = delete;
   TH1DModel(const TH1DModel &) = delete;
   TH1DModel(const ::TH1D &h) : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()), fXUp(h.GetXaxis()->GetXmax())
   {
   }
   TH1DModel(const char *name, const char *title, int nbinsx, double xlow, double xup)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup)
   {
   }
};

class TH2DModel {
public:
   TString fName;
   TString fTitle;
   int fNbinsX;
   double fXLow;
   double fXUp;
   int fNbinsY;
   double fYLow;
   double fYUp;

   TH2DModel() = delete;
   TH2DModel(const TH2DModel &) = delete;
   TH2DModel(const ::TH2D &h) : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()), fXUp(h.GetXaxis()->GetXmax()), fNbinsY(h.GetNbinsY()), fYLow(h.GetYaxis()->GetXmin()), fYUp(h.GetYaxis()->GetXmax())
   {
   }
   TH2DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy, double ylow, double yup)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fNbinsY(nbinsy), fYLow(ylow), fYUp(yup)
   {
   }
};

class TH3DModel {
public:
   TString fName;
   TString fTitle;
   int fNbinsX;
   double fXLow;
   double fXUp;
   int fNbinsY;
   double fYLow;
   double fYUp;
   int fNbinsZ;
   double fZLow;
   double fZUp;

   TH3DModel() = delete;
   TH3DModel(const TH3DModel &) = delete;
   TH3DModel(const ::TH3D &h) : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()), fXUp(h.GetXaxis()->GetXmax()), fNbinsY(h.GetNbinsY()), fYLow(h.GetYaxis()->GetXmin()), fYUp(h.GetYaxis()->GetXmax()), fNbinsZ(h.GetNbinsZ()), fZLow(h.GetZaxis()->GetXmin()), fZUp(h.GetZaxis()->GetXmax())
   {
   }
   TH3DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy, double ylow, double yup, int nbinsz, double zlow, double zup)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fNbinsY(nbinsy), fYLow(ylow), fYUp(yup), fNbinsZ(nbinsz), fZLow(zlow), fZUp(zup)
   {
   }
};


} // ns TDF
} // ns Experimental
} // ns ROOT

#endif // ROOT_TDFHISTOMODELS
