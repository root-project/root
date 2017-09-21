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

#include <TString.h>
#include <memory>

class TH1D;
class TH2D;
class TH3D;

namespace ROOT {
namespace Experimental {
namespace TDF {

struct TH1DModel {
   TString fName;
   TString fTitle;
   int fNbinsX = 128;
   double fXLow = 0.;
   double fXUp = 64.;

   TH1DModel() = default;
   TH1DModel(const TH1DModel &) = default;
   ~TH1DModel();
   TH1DModel(const ::TH1D &h);
   TH1DModel(const char *name, const char *title, int nbinsx, double xlow, double xup);
};

struct TH2DModel {
   TString fName;
   TString fTitle;
   int fNbinsX = 128;
   double fXLow = 0.;
   double fXUp = 64.;
   int fNbinsY = 128;
   double fYLow = 0.;
   double fYUp = 64.;

   TH2DModel() = default;
   TH2DModel(const TH2DModel &) = default;
   ~TH2DModel();
   TH2DModel(const ::TH2D &h);
   TH2DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy, double ylow,
             double yup);
};

struct TH3DModel {
   TString fName;
   TString fTitle;
   int fNbinsX = 128;
   double fXLow = 0.;
   double fXUp = 64.;
   int fNbinsY = 128;
   double fYLow = 0.;
   double fYUp = 64.;
   int fNbinsZ = 128;
   double fZLow = 0.;
   double fZUp = 64.;

   TH3DModel() = default;
   TH3DModel(const TH3DModel &) = default;
   ~TH3DModel();
   TH3DModel(const ::TH3D &h);
   TH3DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy, double ylow,
             double yup, int nbinsz, double zlow, double zup);
};

} // ns TDF
} // ns Experimental
} // ns ROOT

#endif // ROOT_TDFHISTOMODELS
