// Author: Enrico Guiraud, Danilo Piparo CERN  09/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TDFHistoModels.hxx>

#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TProfile.h>
#include <TProfile2D.h>

/**
* \class ROOT::Experimental::TDF::TH1DModel
* \ingroup dataframe
* \brief A struct which stores the parameters of a TH1D
*
* \class ROOT::Experimental::TDF::TH2DModel
* \ingroup dataframe
* \brief A struct which stores the parameters of a TH2D
*
* \class ROOT::Experimental::TDF::TH3DModel
* \ingroup dataframe
* \brief A struct which stores the parameters of a TH3D
*
* \class ROOT::Experimental::TDF::TProfile1DModel
* \ingroup dataframe
* \brief A struct which stores the parameters of a TProfile
*
* \class ROOT::Experimental::TDF::TProfile2DModel
* \ingroup dataframe
* \brief A struct which stores the parameters of a TProfile2D
*/

namespace ROOT {
namespace Experimental {
namespace TDF {

TH1DModel::TH1DModel(const ::TH1D &h)
   : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()),
     fXUp(h.GetXaxis()->GetXmax())
{
}
TH1DModel::TH1DModel(const char *name, const char *title, int nbinsx, double xlow, double xup)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup)
{
}
TH1DModel::~TH1DModel()
{
}

TH2DModel::TH2DModel(const ::TH2D &h)
   : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()),
     fXUp(h.GetXaxis()->GetXmax()), fNbinsY(h.GetNbinsY()), fYLow(h.GetYaxis()->GetXmin()),
     fYUp(h.GetYaxis()->GetXmax())
{
}
TH2DModel::TH2DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy, double ylow,
                     double yup)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fNbinsY(nbinsy), fYLow(ylow), fYUp(yup)
{
}
TH2DModel::~TH2DModel()
{
}

TH3DModel::TH3DModel(const ::TH3D &h)
   : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()),
     fXUp(h.GetXaxis()->GetXmax()), fNbinsY(h.GetNbinsY()), fYLow(h.GetYaxis()->GetXmin()),
     fYUp(h.GetYaxis()->GetXmax()), fNbinsZ(h.GetNbinsZ()), fZLow(h.GetZaxis()->GetXmin()),
     fZUp(h.GetZaxis()->GetXmax())
{
}
TH3DModel::TH3DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy, double ylow,
                     double yup, int nbinsz, double zlow, double zup)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fNbinsY(nbinsy), fYLow(ylow), fYUp(yup),
     fNbinsZ(nbinsz), fZLow(zlow), fZUp(zup)
{
}
TH3DModel::~TH3DModel()
{
}

// Profiles

TProfile1DModel::TProfile1DModel(const ::TProfile &h)
   : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()),
     fXUp(h.GetXaxis()->GetXmax()), fYLow(h.GetYaxis()->GetXmin()), fYUp(h.GetYaxis()->GetXmax()),
     fOption(h.GetErrorOption())
{
}
TProfile1DModel::TProfile1DModel(const char *name, const char *title, int nbinsx, double xlow, double xup,
                                 const char *option)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fOption(option)
{
}

TProfile1DModel::TProfile1DModel(const char *name, const char *title, int nbinsx, double xlow, double xup,
                                 double ylow, double yup, const char *option)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fYLow(ylow), fYUp(yup),
     fOption(option)
{
}

TProfile1DModel::~TProfile1DModel()
{
}

TProfile2DModel::TProfile2DModel(const ::TProfile2D &h)
   : fName(h.GetName()), fTitle(h.GetTitle()), fNbinsX(h.GetNbinsX()), fXLow(h.GetXaxis()->GetXmin()),
     fXUp(h.GetXaxis()->GetXmax()), fNbinsY(h.GetNbinsY()), fYLow(h.GetYaxis()->GetXmin()),
     fYUp(h.GetYaxis()->GetXmax()), fZLow(h.GetZaxis()->GetXmin()), fZUp(h.GetZaxis()->GetXmax()),
     fOption(h.GetErrorOption())
{
}
TProfile2DModel::TProfile2DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy,
                                 double ylow, double yup, const char *option)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fNbinsY(nbinsy), fYLow(ylow), fYUp(yup),
     fOption(option)
{
}

TProfile2DModel::TProfile2DModel(const char *name, const char *title, int nbinsx, double xlow, double xup, int nbinsy,
                                 double ylow, double yup, double zlow, double zup, const char *option)
   : fName(name), fTitle(title), fNbinsX(nbinsx), fXLow(xlow), fXUp(xup), fNbinsY(nbinsy), fYLow(ylow), fYUp(yup),
     fZLow(zlow), fZUp(zup), fOption(option)
{
}

TProfile2DModel::~TProfile2DModel()
{
}

} // ns TDF
} // ns Experimental
} // ns ROOT
