// @(#)root/roostats:$Id$

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_LikelihoodIntervalPlot
#define ROOSTATS_LikelihoodIntervalPlot

#include "RooPrintable.h"
#include "RooArgSet.h"

#include "TNamed.h"

#include "TH2F.h"

#include "RooStats/LikelihoodInterval.h"

namespace RooStats {

 class LikelihoodIntervalPlot : public TNamed, public RooPrintable {

   public:
    LikelihoodIntervalPlot();

    LikelihoodIntervalPlot(LikelihoodInterval* theInterval);

    /// returned plotted object (RooPlot or histograms)
    TObject * GetPlottedObject() const { return fPlotObject; }

    void SetLikelihoodInterval(LikelihoodInterval* theInterval);
    void SetPlotParameters(const RooArgSet *params) ;


    /// set plot range (for 1D plot)
    void SetRange(double x1, double x2) { fXmin = x1; fXmax = x2; }
    /// set plot range (for 2D plot)
    void SetRange(double x1, double y1, double x2, double y2) {
       fXmin = x1; fXmax = x2;
       fYmin = y1; fYmax = y2;
    }

    ///set plot precision (when drawing a RooPlot)
    void SetPrecision(double eps) { fPrecision = eps; }
    /// set the line color for the 1D interval lines or contours (2D)
    void SetLineColor(const Color_t color) {fLineColor = color;}
    /// set the fill contour color
    void SetFillStyle(const Style_t style) {fFillStyle = style;}
    /// set the fill contour color
    void SetContourColor(const Color_t color) {fColor = color;}
    void SetMaximum(const double theMaximum) {fMaximum = theMaximum;}
    void SetNPoints(Int_t np) { fNPoints = np; }


    /// draw the likelihood interval or contour
    /// for the 1D case a RooPlot is drawn by default of the profiled Log-Likelihood ratio
    /// if option "TF1" is used the objects are drawn using a TF1 scanning the LL function in a
    /// grid of the set points (by default
    /// the TF1 can be customized by setting maximum and the number of points to scan
    void Draw(const Option_t *options=nullptr) override;

  private:
     Color_t fColor = 0;        ///< color for the contour (for 2D) or function (in 1D)
     Style_t fFillStyle = 4050; ///< fill style for contours, half transparent by default
     Color_t fLineColor = 0;    ///< line color for the interval (1D) or for other contours (2D)
     Int_t fNdimPlot = 0;
     Int_t fNPoints = 0; ///< number of points used to scan the PL, default depends if 1D or 2D

     double fMaximum = -1; ///< function maximum
     // ranges for plots, default is variable range
     double fXmin = 0;
     double fXmax = -1;
     double fYmin = 0;
     double fYmax = -1;
     double fPrecision = -1; ///< RooCurve precision, use default in case of -1

     LikelihoodInterval *fInterval = nullptr;

     RooArgSet *fParamsPlot = nullptr;
     TObject *fPlotObject = nullptr; ///< plotted object

  protected:

    ClassDefOverride(LikelihoodIntervalPlot,2)  // Class containing the results of the IntervalCalculator
  };
}

#endif
