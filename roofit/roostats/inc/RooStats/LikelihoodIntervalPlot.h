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

    /// Destructor of SamplingDistribution
    ~LikelihoodIntervalPlot() override;


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
    /// the TF1 can be costumized by setting maximum and the number of points to scan
    void Draw(const Option_t *options=0) override;

  private:

    Color_t fColor;      ///< color for the contour (for 2D) or function (in 1D)
    Style_t fFillStyle;  ///< fill style for contours
    Color_t fLineColor;  ///< line color for the interval (1D) or for other contours (2D)
    Int_t fNdimPlot;
    Int_t fNPoints;      ///< number of points used to scan the PL

    double fMaximum;   ///< function maximum
    // ranges for plots
    double fXmin;
    double fXmax;
    double fYmin;
    double fYmax;
    double fPrecision; ///< RooCurve precision

    LikelihoodInterval *fInterval;

    RooArgSet *fParamsPlot;
    TObject * fPlotObject; ///< plotted object


  protected:

    ClassDefOverride(LikelihoodIntervalPlot,2)  // Class containing the results of the IntervalCalculator
  };
}

#endif
