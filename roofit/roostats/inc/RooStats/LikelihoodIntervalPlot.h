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

    // Destructor of SamplingDistribution
    virtual ~LikelihoodIntervalPlot();

    void SetLikelihoodInterval(LikelihoodInterval* theInterval);
    void SetPlotParameters(const RooArgSet *params) ;


    // set plot range (for 1D plot)
    void SetRange(double x1, double x2) { fXmin = x1; fXmax = x2; }
    // set plot range (for 2D plot)
    void SetRange(double x1, double y1, double x2, double y2) {
       fXmin = x1; fXmax = x2; 
       fYmin = y1; fYmax = y2; 
    }

    //set plot precision (when drawing a RooPlot)
    void SetPrecision(double eps) { fPrecision = eps; }

    void SetContourColor(const Color_t color) {fColor = color;}
    void SetLineColor(const Color_t color) {fLineColor = color;}
    void SetMaximum(const Double_t theMaximum) {fMaximum = theMaximum;}
    void SetNPoints(Int_t np) { fNPoints = np; }


    // draw the likelihood interval or contour
    // for the 1D case a RooPlot is drawn by default of the profiled Log-Likelihood ratio
    // if option "TF1" is used the objects is drawn using a TF1 scanning the LL function in a 
    // grid of the setetd points (by default 
    // the TF1 can be costumized by setting maximum and the number of points to scan 
    void Draw(const Option_t *options=0);

  private:

    Int_t fColor;
    Int_t fFillStyle;
    Int_t fLineColor;
    Int_t fNdimPlot;
    Int_t fNPoints; // number of points used to scan the PL 

    Double_t fMaximum;   // function maximum
    // ranges for plots 
    Double_t fXmin;
    Double_t fXmax;
    Double_t fYmin;
    Double_t fYmax;
    Double_t fPrecision;  // RooCurve precision

    LikelihoodInterval *fInterval;

    RooArgSet *fParamsPlot;


  protected:

    ClassDef(LikelihoodIntervalPlot,1)  // Class containing the results of the IntervalCalculator
  };
}

#endif
