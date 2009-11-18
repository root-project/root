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

    void SetContourColor(const Color_t color) {fColor = color;}
    void SetLineColor(const Color_t color) {fLineColor = color;}
    void SetMaximum(const Double_t theMaximum) {fMaximum = theMaximum;}
    void SetNPoints(Int_t np) { fNPoints = np; }

    void Draw(const Option_t *options=0);

  private:

    Int_t fColor;
    Int_t fFillStyle;
    Int_t fLineColor;
    Int_t fNdimPlot;
    Int_t fNPoints; // number of points used to scan the PL 

    Double_t fMaximum;

    LikelihoodInterval *fInterval;

    RooArgSet *fParamsPlot;


  protected:

    ClassDef(LikelihoodIntervalPlot,1)  // Class containing the results of the IntervalCalculator
  };
}

#endif
