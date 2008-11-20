// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   21/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_SPlot
#define RooStats_SPlot

class RooAbsReal;
class RooAbsPdf;
class RooFitResult;
class RooRealVar;
class RooSimultaneous;

#ifndef ROOT_TH1
#include "TH1.h"
#endif
#include "RooFitResult.h"
#include "RooRealVar.h"
#include "RooHist.h"
#include "RooPlot.h"
#include "RooDataSet.h"

namespace RooStats{

 class SPlot : public TH1F {
  public:
    SPlot();
    SPlot(const SPlot &other);
    SPlot(const char* name, const char *title, Int_t nbins, Double_t xmin, Double_t xmax);
    
    static RooDataSet* 
      AddSWeightToData(const RooSimultaneous* pdf, const RooArgList &yieldsTmp, 
		       RooDataSet &data, const RooArgSet &projDeps=RooArgSet()) ;
    
    
    void FillSPlot(const RooDataSet &data, TString varname, TString weightname);
 
    void FillSPlot(const RooAbsReal &x, RooAbsReal &nstar, RooDataSet data, const RooFitResult &fitRes, const RooArgList &pdfList, const RooArgList &yields, Bool_t doErrors, const RooArgSet &projDeps=RooArgSet() );
    void FillSPlot(const RooAbsReal &x, RooAbsReal &nstar, RooDataSet data, const RooFitResult &fitRes, const RooArgList &pdfList, const RooArgList &yields, RooAbsPdf &totalPdf, Bool_t doErrors, const RooArgSet &projDeps=RooArgSet() );
    
    void FillSPlot(const RooAbsReal &x, RooAbsReal &nstar, RooDataSet data, const RooFitResult &fitRes, RooAbsPdf &totalPdf, RooArgList &yields, Bool_t doErrors, const RooArgSet &projDeps=RooArgSet() );
    
    Double_t GetComponentValue(RooAbsPdf &pdf, RooArgList &yieldsTmp, Int_t igood, RooArgSet &normSet);
    
  protected:

    ClassDef(SPlot,1)   // Class used for making sPlots
   };

}
#endif
