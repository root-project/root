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


#include "RooMsgService.h"

#include "RooFitResult.h"
#include "RooRealVar.h"
#include "RooHist.h"
#include "RooPlot.h"
#include "RooDataSet.h"

namespace RooStats{

  class SPlot: public TNamed {

  public:

    ~SPlot();
    SPlot();
    SPlot(const SPlot &other);
    SPlot(const char* name, const char* title);
    SPlot(const char* name, const char* title, const RooDataSet &data);
    SPlot(const char* name, const char* title,RooDataSet& data, RooAbsPdf* pdf,
     const RooArgList &yieldsList,const RooArgSet &projDeps=RooArgSet(),
     bool useWeights=kTRUE, bool copyDataSet = kFALSE, const char* newName = "",
     const RooCmdArg& fitToarg5=RooCmdArg::none(),
     const RooCmdArg& fitToarg6=RooCmdArg::none(),
     const RooCmdArg& fitToarg7=RooCmdArg::none(),
     const RooCmdArg& fitToarg8=RooCmdArg::none());

    RooDataSet* SetSData(RooDataSet* data);

    RooDataSet* GetSDataSet() const;

    RooArgList GetSWeightVars() const;

    Int_t GetNumSWeightVars() const;

    void AddSWeight(RooAbsPdf* pdf, const RooArgList &yieldsTmp,
          const RooArgSet &projDeps=RooArgSet(), bool includeWeights=kTRUE,
          const RooCmdArg& fitToarg5=RooCmdArg::none(),
          const RooCmdArg& fitToarg6=RooCmdArg::none(),
          const RooCmdArg& fitToarg7=RooCmdArg::none(),
          const RooCmdArg& fitToarg8=RooCmdArg::none());

    Double_t GetSumOfEventSWeight(Int_t numEvent) const;

    Double_t GetYieldFromSWeight(const char* sVariable) const;

    Double_t GetSWeight(Int_t numEvent, const char* sVariable) const;



  protected:

     enum {
        kOwnData = BIT(20)
     };

    RooArgList fSWeightVars;

    //  RooListProxy fSWeightVars;

    RooDataSet* fSData;

    ClassDef(SPlot,1)   // Class used for making sPlots


      };

}
#endif
