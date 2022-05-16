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

    ~SPlot() override;
    SPlot();
    SPlot(const SPlot &other);
    SPlot(const char* name, const char* title);
    SPlot(const char* name, const char* title, const RooDataSet &data);
    SPlot(const char* name, const char* title,RooDataSet& data, RooAbsPdf* pdf,
     const RooArgList &yieldsList,const RooArgSet &projDeps=RooArgSet(),
     bool useWeights=true, bool copyDataSet = false, const char* newName = "",
     const RooCmdArg& fitToarg5=RooCmdArg::none(),
     const RooCmdArg& fitToarg6=RooCmdArg::none(),
     const RooCmdArg& fitToarg7=RooCmdArg::none(),
     const RooCmdArg& fitToarg8=RooCmdArg::none());

    RooDataSet* SetSData(RooDataSet* data);

    RooDataSet* GetSDataSet() const;

    RooArgList GetSWeightVars() const;

    Int_t GetNumSWeightVars() const;

    void AddSWeight(RooAbsPdf* pdf, const RooArgList &yieldsTmp,
          const RooArgSet &projDeps=RooArgSet(), bool includeWeights=true,
          const RooCmdArg& fitToarg5=RooCmdArg::none(),
          const RooCmdArg& fitToarg6=RooCmdArg::none(),
          const RooCmdArg& fitToarg7=RooCmdArg::none(),
          const RooCmdArg& fitToarg8=RooCmdArg::none());

    double GetSumOfEventSWeight(Int_t numEvent) const;

    double GetYieldFromSWeight(const char* sVariable) const;

    double GetSWeight(Int_t numEvent, const char* sVariable) const;



  protected:

     enum {
        kOwnData = BIT(20)
     };

    RooArgList fSWeightVars;

    //  RooListProxy fSWeightVars;

    RooDataSet* fSData;

    ClassDefOverride(SPlot,1)   // Class used for making sPlots


      };

}
#endif
