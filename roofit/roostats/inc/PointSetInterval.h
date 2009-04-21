// @(#)root/roostats:$Id: PointSetInterval.cxx 26317 2009-01-13 15:31:05Z cranmer $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_PointSetInterval
#define RooStats_PointSetInterval

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef ROO_TREE_DATA
#include "RooTreeData.h"
#endif
#ifndef RooStats_ConfInterval
#include "RooStats/ConfInterval.h"
#endif


namespace RooStats {
 class PointSetInterval : public ConfInterval {
  private:
    //    RooArgSet* fParameters; // parameter of interest
    Double_t fConfidenceLevel; // confidence level
    RooTreeData* fParameterPointsInInterval; // either a histogram (RooDataHist) or a tree (RooDataSet)

  public:
    // constructors,destructors
    PointSetInterval();
    PointSetInterval(const char* name);
    PointSetInterval(const char* name, const char* title);
    PointSetInterval(const char* name, RooTreeData&);
    PointSetInterval(const char* name, const char* title, RooTreeData&);
    virtual ~PointSetInterval();
        
    virtual Bool_t IsInInterval(RooArgSet&);
    virtual void SetConfidenceLevel(Double_t cl) {fConfidenceLevel = cl;}
    virtual Double_t ConfidenceLevel() const {return fConfidenceLevel;}
 
    // Method to return lower limit on a given parameter 
    //  Double_t LowerLimit(RooRealVar& param) ; // could provide, but misleading?
    //      Double_t UpperLimit(RooRealVar& param) ; // could provide, but misleading?
    
    // do we want it to return list of parameters
    virtual RooArgSet* GetParameters() const;

    // Accessor for making plots
    RooTreeData* GetParameterPoints() const {return (RooTreeData*)fParameterPointsInInterval->Clone();}

    // check if parameters are correct. (dummy implementation to start)
    Bool_t CheckParameters(RooArgSet&) const ;


    
  protected:
    ClassDef(PointSetInterval,1)  // Concrete implementation of ConfInterval for simple 1-D intervals in the form [a,b]
      
  };
}

#endif
