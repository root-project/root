// @(#)root/roostats:$Id$
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
#include "RooAbsData.h"
#endif
#ifndef RooStats_ConfInterval
#include "RooStats/ConfInterval.h"
#endif


namespace RooStats {

 class PointSetInterval : public ConfInterval {

  public:

     // default constructors
    explicit PointSetInterval(const char* name = 0);

    // constructor from name and data set specifying the interval points
    PointSetInterval(const char* name, RooAbsData&);

    // destructor
    virtual ~PointSetInterval();
        

    // check if parameter is in the interval
    virtual Bool_t IsInInterval(const RooArgSet&) const;

    // set the confidence level for the interval
    virtual void SetConfidenceLevel(Double_t cl) {fConfidenceLevel = cl;}

    // return the confidence level for the interval
    virtual Double_t ConfidenceLevel() const {return fConfidenceLevel;}
 
    // Method to return lower limit on a given parameter 
    //  Double_t LowerLimit(RooRealVar& param) ; // could provide, but misleading?
    //      Double_t UpperLimit(RooRealVar& param) ; // could provide, but misleading?
    
    // return a cloned list with the parameter of interest
    virtual RooArgSet* GetParameters() const;

    // return a copy of the data set (points) defining this interval
    RooAbsData* GetParameterPoints() const {return (RooAbsData*)fParameterPointsInInterval->Clone();}

    // return a cloned list with the parameter of interest
    Bool_t CheckParameters(const RooArgSet&) const ;

    // return lower limit on a given parameter 
    Double_t LowerLimit(RooRealVar& param) ;

    // return upper limit on a given parameter 
    Double_t UpperLimit(RooRealVar& param) ;

    
  protected:

    ClassDef(PointSetInterval,1)  // Concrete implementation of ConfInterval for simple 1-D intervals in the form [a,b]

  private:

    //    RooArgSet* fParameters; // parameter of interest
    Double_t fConfidenceLevel; // confidence level
    RooAbsData* fParameterPointsInInterval; // either a histogram (RooDataHist) or a tree (RooDataSet)

      
  };
}

#endif
