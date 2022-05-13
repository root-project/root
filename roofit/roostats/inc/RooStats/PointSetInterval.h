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

#include "RooArgSet.h"
#include "RooAbsData.h"
#include "RooStats/ConfInterval.h"


namespace RooStats {

 class PointSetInterval : public ConfInterval {

  public:

     /// default constructors
    explicit PointSetInterval(const char* name = 0);

    /// constructor from name and data set specifying the interval points
    PointSetInterval(const char* name, RooAbsData&);

    /// destructor
    ~PointSetInterval() override;


    /// check if parameter is in the interval
    bool IsInInterval(const RooArgSet&) const override;

    /// set the confidence level for the interval
    void SetConfidenceLevel(double cl) override {fConfidenceLevel = cl;}

    /// return the confidence level for the interval
    double ConfidenceLevel() const override {return fConfidenceLevel;}

    /// Method to return lower limit on a given parameter
    ///  double LowerLimit(RooRealVar& param) ; // could provide, but misleading?
    ///      double UpperLimit(RooRealVar& param) ; // could provide, but misleading?

    /// return a cloned list with the parameter of interest
    RooArgSet* GetParameters() const override;

    /// return a copy of the data set (points) defining this interval
    RooAbsData* GetParameterPoints() const {return (RooAbsData*)fParameterPointsInInterval->Clone();}

    /// return a cloned list with the parameter of interest
    bool CheckParameters(const RooArgSet&) const override ;

    /// return lower limit on a given parameter
    double LowerLimit(RooRealVar& param) ;

    /// return upper limit on a given parameter
    double UpperLimit(RooRealVar& param) ;


  protected:

    ClassDefOverride(PointSetInterval,1)  // Concrete implementation of ConfInterval for simple 1-D intervals in the form [a,b]

  private:

    double fConfidenceLevel;              ///< confidence level
    RooAbsData* fParameterPointsInInterval; ///< either a histogram (RooDataHist) or a tree (RooDataSet)


  };
}

#endif
