// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_SimpleInterval
#define RooStats_SimpleInterval

#include "RooArgSet.h"
#include "RooStats/ConfInterval.h"

class RooRealVar;

namespace RooStats {
 class SimpleInterval : public ConfInterval {

  public:
    /// default constructor
    explicit SimpleInterval(const char *name = nullptr);

    /// default constructor
    SimpleInterval(const SimpleInterval& other, const char* name);

    /// default constructor
    SimpleInterval& operator=(const SimpleInterval& other) ;

    /// constructor from name, the Parameter of interest and lower/upper bound values
    SimpleInterval(const char* name, const RooRealVar & var, double lower, double upper, double cl);

    /// destructor
    ~SimpleInterval() override;

    /// check if parameter is in the interval
    bool IsInInterval(const RooArgSet&) const override;

    /// set the confidence level for the interval. Simple interval is defined at construction time so this function
    /// has no effect
    void SetConfidenceLevel(double ) override {}

    /// return the confidence interval
    double ConfidenceLevel() const override {return fConfidenceLevel;}

    /// return the interval lower limit
    virtual double LowerLimit() {return fLowerLimit;}
    /// return the interval upper limit
    virtual double UpperLimit() {return fUpperLimit;}

    /// return a cloned list with the parameter of interest
    RooArgSet* GetParameters() const override;

    /// check if parameters are correct (i.e. they are the POI of this interval)
    bool CheckParameters(const RooArgSet&) const override ;



  protected:

    ClassDefOverride(SimpleInterval,1)  // Concrete implementation of ConfInterval for simple 1-D intervals in the form [a,b]

    RooArgSet fParameters;      ///< set containing the parameter of interest
    double  fLowerLimit;      ///< lower interval limit
    double  fUpperLimit;      ///< upper interval limit
    double  fConfidenceLevel; ///< confidence level

  };
}

#endif
