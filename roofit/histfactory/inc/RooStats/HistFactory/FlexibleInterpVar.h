// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_FLEXIBLEINTERPVAR
#define ROOSTATS_FLEXIBLEINTERPVAR

#include <RooAbsReal.h>
#include <RooListProxy.h>

#include <vector>

namespace RooStats{
namespace HistFactory{

  class FlexibleInterpVar : public RooAbsReal {
  public:

    FlexibleInterpVar();

    FlexibleInterpVar(const char *name, const char *title,
            const RooArgList& _paramList,
            double nominal, std::vector<double> const& low, std::vector<double> const& high);

    FlexibleInterpVar(const char *name, const char *title,
            const RooArgList& _paramList, double nominal, std::vector<double> const& low,
            std::vector<double> const& high,std::vector<int> const& code);

    FlexibleInterpVar(const char *name, const char *title);
    FlexibleInterpVar(const FlexibleInterpVar&, const char*);

    void setInterpCode(RooAbsReal& param, int code);
    void setAllInterpCodes(int code);
    void setGlobalBoundary(double boundary) {_interpBoundary = boundary;}
    void setNominal(double newNominal);
    void setLow(RooAbsReal& param, double newLow);
    void setHigh(RooAbsReal& param, double newHigh);

    void printAllInterpCodes();
    const std::vector<int>&  interpolationCodes() const { return _interpCode; }

    TObject* clone(const char* newname=nullptr) const override { return new FlexibleInterpVar(*this, newname); }
    ~FlexibleInterpVar() override ;

    void printMultiline(std::ostream& os, Int_t contents, bool verbose = false, TString indent = "") const override;
    virtual void printFlexibleInterpVars(std::ostream& os) const;

    const RooListProxy& variables() const { return _paramList; }
    double nominal() const { return _nominal; }
    const std::vector<double>& low() const { return _low; }
    const std::vector<double>& high() const { return _high; }
    double globalBoundary() const { return _interpBoundary;}

    void doEval(RooFit::EvalContext &) const override;

  protected:

    RooListProxy _paramList ;
    double _nominal = 0.0;
    std::vector<double> _low;
    std::vector<double> _high;
    std::vector<int> _interpCode;
    double _interpBoundary = 1.0;

    double evaluate() const override;

  private:

    void setInterpCodeForParam(int iParam, int code);

    ClassDefOverride(RooStats::HistFactory::FlexibleInterpVar,2); // flexible interpolation
  };
}
}

#endif
