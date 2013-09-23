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

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include <vector>

class RooRealVar;
class RooArgList ;

namespace RooStats{
namespace HistFactory{

  class FlexibleInterpVar : public RooAbsReal {
  public:

    FlexibleInterpVar() ;
    FlexibleInterpVar(const char *name, const char *title,
		      const RooArgList& _paramList, 
		      double nominal, std::vector<double> low, std::vector<double> high);

    FlexibleInterpVar(const char *name, const char *title,
		      const RooArgList& _paramList, double nominal, std::vector<double> low, 
		      std::vector<double> high,std::vector<int> code);

    FlexibleInterpVar(const char *name, const char *title);
    FlexibleInterpVar(const FlexibleInterpVar&, const char*);

    void setInterpCode(RooAbsReal& param, int code);
    void setAllInterpCodes(int code);
    void setGlobalBoundary(double boundary) {_interpBoundary = boundary;}

    void printAllInterpCodes();

    virtual TObject* clone(const char* newname) const { return new FlexibleInterpVar(*this, newname); }
    virtual ~FlexibleInterpVar() ;

    virtual void printMultiline(ostream& os, Int_t contents, Bool_t verbose = kFALSE, TString indent = "") const;
    virtual void printFlexibleInterpVars(ostream& os) const;

  protected:

    RooListProxy _paramList ;
    Double_t _nominal;
    std::vector<double> _low;
    std::vector<double> _high;
    std::vector<int> _interpCode;
    Double_t _interpBoundary;

    TIterator* _paramIter ;  //! do not persist

    mutable Bool_t         _logInit ; //! 
    mutable std::vector<double> _logLo ; //! cached logs 
    mutable std::vector<double> _logHi ; //! cached logs
    mutable std::vector<double> _powLo ; //! cached powers
    mutable std::vector<double> _powHi ; //! cached powers

    Double_t evaluate() const;

    ClassDef(RooStats::HistFactory::FlexibleInterpVar,2) // flexible interpolation
  };
}
}

#endif
