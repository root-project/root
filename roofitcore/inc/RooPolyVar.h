/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   07-Feb-2000 WV Initial RFC version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_POLY_VAR
#define ROO_POLY_VAR

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooListProxy.hh"

class RooRealVar;
class RooArgList ;

class RooPolyVar : public RooAbsReal {
public:

  RooPolyVar() ;
  RooPolyVar(const char* name, const char* title, RooAbsReal& x) ;
  RooPolyVar(const char *name, const char *title,
		RooAbsReal& _x, const RooArgList& _coefList, Int_t lowestOrder=0) ;

  RooPolyVar(const RooPolyVar& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooPolyVar(*this, newname); }
  inline virtual ~RooPolyVar() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:

  RooRealProxy _x;
  RooListProxy _coefList ;
  Int_t _lowestOrder ;
  TIterator* _coefIter ;  //! do not persist

  Double_t evaluate() const;

  ClassDef(RooPolyVar,1) // Polynomial PDF
};

#endif
