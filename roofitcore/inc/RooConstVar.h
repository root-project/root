/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_CONST_VAR
#define ROO_CONST_VAR

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooListProxy.hh"

class RooArgSet ;

class RooConstVar : public RooAbsReal {
public:
  // Constructors, assignment etc
  inline RooConstVar() { }
  RooConstVar(const char *name, const char *title, Double_t value);
  RooConstVar(const RooConstVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooConstVar(*this,newname); }
  virtual ~RooConstVar();

  virtual Double_t getVal(const RooArgSet* set=0) const { return _value ; }
  void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  virtual Double_t evaluate() const { return _value ; } ;

  Double_t _value ;

  ClassDef(RooConstVar,1) // Real-valued variable, calculated from a string expression formula 
};

#endif
