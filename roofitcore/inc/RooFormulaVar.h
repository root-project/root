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
#ifndef ROO_FORMULA_VAR
#define ROO_FORMULA_VAR

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooFormula.hh"

class RooArgSet ;

class RooFormulaVar : public RooAbsReal {
public:
  // Constructors, assignment etc
  inline RooFormulaVar() { }
  RooFormulaVar(const char *name, const char *title, const RooArgSet& dependents);
  RooFormulaVar(const RooFormulaVar& other, const char* name=0);
  virtual TObject* clone() const { return new RooFormulaVar(*this); }
  virtual ~RooFormulaVar();

  // Formula editing interface
  Bool_t setFormula(const char* formula) ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // Custom dependent checking
  virtual Bool_t checkDependents(const RooDataSet* set) const ;

  // Debugging
  void dumpFormula() { _formula.dump() ; }

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;

  // Post-processing of server redirection
  virtual Bool_t redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;

  virtual Bool_t isValid() const ;
  virtual Bool_t isValid(Double_t value) const ;

  mutable RooFormula _formula ; // Formula engine 

  ClassDef(RooFormulaVar,1) // a real-valued variable and its value
};

#endif
