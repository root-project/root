/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooFormulaVar.h,v 1.29 2007/08/09 19:55:47 wouter Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_FORMULA_VAR
#define ROO_FORMULA_VAR

#include "RooAbsReal.h"
#include "RooFormula.h"
#include "RooArgList.h"
#include "RooListProxy.h"

class RooArgSet ;

class RooFormulaVar : public RooAbsReal {
public:
  // Constructors, assignment etc
  inline RooFormulaVar() : _formula(0), _nset(0) { }
  RooFormulaVar(const char *name, const char *title, const char* formula, const RooArgList& dependents);
  RooFormulaVar(const char *name, const char *title, const RooArgList& dependents);
  RooFormulaVar(const RooFormulaVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooFormulaVar(*this,newname); }
  virtual ~RooFormulaVar();

  inline Bool_t ok() const { return formula().ok() ; }

  inline RooAbsArg* getParameter(const char* name) const { 
    // Return pointer to parameter with given name
    return _actualVars.find(name) ; 
  }
  inline RooAbsArg* getParameter(Int_t index) const { 
    // Return pointer to parameter at given index
    return _actualVars.at(index) ; 
  }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printMultiline(ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent= "") const ;
  void printMetaArgs(ostream& os) const ;

  // Debugging
  void dumpFormula() { formula().dump() ; }

  virtual Double_t defaultErrorLevel() const ;

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;
  RooFormula& formula() const ;

  // Post-processing of server redirection
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;

  virtual Bool_t isValidReal(Double_t value, Bool_t printError) const ;

  RooListProxy _actualVars ;     // Actual parameters used by formula engine
  mutable RooFormula* _formula ; //! Formula engine 
  mutable RooArgSet* _nset ;     //! Normalization set to be passed along to contents
  TString _formExpr ;            // Formula expression string

  ClassDef(RooFormulaVar,1) // Real-valued function of other RooAbsArgs calculated by a TFormula expression
};

#endif
