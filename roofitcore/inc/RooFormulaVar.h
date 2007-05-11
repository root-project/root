/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooFormulaVar.rdl,v 1.27 2005/06/20 15:44:52 wverkerke Exp $
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
  inline RooFormulaVar() { }
  RooFormulaVar(const char *name, const char *title, const char* formula, const RooArgList& dependents);
  RooFormulaVar(const char *name, const char *title, const RooArgList& dependents);
  RooFormulaVar(const RooFormulaVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooFormulaVar(*this,newname); }
  virtual ~RooFormulaVar();

  inline Bool_t ok() const { return formula().ok() ; }

  inline RooAbsArg* getParameter(const char* name) const { return _actualVars.find(name) ; }
  inline RooAbsArg* getParameter(Int_t index) const { return _actualVars.at(index) ; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // Debugging
  void dumpFormula() { formula().dump() ; }

  // In general, we cannot be normalized sensibly so pretend that we are always normalized
//   Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const ;
//   Double_t analyticalIntegral(Int_t code) const ;
  //inline Bool_t selfNormalized() const { return kTRUE; }


  virtual Double_t defaultErrorLevel() const ;

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;
  RooFormula& formula() const ;

  // Post-processing of server redirection
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;

  virtual Bool_t isValidReal(Double_t value, Bool_t printError) const ;

  RooListProxy _actualVars ; 
  mutable RooFormula* _formula ; // Formula engine 
  TString _formExpr ;

  ClassDef(RooFormulaVar,1) // Real-valued variable, calculated from a string expression formula 
};

#endif
