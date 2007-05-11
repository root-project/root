/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGenericPdf.rdl,v 1.19 2005/06/20 15:44:53 wverkerke Exp $
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
#ifndef ROO_GENERIC_PDF
#define ROO_GENERIC_PDF

#include "RooAbsPdf.h"
#include "RooFormula.h"
#include "RooListProxy.h"

class RooArgList ;

class RooGenericPdf : public RooAbsPdf {
public:
  // Constructors, assignment etc
  inline RooGenericPdf() { }
  RooGenericPdf(const char *name, const char *title, const RooArgList& dependents);
  RooGenericPdf(const char *name, const char *title, const char* formula, const RooArgList& dependents);
  RooGenericPdf(const RooGenericPdf& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooGenericPdf(*this,newname); }
  virtual ~RooGenericPdf();

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // Debugging
  void dumpFormula() { _formula.dump() ; }

protected:

  // Function evaluation
  RooListProxy _actualVars ; 
  virtual Double_t evaluate() const ;

  Bool_t setFormula(const char* formula) ;

  // Post-processing of server redirection
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;

  virtual Bool_t isValidReal(Double_t value, Bool_t printError) const ;

  mutable RooFormula _formula ; // Formula engine 

  ClassDef(RooGenericPdf,1) // Generic PDF defined by string expression and list of variables
};

#endif
