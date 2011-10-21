/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsString.h,v 1.26 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_STRING
#define ROO_ABS_STRING

#include "RooAbsArg.h"

class RooArgSet ;
class RooVectorDataStore ;
class TH1F ;

class RooAbsString : public RooAbsArg {
public:

  // Constructors, assignment etc
  RooAbsString() ;
  RooAbsString(const char *name, const char *title, Int_t size=128) ;
  RooAbsString(const RooAbsString& other, const char* name=0);
  virtual ~RooAbsString();

  // Return value and unit accessors
  virtual const char* getVal() const ;
  Bool_t operator==(const char*) const ;
  virtual Bool_t operator==(const RooAbsArg& other) ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printValue(ostream& os) const ;


  RooAbsArg *createFundamental(const char* newname=0) const;

protected:

  // Function evaluation and error tracing
  const char* traceEval() const ;
  virtual Bool_t traceEvalHook(const char* value) const ;
  virtual TString evaluate() const { return "" ; }

  // Internal consistency checking (needed by RooDataSet)
  virtual Bool_t isValid() const ;
  virtual Bool_t isValidString(const char*, Bool_t printError=kFALSE) const ;

  virtual void syncCache(const RooArgSet* nset=0) ;
  void copyCache(const RooAbsArg* source, Bool_t valueOnly=kFALSE) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual void attachToVStore(RooVectorDataStore&) {}
  virtual void fillTreeBranch(TTree& t) ;
  virtual void setTreeBranchStatus(TTree& t, Bool_t active) ;
  Int_t _len ; // Length of _value
  mutable char *_value ; //[_len] Value

  ClassDef(RooAbsString,1) // Abstract string-valued variable
};

#endif
