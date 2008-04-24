/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAcceptReject.h,v 1.16 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ACCEPT_REJECT
#define ROO_ACCEPT_REJECT

#include "TNamed.h"
#include "RooPrintable.h"
#include "RooArgSet.h"

class RooAbsReal;
class RooRealVar;
class RooDataSet;

class RooAcceptReject : public TNamed, public RooPrintable {
public:
  RooAcceptReject(const RooAbsReal &func, const RooArgSet &genVars, const RooAbsReal* maxFuncVal=0, Bool_t verbose= kFALSE);
  Bool_t isValid() const { return _isValid; }
  virtual ~RooAcceptReject();

  inline void setVerbose(Bool_t verbose= kTRUE) { _verbose= verbose; }
  inline Bool_t isVerbose() const { return _verbose; }

  const RooArgSet *generateEvent(UInt_t remaining);

  // ascii printing interface
   inline virtual void Print(Option_t *options= 0) const {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual void printName(ostream& os) const ;
  virtual void printTitle(ostream& os) const ;
  virtual void printClassName(ostream& os) const ;
  virtual void printArgs(ostream& os) const ;

  Double_t getFuncMax() ;
  void attachParameters(const RooArgSet& vars) ;

protected:
  void addEventToCache();
  const RooArgSet *nextAcceptedEvent();

  RooArgSet *_cloneSet;
  RooAbsReal *_funcClone;
  const RooAbsReal *_funcMaxVal ;
  RooArgSet _catVars,_realVars ;
  Bool_t _verbose, _isValid;
  Double_t _maxFuncVal, _funcSum;
  UInt_t _realSampleDim,_catSampleMult,_minTrials,_totalEvents,_eventsUsed;
  RooRealVar *_funcValStore,*_funcValPtr;
  RooDataSet *_cache;
  TIterator *_nextCatVar;
  TIterator *_nextRealVar;

  static const UInt_t _maxSampleDim;
  static const UInt_t _minTrialsArray[];

  ClassDef(RooAcceptReject,0) // Context for generating a dataset from a PDF
};

#endif
