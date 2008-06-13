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
  Bool_t isValid() const { 
    // If true, generator is in a valid state
    return _isValid; 
  }
  virtual ~RooAcceptReject();

  inline void setVerbose(Bool_t verbose= kTRUE) {
    // If flag is true, verbose messaging will be active during generation
    _verbose= verbose; 
  }
  inline Bool_t isVerbose() const { 
    // Return status of verbose messaging flag
    return _verbose; 
  }

  const RooArgSet *generateEvent(UInt_t remaining);

   inline virtual void Print(Option_t *options= 0) const {
     // ascii printing interface     
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

  RooArgSet *_cloneSet;                // Set owning clone of input function
  RooAbsReal *_funcClone;              // Pointer to top level node of cloned function
  const RooAbsReal *_funcMaxVal ;      // Container for maximum function value
  RooArgSet _catVars,_realVars ;       // Sets of discrete and real valued observabeles
  Bool_t _verbose, _isValid;           // Verbose and valid flag
  Double_t _maxFuncVal, _funcSum;      // Maximum function value found, and sum of all samples made
  UInt_t _realSampleDim,_catSampleMult;// Number of real and discrete dimensions to be samplesd
  UInt_t _minTrials;                   // Minimum number of max.finding trials, total number of samples
  UInt_t _totalEvents;                 // Total number of function samples
  UInt_t _eventsUsed;                  // Accepted number of function samples
  RooRealVar *_funcValStore,*_funcValPtr; // RRVs storing function value in context and in output dataset
  RooDataSet *_cache;                  // Dataset holding generared values of observables
  TIterator *_nextCatVar;              // Iterator of categories to be generated
  TIterator *_nextRealVar;             // Iterator over variables to be generated

  static const UInt_t _maxSampleDim;       // Number of filledelements in _minTrialsArray
  static const UInt_t _minTrialsArray[];   // Minimum number of trials samples for 1,2,3 dimensional problems

  ClassDef(RooAcceptReject,0) // Context for generating a dataset from a PDF
};

#endif
