/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAcceptReject.rdl,v 1.8 2001/11/05 18:50:48 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   28-May-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_ACCEPT_REJECT
#define ROO_ACCEPT_REJECT

#include "TNamed.h"
#include "RooFitCore/RooPrintable.hh"
#include "RooFitCore/RooArgSet.hh"

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
  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

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
