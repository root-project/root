/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAcceptReject.rdl,v 1.5 2001/09/24 16:23:12 verkerke Exp $
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
  RooAcceptReject(const RooAbsReal &func, const RooArgSet &genVars, Bool_t verbose= kFALSE);
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

protected:
  void addEventToCache();
  const RooArgSet *nextAcceptedEvent();

  RooArgSet *_cloneSet;
  RooAbsReal *_funcClone;
  RooArgSet _catVars,_realVars ;
  Bool_t _verbose, _isValid;
  Double_t _maxFuncVal, _funcSum;
  UInt_t _realSampleDim,_catSampleMult,_minTrials,_totalEvents,_eventsUsed;
  RooRealVar *_funcValStore,*_funcValPtr;
  RooDataSet *_cache;
  TIterator *_nextCatVar;
  TIterator *_nextRealVar;

  static const int _maxSampleDim, _minTrialsArray[];

  ClassDef(RooAcceptReject,0) // Context for generating a dataset from a PDF
};

#endif
