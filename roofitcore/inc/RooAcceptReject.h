/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.rdl,v 1.1 2001/05/18 00:59:19 david Exp $
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

  void generateEvents(Int_t nEvents, RooDataSet &container);

  // ascii printing interface
  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:
  void addEvent(RooDataSet &cache, TIterator *nextCatVar, TIterator *nextRealVar);

  RooArgSet *_cloneSet;
  RooAbsReal *_funcClone;
  RooArgSet _catVars,_realVars;
  Bool_t _verbose, _isValid;
  Double_t _area, _maxFuncVal, _funcNorm;
  UInt_t _sampleDim,_minTrials,_totalEvents;
  RooRealVar *_funcVal;

  ClassDef(RooAcceptReject,0) // Context for generating a dataset from a PDF
};

#endif
