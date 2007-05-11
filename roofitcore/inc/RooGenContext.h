/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGenContext.rdl,v 1.18 2005/06/20 15:44:52 wverkerke Exp $
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
#ifndef ROO_GEN_CONTEXT
#define ROO_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"

class RooAbsPdf;
class RooDataSet;
class RooRealIntegral;
class RooAcceptReject;
class TRandom;
class RooRealVar ;

class RooGenContext : public RooAbsGenContext {
public:
  RooGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		const RooArgSet* auxProto=0, Bool_t verbose=kFALSE, const RooArgSet* forceDirect=0);
  virtual ~RooGenContext();

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;

protected:

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining);

  RooArgSet *_cloneSet;
  RooAbsPdf *_pdfClone;
  RooArgSet _directVars,_uniformVars,_otherVars;
  Int_t _code;
  Double_t _maxProb, _area, _norm;
  RooRealIntegral *_acceptRejectFunc;
  RooAcceptReject *_generator;
  RooRealVar *_maxVar ;
  TIterator *_uniIter ;
  Int_t _updateFMaxPerEvent ;

  ClassDef(RooGenContext,0) // Context for generating a dataset from a PDF
};

#endif
