/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.rdl,v 1.9 2001/11/05 18:50:49 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-May-2000 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_GEN_CONTEXT
#define ROO_GEN_CONTEXT

#include "RooFitCore/RooAbsGenContext.hh"
#include "RooFitCore/RooArgSet.hh"

class RooAbsPdf;
class RooDataSet;
class RooRealIntegral;
class RooAcceptReject;
class TRandom;

class RooGenContext : public RooAbsGenContext {
public:
  RooGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		Bool_t verbose=kFALSE, const RooArgSet* forceDirect=0);
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
  RooAbsReal *_maxVar ;
  TIterator *_uniIter ;

  ClassDef(RooGenContext,0) // Context for generating a dataset from a PDF
};

#endif
