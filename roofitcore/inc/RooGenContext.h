/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.rdl,v 1.5 2001/10/10 00:22:24 david Exp $
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
class TIterator;

class RooGenContext : public RooAbsGenContext {
public:
  RooGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		Bool_t verbose= kFALSE);
  virtual ~RooGenContext();
  virtual RooDataSet *generate(Int_t nEvents= 0) const;

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:
  const RooArgSet *_origVars;
  const RooDataSet *_prototype;
  RooArgSet *_cloneSet;
  RooAbsPdf *_pdfClone;
  RooArgSet _directVars,_uniformVars,_otherVars,_protoVars,_datasetVars;
  Int_t _code;
  Double_t _maxProb, _area, _norm;
  RooRealIntegral *_acceptRejectFunc;
  RooAcceptReject *_generator;
  TIterator *_protoIterator;
  mutable Int_t _lastProtoIndex;

  ClassDef(RooGenContext,0) // Context for generating a dataset from a PDF
};

#endif
