/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.rdl,v 1.3 2001/08/01 21:30:15 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-May-2000 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_GEN_CONTEXT
#define ROO_GEN_CONTEXT

#include "TNamed.h"
#include "RooFitCore/RooPrintable.hh"
#include "RooFitCore/RooArgSet.hh"

class RooAbsPdf;
class RooDataSet;
class RooRealIntegral;
class RooAcceptReject;
class TRandom;

class RooGenContext : public TNamed, public RooPrintable {
public:
  RooGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		Bool_t _verbose= kFALSE);
  virtual ~RooGenContext();
  virtual RooDataSet *generate(Int_t nEvents= 0) const;

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  Bool_t isValid() const { return _isValid; }

  inline void setVerbose(Bool_t verbose= kTRUE) { _verbose= verbose; }
  inline Bool_t isVerbose() const { return _verbose; }

protected:
  const RooArgSet *_origVars;
  const RooDataSet *_prototype;
  RooArgSet *_cloneSet;
  RooAbsPdf *_pdfClone;
  RooArgSet _directVars,_uniformVars,_otherVars,_protoVars,_datasetVars;
  Bool_t _isValid;
  Int_t _code;
  Double_t _maxProb, _area, _norm;
  RooRealIntegral *_acceptRejectFunc;
  RooAcceptReject *_generator;
  Bool_t _verbose;

  ClassDef(RooGenContext,0) // Context for generating a dataset from a PDF
};

#endif
