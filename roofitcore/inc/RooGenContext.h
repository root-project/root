/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.rdl,v 1.10 2001/05/14 22:54:21 verkerke Exp $
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
class TRandom;

class RooGenContext : public TNamed, public RooPrintable {
public:
  RooGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0);
  virtual ~RooGenContext();
  virtual RooDataSet *generate(Int_t nEvents= 0) const;

  inline Int_t getMaxTrials() const { return _maxTrials; }
  inline void setMaxTrials(Int_t n) { if(n > 0) _maxTrials = n; }

  // static random number generator interface
  static TRandom &randomGenerator();
  static Double_t uniform();

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  Bool_t isValid() const { return _isValid; }

protected:
  RooDataSet *createDataset() const;

  const RooArgSet *_origVars;
  const RooDataSet *_prototype;
  RooArgSet *_cloneSet;
  RooAbsPdf *_pdfClone;
  RooArgSet _directVars,_otherVars,_protoVars,_datasetVars;
  Int_t _maxTrials;
  Bool_t _isValid;

  ClassDef(RooGenContext,0) // Context for generating a dataset from a PDF
};

#endif
