/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsGenContext.rdl,v 1.4 2001/10/17 05:03:57 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_GEN_CONTEXT
#define ROO_ABS_GEN_CONTEXT

#include "TNamed.h"
#include "RooFitCore/RooPrintable.hh"
#include "RooFitCore/RooArgSet.hh"

class RooAbsPdf;
class RooDataSet;

class RooAbsGenContext : public TNamed, public RooPrintable {
public:
  RooAbsGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		   Bool_t _verbose= kFALSE) ;
  virtual ~RooAbsGenContext();

  RooDataSet *generate(Int_t nEvents= 0);

  Bool_t isValid() const { return _isValid; }

  inline void setVerbose(Bool_t verbose= kTRUE) { _verbose= verbose; }
  inline Bool_t isVerbose() const { return _verbose; }

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:

  friend class RooConvGenContext ;
  friend class RooProdGenContext ;
  friend class RooAddGenContext ;
  friend class RooSimGenContext ;

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining) = 0;

  const RooDataSet *_prototype;
  RooArgSet *_theEvent;
  Bool_t _isValid;
  Bool_t _verbose;
  UInt_t _expectedEvents;
  RooArgSet _protoVars;
  Int_t _nextProtoIndex;

  ClassDef(RooAbsGenContext,0) // Abstract context for generating a dataset from a PDF
};

#endif
