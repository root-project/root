/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.rdl,v 1.5 2001/10/10 00:22:24 david Exp $
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

class RooAbsPdf;
class RooDataSet;

class RooAbsGenContext : public TNamed, public RooPrintable {
public:
  RooAbsGenContext(const RooAbsPdf &model, Bool_t _verbose= kFALSE) ;
  virtual ~RooAbsGenContext();
  virtual RooDataSet *generate(Int_t nEvents= 0) const = 0 ;

  Bool_t isValid() const { return _isValid; }

  inline void setVerbose(Bool_t verbose= kTRUE) { _verbose= verbose; }
  inline Bool_t isVerbose() const { return _verbose; }

protected:
  Bool_t _isValid;
  Bool_t _verbose;

  ClassDef(RooAbsGenContext,0) // Abstract context for generating a dataset from a PDF
};

#endif
