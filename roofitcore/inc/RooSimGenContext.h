/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_SIM_GEN_CONTEXT
#define ROO_SIM_GEN_CONTEXT

#include "TList.h"
#include "RooFitCore/RooAbsGenContext.hh"
#include "RooFitCore/RooArgSet.hh"

class RooSimultaneous;
class RooDataSet;

class RooSimGenContext : public RooAbsGenContext {
public:
  RooSimGenContext(const RooSimultaneous &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		Bool_t _verbose= kFALSE);
  virtual ~RooSimGenContext();
  virtual RooDataSet *generate(Int_t nEvents= 0) const;


protected:

  RooSimGenContext(const RooSimGenContext& other) ;

  const RooDataSet *_prototype;  // Prototype data set
  const RooSimultaneous *_pdf ;  // Original PDF
  TList _gcList ;                // List of component generator contexts
  Bool_t _doGenIdx ;             // Flag set if generation of index is requested

  ClassDef(RooSimGenContext,0) // Context for generating a dataset from a PDF
};

#endif
