/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimGenContext.rdl,v 1.3 2001/10/14 07:11:42 verkerke Exp $
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
class RooAbsCategoryLValue ;

class RooSimGenContext : public RooAbsGenContext {
public:
  RooSimGenContext(const RooSimultaneous &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		Bool_t _verbose= kFALSE);
  virtual ~RooSimGenContext();

protected:

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining);

  RooSimGenContext(const RooSimGenContext& other) ;

  RooAbsCategoryLValue* _idxCat ; // Clone of index category
  const RooDataSet *_prototype;   // Prototype data set
  const RooSimultaneous *_pdf ;   // Original PDF
  TList _gcList ;                 // List of component generator contexts
  Bool_t _haveIdxProto ;          // Flag set if generation of index is requested
  TString _idxCatName ;           // Name of index category
  Int_t _numPdf ;                 // Number of generated PDFs
  Double_t* _fracThresh ;         //[_numPdf] Fraction threshold array

  ClassDef(RooSimGenContext,0) // Context for generating a dataset from a PDF
};

#endif
