/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_SIM_SPLIT_GEN_CONTEXT
#define ROO_SIM_SPLIT_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"
#include <vector>

class RooSimultaneous;
class RooDataSet;
class RooAbsCategoryLValue ;

class RooSimSplitGenContext : public RooAbsGenContext {
public:
  RooSimSplitGenContext(const RooSimultaneous &model, const RooArgSet &vars, Bool_t _verbose= kFALSE, Bool_t autoBinned=kTRUE, const char* binnedTag="");
  ~RooSimSplitGenContext() override;
  void setProtoDataOrder(Int_t* lut) override ;

  void attach(const RooArgSet& params) override ;

  void printMultiline(std::ostream &os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const override ;

  RooDataSet *generate(Double_t nEvents= 0, Bool_t skipInit=kFALSE, Bool_t extendedMode=kFALSE) override;

  void setExpectedData(Bool_t) override ;

protected:

  void initGenerator(const RooArgSet &theEvent) override;
  void generateEvent(RooArgSet &theEvent, Int_t remaining) override;

  RooDataSet* createDataSet(const char* name, const char* title, const RooArgSet& obs) override ;

  RooSimSplitGenContext(const RooSimSplitGenContext& other) ;

  RooAbsCategoryLValue* _idxCat ;           ///< Clone of index category
  RooArgSet*            _idxCatSet ;        ///< Owner of index category components
  const RooSimultaneous *_pdf ;             ///< Original PDF
  std::vector<RooAbsGenContext*> _gcList ;  ///< List of component generator contexts
  std::vector<int>               _gcIndex ; ///< Index value corresponding to component
  TString _idxCatName ;                     ///< Name of index category
  Int_t _numPdf ;                           ///< Number of generated PDFs
  Double_t* _fracThresh ;                   ///< fraction thresholds

  RooArgSet _allVarsPdf ; ///< All pdf variables
  TIterator* _proxyIter ; ///< Iterator over pdf proxies

  ClassDefOverride(RooSimSplitGenContext,0) // Context for efficiently generating a dataset from a RooSimultaneous PDF
};

#endif
