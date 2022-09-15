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
#ifndef ROO_BINNED_GEN_CONTEXT
#define ROO_BINNED_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"


class RooDataSet;
class RooDataHist;

class RooBinnedGenContext : public RooAbsGenContext {
public:
  RooBinnedGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= nullptr,
                   const RooArgSet* auxProto=nullptr, bool _verbose= false);
  ~RooBinnedGenContext() override;

  RooDataSet* generate(double nEvents=0.0, bool skipInit=false, bool extendedMode=false) override ;

  void setProtoDataOrder(Int_t*) override  {}

  void attach(const RooArgSet& params) override ;

  void printMultiline(std::ostream &os, Int_t content, bool verbose=false, TString indent="") const override ;

  void setExpectedData(bool) override ;

protected:

  void initGenerator(const RooArgSet &theEvent) override;
  void generateEvent(RooArgSet &theEvent, Int_t remaining) override;

  RooBinnedGenContext(const RooBinnedGenContext& other) ;

  const RooArgSet* _vars ;
  RooArgSet* _pdfSet ;          ///<  Set owned all nodes of internal clone of p.d.f
  RooAbsPdf *_pdf ;             ///<  Pointer to cloned p.d.f
  RooDataHist* _hist ;          ///< Histogram
  bool _expectedData ;        ///< Asimov?

  ClassDefOverride(RooBinnedGenContext,0) // Specialized context for generating a dataset from a binned pdf
};

#endif
