/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGenContext.h,v 1.19 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_GEN_CONTEXT
#define ROO_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"

class RooAbsPdf;
class RooDataSet;
class RooRealIntegral;
class RooAcceptReject;
class TRandom;
class RooRealVar ;
class RooAbsNumGenerator ;

class RooGenContext : public RooAbsGenContext {
public:
  RooGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= nullptr,
      const RooArgSet* auxProto=nullptr, bool verbose=false, const RooArgSet* forceDirect=nullptr);
  ~RooGenContext() override;

  void printMultiline(std::ostream &os, Int_t content, bool verbose=false, TString indent="") const override ;

  void attach(const RooArgSet& params) override ;

protected:

  void initGenerator(const RooArgSet &theEvent) override;
  void generateEvent(RooArgSet &theEvent, Int_t remaining) override;

  RooArgSet _cloneSet;    ///< Clone of all nodes of input p.d.f
  RooAbsPdf *_pdfClone;   ///< Clone of input p.d.f
  RooArgSet _directVars,_uniformVars,_otherVars; ///< List of observables generated internally, randomly, and by accept/reject sampling
  Int_t _code;                                   ///< Internal generation code
  double _maxProb{0.}, _area{0.}, _norm{0.};   ///< Maximum probability, p.d.f area and normalization
  RooRealIntegral *_acceptRejectFunc; ///< Projection function to be passed to accept/reject sampler
  RooAbsNumGenerator *_generator;     ///< MC sampling generation engine
  RooRealVar *_maxVar ;               ///< Variable holding maximum value of p.d.f
  Int_t _updateFMaxPerEvent ;         ///< If true, maximum p.d.f value needs to be recalculated for each event

  ClassDefOverride(RooGenContext,0) // Universal context for generating toy MC data from any p.d.f
};

#endif
