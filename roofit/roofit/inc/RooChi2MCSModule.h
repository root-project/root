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

#ifndef ROO_CHI2_MCS_MODULE
#define ROO_CHI2_MCS_MODULE

#include "RooAbsMCStudyModule.h"

class RooChi2MCSModule : public RooAbsMCStudyModule {
public:

  RooChi2MCSModule() ;
  RooChi2MCSModule(const RooChi2MCSModule& other) ;
  ~RooChi2MCSModule() override ;

  bool initializeInstance() override ;
  bool initializeRun(Int_t /*numSamples*/) override ;
  RooDataSet* finalizeRun() override ;
  bool processAfterFit(Int_t /*sampleNum*/) override  ;

private:

  RooDataSet* _data = nullptr;    // Summary dataset to store results
  RooRealVar* _chi2 = nullptr;    // Chi^2 of function w.r.t. data
  RooRealVar* _ndof = nullptr;    // Number of degrees of freedom
  RooRealVar* _chi2red = nullptr; // Reduced Chi^2 w.r.t data
  RooRealVar* _prob = nullptr;    // Probability of chi^2,nDOF combination

  ClassDefOverride(RooChi2MCSModule,0) // MCStudy module to calculate chi2 between binned data and fit
} ;


#endif

