/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRandomizeParamMCSModule.h,v 1.2 2007/05/11 09:11:30 verkerke Exp $
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

#ifndef ROO_RANDOMIZE_PARAM_MCS_MODULE
#define ROO_RANDOMIZE_PARAM_MCS_MODULE

#include "RooAbsMCStudyModule.h"
#include "RooRealVar.h"
#include <list>

class RooRandomizeParamMCSModule : public RooAbsMCStudyModule {
public:

  RooRandomizeParamMCSModule() ;
  RooRandomizeParamMCSModule(const RooRandomizeParamMCSModule& other) ;
  ~RooRandomizeParamMCSModule() override ;

  void sampleUniform(RooRealVar& param, double lo, double hi) ;
  void sampleGaussian(RooRealVar& param, double mean, double sigma) ;

  void sampleSumUniform(const RooArgSet& paramSet, double lo, double hi) ;
  void sampleSumGauss(const RooArgSet& paramSet, double lo, double hi) ;

  bool initializeInstance() override ;

  bool initializeRun(Int_t /*numSamples*/) override ;
  RooDataSet* finalizeRun() override ;

  bool processBeforeGen(Int_t /*sampleNum*/) override ;

private:

  struct UniParam {
     UniParam() {}
     UniParam(RooRealVar* p, double lo, double hi) : _param(p), _lo(lo), _hi(hi) {}
     bool operator==(const UniParam& other) { return (_param==other._param) ; }
     bool operator<(const UniParam& other) { return (_lo<other._lo) ; }
     RooRealVar* _param ;
     double _lo ;
     double _hi ;
  } ;

  struct UniParamSet {
     UniParamSet() {}
     UniParamSet(const RooArgSet& pset, double lo, double hi) : _pset(pset), _lo(lo), _hi(hi) {}
     bool operator==(const UniParamSet& other) { return (_lo==other._lo) ; }
     bool operator<(const UniParamSet& other) { return (_lo<other._lo) ; }
     RooArgSet _pset ;
     double _lo ;
     double _hi ;
  } ;

  struct GausParam {
     GausParam() {}
     GausParam(RooRealVar* p, double mean, double sigma) : _param(p), _mean(mean), _sigma(sigma) {}
     bool operator==(const GausParam& other) { return (_param==other._param) ; }
     bool operator<(const GausParam& other) { return (_mean<other._mean) ; }
     RooRealVar* _param ;
     double _mean ;
     double _sigma ;
  } ;

  struct GausParamSet {
     GausParamSet() {}
     GausParamSet(const RooArgSet& pset, double mean, double sigma) : _pset(pset), _mean(mean), _sigma(sigma) {}
     bool operator==(const GausParamSet& other) { return (_mean==other._mean) ; }
     bool operator<(const GausParamSet& other) { return (_mean<other._mean) ; }
     RooArgSet _pset ;
     double _mean ;
     double _sigma ;
  } ;

  std::list<UniParam>     _unifParams ; ///<!
  std::list<UniParamSet>  _unifParamSets ; ///<!
  std::list<GausParam>    _gausParams ; ///<!
  std::list<GausParamSet> _gausParamSets ; ///<!

  RooArgSet _genParSet ;
  RooDataSet* _data ;

  ClassDefOverride(RooRandomizeParamMCSModule,0) // MCStudy module to vary one or more input parameters during fit/generation cycle
} ;


#endif

