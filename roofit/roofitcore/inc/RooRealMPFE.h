/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealMPFE.h,v 1.7 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_MPFE
#define ROO_REAL_MPFE

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"
#include "RooArgList.h"
#include "RooMPSentinel.h"
#include "TStopwatch.h"
#include <vector>

class RooArgSet ;
namespace RooFit { class BidirMMapPipe; }

class RooRealMPFE : public RooAbsReal {
public:
  // Constructors, assignment etc
  RooRealMPFE(const char *name, const char *title, RooAbsReal& arg, bool calcInline=false) ;
  RooRealMPFE(const RooRealMPFE& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooRealMPFE(*this,newname); }
  ~RooRealMPFE() override;

  void calculate() const ;
  Double_t getValV(const RooArgSet* nset=0) const override ;
  void standby() ;

  void setVerbose(bool clientFlag=true, bool serverFlag=true) ;

  void applyNLLWeightSquared(bool flag) ;

  void enableOffsetting(bool flag) override ;

  void followAsSlave(RooRealMPFE& master) { _updateMaster = &master ; }

  protected:

  // Function evaluation
  Double_t evaluate() const override ;
  friend class RooAbsTestStatistic ;
  void constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTracking=true) override ;
  virtual Double_t getCarry() const;

  enum State { Initialize,Client,Server,Inline } ;
  State _state ;

  enum Message { SendReal=0, SendCat, Calculate, Retrieve, ReturnValue, Terminate,
       ConstOpt, Verbose, LogEvalError, ApplyNLLW2, EnableOffset, CalculateNoOffset } ;

  void initialize() ;
  void initVars() ;
  void serverLoop() ;

  void doApplyNLLW2(bool flag) ;

  RooRealProxy _arg ; ///< Function to calculate in parallel process
  RooListProxy _vars ;   ///< Variables
  RooArgList _saveVars ;  ///< Copy of variables
  mutable bool _calcInProgress ;
  bool _verboseClient ;
  bool _verboseServer ;
  bool _inlineMode ;
  mutable bool _forceCalc ;
  mutable RooAbsReal::ErrorLoggingMode _remoteEvalErrorLoggingState ;

  RooFit::BidirMMapPipe *_pipe; ///<! connection to child

  mutable std::vector<bool> _valueChanged ; ///<! Flags if variable needs update on server-side
  mutable std::vector<bool> _constChanged ; ///<! Flags if variable needs update on server-side
  RooRealMPFE* _updateMaster ; ///<! Update master
  mutable bool _retrieveDispatched ; ///<!
  mutable Double_t _evalCarry; ///<!

  static RooMPSentinel _sentinel ;

  ClassDefOverride(RooRealMPFE,2) // Multi-process front-end for parallel calculation of a real valued function
};

#endif
