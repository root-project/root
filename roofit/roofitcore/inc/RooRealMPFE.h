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

class RooArgSet ;

class RooRealMPFE : public RooAbsReal {
public:
  // Constructors, assignment etc
  RooRealMPFE(const char *name, const char *title, RooAbsReal& arg, Bool_t calcInline=kFALSE) ;
  RooRealMPFE(const RooRealMPFE& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooRealMPFE(*this,newname); }
  virtual ~RooRealMPFE();

  void calculate() const ;
  virtual Double_t getValV(const RooArgSet* nset=0) const ;
  void standby() ;

  void setVerbose(Bool_t clientFlag=kTRUE, Bool_t serverFlag=kTRUE) ;

  protected:

  // Function evaluation
  virtual Double_t evaluate() const ;
  friend class RooAbsTestStatistic ;
  virtual void constOptimizeTestStatistic(ConstOpCode opcode, Bool_t doAlsoTracking=kTRUE) ;

  enum State { Initialize,Client,Server,Inline } ;
  State _state ;

  enum Message { SendReal=0, SendCat=1, Calculate=2, Retrieve=3, ReturnValue=4, Terminate=5, 
		 ConstOpt=6, Verbose=7, RetrieveErrors=8, SendError=9, LogEvalError=10 } ;
  
  void initialize() ; 
  void initVars() ;
  void serverLoop() ;

  RooRealProxy _arg ; // Function to calculate in parallel process

  RooListProxy _vars ;   // Variables
  RooArgList _saveVars ;  // Copy of variables
  mutable Bool_t _calcInProgress ;
  Bool_t _verboseClient ;
  Bool_t _verboseServer ;
  Bool_t _inlineMode ;
  mutable Bool_t _forceCalc ;
  mutable RooAbsReal::ErrorLoggingMode _remoteEvalErrorLoggingState ;
  Int_t  _pid ;            // PID of child process

  Int_t _pipeToClient[2] ; // Pipe to client process
  Int_t _pipeToServer[2] ; // Pipe to server process

  static RooMPSentinel _sentinel ;

  ClassDef(RooRealMPFE,1) // Multi-process front-end for parallel calculation of a real valued function 
};

#endif
