/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   28-Jun-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_REAL_MPFE
#define ROO_REAL_MPFE

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooListProxy.hh"
#include "RooFitCore/RooArgList.hh"

class RooArgSet ;
class RooMPSentinel ;

class RooRealMPFE : public RooAbsReal {
public:
  // Constructors, assignment etc
  RooRealMPFE(const char *name, const char *title, RooAbsReal& arg, Bool_t calcInline=kFALSE) ;
  RooRealMPFE(const RooRealMPFE& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooRealMPFE(*this,newname); }
  virtual ~RooRealMPFE();

  void calculate() const ;
  virtual Double_t getVal(const RooArgSet* nset=0) const ;
  void standby() ;

  void setVerbose(Bool_t clientFlag=kTRUE, Bool_t serverFlag=kTRUE) ;

  protected:

  // Function evaluation
  virtual Double_t evaluate() const ;
  friend class RooAbsGoodnessOfFit ;
  virtual void constOptimize(ConstOpCode opcode) ;

  enum State { Initialize,Client,Server,Inline } ;
  State _state ;

  enum Message { SendReal, SendCat, Calculate, Retrieve, ReturnValue, Terminate, ConstOpt, Verbose } ;
  
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

  Int_t _pipeToClient[2] ; // Pipe to client process
  Int_t _pipeToServer[2] ; // Pipe to server process

  static RooMPSentinel _sentinel ;

  ClassDef(RooRealMPFE,1) // Multi-process front-end for parallel calculation of a real valued function 
};

#endif
