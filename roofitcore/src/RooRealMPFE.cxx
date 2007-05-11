/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealMPFE.cc,v 1.13 2005/06/20 15:44:56 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooRealMPFE is the multi-processor front-end for parallel calculation
// of RooAbsReal objects. Each RooRealMPFE forks a process that calculates
// the value of the proxies RooAbsReal object. The (re)calculation of
// the proxied object is started asynchronously with the calculate() option.
// A subsequent call to getVal() will return the calculated value when available
// If the calculation is still in progress when getVal() is called it blocks
// the calling process until the calculation is done.
// The forked calculation process is terminated when the front-end object
// is deleted

#ifndef _WIN32
#include "RooFit.h"

#include <unistd.h>
#include <unistd.h>
#endif
#include <errno.h>
#include "RooRealMPFE.h"
#include "RooArgSet.h"
#include "RooAbsCategory.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooMPSentinel.h"

RooMPSentinel RooRealMPFE::_sentinel ;

ClassImp(RooRealMPFE)
  ;

RooRealMPFE::RooRealMPFE(const char *name, const char *title, RooAbsReal& arg, Bool_t calcInline) : 
  RooAbsReal(name,title),
  _state(Initialize),
  _arg("arg","arg",this,arg),
  _vars("vars","vars",this),
  _verboseClient(kFALSE),
  _verboseServer(kFALSE),
  _inlineMode(calcInline)
{  
#ifdef _WIN32
  _inlineMode = kTRUE;
#endif
  initVars() ;
  _sentinel.add(*this) ;
}


RooRealMPFE::RooRealMPFE(const RooRealMPFE& other, const char* name) : 
  RooAbsReal(other, name),
  _state(other._state),
  _arg("arg",this,other._arg),
  _vars("vars",this,other._vars),
  _verboseClient(other._verboseClient),
  _verboseServer(other._verboseServer),
  _inlineMode(other._inlineMode),
  _forceCalc(other._forceCalc)
{
  // Copy constructor
  _saveVars.addClone(other._saveVars) ;
  _sentinel.add(*this) ;
}


RooRealMPFE::~RooRealMPFE() 
{
  // Destructor
  if (_state==Client) {
    standby() ;
  }
  _sentinel.remove(*this) ;
}



void RooRealMPFE::initVars()
{
  // Initialize variable list

  // Empty current lists
  _vars.removeAll() ;
  _saveVars.removeAll() ;

  // Retrieve non-constant parameters
  RooArgSet* vars = _arg.arg().getParameters(RooArgSet()) ;
  RooArgSet* ncVars = (RooArgSet*) vars->selectByAttrib("Constant",kFALSE) ;
  RooArgList varList(*ncVars) ;

  // Save in lists 
  _vars.add(varList) ;
  _saveVars.addClone(varList) ;

  // Force next calculation
  _forceCalc = kTRUE ;

  delete vars ;
  delete ncVars ;
}


void RooRealMPFE::initialize() {

  // Trivial case: Inline mode 
  if (_inlineMode) {
    _state = Inline ;
    return ;
  }

#ifndef _WIN32
  // Fork server process and setup IPC
  
  // Make client/server pipes
  pipe(_pipeToClient) ;
  pipe(_pipeToServer) ;
  
  pid_t pid = fork() ;
  if (pid==0) {

    // Start server loop 
    _state = Server ;
    serverLoop() ;
   
    // Kill server at end of service
    cout << "RooRealMPFE::initialize(" << GetName() 
	 << ") server process terminating" << endl ;
    exit(0) ;

  } else if (pid>0) {
    
    // Client process - fork successul
    cout << "RooRealMPFE::initialize(" << GetName() 
	 << ") successfully forked server process " << pid << endl ;
    _state = Client ;
    _calcInProgress = kFALSE ;

  } else {
    // Client process - fork failed    
    cout << "RooRealMPFE::initialize(" << GetName() << ") ERROR fork() failed" << endl ; 
    _state = Inline ;
  }
#endif // _WIN32
}



void RooRealMPFE::serverLoop() 
{
#ifndef _WIN32
  Bool_t doLoop(kTRUE) ;
  Message msg ;

  Int_t idx, index ;
  Double_t value ;

  while(doLoop) {
    ssize_t n = read(_pipeToServer[0],&msg,sizeof(msg)) ;
    if (n<0&&_verboseServer) perror("read") ;

    switch (msg) {
    case SendReal:
      read(_pipeToServer[0],&idx,sizeof(Int_t)) ;
      read(_pipeToServer[0],&value,sizeof(Double_t)) ;      
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> SendReal [" << idx << "]=" << value << endl ;       
      ((RooRealVar*)_vars.at(idx))->setVal(value) ;
      break ;

    case SendCat:
      read(_pipeToServer[0],&idx,sizeof(Int_t)) ;
      read(_pipeToServer[0],&index,sizeof(Int_t)) ;      
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> SendCat [" << idx << "]=" << index << endl ; 
      ((RooCategory*)_vars.at(idx))->setIndex(index) ;
      break ;

    case Calculate:
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Calculate" << endl ; 
      _value = _arg ;
      break ;

    case Retrieve:
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Retrieve" << endl ; 
      msg = ReturnValue ;
      write(_pipeToClient[1],&msg,sizeof(Message)) ;
      write(_pipeToClient[1],&_value,sizeof(Double_t)) ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC toClient> ReturnValue " << _value << endl ; 
      break ;

    case ConstOpt:
      ConstOpCode code ;
      read(_pipeToServer[0],&code,sizeof(ConstOpCode)) ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> ConstOpt " << code << endl ; 
      ((RooAbsReal&)_arg.arg()).constOptimize(code) ;      
      break ;

    case Verbose:
      Bool_t flag ;
      read(_pipeToServer[0],&flag,sizeof(Bool_t)) ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Verbose " << (flag?1:0) << endl ; 
      _verboseServer = flag ;
      break ;

    case Terminate: 
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Terminate" << endl ; 
      doLoop = kFALSE ;
      break ;

    default:
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Unknown message (code = " << msg << ")" << endl ; 
      break ;
    }
  }
#endif // _WIN32
}



void RooRealMPFE::calculate() const 
{
  // Start asynchronous calculation of arg value

  if (_state==Initialize) {
    const_cast<RooRealMPFE*>(this)->initialize() ;
  }

  // Inline mode -- Calculate value now
  if (_state==Inline) {
    //cout << "RooRealMPFE::calculate(" << GetName() << ") performing Inline calculation NOW" << endl ;
    _value = _arg ;
    clearValueDirty() ;
  }

#ifndef _WIN32
  // Compare current value of variables with saved values and send changes to server
  if (_state==Client) {
    Int_t i ;
    for (i=0 ; i<_vars.getSize() ; i++) {
      RooAbsArg* var = _vars.at(i) ;
      RooAbsArg* saveVar = _saveVars.at(i) ;
      if (!(*var==*saveVar) || _forceCalc) {
	if (_verboseClient) cout << "RooRealMPFE::calculate(" << GetName()
				 << ") variable " << _vars.at(i)->GetName() << " changed" << endl ;
	saveVar->copyCache(var) ;
	
	// send message to server
	if (dynamic_cast<RooAbsReal*>(var)) {
	  Message msg = SendReal ;
	  Double_t val = ((RooAbsReal*)var)->getVal() ;
	  write(_pipeToServer[1],&msg,sizeof(msg)) ;
	  write(_pipeToServer[1],&i,sizeof(Int_t)) ;
	  write(_pipeToServer[1],&val,sizeof(Double_t)) ;
	  if (_verboseServer) cout << "RooRealMPFE::calculate(" << GetName() 
				   << ") IPC toServer> SendReal [" << i << "]=" << val << endl ;
	} else if (dynamic_cast<RooAbsCategory*>(var)) {
	  Message msg = SendCat ;
	  Int_t idx = ((RooAbsCategory*)var)->getIndex() ;
	  write(_pipeToServer[1],&msg,sizeof(msg)) ;
	  write(_pipeToServer[1],&i,sizeof(Int_t)) ;
	  write(_pipeToServer[1],&idx,sizeof(Int_t)) ;	
	  if (_verboseServer) cout << "RooRealMPFE::calculate(" << GetName() 
				   << ") IPC toServer> SendCat [" << i << "]=" << idx << endl ;
	}
      }
    }

    Message msg = Calculate ;
    write(_pipeToServer[1],&msg,sizeof(msg)) ;
    if (_verboseServer) cout << "RooRealMPFE::calculate(" << GetName() 
			     << ") IPC toServer> Calculate " << endl ;

    // Clear dirty state and mark that calculation request was dispatched
    clearValueDirty() ;
    _calcInProgress = kTRUE ;
    _forceCalc = kFALSE ;

  } else if (_state!=Inline) {
    cout << "RooRealMPFE::calculate(" << GetName() 
	 << ") ERROR not in Client or Inline mode" << endl ;
  }
#endif // _WIN32
}




Double_t RooRealMPFE::getVal(const RooArgSet* /*nset*/) const 
{
  if (isValueDirty()) {
    // Cache is dirty, no calculation has been started yet
    //cout << "RooRealMPFE::getVal(" << GetName() << ") cache is dirty, caling calculate and evaluate" << endl ;
    calculate() ;
    _value = evaluate() ;
  } else if (_calcInProgress) {
    //cout << "RooRealMPFE::getVal(" << GetName() << ") calculation in progress, calling evaluate" << endl ;
    // Cache is clean and calculation is in progress
    _value = evaluate() ;    
  } else {
    //cout << "RooRealMPFE::getVal(" << GetName() << ") cache is clean, doing nothing" << endl ;
    // Cache is clean and calculated value is in cache
  }

  return _value ;
}



Double_t RooRealMPFE::evaluate() const
{
  // Retrieve value of arg
  Double_t return_value = 0;
  if (_state==Inline) {
    return_value = _arg ; 
  } else if (_state==Client) {
#ifndef _WIN32
    Message msg ;
    Double_t value ;

    msg = Retrieve ;
    write(_pipeToServer[1],&msg,sizeof(Message)) ;    
    if (_verboseServer) cout << "RooRealMPFE::evaluate(" << GetName() 
			     << ") IPC toServer> Retrieve " << endl ;    
    read(_pipeToClient[0],&msg,sizeof(Message)) ;
    if (msg!=ReturnValue) {
      cout << "RooRealMPFE::evaluate(" << GetName() 
	   << ") ERROR: unexpected message from server process: " << msg << endl ;
      return 0 ;
    }
    read(_pipeToClient[0],&value,sizeof(Double_t)) ;
    if (_verboseServer) cout << "RooRealMPFE::evaluate(" << GetName() 
			     << ") IPC fromServer> ReturnValue " << value << endl ;

    // Mark end of calculation in progress 
    _calcInProgress = kFALSE ;
    return_value = value ;
#endif // _WIN32
  }

  return return_value;
}


void RooRealMPFE::standby()
{
#ifndef _WIN32
  if (_state==Client) {

    // Terminate server process ;
    Message msg = Terminate ;
    write(_pipeToServer[1],&msg,sizeof(msg)) ;
    if (_verboseServer) cout << "RooRealMPFE::standby(" << GetName() 
			     << ") IPC toServer> Terminate " << endl ;  

    // Close pipes
    close(_pipeToClient[0]) ;
    close(_pipeToClient[1]) ;
    close(_pipeToServer[0]) ;
    close(_pipeToServer[1]) ;
    
    // Revert to initialize state 
    _state = Initialize ;
  }
#endif // _WIN32
}


void RooRealMPFE::constOptimize(ConstOpCode opcode) 
{
#ifndef _WIN32
  if (_state==Client) {
    Message msg = ConstOpt ;
    write(_pipeToServer[1],&msg,sizeof(msg)) ;
    write(_pipeToServer[1],&opcode,sizeof(ConstOpCode)) ;
    if (_verboseServer) cout << "RooRealMPFE::constOptimize(" << GetName() 
			     << ") IPC toServer> ConstOpt " << opcode << endl ;  

    initVars() ;
  }
#endif // _WIN32

  if (_state==Inline) {
    ((RooAbsReal&)_arg.arg()).constOptimize(opcode) ;
  }
}



void RooRealMPFE::setVerbose(Bool_t clientFlag, Bool_t serverFlag) 
{
#ifndef _WIN32
  if (_state==Client) {
    Message msg = Verbose ;
    write(_pipeToServer[1],&msg,sizeof(msg)) ;
    write(_pipeToServer[1],&serverFlag,sizeof(Bool_t)) ;
    if (_verboseServer) cout << "RooRealMPFE::setVerbose(" << GetName() 
			     << ") IPC toServer> Verbose " << (serverFlag?1:0) << endl ;      
  }
#endif // _WIN32
  _verboseClient = clientFlag ; _verboseServer = serverFlag ;
}
