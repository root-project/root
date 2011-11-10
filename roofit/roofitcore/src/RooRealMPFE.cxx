/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooRealMPFE is the multi-processor front-end for parallel calculation
// of RooAbsReal objects. Each RooRealMPFE forks a process that calculates
// the value of the proxies RooAbsReal object. The (re)calculation of
// the proxied object is started asynchronously with the calculate() option.
// A subsequent call to getVal() will return the calculated value when available
// If the calculation is still in progress when getVal() is called it blocks
// the calling process until the calculation is done. The forked calculation process 
// is terminated when the front-end object is deleted
// Simple use demonstration
//
// <pre>
// RooAbsReal* slowFunc ;
//
// Double_t val = slowFunc->getVal() // Evaluate slowFunc in current process
//
// RooRealMPFE mpfe("mpfe","frontend to slowFunc",*slowFunc) ;
// mpfe.calculate() ;           // Start calculation of slow-func in remote process
//                              // .. do other stuff here ..
// Double_t val = mpfe.getVal() // Wait for remote calculation to finish and retrieve value
// </pre>
//
// END_HTML
//

#include "Riostream.h"
#include "RooFit.h"

#ifndef _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

#include <errno.h>
#include <sstream>
#include "RooRealMPFE.h"
#include "RooArgSet.h"
#include "RooAbsCategory.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooMPSentinel.h"
#include "RooMsgService.h"

#include "TSystem.h"

RooMPSentinel RooRealMPFE::_sentinel ;

ClassImp(RooRealMPFE)
  ;


//_____________________________________________________________________________
RooRealMPFE::RooRealMPFE(const char *name, const char *title, RooAbsReal& arg, Bool_t calcInline) : 
  RooAbsReal(name,title),
  _state(Initialize),
  _arg("arg","arg",this,arg),
  _vars("vars","vars",this),
  _calcInProgress(kFALSE),
  _verboseClient(kFALSE),
  _verboseServer(kFALSE),
  _inlineMode(calcInline),
  _remoteEvalErrorLoggingState(RooAbsReal::PrintErrors),
  _pid(0)
{  
  // Construct front-end object for object 'arg' whose evaluation will be calculated
  // asynchronously in a separate process. If calcInline is true the value of 'arg'
  // is calculate synchronously in the current process.
#ifdef _WIN32
  _inlineMode = kTRUE;
#endif
  initVars() ;
  _sentinel.add(*this) ;
}



//_____________________________________________________________________________
RooRealMPFE::RooRealMPFE(const RooRealMPFE& other, const char* name) : 
  RooAbsReal(other, name),
  _state(Initialize),
  _arg("arg",this,other._arg),
  _vars("vars",this,other._vars),
  _calcInProgress(kFALSE),
  _verboseClient(other._verboseClient),
  _verboseServer(other._verboseServer),
  _inlineMode(other._inlineMode),
  _forceCalc(other._forceCalc),
  _remoteEvalErrorLoggingState(other._remoteEvalErrorLoggingState),
  _pid(0)
{
  // Copy constructor. Initializes in clean state so that upon eval
  // this instance will create its own server processes

  initVars() ;
  _sentinel.add(*this) ;
}



//_____________________________________________________________________________
RooRealMPFE::~RooRealMPFE() 
{
  // Destructor

  if (_state==Client) {
    standby() ;
  }
  _sentinel.remove(*this) ;
}



//_____________________________________________________________________________
void RooRealMPFE::initVars()
{
  // Initialize list of variables of front-end argument 'arg'

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



//_____________________________________________________________________________
void RooRealMPFE::initialize() 
{
  // Initialize the remote process and message passing
  // pipes between current process and remote process

  // Trivial case: Inline mode 
  if (_inlineMode) {
    _state = Inline ;
    return ;
  }

#ifndef _WIN32
  // Fork server process and setup IPC
  
  // Make client/server pipes
  UInt_t tmp1 = pipe(_pipeToClient) ;
  UInt_t tmp2 = pipe(_pipeToServer) ;
  if (tmp1 || tmp2) perror("pipe") ;
  
  // Clear eval error log prior to forking
  // to avoid confusions...
  clearEvalErrorLog() ;

  _pid = fork() ;

  if (_pid==0) {

    // Start server loop 
    _state = Server ;
    serverLoop() ;
   
    // Kill server at end of service
    coutI(Minimization) << "RooRealMPFE::initialize(" << GetName() 
			<< ") server process terminating" << endl ;
    _exit(0) ;

  } else if (_pid>0) {
 
    // Client process - fork successul
    coutI(Minimization) << "RooRealMPFE::initialize(" << GetName() 
			<< ") successfully forked server process " << _pid << endl ;
    _state = Client ;
    _calcInProgress = kFALSE ;

  } else {
    // Client process - fork failed    
    coutE(Minimization) << "RooRealMPFE::initialize(" << GetName() << ") ERROR fork() failed" << endl ; 
    _state = Inline ;
  }
#endif // _WIN32
}



//_____________________________________________________________________________
void RooRealMPFE::serverLoop() 
{
  // Server loop of remote processes. This function will return
  // only when an incoming TERMINATE message is received.

#ifndef _WIN32
  Bool_t doLoop(kTRUE) ;
  Message msg ;

  Int_t idx, index, numErrors ;
  Double_t value ;
  Bool_t isConst ;
  
  clearEvalErrorLog() ;

  while(doLoop) {
    ssize_t n = read(_pipeToServer[0],&msg,sizeof(msg)) ;
    if (n<0&&_verboseServer) perror("read") ;

    switch (msg) {
    case SendReal:
      {
      UInt_t tmp1 = read(_pipeToServer[0],&idx,sizeof(Int_t)) ;
      UInt_t tmp2 = read(_pipeToServer[0],&value,sizeof(Double_t)) ;      
      UInt_t tmp3 = read(_pipeToServer[0],&isConst,sizeof(Bool_t)) ;      
      if (tmp1+tmp2+tmp3<sizeof(Int_t)+sizeof(Double_t)+sizeof(Bool_t)) perror("read") ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> SendReal [" << idx << "]=" << value << endl ;       
      ((RooRealVar*)_vars.at(idx))->setVal(value) ;
      ((RooRealVar*)_vars.at(idx))->setConstant(isConst) ;
      }
      break ;

    case SendCat:
      {
      UInt_t tmp1 = read(_pipeToServer[0],&idx,sizeof(Int_t)) ;
      UInt_t tmp2 = read(_pipeToServer[0],&index,sizeof(Int_t)) ;      
      if (tmp1+tmp2<sizeof(Int_t)+sizeof(Int_t)) perror("read") ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> SendCat [" << idx << "]=" << index << endl ; 
      ((RooCategory*)_vars.at(idx))->setIndex(index) ;
      }
      break ;

    case Calculate:
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Calculate" << endl ; 
      _value = _arg ;
      break ;

    case Retrieve:
      {
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Retrieve" << endl ; 
      msg = ReturnValue ;
      UInt_t tmp1 = write(_pipeToClient[1],&msg,sizeof(Message)) ;
      UInt_t tmp2 = write(_pipeToClient[1],&_value,sizeof(Double_t)) ;
      numErrors = numEvalErrors() ;
      UInt_t tmp3 = write(_pipeToClient[1],&numErrors,sizeof(Int_t)) ;
      if (tmp1+tmp2+tmp3<sizeof(Message)+sizeof(Double_t)+sizeof(Int_t)) perror("write") ;

      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
 			       << ") IPC toClient> ReturnValue " << _value << " NumError " << numErrors << endl ; 
      }
      break ;

    case ConstOpt:
      {
      ConstOpCode code ; 
      Bool_t doTrack ;
      UInt_t tmp1 = read(_pipeToServer[0],&code,sizeof(ConstOpCode)) ;
      UInt_t tmp2 = read(_pipeToServer[0],&doTrack,sizeof(Bool_t)) ;
      if (tmp1<sizeof(ConstOpCode)) perror("read") ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> ConstOpt " << code << " doTrack = " << (doTrack?"T":"F") << endl ; 
      ((RooAbsReal&)_arg.arg()).constOptimizeTestStatistic(code,doTrack) ;      
      break ;
      }

    case Verbose:
      {
      Bool_t flag ;
      UInt_t tmp1 = read(_pipeToServer[0],&flag,sizeof(Bool_t)) ;
      if (tmp1<sizeof(Bool_t)) perror("read") ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Verbose " << (flag?1:0) << endl ; 
      _verboseServer = flag ;
      }
      break ;

    case Terminate: 
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Terminate" << endl ; 
      doLoop = kFALSE ;
      break ;

    case LogEvalError:
      {
      RooAbsReal::ErrorLoggingMode flag2 ;
      UInt_t tmp1 = read(_pipeToServer[0],&flag2,sizeof(RooAbsReal::ErrorLoggingMode)) ;
      if (tmp1<sizeof(RooAbsReal::ErrorLoggingMode)) perror("read") ;
      RooAbsReal::setEvalErrorLoggingMode(flag2) ;
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> LogEvalError flag = " << flag2 << endl ;       
      }
      break ;

    case RetrieveErrors:

      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> RetrieveErrors" << endl ; 

      // Loop over errors
      {
	std::map<const RooAbsArg*,pair<string,list<EvalError> > >::const_iterator iter = evalErrorIter() ;
	for (int i=0 ; i<numEvalErrorItems() ; i++) {
	  list<EvalError>::const_iterator iter2 = iter->second.second.begin() ;
	  for (;iter2!=iter->second.second.end();++iter2) {
	    
	    // Reply with SendError message
	    msg = SendError ;
	    UInt_t tmp1 = write(_pipeToClient[1],&msg,sizeof(Message)) ;
	    UInt_t tmp2 = write(_pipeToClient[1],&iter->first,sizeof(RooAbsReal*)) ;
	    
	    UInt_t ntext = strlen(iter2->_msg) ;
	    UInt_t tmp3 = write(_pipeToClient[1],&ntext,sizeof(Int_t)) ;
	    UInt_t tmp4 = write(_pipeToClient[1],iter2->_msg,ntext+1) ;
	    
	    UInt_t ntext2 = strlen(iter2->_srvval) ;
	    UInt_t tmp5 = write(_pipeToClient[1],&ntext2,sizeof(Int_t)) ;
	    UInt_t tmp6 = write(_pipeToClient[1],iter2->_srvval,ntext2+1) ;

	    // Format string with object identity as this cannot be evaluated on the other side
	    ostringstream oss2 ;
	    oss2 << "PID" << gSystem->GetPid() << "/" ;
	    printStream(oss2,kName|kClassName|kArgs,kInline)  ;
	    UInt_t ntext3 = strlen(oss2.str().c_str()) ;
	    UInt_t tmp7 = write(_pipeToClient[1],&ntext3,sizeof(Int_t)) ;
	    UInt_t tmp8 = write(_pipeToClient[1],oss2.str().c_str(),ntext3+1) ;

	    if (tmp1+tmp2+tmp3+tmp4+tmp5+tmp6+tmp7+tmp8<
		sizeof(Message)+sizeof(RooAbsReal*)+sizeof(Int_t)+ntext+1+sizeof(Int_t)+ntext2+1+sizeof(Int_t)+ntext3+1) 
	      perror("write") ;
	    	    
	    if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
				     << ") IPC toClient> SendError Arg " << iter->first << " Msg " << iter2->_msg << endl ; 
	  }
	}  
	
	RooAbsReal* null(0) ;
	UInt_t tmp1 = write(_pipeToClient[1],&msg,sizeof(Message)) ;
	UInt_t tmp2 = write(_pipeToClient[1],&null,sizeof(RooAbsReal*)) ;
	if (tmp1+tmp2<sizeof(Message)+sizeof(RooAbsReal*)) perror("read") ;
      
      }
      // Clear error list on local side
      clearEvalErrorLog() ;           
      break ;

    default:
      if (_verboseServer) cout << "RooRealMPFE::serverLoop(" << GetName() 
			       << ") IPC fromClient> Unknown message (code = " << msg << ")" << endl ; 
      break ;
    }
  }
#endif // _WIN32
}



//_____________________________________________________________________________
void RooRealMPFE::calculate() const 
{
  // Client-side function that instructs server process to start
  // asynchronuous (re)calculation of function value. This function
  // returns immediately. The calculated value can be retrieved
  // using getVal()

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
      
      if (!(*var==*saveVar) || (var->isConstant() != saveVar->isConstant()) || _forceCalc) {
	if (_verboseClient) cout << "RooRealMPFE::calculate(" << GetName()
				 << ") variable " << _vars.at(i)->GetName() << " changed" << endl ;

	((RooRealVar*)saveVar)->setConstant(var->isConstant()) ;
	saveVar->copyCache(var) ;
	
	// send message to server
	if (dynamic_cast<RooAbsReal*>(var)) {
	  Message msg = SendReal ;
	  Double_t val = ((RooAbsReal*)var)->getVal() ;
	  Bool_t isC = var->isConstant() ;
	  UInt_t tmp1 = write(_pipeToServer[1],&msg,sizeof(msg)) ;
	  UInt_t tmp2 = write(_pipeToServer[1],&i,sizeof(Int_t)) ;
	  UInt_t tmp3 = write(_pipeToServer[1],&val,sizeof(Double_t)) ;
	  UInt_t tmp4 = write(_pipeToServer[1],&isC,sizeof(Bool_t)) ;
	  if (tmp1+tmp2+tmp3+tmp4<sizeof(Message)+sizeof(Int_t)+sizeof(Double_t)+sizeof(Bool_t)) perror("write") ;

	  if (_verboseServer) cout << "RooRealMPFE::calculate(" << GetName() 
				   << ") IPC toServer> SendReal [" << i << "]=" << val << (isC?" (Constant)":"") <<  endl ;
	} else if (dynamic_cast<RooAbsCategory*>(var)) {
	  Message msg = SendCat ;
	  UInt_t idx = ((RooAbsCategory*)var)->getIndex() ;
	  UInt_t tmp5 = write(_pipeToServer[1],&msg,sizeof(msg)) ;
	  UInt_t tmp6 = write(_pipeToServer[1],&i,sizeof(Int_t)) ;
	  UInt_t tmp7 = write(_pipeToServer[1],&idx,sizeof(Int_t)) ;	
	  if (tmp5+tmp6+tmp7<sizeof(Message)+sizeof(Int_t)+sizeof(Int_t)) perror("write") ;
	  if (_verboseServer) cout << "RooRealMPFE::calculate(" << GetName() 
				   << ") IPC toServer> SendCat [" << i << "]=" << idx << endl ;
	}
      }
    }

    Message msg = Calculate ;
    UInt_t tmp8 = write(_pipeToServer[1],&msg,sizeof(msg)) ;
    if (tmp8<sizeof(Message)) perror("write") ;
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




//_____________________________________________________________________________
Double_t RooRealMPFE::getValV(const RooArgSet* /*nset*/) const 
{
  // If value needs recalculation and calculation has not beed started
  // with a call to calculate() start it now. This function blocks
  // until remote process has finished calculation and returns
  // remote value

  if (isValueDirty()) {
    // Cache is dirty, no calculation has been started yet
    //cout << "RooRealMPFE::getValF(" << GetName() << ") cache is dirty, caling calculate and evaluate" << endl ;
    calculate() ;
    _value = evaluate() ;
  } else if (_calcInProgress) {
    //cout << "RooRealMPFE::getValF(" << GetName() << ") calculation in progress, calling evaluate" << endl ;
    // Cache is clean and calculation is in progress
    _value = evaluate() ;    
  } else {
    //cout << "RooRealMPFE::getValF(" << GetName() << ") cache is clean, doing nothing" << endl ;
    // Cache is clean and calculated value is in cache
  }

  return _value ;
}



//_____________________________________________________________________________
Double_t RooRealMPFE::evaluate() const
{
  // Send message to server process to retrieve output value
  // If error were logged use logEvalError() on remote side
  // transfer those errors to the local eval error queue.

  // Retrieve value of arg
  Double_t return_value = 0;
  if (_state==Inline) {
    return_value = _arg ; 
  } else if (_state==Client) {
#ifndef _WIN32
    Message msg ;
    Double_t value ;

    // If current error loggin state is not the same as remote state
    // update the remote state
    if (evalErrorLoggingMode() != _remoteEvalErrorLoggingState) {
      msg = LogEvalError ;
      UInt_t tmp1 = write(_pipeToServer[1],&msg,sizeof(Message)) ;    
      RooAbsReal::ErrorLoggingMode flag = evalErrorLoggingMode() ;
      UInt_t tmp2 = write(_pipeToServer[1],&flag,sizeof(RooAbsReal::ErrorLoggingMode)) ;      
      _remoteEvalErrorLoggingState = evalErrorLoggingMode() ;
      if (tmp1+tmp2<sizeof(Message)+sizeof(RooAbsReal::ErrorLoggingMode)) perror("write") ;
    }

    msg = Retrieve ;
    UInt_t tmp1 = write(_pipeToServer[1],&msg,sizeof(Message)) ;    
    if (_verboseServer) cout << "RooRealMPFE::evaluate(" << GetName() 
			     << ") IPC toServer> Retrieve " << endl ;    
    if (tmp1<sizeof(Message)) perror("write") ;

    UInt_t tmp2 = read(_pipeToClient[0],&msg,sizeof(Message)) ;
    if (msg!=ReturnValue) {
      cout << "RooRealMPFE::evaluate(" << GetName() 
	   << ") ERROR: unexpected message from server process: " << msg << endl ;
      return 0 ;
    }
    UInt_t tmp3 = read(_pipeToClient[0],&value,sizeof(Double_t)) ;
    if (_verboseServer) cout << "RooRealMPFE::evaluate(" << GetName() 
			     << ") IPC fromServer> ReturnValue " << value << endl ;

    Int_t numError ;
    UInt_t tmp4 = read(_pipeToClient[0],&numError,sizeof(Int_t)) ;
    if (_verboseServer) cout << "RooRealMPFE::evaluate(" << GetName() 
			     << ") IPC fromServer> NumErrors " << numError << endl ;
    if (tmp2+tmp3+tmp4<sizeof(Message)+sizeof(Double_t)+sizeof(Int_t)) perror("read") ;


    // Retrieve remote errors and feed into local error queue
    if (numError>0) {
      msg=RetrieveErrors ;
      UInt_t tmp5 = write(_pipeToServer[1],&msg,sizeof(Message)) ;    
      if (tmp5<sizeof(Message)) perror("write") ;
      if (_verboseServer) cout << "RooRealMPFE::evaluate(" << GetName() 
			     << ") IPC toServer> RetrieveErrors " << endl ;    

      while(true) {
	RooAbsReal* ptr(0) ;
	Int_t ntext1,ntext2,ntext3 ;
	char msgbuf1[1024] ;
	char msgbuf2[1024] ;
	char msgbuf3[1024] ;
	UInt_t tmp12= read(_pipeToClient[0],&msg,sizeof(Message)) ;
	UInt_t tmp13= read(_pipeToClient[0],&ptr,sizeof(RooAbsReal*)) ;
	if (tmp12+tmp13<sizeof(Message)+sizeof(RooAbsReal*)) perror("read") ;
	if (ptr==0) {
	  break ;
	}

	UInt_t tmp6 = read(_pipeToClient[0],&ntext1,sizeof(Int_t)) ;
	if (ntext1>1023) ntext1=1023 ; if (ntext1<0) ntext1=0 ; 
	UInt_t tmp7 = read(_pipeToClient[0],msgbuf1,ntext1+1) ;
	UInt_t tmp8 = read(_pipeToClient[0],&ntext2,sizeof(Int_t)) ;
	if (ntext2>1023) ntext2=1023 ; if (ntext2<0) ntext2=0 ; 
	UInt_t tmp9 = read(_pipeToClient[0],msgbuf2,ntext2+1) ;
	UInt_t tmp10= read(_pipeToClient[0],&ntext3,sizeof(Int_t)) ;
	if (ntext3>1023) ntext3=1023 ; if (ntext3<0) ntext3=0 ; 
	UInt_t tmp11= read(_pipeToClient[0],msgbuf3,ntext3+1) ;
	if (tmp6+tmp7+tmp8+tmp9+tmp10+tmp11<sizeof(Int_t)+ntext1+1+sizeof(Int_t)+ntext2+1+sizeof(Int_t)+ntext3+1) perror("read") ;
	
	if (_verboseServer) cout << "RooRealMPFE::evaluate(" << GetName() 
				 << ") IPC fromServer> SendError Arg " << ptr << " Msg " << msgbuf1 << endl ;    
	
	// WVE: null char is read from pipe  
	// coverity[STRING_NULL]
	logEvalError(ptr,msgbuf3,msgbuf1,msgbuf2) ;
      }
	
    }

    // Mark end of calculation in progress 
    _calcInProgress = kFALSE ;
    return_value = value ;
#endif // _WIN32
  }

  return return_value;
}



//_____________________________________________________________________________
void RooRealMPFE::standby()
{
  // Terminate remote server process and return front-end class
  // to standby mode. Calls to calculate() or evaluate() after
  // this call will automatically recreated the server process.

#ifndef _WIN32
  if (_state==Client) {

    // Terminate server process ;
    Message msg = Terminate ;
    UInt_t tmp1 = write(_pipeToServer[1],&msg,sizeof(msg)) ;
    if (tmp1<sizeof(Message)) perror("write") ;
    if (_verboseServer) cout << "RooRealMPFE::standby(" << GetName() 
			     << ") IPC toServer> Terminate " << endl ;  

    // Close pipes
    close(_pipeToClient[0]) ;
    close(_pipeToClient[1]) ;
    close(_pipeToServer[0]) ;
    close(_pipeToServer[1]) ;

    // Release resource of child process
    waitpid(_pid,0,0) ;
    
    // Revert to initialize state 
    _state = Initialize ;
  }
#endif // _WIN32
}



//_____________________________________________________________________________
void RooRealMPFE::constOptimizeTestStatistic(ConstOpCode opcode, Bool_t doAlsoTracking) 
{
  // Intercept call to optimize constant term in test statistics
  // and forward it to object on server side.

#ifndef _WIN32
  if (_state==Client) {
    Message msg = ConstOpt ;
    UInt_t tmp1 = write(_pipeToServer[1],&msg,sizeof(msg)) ;
    UInt_t tmp2 = write(_pipeToServer[1],&opcode,sizeof(ConstOpCode)) ;
    UInt_t tmp3 = write(_pipeToServer[2],&doAlsoTracking,sizeof(Bool_t)) ;
    if (tmp1+tmp2+tmp3<sizeof(Message)+sizeof(ConstOpCode)+sizeof(Bool_t)) perror("write") ;
    if (_verboseServer) cout << "RooRealMPFE::constOptimize(" << GetName() 
			     << ") IPC toServer> ConstOpt " << opcode << endl ;  

    initVars() ;
  }
#endif // _WIN32

  if (_state==Inline) {
    ((RooAbsReal&)_arg.arg()).constOptimizeTestStatistic(opcode) ;
  }
}



//_____________________________________________________________________________
void RooRealMPFE::setVerbose(Bool_t clientFlag, Bool_t serverFlag) 
{
  // Control verbose messaging related to inter process communication
  // on both client and server side

#ifndef _WIN32
  if (_state==Client) {
    Message msg = Verbose ;
    UInt_t tmp1 = write(_pipeToServer[1],&msg,sizeof(msg)) ;
    UInt_t tmp2 = write(_pipeToServer[1],&serverFlag,sizeof(Bool_t)) ;
    if (tmp1+tmp2<sizeof(Message)+sizeof(Bool_t)) perror("write") ;
    if (_verboseServer) cout << "RooRealMPFE::setVerbose(" << GetName() 
			     << ") IPC toServer> Verbose " << (serverFlag?1:0) << endl ;      
  }
#endif // _WIN32
  _verboseClient = clientFlag ; _verboseServer = serverFlag ;
}
