/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealIntegral.cc,v 1.11 2001/05/14 05:22:55 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include "TObjString.h"
#include "TH1.h"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooMultiCatIter.hh"
#include "RooFitCore/RooIntegrator1D.hh"

ClassImp(RooRealIntegral) 
;


RooRealIntegral::RooRealIntegral(const char *name, const char *title, 
				 const RooAbsReal& function, RooArgSet& depList,
				 Int_t maxSteps, Double_t eps) : 
  RooAbsReal(name,title), _function((RooAbsReal*)&function), _mode(0),
  _intList("intList"), _sumList("sumList"), _numIntEngine(0) 
{
  RooArgSet intDepList ;

  // Loop over list of servers of integrand
  // to determine list of variables to integrate over
  TIterator *sIter = function.serverIterator() ;
  RooAbsArg *arg ;
  while(arg=(RooAbsArg*)sIter->Next()) {

    // Dependent or parameter?
    if (!arg->dependsOn(depList)) {
      // Add to parameter as value server
      addServer(*arg,kTRUE,kFALSE) ;
      continue ;
    }
    
    Bool_t expandArg(kTRUE) ;

    // Check for integratable AbsRealLValue
    if (arg->isDerived()) {
      RooAbsRealLValue    *realArgLV = dynamic_cast<RooAbsRealLValue*>(arg) ;
      RooAbsCategoryLValue *catArgLV = dynamic_cast<RooAbsCategoryLValue*>(arg) ;
      if ((realArgLV && realArgLV->isSafeForIntegration(depList)) || catArgLV) {
	
	// check for overlaps
	Bool_t overlapOK = kTRUE ;
	RooAbsArg *otherArg ;
	TIterator* sIter2 = function.serverIterator() ;
	
	while(otherArg=(RooAbsArg*)sIter2->Next()) {
	  // skip comparison with self
	  if (arg==otherArg) continue ;
	  if (arg->overlaps(*otherArg)) {
	    overlapOK=kFALSE ;
	  }
	}
	
	// All clear for integration over this lvalue
	if (overlapOK) expandArg=kFALSE ;      
	delete sIter2 ;
      }
    }
    
    // Add server (directly or indirectly) to integration list
    if (expandArg && arg->isDerived()) {
      // Add final dependents of this direct server to integration list
      RooArgSet *argDeps = arg->getDependents(&depList) ;
      intDepList.add(*argDeps) ;
      argDeps->Print() ;
      delete argDeps ;
    } else {
      // Add server to integration list
      intDepList.add(*arg) ;
    }
  }
  
  // Register all args in intDepList as shape server
  TIterator* iIter = intDepList.MakeIterator() ;
  while (arg=(RooAbsArg*)iIter->Next()) {
    addServer(*arg,kFALSE,kTRUE) ;
  }
  delete iIter ;

  // Determine which parts needs to be integrated numerically
  RooArgSet numIntDepList("numIntDepList") ;
  _mode = _function->getAnalyticalIntegral(intDepList,numIntDepList) ;    
  
  // Split numeric integration list in summation and integration lists
  TIterator* numIter=numIntDepList.MakeIterator() ;
  while (arg=(RooAbsArg*)numIter->Next()) {

    if (arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
      _intList.add(*arg) ;
    } else if (arg->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
      _sumList.add(*arg) ;
    }
  }
  delete numIter ;

  initNumIntegrator() ;
}


void RooRealIntegral::initNumIntegrator() 
{
  if (_numIntEngine) delete _numIntEngine ;

  // Initialize numerical integration part, if necessary 
  switch(_intList.GetSize()) {
  case 0: 
    // No numerical integration required
    _numIntEngine = 0 ; 
    break ;    
  case 1: 
    // 1-dimensional integration required
    _numIntEngine = new RooIntegrator1D(*_function,_mode,*((RooRealVar*)_intList.First())) ;
    break ;
  default: 
    // multi-dimensional integration required (not supported currently)
    cout << "RooRealIntegral::" << GetName() << ": Numerical integrals in >1 dimension not supported" << endl ;
    assert(0) ;
  }  
}

RooRealIntegral::RooRealIntegral(const RooRealIntegral& other, const char* name) : 
  RooAbsReal(other,name), _function(other._function), _mode(other._mode),
  _intList("intList"), _sumList("sumList") 
{
  copyList(_intList,other._intList) ;
  copyList(_sumList,other._sumList) ;
  initNumIntegrator() ;
}


RooRealIntegral::~RooRealIntegral()
{
  if (_numIntEngine) delete _numIntEngine ;
}


Double_t RooRealIntegral::evaluate() const 
{
  // Save current integral dependent values 
  RooArgSet *saveInt = _intList.snapshot() ;
  RooArgSet *saveSum = _sumList.snapshot() ;

  // Evaluate integral
  Double_t retVal = sum() ;

  // Restore integral dependent values
  _intList=*saveInt ;
  _sumList=*saveSum ;
  delete saveInt ;
  delete saveSum ;

  return retVal ;
}


Double_t RooRealIntegral::sum() const
{
  if (_sumList.GetSize()!=0) {

    // Add integrals for all permutations of categories summed over
    Double_t total(0) ;
    RooMultiCatIter sumIter(_sumList) ;
    while(sumIter.Next()) {
      total += integrate() ;
    }
    return total ;

  } else {

    // Simply return integral 
    return integrate() ;
  }
}



// Default implementation does numerical integration
Double_t RooRealIntegral::integrate() const
{
  // Trivial case, fully analytical integration
  if (!_numIntEngine) return _function->analyticalIntegral(_mode) ;

  // Partial or complete numerical integration
  return _numIntEngine->integral() ;
}




Bool_t RooRealIntegral::isValid(Double_t value) const 
{
  return kTRUE ;
}


Bool_t RooRealIntegral::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  RooAbsArg* arg ;
  TIterator* iter ;
  Bool_t error(kFALSE) ;

  // Update contents of integration list
  iter = _intList.MakeIterator() ;
  while (arg=(RooAbsArg*)iter->Next()) {
    RooAbsArg* newServer = newServerList.find(arg->GetName()) ;
    if (newServer) _intList.replace(*arg,*newServer) ;
    error |= (!newServer && mustReplaceAll) ;    
  }
  delete iter ;
  
  // Update contents of summing list
  iter = _sumList.MakeIterator() ;
  while (arg=(RooAbsArg*)iter->Next()) {
    RooAbsArg* newServer = newServerList.find(arg->GetName()) ;
    if (newServer) _sumList.replace(*arg,*newServer) ;
    error |= (!newServer && mustReplaceAll) ;    
  }
  delete iter ;

  // Restart numerical integrator engine, which uses _intlist
  initNumIntegrator() ;
  
  return kFALSE ;
}


void RooRealIntegral::printToStream(ostream& os, PrintOption opt, TString indent) const
{

  if (opt==Verbose) {
    RooAbsArg::printToStream(os,Verbose,indent) ;
    return ;
  }

  //Print object contents
  os << indent << "RooRealIntegral: " << GetName() << " =" ;

  RooAbsArg* arg ;
  Bool_t first(kTRUE) ;

  if (_sumList.First()) {
    TIterator* sIter = _sumList.MakeIterator() ;
    os << " Sum(" ;
    while (arg=(RooAbsArg*)sIter->Next()) {
      os << (first?"":",") << arg->GetName() ;
      first=kFALSE ;
    }
    delete sIter ;
    os << ")" ;
  }

  first=kTRUE ;
  if (_intList.First()) {
    TIterator* iIter = _intList.MakeIterator() ;
    os << " Int(" ;
    while (arg=(RooAbsArg*)iIter->Next()) {
      os << (first?"":",") << arg->GetName() ;
      first=kFALSE ;
    }
    delete iIter ;
    os << ")" ;
  }


  os << " " << _function->GetName() << " = " << getVal();
  if(!_unit.IsNull()) os << ' ' << _unit;
  os << " : \"" << fTitle << "\"" ;

  printAttribList(os) ;
  os << endl ;
} 


