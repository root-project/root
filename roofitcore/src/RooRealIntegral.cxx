/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealIntegral.cc,v 1.2 2001/04/18 20:38:03 verkerke Exp $
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
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooMultiCatIter.hh"
#include "RooFitCore/RooIntegrator1D.hh"

ClassImp(RooRealIntegral) 
;


RooRealIntegral::RooRealIntegral(const char *name, const char *title, 
				 RooDerivedReal& function, RooArgSet& depList,
				 Int_t maxSteps, Double_t eps) : 
  RooDerivedReal(name,title), _function(&function), _mode(0),
  _intList("intList"), _sumList("sumList"), _depList("depList"), _init(kFALSE)
{

  // Filter out junk variables from dependent list
  TIterator* depIter=depList.MakeIterator() ;
  RooAbsArg *arg ;
  while (arg=(RooAbsArg*)depIter->Next()) {

    if (!function.dependsOn(*arg)) {
      cout << "RooRealIntegral::RooIntegral(" << name << "): integrand " << arg->GetName()
	   << " doesn't depend on function " << function.GetName() << ", ignored" << endl ;
      continue ;
    }
    
    if (!arg->IsA()->InheritsFrom(RooRealVar::Class()) &&
	!arg->IsA()->InheritsFrom(RooCategory::Class())) {
      cout << "RooRealIntegral::RooIntegral(" << name << "): integrand " << arg->GetName()
	   << " is neither a RooCategory nor a RooRealVar, ignored" << endl ;
      continue ;
    }
    
    // Add integrand as shape server 
    addServer(*arg,kFALSE,kTRUE) ;
    _depList.add(*arg) ;
  }
  delete depIter ;

  // Register all non-integrands of functions as value servers
  TIterator* sIter = function.serverIterator() ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (!_depList.FindObject(arg))
      addServer(*arg,kTRUE,kFALSE) ;
  }
  delete sIter ;

  // Remaining initialization deferred (implemented in deferredInit())
}


void RooRealIntegral::deferredInit() 
{
  if (_init) return ;

  // Determine which parts needs to be integrated numerically
  RooArgSet numDepList("numDepList") ;
  _mode = _function->getAnalyticalIntegral(_depList,numDepList) ;    
  
  // Split numeric integration list in summation and integration lists
  RooAbsArg* arg ;
  TIterator* numIter=numDepList.MakeIterator() ;
  while (arg=(RooAbsArg*)numIter->Next()) {
  
    if (arg->IsA()->InheritsFrom(RooRealVar::Class())) {
      _intList.add(*arg) ;
    } else if (arg->IsA()->InheritsFrom(RooCategory::Class())) {
      _sumList.add(*arg) ;
    }
  }
  delete numIter ;

  initNumIntegrator() ;
  _init = kTRUE ;
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

RooRealIntegral::RooRealIntegral(const char* name, const RooRealIntegral& other) : 
  RooDerivedReal(name,other), _function(other._function), 
  _intList("intList"), _sumList("sumList"), _depList("depList"), _init(other._init) 
{
  copyList(_intList,other._intList) ;
  copyList(_sumList,other._sumList) ;
  copyList(_depList,other._depList) ;  
  if (other._init) initNumIntegrator() ;
}


RooRealIntegral::RooRealIntegral(const RooRealIntegral& other) :
  RooDerivedReal(other), _function(other._function),
  _intList("intList"), _sumList("sumList"), _init(other._init)
{
  copyList(_intList,other._intList) ;
  copyList(_sumList,other._sumList) ;
  copyList(_depList,other._depList) ;
  if (other._init) initNumIntegrator() ;
}



RooRealIntegral::~RooRealIntegral()
{
  if (_numIntEngine) delete _numIntEngine ;
}


RooRealIntegral& RooRealIntegral::operator=(const RooRealIntegral& other)
{
  RooDerivedReal::operator=(other) ;
  copyList(_intList,other._intList) ;
  copyList(_sumList,other._sumList) ;
  copyList(_depList,other._depList) ;
  _function = other._function ;
  setValueDirty(kTRUE) ;
  return *this ;
}


RooAbsArg& RooRealIntegral::operator=(const RooAbsArg& aother)
{
  return operator=((const RooRealIntegral&)aother) ;
}



Double_t RooRealIntegral::evaluate() const 
{
  // Perform deferred initialization
  if (!_init) deferredInit() ;

  // Save current integrand values 
  RooArgSet saveInt("saveInt",_intList), saveSum("saveSum",_sumList) ;

  // Evaluate integral
  Double_t retVal = sum() ;

  // Restore integrand values
  _intList=saveInt ;
  _sumList=saveSum ;

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


Bool_t RooRealIntegral::redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  return kFALSE ;
}


void RooRealIntegral::printToStream(ostream& os, PrintOption opt=Standard) const
{

  if (opt==Verbose) {
    RooAbsArg::printToStream(os,Verbose) ;
    return ;
  }

  //Print object contents
  os << "RooRealIntegral: " << GetName() << " =" ;

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


