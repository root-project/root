/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealIntegral.cc,v 1.8 2001/05/10 21:26:09 verkerke Exp $
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

  // Filter out junk variables from dependent list
  TIterator* depIter=depList.MakeIterator() ;
  RooAbsArg *arg ;
  while (arg=(RooAbsArg*)depIter->Next()) {

    if (!function.dependsOn(*arg)) {
      cout << "RooRealIntegral::RooIntegral(" << name << "): integrand " << arg->GetName()
	   << " doesn't depend on function " << function.GetName() << ", ignored" << endl ;
      continue ;
    }
    
    if (!arg->IsA()->InheritsFrom(RooAbsRealLValue::Class()) &&
	!arg->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
      cout << "RooRealIntegral::RooIntegral(" << name << "): integrand " << arg->GetName()
	   << " is neither a RooAbsCategoryLValue nor a RooAbsRealLValue, ignored" << endl ;
      continue ;
    }
    
    // Add integrand as shape server 
    addServer(*arg,kFALSE,kTRUE) ;
  }
  delete depIter ;

  // Register all non-integrands of functions as value servers
  TIterator* sIter = function.serverIterator() ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (!depList.FindObject(arg))
      addServer(*arg,kTRUE,kFALSE) ;
  }
  delete sIter ;

  // Determine which parts needs to be integrated numerically
  RooArgSet numDepList("numDepList") ;
  _mode = _function->getAnalyticalIntegral(depList,numDepList) ;    
  
  // Split numeric integration list in summation and integration lists
  TIterator* numIter=numDepList.MakeIterator() ;
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
  // Save current integrand values 
  RooArgSet saveInt(_intList), saveSum(_sumList) ;

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


Bool_t RooRealIntegral::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll)
{
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


