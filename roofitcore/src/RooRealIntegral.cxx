/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealIntegral.cc,v 1.16 2001/06/06 00:06:39 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooRealIntegral performs hybrid numerical/analytical integrals of RooAbsPdf objects
// The class performs none of the actual integration, but only manages the logic
// of what variables can be integrated analytically, accounts for eventual jacobian
// terms and defines what numerical integrations needs to be done to complement the
// analytical integral.
//
// The actual analytical integrations (if any) are done in the PDF themselves, the numerical
// integration is performed in the various implemenations of the RooAbsIntegrator base class.

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
				 const RooAbsPdf& function, RooArgSet& depList,
				 Int_t maxSteps, Double_t eps) : 
  RooAbsReal(name,title), _mode(0),
  _function("function","Function to be integrated",this,(RooAbsPdf&)function,kFALSE,kFALSE), 
  _sumList("sumList","Categories to be summed numerically",this,kFALSE,kFALSE), 
  _intList("intList","Variables to be integrated numerically",this,kFALSE,kFALSE), 
  _anaList("anaList","Variables to be integrated analytically",this,kFALSE,kFALSE), 
  _jacList("jacList","Jacobian product term",this,kFALSE,kFALSE), 
  _numIntEngine(0) 
{
  // Constructor
  RooArgSet intDepList ;

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * A) Make list of servers that can be integrated analytically *
  //      Add all parameters/dependents as value/shape servers     *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  RooArgSet anIntOKDepList ;
  TIterator *sIter = function.serverIterator() ;
  RooAbsArg *arg ;
  while(arg=(RooAbsArg*)sIter->Next()) {

    // Dependent or parameter?
    if (!arg->dependsOn(depList)) {

      // Add parameter as value server
      addServer(*arg,kTRUE,kFALSE) ;
      continue ;

    } else {

      // Add final dependents of arg as shape servers
      RooArgSet *argDeps = arg->getDependents(&depList) ;
      TIterator *adIter = argDeps->MakeIterator() ;
      RooAbsArg *argDep ;
      while(argDep = (RooAbsArg*)adIter->Next()) {
	addServer(*argDep,kFALSE,kTRUE) ;
      }
      delete argDeps ;      
    }

    // If this dependent arg is self-normalized, stop here
    if (function.selfNormalized(*arg)) continue ;
    
    Bool_t depOK(kFALSE) ;
    // Check for integratable AbsRealLValue
    if (arg->isDerived()) {
      RooAbsRealLValue    *realArgLV = dynamic_cast<RooAbsRealLValue*>(arg) ;
      RooAbsCategoryLValue *catArgLV = dynamic_cast<RooAbsCategoryLValue*>(arg) ;
      if ((realArgLV && (realArgLV->isJacobianOK(depList)!=0)) || catArgLV) {
	
	// Derived LValue with valid jacobian
	depOK = kTRUE ;
	
	// Now, check for overlaps
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
	if (!overlapOK) depOK=kFALSE ;      

	delete sIter2 ;
      }
    } else {
      // Fundamental types are always OK
      depOK = kTRUE ;
    }
    
    // Add server to list of dependents that are OK for analytical integration
    if (depOK) anIntOKDepList.add(*arg) ;
  }

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * B) interact with function to make list of objects actually integrated analytically  *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  RooArgSet anIntDepList ;
  _mode = ((RooAbsPdf&)_function.arg()).getAnalyticalIntegral(anIntOKDepList,_anaList) ;    

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * C) Make list of numerical integration variables consisting of:            *  
  // *   - Category dependents of RealLValues in analytical integration          *  
  // *   - Expanded server lists of server that are not analytically integrated  *
  // *    Make Jacobian list with analytically integrated RealLValues            *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  RooArgSet numIntDepList ;

  // Loop over actually analytically integrated dependents
  TIterator* aiIter = _anaList.MakeIterator() ;
  while (arg=(RooAbsArg*)aiIter->Next()) {    

    // Process only derived RealLValues
    if (arg->IsA()->InheritsFrom(RooAbsRealLValue::Class()) && arg->isDerived()) {

      // Add to list of Jacobians to calculate
      _jacList.add(*arg) ;

      // Add category dependent of LValueReal used in integration
      RooAbsArg *argDep ;
      RooArgSet *argDepList = arg->getDependents(&depList) ;
      TIterator *adIter = argDepList->MakeIterator() ;
      while (argDep=(RooAbsArg*)adIter->Next()) {
	if (argDep->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
	  numIntDepList.add(*argDep) ;
	}
      }
      delete adIter ;
      delete argDepList ;
    }
  }
  delete aiIter ;

  // Loop again over function servers to add remaining numeric integrations
  sIter->Reset() ;
  while(arg=(RooAbsArg*)sIter->Next()) {

    // Process only servers that are not treated analytically
    if (!_anaList.find(arg->GetName()) && arg->dependsOn(depList)) {
      
      // Expand server in final dependents and add to numerical integration list      
      RooArgSet *argDeps = arg->getDependents(&depList) ;
      numIntDepList.add(*argDeps) ;
      delete argDeps ; 
    }
  }

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // * D) Split numeric list in integration list and summation list  *
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

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
  // (Re)Initialize numerical integration engine

  if (_numIntEngine) delete _numIntEngine ;

  // Initialize numerical integration part, if necessary 
  switch(_intList.GetSize()) {
  case 0: 
    // No numerical integration required
    _numIntEngine = 0 ; 
    break ;    
  case 1: 
    // 1-dimensional integration required
    _numIntEngine = new RooIntegrator1D((RooAbsPdf&)_function.arg(),_mode,*((RooRealVar*)_intList.First())) ;
    break ;
  default: 
    // multi-dimensional integration required (not supported currently)
    cout << "RooRealIntegral::" << GetName() << ": Numerical integrals in >1 dimension not supported" << endl ;
    assert(0) ;
  }  
}

RooRealIntegral::RooRealIntegral(const RooRealIntegral& other, const char* name) : 
  RooAbsReal(other,name), _mode(other._mode),
 _function("function",this,other._function), 
  _intList("intList",this,other._intList), 
  _sumList("sumList",this,other._sumList),
  _anaList("anaList",this,other._anaList),
  _jacList("jacList",this,other._jacList) 
{
  // Copy constructor
  initNumIntegrator() ;
}


RooRealIntegral::~RooRealIntegral()
  // Destructor
{
  if (_numIntEngine) delete _numIntEngine ;
}


Double_t RooRealIntegral::evaluate(const RooDataSet* dset) const 
{
  // Calculate integral

  // Save current integral dependent values 
  RooArgSet *saveInt = _intList.snapshot() ;
  RooArgSet *saveSum = _sumList.snapshot() ;

  // Evaluate sum/integral
  Double_t retVal = sum() / jacobianProduct() ;

  // Restore integral dependent values
  _intList=*saveInt ;
  _sumList=*saveSum ;
  delete saveInt ;
  delete saveSum ;

  return retVal ;
}


Double_t RooRealIntegral::jacobianProduct() const 
{
  // Return product of jacobian terms originating from analytical integration
  Double_t jacProd(1) ;

  TIterator *jIter = _jacList.MakeIterator() ;
  RooAbsRealLValue* arg ;
  while (arg=(RooAbsRealLValue*)jIter->Next()) {
    jacProd *= arg->jacobian() ;
  }
  delete jIter ;

  return jacProd ;
}


Double_t RooRealIntegral::sum() const
{
  // Perform summation of list of category dependents to be integrated
  if (_sumList.GetSize()!=0) {

    // Add integrals for all permutations of categories summed over
    Double_t total(0) ;
    RooMultiCatIter sumIter(_sumList) ;
    while(sumIter.Next()) {
      total += integrate() / jacobianProduct() ;
    }
    return total ;

  } else {

    // Simply return integral 
    return integrate() ;
  }
}




Double_t RooRealIntegral::integrate() const
{
  // Perform hybrid numerical/analytical integration over all real-valued dependents

  // Trivial case, fully analytical integration
  if (!_numIntEngine) return ((RooAbsPdf&)_function.arg()).analyticalIntegral(_mode) ;

  // Partial or complete numerical integration
  return _numIntEngine->integral() ;
}




Bool_t RooRealIntegral::isValid(Double_t value) const 
{
  // Check if current value is valid
  return kTRUE ;
}


Bool_t RooRealIntegral::redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll)
{
  // Restart numerical integrator engine, which uses _intlist
  initNumIntegrator() ;
  
  return kFALSE ;
}


void RooRealIntegral::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print the state of this object to the specified output stream.

  if (opt==Verbose) {
    RooAbsReal::printToStream(os,Verbose,indent) ;
    os << indent << "--- RooRealIntegral ---" << endl;
    os << indent << "  Integrates ";
    _function.arg().printToStream(os,Standard);
    TString deeper(indent);
    deeper.Append("  ");
    os << indent << "  Summed discrete args are ";
    _sumList.printToStream(os,Standard,deeper);
    os << indent << "  Numerically integrated args are ";
    _intList.printToStream(os,Standard,deeper);
    os << indent << "  Analytically integrated args using mode " << _mode << " are ";
    _anaList.printToStream(os,Standard,deeper);
    os << indent << "  Arguments included in Jacobean are ";
    _jacList.printToStream(os,Standard,deeper);
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
    os << " NumInt(" ;
    while (arg=(RooAbsArg*)iIter->Next()) {
      os << (first?"":",") << arg->GetName() ;
      first=kFALSE ;
    }
    delete iIter ;
    os << ")" ;
  }

  first=kTRUE ;
  if (_anaList.First()) {
    TIterator* aIter = _anaList.MakeIterator() ;
    os << " AnaInt(" ;
    while (arg=(RooAbsArg*)aIter->Next()) {
      os << (first?"":",") << arg->GetName() ;
      first=kFALSE ;
    }
    delete aIter ;
    os << ")" ;
  }


  os << " " << _function.arg().GetName() << " = " << getVal();
  if(!_unit.IsNull()) os << ' ' << _unit;
  os << " : \"" << fTitle << "\"" ;

  printAttribList(os) ;
  os << endl ;
} 


