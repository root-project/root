/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDerivedReal.cc,v 1.4 2001/04/21 02:42:43 verkerke Exp $
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
#include "RooFitCore/RooDerivedReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgProxy.hh"

ClassImp(RooDerivedReal) 
;


RooDerivedReal::RooDerivedReal(const char *name, const char *title, const char *unit) : 
  RooAbsReal(name,title,unit)
{
}

RooDerivedReal::RooDerivedReal(const char *name, const char *title, Double_t minVal,
		       Double_t maxVal, const char *unit) :
  RooAbsReal(name,title,minVal,maxVal,unit)
{
}


RooDerivedReal::RooDerivedReal(const RooDerivedReal& other, const char* name) : 
  RooAbsReal(other,name)
{
}



RooDerivedReal::~RooDerivedReal()
{
}


RooDerivedReal& RooDerivedReal::operator=(const RooDerivedReal& other)
{
  RooAbsReal::operator=(other) ;
  return *this ;
}


RooAbsArg& RooDerivedReal::operator=(const RooAbsArg& aother)
{
  return operator=((const RooDerivedReal&)aother) ;
}


Double_t RooDerivedReal::getVal(const RooDataSet* dset) const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;
    setValueDirty(kFALSE) ;
    setShapeDirty(kFALSE) ;
  } 
  
  return _value ;
}


Double_t RooDerivedReal::traceEval() const
{
  Double_t value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValid(value)) {
    cout << "RooDerivedReal::traceEval(" << GetName() 
	 << "): validation failed: " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}


Int_t RooDerivedReal::getAnalyticalIntegral(RooArgSet& allDeps, RooArgSet& numDeps) const
{
  // By default we do supply any analytical integrals

  // Indicate all variables need to be integrated numerically
  TIterator* iter = allDeps.MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    numDeps.add(*arg) ;
  }

  return 0 ;
}



Bool_t RooDerivedReal::tryIntegral(const RooArgSet& allDeps, RooArgSet& numDeps, const RooArgProxy& a) const
{
  Bool_t match = kFALSE ;
  TString name(a.absArg()->GetName()) ;

  TIterator* iter = allDeps.MakeIterator()  ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()){    
    if (!name.CompareTo(arg->GetName())) {
      match = kTRUE ;
    } else {
      numDeps.add(*arg) ;
    }
  }
  delete iter ;

  return match ;  
}



Double_t RooDerivedReal::analyticalIntegral(Int_t code) const
{
  // By default no analytical integrals are implemented
  return getVal() ;
}



