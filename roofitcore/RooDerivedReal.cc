/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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

ClassImp(RooDerivedReal) 
;


RooDerivedReal::RooDerivedReal(const char *name, const char *title, const char *unit= "") : 
  RooAbsReal(name,title,unit)
{
}

RooDerivedReal::RooDerivedReal(const char *name, const char *title, Double_t minVal,
		       Double_t maxVal, const char *unit= "") :
  RooAbsReal(name,title,minVal,maxVal,unit)
{
}


RooDerivedReal::RooDerivedReal(const char* name, const RooDerivedReal& other) : 
  RooAbsReal(name,other)
{
}


RooDerivedReal::RooDerivedReal(const RooDerivedReal& other) :
  RooAbsReal(other)
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


Double_t RooDerivedReal::getVal() const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty() || isShapeDirty()) {
    setValueDirty(kFALSE) ;
    setShapeDirty(kFALSE) ;
    _value = traceEval() ;
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


Int_t RooDerivedReal::getAnalyticalIntegral(RooArgSet& integrands)
{
  // By default we do supply any analytical integrals
  return 0 ;
}


Double_t RooDerivedReal::analyticalIntegral(Int_t code) 
{
  // By default no analytical integrals are implemented
  cout << "RooDerivedReal::analyticalIntegal(" << GetName() 
       << "): code " << code << " not implemented" << endl ;

  return 0 ;
}



