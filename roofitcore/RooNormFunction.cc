/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNormFunction.cc,v 1.1 2001/04/18 20:38:03 verkerke Exp $
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
#include "RooFitCore/RooNormFunction.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooNormFunction) 
;


RooNormFunction::RooNormFunction(const char *name, const char *title, RooArgSet& depList, 
				 RooArgSet& parList, const char *unit= "") : 
  RooDerivedReal(name,title,unit)
{
  init(depList,parList) ;
}


RooNormFunction::RooNormFunction(const char *name, const char *title, Double_t plotMin,
		       Double_t plotMax,  RooArgSet& depList, RooArgSet& parList, const char *unit= "") :
  RooDerivedReal(name,title,plotMin,plotMax,unit)
{
  init(depList,parList) ;
}



RooNormFunction::~RooNormFunction()
{
  // Clear depedent/parameter attributes in our servers
  _depList.setAttribAll(depAttribName(),kFALSE) ;
  _parList.setAttribAll(parAttribName(),kFALSE) ;

  delete _norm ;
}



void RooNormFunction::init(RooArgSet& depList, RooArgSet& parList) 
{
  addServerList(depList,kTRUE,kFALSE) ;
  addServerList(parList,kTRUE,kFALSE) ;

  copyList(_depList,depList) ;
  copyList(_parList,parList) ;

  depList.setAttribAll(depAttribName()) ;
  parList.setAttribAll(parAttribName()) ;

  _norm = new RooRealIntegral(TString(GetName()).Append("Norm"),TString(GetTitle()).Append(" Integral"),*this,depList) ;
}



const char* RooNormFunction::parAttribName() const 
{
  static char buf[1024] ;
  sprintf(buf,"Par(%s,%x)",GetName(),this) ;  
  return buf ;
}

const char* RooNormFunction::depAttribName() const 
{
  static char buf[1024] ;
  sprintf(buf,"Dep(%s,%x)",GetName(),this) ;  
  return buf ;
}



RooNormFunction::RooNormFunction(const char* name, const RooNormFunction& other) : 
  RooDerivedReal(name,other)
{
  copyList(_depList,other._depList) ;
  copyList(_parList,other._parList) ;

  _depList.setAttribAll(depAttribName()) ;
  _parList.setAttribAll(parAttribName()) ;

  _norm = new RooRealIntegral(TString(name).Append("Norm"),*other._norm) ;
}




RooNormFunction::RooNormFunction(const RooNormFunction& other) :
  RooDerivedReal(other)
{
  copyList(_depList,other._depList) ;
  copyList(_parList,other._parList) ;

  _norm = new RooRealIntegral(*other._norm) ;
}



RooNormFunction& RooNormFunction::operator=(const RooNormFunction& other)
{
  RooAbsReal::operator=(other) ;
  return *this ;
}



RooAbsArg& RooNormFunction::operator=(const RooAbsArg& aother)
{
  return operator=((const RooNormFunction&)aother) ;
}






