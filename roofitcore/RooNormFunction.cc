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
#include "RooFitCore/RooNormFunction.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooNormFunction) 
;


RooNormFunction::RooNormFunction(const char *name, const char *title, RooArgSet& depList, 
				 RooArgSet& parList, const char *unit= "") : 
  RooDerivedReal(name,title,unit)
{
  addServerList(depList,kTRUE,kFALSE) ;
  addServerList(parList,kTRUE,kFALSE) ;

  _norm = new RooRealIntegral(TString(name).Append("Norm"),TString(title).Append(" Integral"),*this,depList) ;
}

RooNormFunction::RooNormFunction(const char *name, const char *title, Double_t plotMin,
		       Double_t plotMax,  RooArgSet& depList, RooArgSet& parList, const char *unit= "") :
  RooDerivedReal(name,title,plotMin,plotMax,unit)
{
  addServerList(depList,kTRUE,kFALSE) ;
  addServerList(parList,kTRUE,kFALSE) ;

  _norm = new RooRealIntegral(TString(name).Append("Norm"),TString(title).Append(" Integral"),*this,depList) ;
}


RooNormFunction::RooNormFunction(const char* name, const RooNormFunction& other) : 
  RooDerivedReal(name,other)
{
  _norm = new RooRealIntegral(TString(name).Append("Norm"),*other._norm) ;
}


RooNormFunction::RooNormFunction(const RooNormFunction& other) :
  RooDerivedReal(other)
{
  _norm = new RooRealIntegral(*other._norm) ;
}



RooNormFunction::~RooNormFunction()
{
  delete _norm ;
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






