/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNormFunction.cc,v 1.2 2001/04/20 01:51:39 verkerke Exp $
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
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooNormFunction) 
;


RooNormFunction::RooNormFunction(const char *name, const char *title, const char *unit= "") : 
  RooDerivedReal(name,title,unit), _norm(0), _lastDataSet(0)
{
}


RooNormFunction::RooNormFunction(const char *name, const char *title, 
				 Double_t plotMin, Double_t plotMax, const char *unit= "") :
  RooDerivedReal(name,title,plotMin,plotMax,unit), _norm(0), _lastDataSet(0)
{
}



RooNormFunction::~RooNormFunction()
{
  if (_norm) delete _norm ;
}



RooNormFunction::RooNormFunction(const char* name, const RooNormFunction& other) : 
  RooDerivedReal(name,other), _norm(0), _lastDataSet(0)
{
}




RooNormFunction::RooNormFunction(const RooNormFunction& other) :
  RooDerivedReal(other), _norm(0), _lastDataSet(0)
{
}


Double_t RooNormFunction::getNorm(const RooDataSet* dset) const
{
  // Trivial case: normalization cache still valid
  if (dset==_lastDataSet) return _norm->getVal() ;

  // Destroy old normalization
  if (_norm) delete _norm ;

  // Create new normalization
  RooArgSet* depList = getDependents(dset) ;
  _lastDataSet = (RooDataSet*) dset ;
  _norm = new RooRealIntegral(TString(GetName()).Append("Norm"),
			      TString(GetTitle()).Append(" Integral"),*this,*depList) ;
  delete depList ;
  
  return _norm->getVal() ;
}



Int_t RooNormFunction::getNPar(const RooDataSet* set) 
{
  RooArgSet* parList = getParameters(set) ;
  Int_t npar = parList->GetSize() ;
  delete parList ;
  
  return npar ;
}



RooArgSet* RooNormFunction::getParameters(const RooDataSet* set) const 
{
  RooArgSet* parList = new RooArgSet("parameters") ;
  const RooArgSet* dataList = set->get() ;

  TIterator* sIter = serverIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (!dataList->FindObject(arg->GetName())) {
      parList->add(*arg) ;
    }
  }

  return parList ;
}



RooArgSet* RooNormFunction::getDependents(const RooDataSet* set) const 
{
  RooArgSet* depList = new RooArgSet("parameters") ;
  const RooArgSet* dataList = set->get() ;

  TIterator* sIter = serverIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)sIter->Next()) {
    if (dataList->FindObject(arg->GetName())) {
      depList->add(*arg) ;
    }
  }

  return depList ;
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






