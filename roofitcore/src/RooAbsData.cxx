/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsData.cc,v 1.1 2001/08/17 00:35:56 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   15-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooAbsData.hh"

ClassImp(RooAbsData)
;


RooAbsData::RooAbsData() 
{
}


RooAbsData::RooAbsData(const char *name, const char *title, const RooArgSet& vars) :
  TNamed(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), 
  _doDirtyProp(kTRUE)

{
  // clone the fundamentals of the given data set into internal buffer
  TIterator* iter = vars.MakeIterator() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)iter->Next())) {
    if (!var->isFundamental()) {
      cout << "RooDataSet::initialize(" << GetName() 
	   << "): Data set cannot contain non-fundamental types, ignoring " 
	   << var->GetName() << endl ;
    } else {
      RooAbsArg* varClone = (RooAbsArg*) var->Clone() ;
      _vars.add(*varClone) ;
    }
  }
  delete iter ;

  _iterator= _vars.MakeIterator();
  _cacheIter = _cachedVars.MakeIterator() ;
}


RooAbsData::~RooAbsData() 
{
  delete _iterator ;
  delete _cacheIter ;
}


