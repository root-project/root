/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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
  _iterator= _vars.MakeIterator();
  _cacheIter = _cachedVars.MakeIterator() ;
}


RooAbsData::~RooAbsData() 
{
  delete _iterator ;
  delete _cacheIter ;
}


