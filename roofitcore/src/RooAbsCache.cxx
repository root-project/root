/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "RooFit.h"
#include "RooAbsCache.h"
#include "RooAbsArg.h"
#include "RooArgList.h"

ClassImp(RooAbsCache) 
   ;


RooAbsCache::RooAbsCache(RooAbsArg* owner) : _owner(owner) 
{ 
  if (_owner) {
    _owner->registerCache(*this) ; 
  }
} 


RooAbsCache::RooAbsCache(const RooAbsCache&, RooAbsArg* owner ) : _owner(owner) 
{ 
  if (_owner) {
    owner->registerCache(*this) ; 
  }
}


RooAbsCache::~RooAbsCache() 
{ 
  if (_owner) {
    _owner->unRegisterCache(*this) ; 
  }
}


void RooAbsCache::optimizeCacheMode(const RooArgSet& /*obs*/, RooArgSet&, RooLinkedList& ) 
{
}


Bool_t RooAbsCache::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{ 
  return kFALSE ; 
} 


void RooAbsCache::operModeHook() 
{
} 


void RooAbsCache::findConstantNodes(const RooArgSet&, RooArgSet&, RooLinkedList& ) 
{  
}


void RooAbsCache::printCompactTreeHook(std::ostream&, const char *)
{
}

