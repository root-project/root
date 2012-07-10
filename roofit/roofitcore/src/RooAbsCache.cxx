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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooAbsCache is the abstract base class for data members of RooAbsArgs
// that cache other (composite) RooAbsArg expressions. The RooAbsCache
// interface defines the interaction between the owning RooAbsArg object
// and the cache data member to communicate server redirects, operation
// mode changes and constant term optimization management calls.
// END_HTML
//
//


#include "RooFit.h"
#include "RooAbsCache.h"
#include "RooAbsArg.h"
#include "RooArgList.h"

using namespace std;

ClassImp(RooAbsCache) 
   ;


//_____________________________________________________________________________
RooAbsCache::RooAbsCache(RooAbsArg* owner) : _owner(owner) 
{ 
  // Constructor. Takes owner as argument and register cache with owner
  if (_owner) {
    _owner->registerCache(*this) ; 
  }
} 



//_____________________________________________________________________________
RooAbsCache::RooAbsCache(const RooAbsCache&, RooAbsArg* owner ) : _owner(owner) 
{ 
  // Copy constructor. Takes owner as argument and registers cache with owne
  if (_owner) {
    owner->registerCache(*this) ; 
  }
}



//_____________________________________________________________________________
RooAbsCache::~RooAbsCache() 
{ 
  // Destructor. Unregisters cache with owner
  if (_owner) {
    _owner->unRegisterCache(*this) ; 
  }
}



//_____________________________________________________________________________
void RooAbsCache::optimizeCacheMode(const RooArgSet& /*obs*/, RooArgSet&, RooLinkedList& ) 
{
  // Interface for processing of cache mode optimization calls
}



//_____________________________________________________________________________
Bool_t RooAbsCache::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{ 
  // Interface for server redirect calls
  return kFALSE ; 
} 



//_____________________________________________________________________________
void RooAbsCache::operModeHook() 
{
  // Interface for operation mode changes
} 



//_____________________________________________________________________________
void RooAbsCache::findConstantNodes(const RooArgSet&, RooArgSet&, RooLinkedList& ) 
{  
  // Interface for constant term node finding calls
}



//_____________________________________________________________________________
void RooAbsCache::printCompactTreeHook(std::ostream&, const char *)
{
  // Interface for printing of cache guts in tree mode printing
}

