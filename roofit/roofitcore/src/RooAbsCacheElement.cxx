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
// RooAbsCacheElement is the abstract base class for objects to be stored
// in RooAbsCache cache manager objects. Each storage element has an 
// interface to pass on calls for server redirection, operation mode
// change calls and constant term optimization management calls
// END_HTML
//
//
//


#include "RooFit.h"
#include "RooAbsCacheElement.h"
#include "RooAbsArg.h"
#include "RooArgList.h"

ClassImp(RooAbsCacheElement) 
   ;


//_____________________________________________________________________________
Bool_t RooAbsCacheElement::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, 
					       Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{ 
  // Interface for server redirect calls
  return kFALSE ; 
} 



//_____________________________________________________________________________
void RooAbsCacheElement::printCompactTreeHook(std::ostream&, const char *, Int_t , Int_t )
{
  // Hook function to print cache guts in tree printing mode of RooAbsArgs
}



//_____________________________________________________________________________
void RooAbsCacheElement::operModeHook(RooAbsArg::OperMode) 
{
  // Interface for operation mode change calls
} 



//_____________________________________________________________________________
void RooAbsCacheElement::optimizeCacheMode(const RooArgSet& obs, RooArgSet& optNodes, RooLinkedList& processedNodes) 
{
  // Interface for cache optimization calls. The default implementation is to forward all these
  // calls to all contained RooAbsArg objects as publicized through containedArg()
  
  RooArgList list = containedArgs(OptimizeCaching) ;
  TIterator* iter = list.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {    
    arg->optimizeCacheMode(obs, optNodes, processedNodes) ;
  }
  delete iter ;
}



//_____________________________________________________________________________
void RooAbsCacheElement::findConstantNodes(const RooArgSet& obs, RooArgSet& cacheList, RooLinkedList& processedNodes) 
{
  // Interface for constant term optimization calls. The default implementation is to forward all these
  // calls to all contained RooAbsArg objects as publicized through containedArg()

  RooArgList list = containedArgs(FindConstantNodes) ;
  TIterator* iter = list.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {    
    arg->findConstantNodes(obs,cacheList, processedNodes) ;
  }
  delete iter ;
}
