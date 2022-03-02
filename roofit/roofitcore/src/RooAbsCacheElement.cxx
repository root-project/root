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

/**
\file RooAbsCacheElement.cxx
\class RooAbsCacheElement
\ingroup Roofitcore

RooAbsCacheElement is the abstract base class for objects to be stored
in RooAbsCache cache manager objects. Each storage element has an
interface to pass on calls for server redirection, operation mode
change calls and constant term optimization management calls
**/


#include "RooFit.h"
#include "RooAbsCacheElement.h"
#include "RooAbsArg.h"
#include "RooArgList.h"

using namespace std;

ClassImp(RooAbsCacheElement);
   ;


////////////////////////////////////////////////////////////////////////////////
/// Interface for server redirect calls

Bool_t RooAbsCacheElement::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/,
                      Bool_t /*nameChange*/, Bool_t /*isRecursive*/)
{
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Hook function to print cache guts in tree printing mode of RooAbsArgs

void RooAbsCacheElement::printCompactTreeHook(std::ostream&, const char *, Int_t , Int_t )
{
}



////////////////////////////////////////////////////////////////////////////////
/// Interface for cache optimization calls. The default implementation is to forward all these
/// calls to all contained RooAbsArg objects as publicized through containedArg()

void RooAbsCacheElement::optimizeCacheMode(const RooArgSet& obs, RooArgSet& optNodes, RooLinkedList& processedNodes)
{
  RooArgList list = containedArgs(OptimizeCaching) ;
  TIterator* iter = list.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    arg->optimizeCacheMode(obs, optNodes, processedNodes) ;
  }
  delete iter ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface for constant term optimization calls. The default implementation is to forward all these
/// calls to all contained RooAbsArg objects as publicized through containedArg()

void RooAbsCacheElement::findConstantNodes(const RooArgSet& obs, RooArgSet& cacheList, RooLinkedList& processedNodes)
{
  RooArgList list = containedArgs(FindConstantNodes) ;
  for (const auto arg : list) {
    arg->findConstantNodes(obs, cacheList, processedNodes);
  }
}
