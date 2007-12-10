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
#include "RooAbsCacheElement.h"
#include "RooAbsArg.h"
#include "RooArgList.h"

ClassImp(RooAbsCacheElement) 
   ;


Bool_t RooAbsCacheElement::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{ 
  return kFALSE ; 
} 


void RooAbsCacheElement::printCompactTreeHook(std::ostream&, const char *, Int_t , Int_t )
{
}



void RooAbsCacheElement::operModeHook(RooAbsArg::OperMode) 
{
  //   RooArgList list = containedArgs(OperModeChange) ;
  //   TIterator* iter = list.createIterator() ;
  //   RooAbsArg* arg ;
  //   while((arg=(RooAbsArg*)iter->Next())) {
  //     arg->setOperMode(newMode) ;
  //   }
  //   delete iter ;
} 


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


void RooAbsCacheElement::findConstantNodes(const RooArgSet& obs, RooArgSet& cacheList, RooLinkedList& processedNodes) 
{
  RooArgList list = containedArgs(FindConstantNodes) ;
  TIterator* iter = list.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {    
    arg->findConstantNodes(obs,cacheList, processedNodes) ;
  }
  delete iter ;
}
