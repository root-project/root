 /***************************************************************************** 
  * Project: RooFit                                                           * 
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
// RooAbsSelfCachedReal is an abstract base class for functions whose
// output is cached in terms of a histogram in all observables between
// getVal() and evaluate(). For certain p.d.f.s that are very
// expensive to calculate it may be beneficial to implement them as a
// RooAbsSelfCachedReal rather than a RooAbsReal. Class
// RooAbsSelfCachedReal is designed to have its interface identical to
// that of RooAbsReal, so any p.d.f can make use of its caching
// functionality by merely switching its base class.  Existing
// RooAbsReal objects can also be cached a posteriori with the
// RooCachedReal wrapper function that takes any RooAbsReal object as
// input.
// END_HTML
//
//

#include "Riostream.h" 

#include "RooFit.h"
#include "RooAbsSelfCachedReal.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

using namespace std ;

ClassImp(RooAbsSelfCachedReal) 



//_____________________________________________________________________________
RooAbsSelfCachedReal::RooAbsSelfCachedReal(const char *name, const char *title, Int_t ipOrder) :
  RooAbsCachedReal(name,title,ipOrder)
 { 
   // Constructor
 } 



//_____________________________________________________________________________
RooAbsSelfCachedReal::RooAbsSelfCachedReal(const RooAbsSelfCachedReal& other, const char* name) :  
   RooAbsCachedReal(other,name)
 { 
   // Copy constructor
 } 



//_____________________________________________________________________________
RooAbsSelfCachedReal::~RooAbsSelfCachedReal() 
{
  // Destructor
}



//_____________________________________________________________________________
void RooAbsSelfCachedReal::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const 
{
  // Fill cache with sampling of function as defined by the evaluate() implementation

  RooDataHist& cacheHist = *cache.hist() ;

  // Make deep clone of self in non-caching mde and attach to dataset observables
  RooArgSet* cloneSet = (RooArgSet*) RooArgSet(*this).snapshot(kTRUE) ;
  RooAbsSelfCachedReal* clone2 = (RooAbsSelfCachedReal*) cloneSet->find(GetName()) ;
  clone2->disableCache(kTRUE) ;
  clone2->attachDataSet(cacheHist) ;

  // Iterator over all bins of RooDataHist and fill weights
  for (Int_t i=0 ; i<cacheHist.numEntries() ; i++) {
    const RooArgSet* obs = cacheHist.get(i) ;
    Double_t wgt = clone2->getVal(obs) ;
    cacheHist.set(wgt) ;
  }

  delete cloneSet ;
}



//_____________________________________________________________________________
RooArgSet* RooAbsSelfCachedReal::actualObservables(const RooArgSet& nset) const 
{
  // Defines observables to be cached, given a set of user defined observables
  // Returns the subset of nset that are observables this p.d.f

  // Make list of servers
  RooArgSet servers ;

  TIterator* siter = serverIterator() ;
  siter->Reset() ;
  RooAbsArg* server ;
  while((server=(RooAbsArg*)siter->Next())) {
    servers.add(*server) ;
  }
  
  // Return servers that are in common with given normalization set
  return (RooArgSet*) servers.selectCommon(nset) ;
  
}


//_____________________________________________________________________________
RooArgSet* RooAbsSelfCachedReal::actualParameters(const RooArgSet& nset) const 
{  
  // Defines parameters on which cache contents depends. Returns
  // subset of variables of self that is not contained in the
  // supplied nset

  // Make list of servers
  RooArgSet *servers = new RooArgSet ;

  TIterator* siter = serverIterator() ;
  siter->Reset() ;
  RooAbsArg* server ;
  while((server=(RooAbsArg*)siter->Next())) {
    servers->add(*server) ;
  }
  
  // Remove all given observables from server list
  servers->remove(nset,kTRUE,kTRUE) ;

  return servers ;
}







