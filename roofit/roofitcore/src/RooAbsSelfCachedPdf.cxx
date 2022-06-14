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

/**
\file RooAbsSelfCachedPdf.cxx
\class RooAbsSelfCachedPdf
\ingroup Roofitcore

RooAbsSelfCachedPdf is an abstract base class for probability
density functions whose output is cached in terms of a histogram in
all observables between getVal() and evaluate(). For certain
p.d.f.s that are very expensive to calculate it may be beneficial
to implement them as a RooAbsSelfCachedPdf rather than a
RooAbsPdf. Class RooAbsSelfCachedPdf is designed to have its
interface identical to that of RooAbsPdf, so any p.d.f can make use
of its caching functionality by merely switching its base class.
Existing RooAbsPdf objects can also be cached a posteriori with the
RooCachedPdf wrapper p.d.f. that takes any RooAbsPdf object as
input.
**/

#include "Riostream.h"

#include "RooAbsSelfCachedPdf.h"
#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

using namespace std ;

ClassImp(RooAbsSelfCachedPdf);



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsSelfCachedPdf::RooAbsSelfCachedPdf(const char *name, const char *title, Int_t ipOrder) :
  RooAbsCachedPdf(name,title,ipOrder)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsSelfCachedPdf::RooAbsSelfCachedPdf(const RooAbsSelfCachedPdf& other, const char* name) :
   RooAbsCachedPdf(other,name)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsSelfCachedPdf::~RooAbsSelfCachedPdf()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Fill cache with sampling of p.d.f as defined by the evaluate() implementation

void RooAbsSelfCachedPdf::fillCacheObject(RooAbsCachedPdf::PdfCacheElem& cache) const
{
  RooDataHist& cacheHist = *cache.hist() ;

  // Make deep clone of self in non-caching mde and attach to dataset observables
  RooArgSet* cloneSet = (RooArgSet*) RooArgSet(*this).snapshot(true) ;
  RooAbsSelfCachedPdf* clone2 = (RooAbsSelfCachedPdf*) cloneSet->find(GetName()) ;
  clone2->disableCache(true) ;
  clone2->attachDataSet(cacheHist) ;

  // Iterator over all bins of RooDataHist and fill weights
  for (Int_t i=0 ; i<cacheHist.numEntries() ; i++) {
    const RooArgSet* obs = cacheHist.get(i) ;
    double wgt = clone2->getVal(obs) ;
    cacheHist.set(i, wgt, 0.);
  }

  cache.pdf()->setUnitNorm(true) ;

  delete cloneSet ;
}



////////////////////////////////////////////////////////////////////////////////
/// Defines observables to be cached, given a set of user defined observables
/// Returns the subset of nset that are observables this p.d.f

RooArgSet* RooAbsSelfCachedPdf::actualObservables(const RooArgSet& /*nset*/) const
{
  // Make list of servers
  RooArgSet *serverSet = new RooArgSet;

  for (auto server : _serverList) {
    serverSet->add(*server) ;
  }

  // Return servers that are in common with given normalization set
  return serverSet;
}



////////////////////////////////////////////////////////////////////////////////
/// Defines parameters on which cache contents depends. Returns
/// subset of variables of self that is not contained in the
/// supplied nset

RooArgSet* RooAbsSelfCachedPdf::actualParameters(const RooArgSet& nset) const
{
  RooArgSet *serverSet = new RooArgSet;

  for (auto server : _serverList) {
    serverSet->add(*server) ;
  }

  // Remove all given observables from server list
  serverSet->remove(nset,true,true);

  return serverSet;
}







