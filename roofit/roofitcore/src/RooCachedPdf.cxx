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
\file RooCachedPdf.cxx
\class RooCachedPdf
\ingroup Roofitcore

Implementation of RooAbsCachedPdf that can cache
any external RooAbsPdf input function provided in the constructor.
**/

#include "Riostream.h"

#include "RooAbsPdf.h"
#include "RooCachedPdf.h"
#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

using std::endl;



////////////////////////////////////////////////////////////////////////////////
/// Constructor taking name, title and function to be cached. To control
/// granularity of the binning of the cache histogram set the desired properties
/// in the binning named "cache" in the observables of the function. The dimensions
/// of the cache are automatically matched to the number of observables used
/// in each use context. Multiple cache in different observable may exists
/// simultaneously if the cached p.d.f is used with multiple observable
/// configurations simultaneously

RooCachedPdf::RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf) :
   RooAbsCachedPdf(name,title),
   pdf("pdf","pdf",this,_pdf),
   _cacheObs("cacheObs","cacheObs",this,false,false)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Constructor taking name, title and function to be cached and
/// fixed choice of variable to cache. To control granularity of the
/// binning of the cache histogram set the desired properties in the
/// binning named "cache" in the observables of the function.
/// If the fixed set of cache observables does not match the observables
/// defined in the use context of the p.d.f the cache is still filled
/// completely. Ee.g. when it is specified to cache x and p and only x
/// is a observable in the given use context the cache histogram will
/// store sampled values for all values of observable x and parameter p.
/// In such a mode of operation the cache will also not be recalculated
/// if the observable p changes

RooCachedPdf::RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf, const RooArgSet& cacheObs) :
   RooAbsCachedPdf(name,title),
   pdf("pdf","pdf",this,_pdf),
   _cacheObs("cacheObs","cacheObs",this,false,false)
 {
   _cacheObs.add(cacheObs) ;
 }



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooCachedPdf::RooCachedPdf(const RooCachedPdf& other, const char* name) :
   RooAbsCachedPdf(other,name),
   pdf("pdf",this,other.pdf),
   _cacheObs("cacheObs",this,other._cacheObs)
 {
 }

////////////////////////////////////////////////////////////////////////////////
/// Update contents of cache histogram by resampling the input p.d.f. Note that
/// the cache is filled with normalized p.d.f values so that the RooHistPdf
/// that represents the cache contents can be explicitly declared as self normalized
/// eliminating the need for superfluous numeric calculations of unit normalization.s

void RooCachedPdf::fillCacheObject(RooAbsCachedPdf::PdfCacheElem& cache) const
{

  if (cache.hist()->get()->size()>1) {
    coutP(Eval) << "RooCachedPdf::fillCacheObject(" << GetName() << ") filling multi-dimensional cache" ;
  }

  // Update contents of histogram
  (const_cast<RooAbsPdf &>(static_cast<RooAbsPdf const&>(pdf.arg()))).fillDataHist(cache.hist(),&cache.nset(),1.0,false,true) ;

  if (cache.hist()->get()->size()>1) {
    ccoutP(Eval) << std::endl ;
  }

  cache.pdf()->setUnitNorm(true) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Defer preferred scan order to cached pdf preference

void RooCachedPdf::preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const
{
  pdf.arg().preferredObservableScanOrder(obs,orderedObs) ;
}



////////////////////////////////////////////////////////////////////////////////
/// If this pdf is operated with a fixed set of observables, return
/// the subset of the fixed observables that are actual dependents
/// of the external input p.d.f. If this p.d.f is operated without
/// a fixed set of cache observables, return the actual observables
/// of the external input p.d.f given the choice of observables defined
/// in nset

RooFit::OwningPtr<RooArgSet> RooCachedPdf::actualObservables(const RooArgSet& nset) const
{
  if (!_cacheObs.empty()) {
    return pdf->getObservables(_cacheObs);
  }

  return pdf->getObservables(nset);
}



////////////////////////////////////////////////////////////////////////////////
/// If this p.d.f is operated with a fixed set of observables, return
/// all variables of the external input p.d.f that are not one of
/// the cache observables. If this p.d.f is operated in automatic mode,
/// return the parameters of the external input p.d.f

RooFit::OwningPtr<RooArgSet> RooCachedPdf::actualParameters(const RooArgSet& nset) const
{
   return pdf.arg().getParameters(_cacheObs.empty() ? nset : _cacheObs) ;
}


