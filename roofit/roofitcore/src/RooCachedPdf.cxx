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

 // -- CLASS DESCRIPTION [PDF] -- 
 // Your description goes here... 

#include "Riostream.h" 

#include "RooAbsPdf.h"
#include "RooCachedPdf.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

ClassImp(RooCachedPdf) 
  ;


RooCachedPdf::RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf) :
   RooAbsCachedPdf(name,title), 
   pdf("pdf","pdf",this,_pdf),
   _cacheObs("cacheObs","cacheObs",this,kFALSE,kFALSE)  
 { 
 } 

RooCachedPdf::RooCachedPdf(const char *name, const char *title, RooAbsPdf& _pdf, const RooArgSet& cacheObs) :
   RooAbsCachedPdf(name,title), 
   pdf("pdf","pdf",this,_pdf),
   _cacheObs("cacheObs","cacheObs",this,kFALSE,kFALSE)  
 { 
   _cacheObs.add(cacheObs) ;
 } 




RooCachedPdf::RooCachedPdf(const RooCachedPdf& other, const char* name) :  
   RooAbsCachedPdf(other,name), 
   pdf("pdf",this,other.pdf),
   _cacheObs("cacheObs",this,other._cacheObs)
 { 
 } 


RooCachedPdf::~RooCachedPdf() 
{
}



void RooCachedPdf::fillCacheObject(RooAbsCachedPdf::PdfCacheElem& cache) const 
{

  if (cache.hist()->get()->getSize()>1) {
    coutP(Eval) << "RooCachedPdf::fillCacheObject(" << GetName() << ") filling multi-dimensional cache" ;
  }

  // Update contents of histogram
  ((RooAbsPdf&)pdf.arg()).fillDataHist(cache.hist(),&cache.nset(),1.0,kFALSE,kTRUE) ;

  if (cache.hist()->get()->getSize()>1) {
    ccoutP(Eval) << endl ;
  }

  cache.pdf()->setUnitNorm(kTRUE) ;
}

void RooCachedPdf::preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const
{
  // Defer to cached pdf
  pdf.arg().preferredObservableScanOrder(obs,orderedObs) ;
}


RooArgSet* RooCachedPdf::actualObservables(const RooArgSet& nset) const 
{ 
  if (_cacheObs.getSize()>0) {
    return pdf.arg().getObservables(_cacheObs) ;
  } 

  return pdf.arg().getObservables(nset) ; 
}


RooArgSet* RooCachedPdf::actualParameters(const RooArgSet& nset) const 
{ 
  if (_cacheObs.getSize()>0) {
    return pdf.arg().getParameters(_cacheObs) ;
  } 
  return pdf.arg().getParameters(nset) ; 
}


