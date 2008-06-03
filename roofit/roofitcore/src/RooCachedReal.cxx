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
#include "RooCachedReal.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"

ClassImp(RooCachedReal) 
  ;


RooCachedReal::RooCachedReal(const char *name, const char *title, RooAbsReal& _func) :
   RooAbsCachedReal(name,title), 
   func("func","func",this,_func),
   _useCdfBoundaries(kFALSE)
 { 
 } 




RooCachedReal::RooCachedReal(const RooCachedReal& other, const char* name) :  
   RooAbsCachedReal(other,name), 
   func("func",this,other.func),
   _useCdfBoundaries(other._useCdfBoundaries)
 { 
 } 


RooCachedReal::~RooCachedReal() 
{
}



void RooCachedReal::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const 
{
  // Update contents of histogram
  if (cache.hist()->get()->getSize()>1) {
    coutP(Eval) << "RooCachedReal::fillCacheObject(" << GetName() << ") filling multi-dimensional cache" ;
  }

  func.arg().fillDataHist(cache.hist(),0,1.0,kFALSE,kTRUE) ;
  cache.func()->setCdfBoundaries(_useCdfBoundaries) ;

  if (cache.hist()->get()->getSize()>1) {
    ccoutP(Eval) << endl ;
  }

}


