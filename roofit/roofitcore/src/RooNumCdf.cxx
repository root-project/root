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
#include "RooNumCdf.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooRealVar.h"

ClassImp(RooNumCdf) 
  ;


RooNumCdf::RooNumCdf(const char *name, const char *title, RooAbsPdf& _pdf, RooRealVar& _x, const char* bname) :
   RooNumRunningInt(name,title,_pdf,_x,bname)
 { 
 } 


RooNumCdf::RooNumCdf(const RooNumCdf& other, const char* name) :  
   RooNumRunningInt(other,name)
 { 
 } 


RooNumCdf::~RooNumCdf() 
{
}


void RooNumCdf::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const 
{
  RICacheElem& riCache = static_cast<RICacheElem&>(cache) ;  
  riCache.calculate(kTRUE) ;
}


