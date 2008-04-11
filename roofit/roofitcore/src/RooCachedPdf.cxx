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
   pdf("pdf","pdf",this,_pdf)
 { 
 } 




RooCachedPdf::RooCachedPdf(const RooCachedPdf& other, const char* name) :  
   RooAbsCachedPdf(other,name), 
   pdf("pdf",this,other.pdf)
 { 
 } 


RooCachedPdf::~RooCachedPdf() 
{
}



void RooCachedPdf::fillCacheObject(RooAbsCachedPdf::CacheElem& cache) const 
{
  // Update contents of histogram
  cache._pdf->fillDataHist(cache._hist,1.0) ;
}



Int_t RooCachedPdf::getAnalyticalIntegral(RooArgSet& /*allVars*/, RooArgSet& /*analVars*/, const char* /*rangeName*/) const  
 { 
   return 0 ; 
 } 





Double_t RooCachedPdf::analyticalIntegral(Int_t /*code*/, const char* /*rangeName*/) const  
 { 
   return 0 ; 
 } 





Int_t RooCachedPdf::getGenerator(const RooArgSet& /*directVars*/, RooArgSet &/*generateVars*/, Bool_t /*staticInitOK*/) const 
 { 
   return 0 ; 
 } 





void RooCachedPdf::generateEvent(Int_t /*code*/) 
 { 
   return; 
 } 


