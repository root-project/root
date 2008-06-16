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
// Class RooNumCdf is an implementation of RooNumRunningInt specialized
// to calculate cumulative distribution functions from p.d.f.s. The main
// difference between RooNumCdf and RooNumRunningInt is that this class
// imposes special end-point conditions on the interpolated histogram
// that represents the output so that the value at the lower bound is
// guaranteed to converge to exactly zero and that the value at the
// upper bound is guaranteed to converge to exactly one, at all interpolation
// orders.
// END_HTML
//

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



//_____________________________________________________________________________
RooNumCdf::RooNumCdf(const char *name, const char *title, RooAbsPdf& _pdf, RooRealVar& _x, const char* bname) :
   RooNumRunningInt(name,title,_pdf,_x,bname)
 { 
   // Construct a cumulative distribution function from given input p.d.f over observable x.
   // using a numeric sampling algorithm. Use binning named 'bname' to control sampling
   // granularity
 } 



//_____________________________________________________________________________
RooNumCdf::RooNumCdf(const RooNumCdf& other, const char* name) :  
   RooNumRunningInt(other,name)
 { 
   // Copy constructor
 } 



//_____________________________________________________________________________
RooNumCdf::~RooNumCdf() 
{
  // Destructor
}



//_____________________________________________________________________________
void RooNumCdf::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const 
{
  // Fill cache using running integral cache elements calculate()
  // method with specification of cdf-specific boundary conditions

  RICacheElem& riCache = static_cast<RICacheElem&>(cache) ;  
  riCache.calculate(kTRUE) ;
}


