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
\file RooNumCdf.cxx
\class RooNumCdf
\ingroup Roofitcore

Class RooNumCdf is an implementation of RooNumRunningInt specialized
to calculate cumulative distribution functions from p.d.f.s. The main
difference between RooNumCdf and RooNumRunningInt is that this class
imposes special end-point conditions on the interpolated histogram
that represents the output so that the value at the lower bound is
guaranteed to converge to exactly zero and that the value at the
upper bound is guaranteed to converge to exactly one, at all interpolation
orders.
**/

#include "Riostream.h"

#include "RooAbsPdf.h"
#include "RooNumCdf.h"
#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooRealVar.h"

using namespace std;

ClassImp(RooNumCdf);
  ;



////////////////////////////////////////////////////////////////////////////////
/// Construct a cumulative distribution function from given input p.d.f over observable x.
/// using a numeric sampling algorithm. Use binning named 'bname' to control sampling
/// granularity

RooNumCdf::RooNumCdf(const char *name, const char *title, RooAbsPdf& _pdf, RooRealVar& _x, const char* bname) :
   RooNumRunningInt(name,title,_pdf,_x,bname)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooNumCdf::RooNumCdf(const RooNumCdf& other, const char* name) :
   RooNumRunningInt(other,name)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooNumCdf::~RooNumCdf()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Fill cache using running integral cache elements calculate()
/// method with specification of cdf-specific boundary conditions

void RooNumCdf::fillCacheObject(RooAbsCachedReal::FuncCacheElem& cache) const
{
  RICacheElem& riCache = static_cast<RICacheElem&>(cache) ;
  riCache.calculate(kTRUE) ;
}


