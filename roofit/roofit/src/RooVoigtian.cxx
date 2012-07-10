/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   TS, Thomas Schietinger, SLAC,           schieti@slac.stanford.edu       *
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
// RooVoigtian is an efficient implementation of the convolution of a 
// Breit-Wigner with a Gaussian, making use of the complex error function.
// RooFitCore provides two algorithms for the evaluation of the complex error 
// function (the default CERNlib C335 algorithm, and a faster, look-up-table 
// based method). By default, RooVoigtian employs the default (CERNlib) 
// algorithm. Select the faster algorithm either in the constructor, or with
// the selectFastAlgorithm() method.
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooVoigtian.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooComplex.h"
#include "RooMath.h"

using namespace std;

ClassImp(RooVoigtian)


//_____________________________________________________________________________
RooVoigtian::RooVoigtian(const char *name, const char *title,
			 RooAbsReal& _x, RooAbsReal& _mean,
			 RooAbsReal& _width, RooAbsReal& _sigma,
    			 Bool_t doFast) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  width("width","Breit-Wigner Width",this,_width),
  sigma("sigma","Gauss Width",this,_sigma),
  _doFast(doFast)
{
  _invRootPi= 1./sqrt(atan2(0.,-1.));
}



//_____________________________________________________________________________
RooVoigtian::RooVoigtian(const RooVoigtian& other, const char* name) : 
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  width("width",this,other.width),sigma("sigma",this,other.sigma),
  _doFast(other._doFast)
{
  _invRootPi= 1./sqrt(atan2(0.,-1.));
}



//_____________________________________________________________________________
Double_t RooVoigtian::evaluate() const
{
  Double_t s = (sigma>0) ? sigma : -sigma ;
  Double_t w = (width>0) ? width : -width ;

  Double_t coef= -0.5/(s*s);
  Double_t arg = x - mean;

  // return constant for zero width and sigma
  if (s==0. && w==0.) return 1.;

  // Breit-Wigner for zero sigma
  if (s==0.) return (1./(arg*arg+0.25*w*w));

  // Gauss for zero width
  if (w==0.) return exp(coef*arg*arg);

  // actual Voigtian for non-trivial width and sigma
  Double_t c = 1./(sqrt(2.)*s);
  Double_t a = 0.5*c*w;
  Double_t u = c*arg;
  RooComplex z(u,a) ;
  RooComplex v(0.) ;

  if (_doFast) {
    v = RooMath::FastComplexErrFunc(z);
  } else {
    v = RooMath::ComplexErrFunc(z);
  }
  return c*_invRootPi*v.re();

}
