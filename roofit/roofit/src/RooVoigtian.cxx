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

/** \class RooVoigtian
    \ingroup Roofit

RooVoigtian is an efficient implementation of the convolution of a
Breit-Wigner with a Gaussian, making use of the complex error function.
RooFitCore provides two algorithms for the evaluation of the complex error
function (the default CERNlib C335 algorithm, and a faster, look-up-table
based method). By default, RooVoigtian employs the default (CERNlib)
algorithm. Select the faster algorithm either in the constructor, or with
the selectFastAlgorithm() method.

\note The "width" parameter that determines the Breit-Wigner shape
      represents the **full width at half maximum (FWHM)** of the
      Breit-Wigner (often referred to as \f$\Gamma\f$ or \f$2\gamma\f$).
**/

#include <RooVoigtian.h>

#include <RooMath.h>
#include <RooBatchCompute.h>

#include <cmath>
#include <complex>


////////////////////////////////////////////////////////////////////////////////
/// Construct a RooVoigtian PDF, which represents the convolution of a
/// Breit-Wigner with a Gaussian.
/// \param name Name that identifies the PDF in computations.
/// \param title Title for plotting.
/// \param _x The observable for the PDF.
/// \param _mean The mean of the distribution.
/// \param _width The **full width at half maximum (FWHM)** of the Breit-Wigner
///               (often referred to as \f$\Gamma\f$ or \f$2\gamma\f$).
/// \param _sigma The width of the Gaussian distribution.
/// \param doFast Use the faster look-up-table-based method for the evaluation
///               of the complex error function.

RooVoigtian::RooVoigtian(const char *name, const char *title,
          RooAbsReal& _x, RooAbsReal& _mean,
          RooAbsReal& _width, RooAbsReal& _sigma,
              bool doFast) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  width("width","Breit-Wigner Width",this,_width),
  sigma("sigma","Gauss Width",this,_sigma),
  _doFast(doFast)
{

}

////////////////////////////////////////////////////////////////////////////////

RooVoigtian::RooVoigtian(const RooVoigtian& other, const char* name) :
  RooAbsPdf(other,name), x("x",this,other.x), mean("mean",this,other.mean),
  width("width",this,other.width),sigma("sigma",this,other.sigma),
  _doFast(other._doFast)
{

}

////////////////////////////////////////////////////////////////////////////////

double RooVoigtian::evaluate() const
{
  double s = (sigma>0) ? sigma : -sigma ;
  double w = (width>0) ? width : -width ;

  double coef= -0.5/(s*s);
  double arg = x - mean;

  // return constant for zero width and sigma
  if (s==0. && w==0.) return 1.;

  // Breit-Wigner for zero sigma
  if (s==0.) return (1./(arg*arg+0.25*w*w));

  // Gauss for zero width
  if (w==0.) return std::exp(coef*arg*arg);

  // actual Voigtian for non-trivial width and sigma
  double c = 1./(sqrt(2.)*s);
  double a = 0.5*c*w;
  double u = c*arg;
  std::complex<double> z(u,a) ;
  std::complex<double> v(0.) ;

  if (_doFast) {
    v = RooMath::faddeeva_fast(z);
  } else {
    v = RooMath::faddeeva(z);
  }
  return c * v.real();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute multiple values of Voigtian distribution.
void RooVoigtian::doEval(RooFit::EvalContext &ctx) const
{
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::Voigtian, ctx.output(),
                            {ctx.at(x), ctx.at(mean), ctx.at(width), ctx.at(sigma)});
}
