/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.cc,v 1.11 2001/09/15 00:26:02 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   05-Aug-2001 DK Adapted to use RooAbsFunc interface
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooIntegrator1D implements an adaptive one-dimensional 
// numerical integration algorithm.


#include "RooFitCore/RooIntegrator1D.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooNumber.hh"

#include <assert.h>

ClassImp(RooIntegrator1D)
;

RooIntegrator1D::RooIntegrator1D(const RooAbsFunc& function, SummationRule rule,
				 Int_t maxSteps, Double_t eps) : 
  RooAbsIntegrator(function), _rule(rule), _maxSteps(maxSteps), _eps(eps)
{
  // Use this form of the constructor to integrate over the function's default range.

  _useIntegrandLimits= kTRUE;
  _valid= initialize();
} 

RooIntegrator1D::RooIntegrator1D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
				 SummationRule rule, Int_t maxSteps, Double_t eps) : 
  RooAbsIntegrator(function), _rule(rule), _maxSteps(maxSteps), _eps(eps)
{
  // Use this form of the constructor to override the function's default range.

  _useIntegrandLimits= kFALSE;
  _xmin= xmin;
  _xmax= xmax;
  _valid= initialize();
} 

Bool_t RooIntegrator1D::initialize()
{
  // apply defaults if necessary
  if(_maxSteps <= 0) {
    _maxSteps= (_rule == Trapezoid) ? 20 : 14;
  }
  if(_eps <= 0) _eps= 1e-6;
  // check that the integrand is a valid function
  if(!isValid()) {
    cout << "RooIntegrator1D::initialize: cannot integrate invalid function" << endl;
    return kFALSE;
  }
  // check that the function is one dimensional
  if(_function->getDimension() != 1) {
    cout << "RooIntegrator1D::initialize: cannot integrate function of dimension "
	 << _function->getDimension() << endl;
    return kFALSE;
  }

  // Allocate workspace for numerical integration engine
  _h= new Double_t[_maxSteps + 2];
  _s= new Double_t[_maxSteps + 2];
  _c= new Double_t[_nPoints + 1];
  _d= new Double_t[_nPoints + 1];

  return checkLimits();
}

RooIntegrator1D::~RooIntegrator1D()
{
  // Release integrator workspace
  if(_h) delete[] _h;
  if(_s) delete[] _s;
  if(_c) delete[] _c;
  if(_d) delete[] _d;
}

Bool_t RooIntegrator1D::setLimits(Double_t xmin, Double_t xmax) {
  // Change our integration limits. Return kTRUE if the new limits are
  // ok, or otherwise kFALSE. Always returns kFALSE and does nothing
  // if this object was constructed to always use our integrand's limits.

  if(_useIntegrandLimits) {
    cout << "RooIntegrator1D::setLimits: cannot override integrand's limits" << endl;
    return kFALSE;
  }
  _xmin= xmin;
  _xmax= xmax;
  return checkLimits();
}

Bool_t RooIntegrator1D::checkLimits() const {
  // Check that our integration range is finite and otherwise return kFALSE.
  // Update the limits from the integrand if requested.

  if(_useIntegrandLimits) {
    assert(0 != integrand() && integrand()->isValid());
    _xmin= integrand()->getMinLimit(0);
    _xmax= integrand()->getMaxLimit(0);
  }
  _range= _xmax - _xmin;
  if(_range <= 0) {
    cout << "RooIntegrator1D::checkLimits: bad range with min >= max" << endl;
    return kFALSE;
  }
  return (RooNumber::isInfinite(_xmin) || RooNumber::isInfinite(_xmax)) ? kFALSE : kTRUE;
}

Double_t RooIntegrator1D::integral() 
{
  assert(isValid());

  Int_t j;
  _h[1]=1.0;
  for(j= 1; j<=_maxSteps; j++) {
    // refine our estimate using the appropriate summation rule
    _s[j]= (_rule == Trapezoid) ? addTrapezoids(j) : addMidpoints(j);
    if(j >= _nPoints) {
      // extrapolate the results of recent refinements and check for a stable result
      extrapolate(j);
      if(fabs(_extrapError) <= _eps*fabs(_extrapValue)) return _extrapValue;
    }
    // update the step size for the next refinement of the summation
    _h[j+1]= (_rule == Trapezoid) ? _h[j]/4. : _h[j]/9.;
  }

  cout << "RooIntegrator1D::integral: integral over range (" << _xmin << "," << _xmax << ") did not converge after " 
       << _maxSteps << " steps" << endl;
  for(j= 1; j <= _maxSteps; j++) {
    cout << "   [" << j << "] h = " << _h[j] << " , s = " << _s[j] << endl;
  }
  return 0;
}

Double_t RooIntegrator1D::addMidpoints(Int_t n)
{
  // Calculate the n-th stage of refinement of the Second Euler-Maclaurin
  // summation rule which has the useful property of not evaluating the
  // integrand at either of its endpoints but requires more function
  // evaluations than the trapezoidal rule. This rule can be used with
  // a suitable change of variables to estimate improper integrals.

  Double_t x,tnm,sum,del,ddel;
  Int_t it,j;

  if(n == 1) {
    Double_t xmid= 0.5*(_xmin + _xmax);
    return (_savedResult= _range*integrand(&xmid));
  }
  else {
    for(it=1, j=1; j < n-1; j++) it*= 3;
    tnm= it;
    del= _range/(3.*tnm);
    ddel= del+del;
    x= _xmin + 0.5*del;
    for(sum= 0, j= 1; j <= it; j++) {
      sum+= integrand(&x);
      x+= ddel;
      sum+= integrand(&x);
      x+= del;
    }      
    return (_savedResult= (_savedResult + _range*sum/tnm)/3.);
  }
}

Double_t RooIntegrator1D::addTrapezoids(Int_t n)
{
  // Calculate the n-th stage of refinement of the extended trapezoidal
  // summation rule. This is the most efficient rule for a well behaved
  // integrand that can be evaluated over its entire range, including the
  // endpoints.

  Double_t x,tnm,sum,del;
  Int_t it,j;

  if (n == 1) {
    // use a single trapezoid to cover the full range
    return (_savedResult= 0.5*_range*(integrand(&_xmin) + integrand(&_xmax)));
  }
  else {
    // break the range down into several trapezoids using 2**(n-2)
    // equally-spaced interior points
    for(it=1, j=1; j < n-1; j++) it <<= 1;
    tnm= it;
    del= _range/tnm;
    x= _xmin + 0.5*del;
    for(sum=0.0, j=1; j<=it; j++, x+=del) sum += integrand(&x);
    return (_savedResult= 0.5*(_savedResult + _range*sum/tnm));
  }
}


void RooIntegrator1D::extrapolate(Int_t n)
{
  Double_t *xa= &_h[n-_nPoints];
  Double_t *ya= &_s[n-_nPoints];
  Int_t i,m,ns=1;
  Double_t den,dif,dift,ho,hp,w;

  dif=fabs(xa[1]);
  for (i= 1; i <= _nPoints; i++) {
    if ((dift=fabs(xa[i])) < dif) {
      ns=i;
      dif=dift;
    }
    _c[i]=ya[i];
    _d[i]=ya[i];
  }
  _extrapValue= ya[ns--];
  for(m= 1; m < _nPoints; m++) {
    for(i= 1; i <= _nPoints-m; i++) {
      ho=xa[i];
      hp=xa[i+m];
      w=_c[i+1]-_d[i];
      if((den=ho-hp) == 0.0) {
	cout << "RooIntegrator1D::extrapolate: internal error" << endl;
      }
      den=w/den;
      _d[i]=hp*den;
      _c[i]=ho*den;
    }
    _extrapValue += (_extrapError=(2*ns < (_nPoints-m) ? _c[ns+1] : _d[ns--]));
  }
}
