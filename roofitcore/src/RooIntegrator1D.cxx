/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.cc,v 1.6 2001/07/31 20:54:07 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooIntegrator1D implements an adaptive one-dimensional 
// numerical integration algorithm.


#include "RooFitCore/RooIntegrator1D.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooIntegrator1D)
;

RooIntegrator1D::RooIntegrator1D(const RooAbsReal& function, Int_t mode, RooRealVar& var,
				 Int_t maxSteps, Double_t eps) : 
  RooAbsIntegrator(function, mode), _var(&var), _maxSteps(maxSteps), _eps(eps) 
{
  initialize() ;
} 



RooIntegrator1D::RooIntegrator1D(const RooIntegrator1D& other) : 
  RooAbsIntegrator(other), _var(other._var), _maxSteps(other._maxSteps), _eps(other._eps)
{
  initialize() ;
}


void RooIntegrator1D::initialize()
{
  // Allocate workspace for numerical integration engine
  _h= new Double_t[_maxSteps + 1];
  _s= new Double_t[_maxSteps + 2];
  _c= new Double_t[_nPoints + 1];
  _d= new Double_t[_nPoints + 1];
}



RooIntegrator1D::~RooIntegrator1D()
{
  // Release integrator workspace
  if(_h) delete[] _h;
  if(_s) delete[] _s;
  if(_c) delete[] _c;
  if(_d) delete[] _d;
}



Double_t RooIntegrator1D::integral() 
{
  _xmin= _var->getFitMin() ;
  _xmax= _var->getFitMax() ;
  _range= _var->getFitMax() - _var->getFitMin() ;

  Int_t j;
  _h[1]=1.0;
  for(j= 1; j<=_maxSteps; j++) {
    _s[j]= addTrapezoids(j);
    if(j >= _nPoints) {
      extrapolate(j);
      if(fabs(_extrapError) <= _eps*fabs(_extrapValue)) return _extrapValue;
    }
    _h[j+1]=0.25*_h[j];
  }

  cout << "RooIntegrator1D::integral: did not converge after " 
       << _maxSteps << " steps" << endl;
  for(j= 1; j <= _maxSteps; j++) {
    cout << "   [" << j << "] h = " << _h[j] << " , s = " << _s[j] << endl;
  }
  return 0;
}


Double_t RooIntegrator1D::evalAt(Double_t x) const 
{
  _var->setVal(x) ;
  return eval() ;
}


Double_t RooIntegrator1D::addTrapezoids(Int_t n)
{
  Double_t x,tnm,sum,del;
  Int_t it,j;

  if (n == 1) {
    // use a single trapezoid to cover the full range
    return (_savedResult= 0.5*_range*(evalAt(_xmin) + evalAt(_xmax)));
  }
  else {
    // break the range down into several trapezoids using 2**(n-2)
    // equally-spaced interior points
    for(it=1, j=1; j < n-1; j++) it <<= 1;
    tnm= it;
    del= _range/tnm;
    x= _xmin + 0.5*del;
    for(sum=0.0, j=1; j<=it; j++, x+=del) sum += evalAt(x);
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



