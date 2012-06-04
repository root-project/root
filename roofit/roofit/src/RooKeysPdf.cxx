/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   UC San Diego,        raven@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#include "RooFit.h"

#include <math.h>
#include "Riostream.h"
#include "TMath.h"

#include "RooKeysPdf.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooDataSet.h"

using namespace std;

ClassImp(RooKeysPdf)


//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Class RooKeysPdf implements a one-dimensional kernel estimation p.d.f which model the distribution
// of an arbitrary input dataset as a superposition of Gaussian kernels, one for each data point,
// each contributing 1/N to the total integral of the p.d.f.
// <p>
// If the 'adaptive mode' is enabled, the width of the Gaussian is adaptively calculated from the
// local density of events, i.e. narrow for regions with high event density to preserve details and
// wide for regions with log event density to promote smoothness. The details of the general algorithm
// are described in the following paper: 
// <p>
// Cranmer KS, Kernel Estimation in High-Energy Physics.  
//             Computer Physics Communications 136:198-207,2001 - e-Print Archive: hep ex/0011057
// <p>
// END_HTML
//


//_____________________________________________________________________________
  RooKeysPdf::RooKeysPdf() : _nEvents(0), _dataPts(0), _dataWgts(0), _weights(0), _sumWgt(0),
			     _mirrorLeft(kFALSE), _mirrorRight(kFALSE), 
			     _asymLeft(kFALSE), _asymRight(kFALSE)
{ 
  // coverity[UNINIT_CTOR]
}


//_____________________________________________________________________________
RooKeysPdf::RooKeysPdf(const char *name, const char *title,
                       RooAbsReal& x, RooDataSet& data,
                       Mirror mirror, Double_t rho) :
  RooAbsPdf(name,title),
  _x("x","Dependent",this,x),
  _nEvents(0),
  _dataPts(0),
  _dataWgts(0),
  _weights(0),
  _mirrorLeft(mirror==MirrorLeft || mirror==MirrorBoth || mirror==MirrorLeftAsymRight),
  _mirrorRight(mirror==MirrorRight || mirror==MirrorBoth || mirror==MirrorAsymLeftRight),
  _asymLeft(mirror==MirrorAsymLeft || mirror==MirrorAsymLeftRight || mirror==MirrorAsymBoth),
  _asymRight(mirror==MirrorAsymRight || mirror==MirrorLeftAsymRight || mirror==MirrorAsymBoth),
  _rho(rho)
{
  // cache stuff about x
  snprintf(_varName, 128,"%s", x.GetName());
  RooRealVar real= (RooRealVar&)(_x.arg());
  _lo = real.getMin();
  _hi = real.getMax();
  _binWidth = (_hi-_lo)/(_nPoints-1);

  // form the lookup table
  LoadDataSet(data);
}



//_____________________________________________________________________________
RooKeysPdf::RooKeysPdf(const RooKeysPdf& other, const char* name):
  RooAbsPdf(other,name), _x("x",this,other._x), _nEvents(other._nEvents),
  _dataPts(0), _dataWgts(0), _weights(0), _sumWgt(0),
  _mirrorLeft( other._mirrorLeft ), _mirrorRight( other._mirrorRight ),
  _asymLeft(other._asymLeft), _asymRight(other._asymRight),
  _rho( other._rho ) {

  // cache stuff about x
  snprintf(_varName, 128, "%s", other._varName );
  _lo = other._lo;
  _hi = other._hi;
  _binWidth = other._binWidth;

  // copy over data and weights... not necessary, commented out for speed
//    _dataPts = new Double_t[_nEvents];
//    _weights = new Double_t[_nEvents];  
//    for (Int_t i= 0; i<_nEvents; i++) {
//      _dataPts[i]= other._dataPts[i];
//      _weights[i]= other._weights[i];
//    }

  // copy over the lookup table
  for (Int_t i= 0; i<_nPoints+1; i++)
    _lookupTable[i]= other._lookupTable[i];
  
}


//_____________________________________________________________________________
RooKeysPdf::~RooKeysPdf() {
  delete[] _dataPts;
  delete[] _dataWgts;
  delete[] _weights;
}


void

//_____________________________________________________________________________
RooKeysPdf::LoadDataSet( RooDataSet& data) {
  delete[] _dataPts;
  delete[] _dataWgts;
  delete[] _weights;

  // make new arrays for data and weights to fill
  _nEvents= (Int_t)data.numEntries();
  if (_mirrorLeft) _nEvents += data.numEntries();
  if (_mirrorRight) _nEvents += data.numEntries();

  _dataPts  = new Double_t[_nEvents];
  _dataWgts = new Double_t[_nEvents];
  _weights  = new Double_t[_nEvents];
  _sumWgt = 0 ;

  Double_t x0(0);
  Double_t x1(0);
  Double_t x2(0);

  Int_t i, idata=0;
  for (i=0; i<data.numEntries(); i++) {
    const RooArgSet *values= data.get(i);
    RooRealVar real= (RooRealVar&)(values->operator[](_varName));

    _dataPts[idata]= real.getVal();
    _dataWgts[idata] = data.weight() ;
    x0 += _dataWgts[idata] ; x1+=_dataWgts[idata]*_dataPts[idata]; x2+=_dataWgts[idata]*_dataPts[idata]*_dataPts[idata];
    idata++;
    _sumWgt+= data.weight() ;

    if (_mirrorLeft) {
      _dataPts[idata]= 2*_lo - real.getVal();
      _dataWgts[idata]= data.weight() ;
      _sumWgt+= data.weight() ;
      idata++;
    }

    if (_mirrorRight) {
      _dataPts[idata]  = 2*_hi - real.getVal();
      _dataWgts[idata] = data.weight() ;
      _sumWgt+= data.weight() ;
      idata++;
    }
  }

  Double_t meanv=x1/x0;
  Double_t sigmav=sqrt(x2/x0-meanv*meanv);
  Double_t h=TMath::Power(Double_t(4)/Double_t(3),0.2)*TMath::Power(_nEvents,-0.2)*_rho;
  Double_t hmin=h*sigmav*sqrt(2.)/10;
  Double_t norm=h*sqrt(sigmav)/(2.0*sqrt(3.0));

  _weights=new Double_t[_nEvents];
  for(Int_t j=0;j<_nEvents;++j) {
    _weights[j]=norm/sqrt(g(_dataPts[j],h*sigmav));
    if (_weights[j]<hmin) _weights[j]=hmin;
  }
  
  for (i=0;i<_nPoints+1;++i) 
    _lookupTable[i]=evaluateFull( _lo+Double_t(i)*_binWidth );

  
}



//_____________________________________________________________________________
Double_t RooKeysPdf::evaluate() const {
  Int_t i = (Int_t)floor((Double_t(_x)-_lo)/_binWidth);
  if (i<0) {
    cerr << "got point below lower bound:"
	 << Double_t(_x) << " < " << _lo
	 << " -- performing linear extrapolation..." << endl;
    i=0;
  }
  if (i>_nPoints-1) {
    cerr << "got point above upper bound:"
	 << Double_t(_x) << " > " << _hi
	 << " -- performing linear extrapolation..." << endl;
    i=_nPoints-1;
  }
  Double_t dx = (Double_t(_x)-(_lo+i*_binWidth))/_binWidth;
  
  // for now do simple linear interpolation.
  // one day replace by splines...
  return (_lookupTable[i]+dx*(_lookupTable[i+1]-_lookupTable[i]));
}


//_____________________________________________________________________________
Double_t RooKeysPdf::evaluateFull( Double_t x ) const {
  Double_t y=0;

  for (Int_t i=0;i<_nEvents;++i) {
    Double_t chi=(x-_dataPts[i])/_weights[i];
    y+=_dataWgts[i]*exp(-0.5*chi*chi)/_weights[i];

    // if mirroring the distribution across either edge of
    // the range ("Boundary Kernels"), pick up the additional
    // contributions
//      if (_mirrorLeft) {
//        chi=(x-(2*_lo-_dataPts[i]))/_weights[i];
//        y+=exp(-0.5*chi*chi)/_weights[i];
//      }
    if (_asymLeft) {
      chi=(x-(2*_lo-_dataPts[i]))/_weights[i];
      y-=_dataWgts[i]*exp(-0.5*chi*chi)/_weights[i];
    }
//      if (_mirrorRight) {
//        chi=(x-(2*_hi-_dataPts[i]))/_weights[i];
//        y+=exp(-0.5*chi*chi)/_weights[i];
//      }
    if (_asymRight) {
      chi=(x-(2*_hi-_dataPts[i]))/_weights[i];
      y-=_dataWgts[i]*exp(-0.5*chi*chi)/_weights[i];
    }
  }
  
  static const Double_t sqrt2pi(sqrt(2*TMath::Pi()));  
  return y/(sqrt2pi*_sumWgt);
}


//_____________________________________________________________________________
Double_t RooKeysPdf::g(Double_t x,Double_t sigmav) const {
  
  Double_t c=Double_t(1)/(2*sigmav*sigmav);

  Double_t y=0;
  for (Int_t i=0;i<_nEvents;++i) {
    Double_t r=x-_dataPts[i];
    y+=exp(-c*r*r);
  }
  
  static const Double_t sqrt2pi(sqrt(2*TMath::Pi()));  
  return y/(sigmav*sqrt2pi*_nEvents);
}
