/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooKeysPdf.cc,v 1.2 2002/02/06 15:55:31 giraudpf Exp $
 * Authors:
 *   GR, Gerhard Raven, UC, San Diego , Gerhard.Raven@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   04-Jul-2000 GR Created initial version
 *   01-Sep-2000 DK Override useParameters() method to fix the normalization
 *
 * Copyright (C) 2000 UC, San Diego
 *****************************************************************************/
#include "BaBar/BaBar.hh"
#include <math.h>
#include <iostream.h>

#include "RooFitModels/RooKeysPdf.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRandom.hh"
#include "RooFitCore/RooDataSet.hh"

ClassImp(RooKeysPdf)

RooKeysPdf::RooKeysPdf(const char *name, const char *title,
                       RooAbsReal& x, RooDataSet& data,
                       Mirror mirror, Double_t rho) :
  RooAbsPdf(name,title),
  _x("x","Dependent",this,x),
  _nEvents(0),
  _dataPts(0),
  _weights(0),
  _mirrorLeft( mirror==MirrorLeft || mirror==MirrorBoth ),
  _mirrorRight( mirror==MirrorRight || mirror==MirrorBoth ),
  _rho(rho)
{
  // cache stuff about x
  sprintf(_varName, "%s", x.GetName());
  RooRealVar real= (RooRealVar&)(_x.arg());
  _lo = real.getFitMin();
  _hi = real.getFitMax();
  _binWidth = (_hi-_lo)/(_nPoints-1);

  // form the lookup table
  LoadDataSet(data);
}

RooKeysPdf::RooKeysPdf(const RooKeysPdf& other, const char* name):
  RooAbsPdf(other,name), _x("x",this,other._x), _nEvents(other._nEvents),
  _dataPts(0), _weights(0),
  _mirrorLeft( other._mirrorLeft ), _mirrorRight( other._mirrorRight ),
  _rho( other._rho ) {

  // cache stuff about x
  sprintf(_varName, "%s", other._varName );
  RooRealVar real= (RooRealVar&)(_x.arg());
  _lo = real.getFitMin();
  _hi = real.getFitMax();
  _binWidth = (_hi-_lo)/(_nPoints-1);

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

RooKeysPdf::~RooKeysPdf() {
  delete[] _dataPts;
  delete[] _weights;
}


void
RooKeysPdf::LoadDataSet( RooDataSet& data) {
  delete[] _dataPts;
  delete[] _weights;

  // make new arrays for data and weights to fill
  _nEvents= (Int_t)data.numEntries();
  _dataPts = new Double_t[_nEvents];
  _weights = new Double_t[_nEvents];

  Double_t x0(0);
  Double_t x1(0);
  Double_t x2(0);

  for (Int_t i=0; i<_nEvents; i++) {
    const RooArgSet *values= data.get(i);
    RooRealVar real= (RooRealVar&)(values->operator[](_varName));
    _dataPts[i]= real.getVal();
    x0++; x1+=_dataPts[i]; x2+=_dataPts[i]*_dataPts[i];
  }

  Double_t mean=x1/x0;
  Double_t sigma=sqrt(x2/_nEvents-mean*mean);
  Double_t h=pow(Double_t(4)/Double_t(3),0.2)*pow(_nEvents,-0.2)*_rho;
  Double_t hmin=h*sigma*sqrt(2)/10;
  Double_t norm=h*sqrt(sigma)/(2.0*sqrt(3.0));

  _weights=new Double_t[_nEvents];
  for(Int_t j=0;j<_nEvents;++j) {
    _weights[j]=norm/sqrt(g(_dataPts[j],h*sigma));
    if (_weights[j]<hmin) _weights[j]=hmin;
  }
  
  for (Int_t i=0;i<_nPoints+1;++i) 
    _lookupTable[i]=evaluateFull( _lo+Double_t(i)*_binWidth );

  
}


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

Double_t RooKeysPdf::evaluateFull( Double_t x ) const {
  Double_t y=0;

  for (Int_t i=0;i<_nEvents;++i) {
    Double_t chi=(x-_dataPts[i])/_weights[i];
    y+=exp(-0.5*chi*chi)/_weights[i];

    // if mirroring the distribution across either edge of
    // the range ("Boundary Kernals"), pick up the additional
    // contributions
    if (_mirrorLeft) {
      chi=(x-(2*_lo-_dataPts[i]))/_weights[i];
      y+=exp(-0.5*chi*chi)/_weights[i];
    }
    if (_mirrorRight) {
      chi=(x-(2*_hi-_dataPts[i]))/_weights[i];
      y+=exp(-0.5*chi*chi)/_weights[i];
    }
  }
  
  static const Double_t sqrt2pi(sqrt(2*M_PI));  
  return y/(sqrt2pi*_nEvents);
}

Double_t RooKeysPdf::g(Double_t x,Double_t sigma) const {
  
  Double_t c=Double_t(1)/(2*sigma*sigma);

  Double_t y=0;
  for (Int_t i=0;i<_nEvents;++i) {
    Double_t r=x-_dataPts[i];
    y+=exp(-c*r*r);
  }
  
  static const Double_t sqrt2pi(sqrt(2*M_PI));  
  return y/(sigma*sqrt2pi*_nEvents);
}
