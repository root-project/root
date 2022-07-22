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

/** \class RooKeysPdf
    \ingroup Roofit

Class RooKeysPdf implements a one-dimensional kernel estimation p.d.f which model the distribution
of an arbitrary input dataset as a superposition of Gaussian kernels, one for each data point,
each contributing 1/N to the total integral of the pdf.
If the 'adaptive mode' is enabled, the width of the Gaussian is adaptively calculated from the
local density of events, i.e. narrow for regions with high event density to preserve details and
wide for regions with low event density to promote smoothness. The details of the general algorithm
are described in the following paper:

Cranmer KS, Kernel Estimation in High-Energy Physics.
            Computer Physics Communications 136:198-207,2001 - e-Print Archive: hep ex/0011057
**/

#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "TMath.h"
#include "snprintf.h"
#include "RooKeysPdf.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooDataSet.h"
#include "RooTrace.h"

#include "TError.h"

using namespace std;

ClassImp(RooKeysPdf);

const double RooKeysPdf::_nSigma = std::sqrt(-2. *
    std::log(std::numeric_limits<double>::epsilon()));

////////////////////////////////////////////////////////////////////////////////
/// coverity[UNINIT_CTOR]

  RooKeysPdf::RooKeysPdf() : _nEvents(0), _dataPts(0), _dataWgts(0), _weights(0), _sumWgt(0),
              _mirrorLeft(false), _mirrorRight(false),
              _asymLeft(false), _asymRight(false)
{
  TRACE_CREATE
}

////////////////////////////////////////////////////////////////////////////////
/// cache stuff about x

RooKeysPdf::RooKeysPdf(const char *name, const char *title,
                       RooAbsReal& x, RooDataSet& data,
                       Mirror mirror, double rho) :
  RooAbsPdf(name,title),
  _x("x","observable",this,x),
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
  snprintf(_varName, 128,"%s", x.GetName());
  RooAbsRealLValue& real= (RooRealVar&)(_x.arg());
  _lo = real.getMin();
  _hi = real.getMax();
  _binWidth = (_hi-_lo)/(_nPoints-1);

  // form the lookup table
  LoadDataSet(data);
  TRACE_CREATE
}

////////////////////////////////////////////////////////////////////////////////
/// cache stuff about x

RooKeysPdf::RooKeysPdf(const char *name, const char *title,
                       RooAbsReal& xpdf, RooRealVar& xdata, RooDataSet& data,
                       Mirror mirror, double rho) :
  RooAbsPdf(name,title),
  _x("x","Observable",this,xpdf),
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
  snprintf(_varName, 128,"%s", xdata.GetName());
  RooAbsRealLValue& real= (RooRealVar&)(xdata);
  _lo = real.getMin();
  _hi = real.getMax();
  _binWidth = (_hi-_lo)/(_nPoints-1);

  // form the lookup table
  LoadDataSet(data);
  TRACE_CREATE
}

////////////////////////////////////////////////////////////////////////////////

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
//    _dataPts = new double[_nEvents];
//    _weights = new double[_nEvents];
//    for (Int_t i= 0; i<_nEvents; i++) {
//      _dataPts[i]= other._dataPts[i];
//      _weights[i]= other._weights[i];
//    }

  // copy over the lookup table
  for (Int_t i= 0; i<_nPoints+1; i++)
    _lookupTable[i]= other._lookupTable[i];

  TRACE_CREATE
}

////////////////////////////////////////////////////////////////////////////////

RooKeysPdf::~RooKeysPdf() {
  delete[] _dataPts;
  delete[] _dataWgts;
  delete[] _weights;

  TRACE_DESTROY
}

////////////////////////////////////////////////////////////////////////////////
/// small helper structure

namespace {
  struct Data {
    double x;
    double w;
  };
  // helper to order two Data structures
  struct cmp {
    inline bool operator()(const struct Data& a, const struct Data& b) const
    { return a.x < b.x; }
  };
}
void RooKeysPdf::LoadDataSet( RooDataSet& data) {
  delete[] _dataPts;
  delete[] _dataWgts;
  delete[] _weights;

  std::vector<Data> tmp;
  tmp.reserve((1 + _mirrorLeft + _mirrorRight) * data.numEntries());
  double x0 = 0., x1 = 0., x2 = 0.;
  _sumWgt = 0.;
  // read the data set into tmp and accumulate some statistics
  RooRealVar& real = (RooRealVar&)(data.get()->operator[](_varName));
  for (Int_t i = 0; i < data.numEntries(); ++i) {
    data.get(i);
    const double x = real.getVal();
    const double w = data.weight();
    x0 += w;
    x1 += w * x;
    x2 += w * x * x;
    _sumWgt += double(1 + _mirrorLeft + _mirrorRight) * w;

    Data p;
    p.x = x, p.w = w;
    tmp.push_back(p);
    if (_mirrorLeft) {
      p.x = 2. * _lo - x;
      tmp.push_back(p);
    }
    if (_mirrorRight) {
      p.x = 2. * _hi - x;
      tmp.push_back(p);
    }
  }
  // sort the entire data set so that values of x are increasing
  std::sort(tmp.begin(), tmp.end(), cmp());

  // copy the sorted data set to its final destination
  _nEvents = tmp.size();
  _dataPts  = new double[_nEvents];
  _dataWgts = new double[_nEvents];
  for (unsigned i = 0; i < tmp.size(); ++i) {
    _dataPts[i] = tmp[i].x;
    _dataWgts[i] = tmp[i].w;
  }
  {
    // free tmp
    std::vector<Data> tmp2;
    tmp2.swap(tmp);
  }

  double meanv=x1/x0;
  double sigmav=std::sqrt(x2/x0-meanv*meanv);
  double h=std::pow(double(4)/double(3),0.2)*std::pow(_sumWgt,-0.2)*_rho;
  double hmin=h*sigmav*std::sqrt(2.)/10;
  double norm=h*std::sqrt(sigmav * _sumWgt)/(2.0*std::sqrt(3.0));

  _weights=new double[_nEvents];
  for(Int_t j=0;j<_nEvents;++j) {
    _weights[j] = norm / std::sqrt(_dataWgts[j] * g(_dataPts[j],h*sigmav));
    if (_weights[j]<hmin) _weights[j]=hmin;
  }

  // The idea below is that beyond nSigma sigma, the value of the exponential
  // in the Gaussian is well below the machine precision of a double, so it
  // does not contribute any more. That way, we can limit how many bins of the
  // binned approximation in _lookupTable we have to touch when filling it.
  for (Int_t i=0;i<_nPoints+1;++i) _lookupTable[i] = 0.;
  for(Int_t j=0;j<_nEvents;++j) {
      const double xlo = std::min(_hi,
         std::max(_lo, _dataPts[j] - _nSigma * _weights[j]));
      const double xhi = std::max(_lo,
         std::min(_hi, _dataPts[j] + _nSigma * _weights[j]));
      if (xlo >= xhi) continue;
      const double chi2incr = _binWidth / _weights[j] / std::sqrt(2.);
      const double weightratio = _dataWgts[j] / _weights[j];
      const Int_t binlo = static_cast<Int_t>(std::floor((xlo - _lo) / _binWidth));
      const Int_t binhi = static_cast<Int_t>(_nPoints - std::floor((_hi - xhi) / _binWidth));
      const double x = (double(_nPoints - binlo) * _lo +
         double(binlo) * _hi) / double(_nPoints);
      double chi = (x - _dataPts[j]) / _weights[j] / std::sqrt(2.);
      for (Int_t k = binlo; k <= binhi; ++k, chi += chi2incr) {
     _lookupTable[k] += weightratio * std::exp(- chi * chi);
      }
  }
  if (_asymLeft) {
      for(Int_t j=0;j<_nEvents;++j) {
     const double xlo = std::min(_hi,
        std::max(_lo, 2. * _lo - _dataPts[j] + _nSigma * _weights[j]));
     const double xhi = std::max(_lo,
        std::min(_hi, 2. * _lo - _dataPts[j] - _nSigma * _weights[j]));
     if (xlo >= xhi) continue;
     const double chi2incr = _binWidth / _weights[j] / std::sqrt(2.);
     const double weightratio = _dataWgts[j] / _weights[j];
     const Int_t binlo = static_cast<Int_t>(std::floor((xlo - _lo) / _binWidth));
     const Int_t binhi = static_cast<Int_t>(_nPoints - std::floor((_hi - xhi) / _binWidth));
     const double x = (double(_nPoints - binlo) * _lo +
        double(binlo) * _hi) / double(_nPoints);
     double chi = (x - (2. * _lo - _dataPts[j])) / _weights[j] / std::sqrt(2.);
     for (Int_t k = binlo; k <= binhi; ++k, chi += chi2incr) {
         _lookupTable[k] -= weightratio * std::exp(- chi * chi);
     }
      }
  }
  if (_asymRight) {
      for(Int_t j=0;j<_nEvents;++j) {
     const double xlo = std::min(_hi,
        std::max(_lo, 2. * _hi - _dataPts[j] + _nSigma * _weights[j]));
     const double xhi = std::max(_lo,
        std::min(_hi, 2. * _hi - _dataPts[j] - _nSigma * _weights[j]));
     if (xlo >= xhi) continue;
     const double chi2incr = _binWidth / _weights[j] / std::sqrt(2.);
     const double weightratio = _dataWgts[j] / _weights[j];
     const Int_t binlo = static_cast<Int_t>(std::floor((xlo - _lo) / _binWidth));
     const Int_t binhi = static_cast<Int_t>(_nPoints - std::floor((_hi - xhi) / _binWidth));
     const double x = (double(_nPoints - binlo) * _lo +
        double(binlo) * _hi) / double(_nPoints);
     double chi = (x - (2. * _hi - _dataPts[j])) / _weights[j] / std::sqrt(2.);
     for (Int_t k = binlo; k <= binhi; ++k, chi += chi2incr) {
         _lookupTable[k] -= weightratio * std::exp(- chi * chi);
     }
      }
  }
  static const double sqrt2pi(std::sqrt(2*TMath::Pi()));
  for (Int_t i=0;i<_nPoints+1;++i)
    _lookupTable[i] /= sqrt2pi * _sumWgt;
}

////////////////////////////////////////////////////////////////////////////////

double RooKeysPdf::evaluate() const {
  Int_t i = (Int_t)floor((double(_x)-_lo)/_binWidth);
  if (i<0) {
//     cerr << "got point below lower bound:"
//     << double(_x) << " < " << _lo
//     << " -- performing linear extrapolation..." << endl;
    i=0;
  }
  if (i>_nPoints-1) {
//     cerr << "got point above upper bound:"
//     << double(_x) << " > " << _hi
//     << " -- performing linear extrapolation..." << endl;
    i=_nPoints-1;
  }
  double dx = (double(_x)-(_lo+i*_binWidth))/_binWidth;

  // for now do simple linear interpolation.
  // one day replace by splines...
  double ret = (_lookupTable[i]+dx*(_lookupTable[i+1]-_lookupTable[i]));
  if (ret<0) ret=0 ;
  return ret ;
}

Int_t RooKeysPdf::getAnalyticalIntegral(
   RooArgSet& allVars, RooArgSet& analVars, const char* /* rangeName */) const
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}

double RooKeysPdf::analyticalIntegral(Int_t code, const char* rangeName) const
{
  R__ASSERT(1 == code);
  // this code is based on _lookupTable and uses linear interpolation, just as
  // evaluate(); integration is done using the trapez rule
  const double xmin = std::max(_lo, _x.min(rangeName));
  const double xmax = std::min(_hi, _x.max(rangeName));
  const Int_t imin = (Int_t)floor((xmin - _lo) / _binWidth);
  const Int_t imax = std::min((Int_t)floor((xmax - _lo) / _binWidth),
      _nPoints - 1);
  double sum = 0.;
  // sum up complete bins in middle
  if (imin + 1 < imax)
    sum += _lookupTable[imin + 1] + _lookupTable[imax];
  for (Int_t i = imin + 2; i < imax; ++i)
    sum += 2. * _lookupTable[i];
  sum *= _binWidth * 0.5;
  // treat incomplete bins
  const double dxmin = (xmin - (_lo + imin * _binWidth)) / _binWidth;
  const double dxmax = (xmax - (_lo + imax * _binWidth)) / _binWidth;
  if (imin < imax) {
    // first bin
    sum += _binWidth * (1. - dxmin) * 0.5 * (_lookupTable[imin + 1] +
   _lookupTable[imin] + dxmin *
   (_lookupTable[imin + 1] - _lookupTable[imin]));
    // last bin
    sum += _binWidth * dxmax * 0.5 * (_lookupTable[imax] +
   _lookupTable[imax] + dxmax *
   (_lookupTable[imax + 1] - _lookupTable[imax]));
  } else if (imin == imax) {
    // first bin == last bin
    sum += _binWidth * (dxmax - dxmin) * 0.5 * (
   _lookupTable[imin] + dxmin *
   (_lookupTable[imin + 1] - _lookupTable[imin]) +
   _lookupTable[imax] + dxmax *
   (_lookupTable[imax + 1] - _lookupTable[imax]));
  }
  return sum;
}

Int_t RooKeysPdf::getMaxVal(const RooArgSet& vars) const
{
  if (vars.contains(*_x.absArg())) return 1;
  return 0;
}

double RooKeysPdf::maxVal(Int_t code) const
{
  R__ASSERT(1 == code);
  double max = -std::numeric_limits<double>::max();
  for (Int_t i = 0; i <= _nPoints; ++i)
    if (max < _lookupTable[i]) max = _lookupTable[i];
  return max;
}

////////////////////////////////////////////////////////////////////////////////

double RooKeysPdf::g(double x,double sigmav) const {
  double y=0;
  // since data is sorted, we can be a little faster because we know which data
  // points contribute
  double* it = std::lower_bound(_dataPts, _dataPts + _nEvents,
      x - _nSigma * sigmav);
  if (it >= (_dataPts + _nEvents)) return 0.;
  double* iend = std::upper_bound(it, _dataPts + _nEvents,
      x + _nSigma * sigmav);
  for ( ; it < iend; ++it) {
    const double r = (x - *it) / sigmav;
    y += std::exp(-0.5 * r * r);
  }

  static const double sqrt2pi(std::sqrt(2*TMath::Pi()));
  return y/(sigmav*sqrt2pi);
}
