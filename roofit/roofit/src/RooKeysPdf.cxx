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
It was inspired by Kyle Cranmer's KEYS package, see
[the original web page](https://web.archive.org/web/20020705034344/https://www-wisconsin.cern.ch/~cranmer/keys.html).

\note KEYS stands for Kernel Estimating Your Shapes, see
[the KEYS write-up](https://web.archive.org/web/20010604031632/http://www-wisconsin.cern.ch/~cranmer/KEYS.pdf).

If the 'adaptive mode' is enabled, the width of the Gaussian is adaptively calculated from the
local density of events, i.e. narrow for regions with high event density to preserve details and
wide for regions with low event density to promote smoothness. The details of the general algorithm
are described in the following paper:

Cranmer KS, Kernel Estimation in High-Energy Physics.
            Computer Physics Communications 136:198-207,2001 - e-Print Archive: hep-ex/0011057,
            [doi:10.1016/S0010-4655(00)00243-5](https://doi.org/10.1016/S0010-4655(00)00243-5)

The `rho` parameter (default 1) is an overall scale factor for the width of the
kernels. Values larger than 1 make the kernels wider and give a smoother
estimate, while values smaller than 1 make them narrower and keep more detail.
The default corresponds to the usual normal-reference ("rule of thumb")
bandwidth.

Close to the edges of the observable range the estimate is biased: the kernels
of events near an edge have no data on the other side to balance them, so the
density "leaks" out of the range. The `mirror` parameter selects an optional
boundary correction that reflects the data across an edge. Symmetric mirroring
adds the reflected events, which is appropriate when the true density is flat at
the boundary (the estimate keeps a non-zero value there, with zero slope). Asymmetric mirroring
subtracts the reflected events, which is appropriate when the true density is
expected to vanish at the boundary. See the RooKeysPdf::Mirror enum for the list
of options.

For a multi-dimensional version of this pdf, see RooNDKeysPdf.
**/

#include "TMath.h"
#include "RooKeysPdf.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooDataSet.h"

#include "TError.h"
#include "TMath.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>


const double RooKeysPdf::_nSigma = std::sqrt(-2. *
    std::log(std::numeric_limits<double>::epsilon()));

////////////////////////////////////////////////////////////////////////////////
/// coverity[UNINIT_CTOR]

RooKeysPdf::RooKeysPdf()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Construct a kernel estimation pdf of the observable `xpdf` from its
/// distribution in `data`.
///
/// \param[in] name   Name of the pdf.
/// \param[in] title  Title of the pdf, used for plotting.
/// \param[in] xpdf   Observable the pdf is defined in. Its range sets the
///                   boundaries used for the mirror correction and for the
///                   internal binned lookup table.
/// \param[in] data   Dataset whose distribution of `xpdf` is modelled. The width
///                   of each kernel is adapted to the local event density.
/// \param[in] mirror Optional boundary correction, see the Mirror enum.
/// \param[in] rho    Overall scale factor for the kernel width (default 1);
///                   larger values give a smoother estimate.

RooKeysPdf::RooKeysPdf(const char *name, const char *title, RooAbsReal &xpdf, RooDataSet &data, Mirror mirror, double rho)
   : RooKeysPdf(name, title, xpdf, static_cast<RooRealVar &>(xpdf), data, mirror, rho)
{
}

////////////////////////////////////////////////////////////////////////////////
/// As above, but reading the input values from a dataset variable `xdata` that
/// can be different from the observable `xpdf` the pdf depends on.
///
/// \param[in] name   Name of the pdf.
/// \param[in] title  Title of the pdf, used for plotting.
/// \param[in] xpdf   Observable the pdf is defined in.
/// \param[in] xdata  Variable in `data` whose distribution is modelled. Its
///                   range sets the boundaries used for the mirror correction
///                   and for the internal binned lookup table.
/// \param[in] data   Dataset holding the values of `xdata` to model.
/// \param[in] mirror Optional boundary correction, see the Mirror enum.
/// \param[in] rho    Overall scale factor for the kernel width (default 1);
///                   larger values give a smoother estimate.

RooKeysPdf::RooKeysPdf(const char *name, const char *title, RooAbsReal &xpdf, RooRealVar &xdata, RooDataSet &data,
                       Mirror mirror, double rho)
   : RooAbsPdf(name, title),
     _x("x", "Observable", this, xpdf),
     _mirrorLeft(mirror == MirrorLeft || mirror == MirrorBoth || mirror == MirrorLeftAsymRight),
     _mirrorRight(mirror == MirrorRight || mirror == MirrorBoth || mirror == MirrorAsymLeftRight),
     _asymLeft(mirror == MirrorAsymLeft || mirror == MirrorAsymLeftRight || mirror == MirrorAsymBoth),
     _asymRight(mirror == MirrorAsymRight || mirror == MirrorLeftAsymRight || mirror == MirrorAsymBoth),
     _lo(xdata.getMin()),
     _hi(xdata.getMax()),
     _binWidth((_hi - _lo) / (_nPoints - 1)),
     _rho(rho)
{
  snprintf(_varName, 128,"%s", xdata.GetName());

  // form the lookup table
  LoadDataSet(data);
}

////////////////////////////////////////////////////////////////////////////////

RooKeysPdf::RooKeysPdf(const RooKeysPdf &other, const char *name)
   : RooAbsPdf(other, name),
     _x("x", this, other._x),
     _nEvents(other._nEvents),
     _mirrorLeft(other._mirrorLeft),
     _mirrorRight(other._mirrorRight),
     _asymLeft(other._asymLeft),
     _asymRight(other._asymRight),
     _lo(other._lo),
     _hi(other._hi),
     _binWidth(other._binWidth),
     _rho(other._rho)
{
  // cache stuff about x
  snprintf(_varName, 128, "%s", other._varName );

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

}

////////////////////////////////////////////////////////////////////////////////

RooKeysPdf::~RooKeysPdf() {
  delete[] _dataPts;
  delete[] _dataWgts;
  delete[] _weights;

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
  double x0 = 0.;
  double x1 = 0.;
  double x2 = 0.;
  _sumWgt = 0.;
  // read the data set into tmp and accumulate some statistics
  RooRealVar& real = static_cast<RooRealVar&>(data.get()->operator[](_varName));
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
  // Dividing by 2*sqrt(3) = sqrt(12) turns a width into the standard deviation
  // of a uniform distribution of that width. Per the original author, this goes
  // back to inputs that were finely binned histograms rather than unbinned data:
  // entries spread uniformly over a bin get aggregated into a single sample with
  // no variance, so the bin width was taken as the spread of that sample.
  //
  // Beware that no bin width enters the expression below, so that rationale does
  // not map onto the code as it stands: what remains is an extra factor of
  // sqrt(12) with respect to hep-ex/0011057, kept for backwards compatibility.
  // The same factor appears in RooNDKeysPdf::calculateBandWidth().
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
//     << " -- performing linear extrapolation..." << std::endl;
    i=0;
  }
  if (i>_nPoints-1) {
//     cerr << "got point above upper bound:"
//     << double(_x) << " > " << _hi
//     << " -- performing linear extrapolation..." << std::endl;
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
