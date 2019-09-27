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
each contributing 1/N to the total integral of the p.d.f..
If the 'adaptive mode' is enabled, the width of the Gaussian is adaptively calculated from the
local density of events, i.e. narrow for regions with high event density to preserve details and
wide for regions with low event density to promote smoothness. The details of the general algorithm
are described in the following paper:

Cranmer KS, Kernel Estimation in High-Energy Physics.
            Computer Physics Communications 136:198-207,2001 - e-Print Archive: hep ex/0011057
**/

#include "RooKeysPdf.h"
#include "RooFit.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooVDTHeaders.h"

#include <algorithm>
#include <cmath>

ClassImp(RooKeysPdf);

////////////////////////////////////////////////////////////////////////////////

  RooKeysPdf::RooKeysPdf() : 
    nEvents           (0),
    nBins             (1000), 
    mirrorLeft        (false), 
    mirrorRight       (false),
    asymLeft          (false), 
    asymRight         (false)
{
}

////////////////////////////////////////////////////////////////////////////////

RooKeysPdf::RooKeysPdf(const char* name, const char* title,
                       RooAbsReal& x, RooDataSet& data,
                       Mirror mirror, double rho, int nBins) :
  RooAbsPdf     (name,title),
  x             ("x","observable",this,x),
  nEvents       (0),
  nBins         (nBins),
  mirrorLeft    (mirror==MirrorLeft || mirror==MirrorBoth || mirror==MirrorLeftAsymRight),
  mirrorRight   (mirror==MirrorRight || mirror==MirrorBoth || mirror==MirrorAsymLeftRight),
  asymLeft      (mirror==MirrorAsymLeft || mirror==MirrorAsymLeftRight || mirror==MirrorAsymBoth),
  asymRight     (mirror==MirrorAsymRight || mirror==MirrorLeftAsymRight || mirror==MirrorAsymBoth),
  rho           (rho),
  varName       (x.GetName()),
  lookupTable   (std::vector<double>(nBins+1,0))
{
  RooAbsRealLValue& real= (RooRealVar&)(this->x.arg());
  lo = real.getMin();
  hi = real.getMax();
  binWidth = (hi-lo) / nBins;

  // form the lookup table
  LoadDataSet(data);
}

////////////////////////////////////////////////////////////////////////////////
/// cache stuff about x

RooKeysPdf::RooKeysPdf(const char* name, const char* title,
                       RooAbsReal& xpdf, RooRealVar& xdata, RooDataSet& data,
                       Mirror mirror, double rho, int nBins) :
  RooAbsPdf     (name,title),
  x             ("x","Observable",this,xpdf),
  nEvents       (0),
  nBins         (nBins),
  mirrorLeft    (mirror==MirrorLeft || mirror==MirrorBoth || mirror==MirrorLeftAsymRight),
  mirrorRight   (mirror==MirrorRight || mirror==MirrorBoth || mirror==MirrorAsymLeftRight),
  asymLeft      (mirror==MirrorAsymLeft || mirror==MirrorAsymLeftRight || mirror==MirrorAsymBoth),
  asymRight     (mirror==MirrorAsymRight || mirror==MirrorLeftAsymRight || mirror==MirrorAsymBoth),
  rho           (rho),
  varName       (xdata.GetName()),
  lookupTable   (std::vector<double>(nBins+1,0))
{
  RooAbsRealLValue& real= (RooRealVar&)(xdata);
  lo = real.getMin();
  hi = real.getMax();
  binWidth = (hi-lo) / nBins;
  /* Trick: we actually divide the hi-lo range into nBins. However,
   * when the time comes to do linear extrapolation, if we need the 
   * estimated value for a point that belongs to bin nBins-1, we also
   * need the value at point hi. This corresponds to the limit of the
   * bins nBin-1 and nBin (indexing begins from 0). So we need one extra bin.
   * Check the evaluate function for a better understanding.
   */

  // form the lookup table
  LoadDataSet(data);
}

////////////////////////////////////////////////////////////////////////////////

RooKeysPdf::RooKeysPdf(const RooKeysPdf& other, const char* name):
  RooAbsPdf     (other,name), 
  x             ("x",this,other.x), 
  nEvents       (other.nEvents),
  nBins         (other.nBins),
  lookupTable   (other.lookupTable), //copy the lookup table
  mirrorLeft    (other.mirrorLeft), 
  mirrorRight   (other.mirrorRight),
  asymLeft      (other.asymLeft), 
  asymRight     (other.asymRight),
  rho           (other.rho),
  varName       (other.varName),
  lo            (other.lo),
  binWidth      (other.binWidth)
{
}

////////////////////////////////////////////////////////////////////////////////
RooKeysPdf::~RooKeysPdf()
{
}

////////////////////////////////////////////////////////////////////////////////

void RooKeysPdf::LoadDataSet( RooDataSet& data) {
  nEvents = data.numEntries(); // before mirroring
  const size_t duplicateFactor = 1 +mirrorLeft +mirrorRight;
  nTotalEvents = duplicateFactor*nEvents; // after mirroring
  dataArr.resize(nTotalEvents);
  adaptedWidthArr.resize(nTotalEvents);
  
  auto xFromDataSet = static_cast<RooRealVar*>(data.get()->find(x.arg()));
  if (!xFromDataSet)
    throw std::runtime_error("Couldn't find x, Blabla");
  data.attachBuffers(RooArgSet(*xFromDataSet));

  RooSpan<const double> pointData = xFromDataSet->getValBatch(0, nEvents);
  RooSpan<const double> weightData = data.getWeightBatch(0, nEvents);
  if (pointData.empty())
    throw std::runtime_error("No data to construct density from.");
    
  // x0, x1, x2 are the 0th, 1st and 2nd momentums
  double x0=0.0, x1=0.0, x2=0.0, sumWgt=0.0;
    
  for (size_t i=0, iii=0; i<nEvents; i++, iii+=duplicateFactor) {
    // i for reading from the spans, iii for writing to the arrays  
    const double x = pointData[i]; 
    const double w = 1; //weightData[i]; -------------------------!!! To be fixed
    
    dataArr[iii].x = x;
    dataArr[iii].w = w;
    x0 += w;
    x1 += w*x;
    x2 += w*x*x;

    if (mirrorLeft && mirrorRight) {
      dataArr[iii+1].x = 2*lo-x;
      dataArr[iii+2].x = 2*hi-x;
      dataArr[iii+1].w = dataArr[iii+2].w = w;
    } else if (mirrorLeft) {
      dataArr[iii+1].x = 2*lo-x;
      dataArr[iii+1].w = w;
    } else if (mirrorRight) {
      dataArr[iii+1].x = 2*hi-x;
      dataArr[iii+1].w = w;
    }
  }
    
  // sort the entire data set so that values of x are increasing
  std::sort(dataArr.begin(), dataArr.end(), Data::compare);
    
  sumWgt = duplicateFactor*x0;
  double mean = x1/x0;
  double sigma = std::sqrt(x2/x0 -mean*mean);
  double h = std::pow(nTotalEvents,-0.2)*sigma*rho*1.05922384104881225329; //(4/3)^(1/5)
  double hmin = h*0.14142135623730950488; // sqrt(2)/10
  double norm = h/std::sqrt(sigma)/ 3.46410161513775458705; //3.5449077018110320546; // 2sqrt(pi)
  
  
  printf("New =================================================\n");
  
  //~ for (auto i: {mean,sigma,h,hmin,norm}) printf("%lf ",i);
  //~ printf("\n");

  unsigned int __start=0, __end=0; //helper variables only used in gaussian()
  for(size_t i=0; i<nTotalEvents; ++i) {
    adaptedWidthArr[i] = norm/std::sqrt( gaussian(dataArr[i].x,h,__start,__end) );
    if (adaptedWidthArr[i]<hmin) adaptedWidthArr[i]=hmin;
    //~ printf("%lf ",adaptedWidthArr[i]);
  }
  //~ printf("\n");

  // The idea below is that beyond sigmaLowLimit sigma, the value of the exponential
  // in the Gaussian is well below the machine precision of a double, so it
  // does not contribute any more. That way, we can limit how many bins of the
  // binned approximation in lookupTable we have to touch when filling it.
  for (int i=0; i<nBins+1; ++i) {
    lookupTable[i] = 0.0;
  }
  //~ printf("SigmaLowLimit=%lf\n",sigmaLowLimit);
  for(size_t i=0; i<nTotalEvents; ++i) {
    fillLookupTable(dataArr[i].x, dataArr[i].w, adaptedWidthArr[i], +1);
    if (asymLeft)  fillLookupTable(2*lo -dataArr[i].x, dataArr[i].w, adaptedWidthArr[i], -1);
    if (asymRight) fillLookupTable(2*hi -dataArr[i].x, dataArr[i].w, adaptedWidthArr[i], -1);
  }
  for (int i=0; i<nBins+1; ++i) {
    lookupTable[i] /= 2.50662827463100050242 * sumWgt; //sqrt(2pi)
    //~ printf(i%10 ? "%lf " : "%lf\n",lookupTable[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////

double RooKeysPdf::evaluate() const {
  const double realX = x;
  int i = static_cast<int>( (realX-lo)/binWidth );
  if (i<0) {
//     cerr << "got point below lower bound:"
//     << double(x) << " < " << lo
//     << " -- performing linear extrapolation..." << endl;
    i=0;
  }
  if (i>nBins-1) {
//     cerr << "got point above upper bound:"
//     << double(x) << " > " << hi
//     << " -- performing linear extrapolation..." << endl;
    i=nBins-1;
  }
  double dx = (realX-lo)/binWidth -i;
  
  // for now do simple linear interpolation.
  // one day replace by splines...
  double ret = lookupTable[i] +dx*(lookupTable[i+1]-lookupTable[i]) ;
  if (ret<0) ret=0;
  printf("%lf\n",ret);
  return ret;
}

int RooKeysPdf::getAnalyticalIntegral(
   RooArgSet& allVars, RooArgSet& analVars, const char* /* rangeName */) const
{
  if (matchArgs(allVars, analVars, x)) return 1;
  return 0;
}

double RooKeysPdf::analyticalIntegral(int code, const char* rangeName) const
{
  assert(1 == code);
  // this code is based on lookupTable and uses linear interpolation, just as
  // evaluate(); integration is done using the trapez rule
  const double xmin = std::max(lo, x.min(rangeName));
  const double xmax = std::min(hi, x.max(rangeName));
  const int imin = static_cast<int>( (xmin-lo)/binWidth );
  const int imax = std::min( static_cast<int>( (xmax-lo)/binWidth ), nBins-1);
  double sum = 0.0;
  // sum up complete bins in middle
  if (imin + 1 < imax)
    sum += lookupTable[imin + 1] + lookupTable[imax];
  for (int i = imin + 2; i < imax; ++i)
    sum += 2. * lookupTable[i];
  sum *= binWidth * 0.5;
  // treat incomplete bins
  const double dxmin = (xmin - (lo + imin * binWidth)) / binWidth;
  const double dxmax = (xmax - (lo + imax * binWidth)) / binWidth;
  if (imin < imax) {
    // first bin
    sum += binWidth * (1. - dxmin) * 0.5 * (lookupTable[imin + 1] +
   lookupTable[imin] + dxmin *
   (lookupTable[imin + 1] - lookupTable[imin]));
    // last bin
    sum += binWidth * dxmax * 0.5 * (lookupTable[imax] +
   lookupTable[imax] + dxmax *
   (lookupTable[imax + 1] - lookupTable[imax]));
  } else if (imin == imax) {
    // first bin == last bin
    sum += binWidth * (dxmax - dxmin) * 0.5 * (
   lookupTable[imin] + dxmin *
   (lookupTable[imin + 1] - lookupTable[imin]) +
   lookupTable[imax] + dxmax *
   (lookupTable[imax + 1] - lookupTable[imax]));
  }
  return sum;
}

int RooKeysPdf::getMaxVal(const RooArgSet& vars) const
{
  if (vars.contains(*x.absArg())) return 1;
  return 0;
}

double RooKeysPdf::maxVal(int code) const
{
  assert(1 == code);
  double max = -std::numeric_limits<double>::max();
  for (int i = 0; i < nBins; ++i)
    if (max < lookupTable[i]) max = lookupTable[i];
  return max;
}

////////////////////////////////////////////////////////////////////////////////

double RooKeysPdf::gaussian(double x, double sigma, unsigned int& start, unsigned int& end) const {
  /* The value computed for each point and added to ret is only significant
   * for the points within the range [x-sigmaLowLimit*sigma, x+sigmaLowLimit*sigma].
   * For the rest points, the value is zero, due to machine precision limits.
   * start will be the index from first point to lie within this range.
   * end will be the index after the last point to lie within this range.
   * The data is sorted, so two variables that constantly increase will do the job.
   */
  while ( start < nTotalEvents 
       && dataArr[start].x < x -sigmaLowLimit*sigma) {
    start++;
  }
  while ( end < nTotalEvents
       && dataArr[end].x <= x +sigmaLowLimit*sigma) {
    end++;
  }
  
  double ret=0;
  const double halfOverSigma2 = -0.5/(sigma*sigma);
  for (unsigned int i=start; i<end; i++) {
    const double r = x-dataArr[i].x;
    ret += _rf_fast_exp(halfOverSigma2*r*r);
  }
  return ret / (sigma*nTotalEvents*2.50662827463100050242); //sqrt(2pi)
}

////////////////////////////////////////////////////////////////////////////////

void RooKeysPdf::fillLookupTable(double x, double weight, double adaptedWidth, double sign) {
    //~ printf("%lf %lf %lf\n",x,weight,adaptedWidth);
    int binlo = static_cast<int>( (x -sign*sigmaLowLimit*adaptedWidth -lo) / binWidth );
    if (binlo >= nBins) {
      //~ printf("dropped\n");
      return;
    }
    if (binlo < 0) binlo = 0;
    
    int binhi = static_cast<int>( (x +sign*sigmaLowLimit*adaptedWidth -lo) / binWidth );
    if (binhi < 0) {
      //~ printf("dropped\n");
      return;
    }
    if (binhi > nBins) binhi = nBins;
    //we well want to write to the extra bin
    
    const double startOfBinlo = lo +binlo*binWidth;
    double dist = (startOfBinlo-x) / adaptedWidth / 1.4142135623730950488;
    const double distInc = binWidth / adaptedWidth / 1.4142135623730950488; //sqrt(2)
    const double weightratio = weight / adaptedWidth;
    
    //~ printf("%d %d\n",binlo,binhi);
    for (int i=binlo; i<=binhi; i++) {
      lookupTable[i] += sign*weightratio*_rf_fast_exp( -dist*dist );
      dist += distInc;
    }
}
