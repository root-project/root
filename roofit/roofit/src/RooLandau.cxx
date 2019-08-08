/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooLandau
    \ingroup Roofit

Landau distribution p.d.f
\image html RF_Landau.png "PDF of the Landau distribution."
**/

#include "RooLandau.h"
#include "TMath.h"
#include "RooFit.h"

#include "RooRandom.h"
#include "BatchHelpers.h"

#include "TError.h"

using namespace std;
using namespace BatchHelpers;

ClassImp(RooLandau);

////////////////////////////////////////////////////////////////////////////////

RooLandau::RooLandau(const char *name, const char *title, RooAbsReal& _x, RooAbsReal& _mean, RooAbsReal& _sigma) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{
}

////////////////////////////////////////////////////////////////////////////////

RooLandau::RooLandau(const RooLandau& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooLandau::evaluate() const
{
  return TMath::Landau(x, mean, sigma);
}

////////////////////////////////////////////////////////////////////////////////

namespace LandauBatchEvaluate {
  
//Author: Emmanouil Michalainas, CERN 26 JULY 2019
/* Actual computation of Landau(x,mean,sigma) in a vectorization-friendly way
 * Code copied from function landau_pdf (math/mathcore/src/PdfFuncMathCore.cxx)
 * and rewritten to take advantage for the most popular case
 * which is -1 < (x-mean)/sigma < 1. The rest cases are handled in scalar way
 */  
   
constexpr double p1[5] = {0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253};
constexpr double q1[5] = {1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063};

constexpr double p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211};
constexpr double q2[5] = {1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714};

constexpr double p3[5] = {0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101};
constexpr double q3[5] = {1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675};

constexpr double p4[5] = {0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186};
constexpr double q4[5] = {1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511};

constexpr double p5[5] = {1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910};
constexpr double q5[5] = {1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357};

constexpr double p6[5] = {1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109};
constexpr double q6[5] = {1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939};

constexpr double a1[3] = {0.04166666667,-0.01996527778, 0.02709538966};
constexpr double a2[2] = {-1.845568670,-4.284640743};
  
template<class Tx, class TMean, class TSig>
void compute(RooSpan<double> output, Tx x, TMean mean, TSig sigma) {
  const int n = output.size();
  const double NaN = std::nan("");
  constexpr int block=256;
  double v[block];
  
  for (int i=0; i<n; i+=block) { //CHECK_VECTORISE
    const int stop = (i+block < n) ? block : n-i ; 
    
    for (int j=0; j<stop; j++) { //CHECK_VECTORISE
      v[j] = (x[i+j]-mean[i+j]) / sigma[i+j];
      output[i+j] = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v[j])*v[j])*v[j])*v[j]) /
                 (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v[j])*v[j])*v[j])*v[j]);
    }
    
    for (int j=0; j<stop; j++) { //CHECK_VECTORISE
      const bool mask = sigma[i+j] > 0;
      /*  comparison with NaN will give result false, so the next
       *  loop won't affect output, for cases where sigma <=0
       */
      if (!mask) v[j] = NaN; 
      output[i+j] *= mask;
    }
  
    double u, ue, us;
    for (int j=0; j<stop; j++) { //CHECK_VECTORISE
      // if branch written in way to quickly process the most popular case -1 <= v[j] < 1
      if (v[j] >= 1) {
        if (v[j] < 5) {
          output[i+j] = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v[j])*v[j])*v[j])*v[j]) /
                   (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v[j])*v[j])*v[j])*v[j]);
        } else if (v[j] < 12) {
            u   = 1/v[j];
            output[i+j] = u*u*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u) /
                    (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
        } else if (v[j] < 50) {
            u   = 1/v[j];
            output[i+j] = u*u*(p5[0]+(p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)*u) /
                     (q5[0]+(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)*u);
        } else if (v[j] < 300) {
            u   = 1/v[j];
            output[i+j] = u*u*(p6[0]+(p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)*u) /
                     (q6[0]+(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)*u);
        } else {
            u   = 1 / (v[j] -v[j]*std::log(v[j])/(v[j]+1) );
            output[i+j] = u*u*(1 +(a2[0] +a2[1]*u)*u );
        }
      } else if (v[j] < -1) {
          if (v[j] >= -5.5) {
            u   = std::exp(-v[j]-1);
            output[i+j] = std::exp(-u)*std::sqrt(u)*
              (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v[j])*v[j])*v[j])*v[j])/
              (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v[j])*v[j])*v[j])*v[j]);
          } else  {
              u   = std::exp(v[j]+1.0);
              if (u < 1e-10) output[i+j] = 0.0;
              else {
                ue  = std::exp(-1/u);
                us  = std::sqrt(u);
                output[i+j] = 0.3989422803*(ue/us)*(1+(a1[0]+(a1[1]+a1[2]*u)*u)*u);
              }
          } 
        }
    }
  }
}
};

////////////////////////////////////////////////////////////////////////////////

/// Compute \f$ Landau(x,mean,sigma) \f$ in batches.
/// The local proxies {x, mean, sigma} will be searched for batch input data,
/// and if found, the computation will be batched over their
/// values. If batch data are not found for one of the proxies, the proxies value is assumed to
/// be constant over the batch.
/// \param[in] batchIndex Index of the batch to be computed.
/// \param[in] batchSize Size of each batch. The last batch may be smaller.
/// \return A span with the computed values.

RooSpan<double> RooLandau::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace LandauBatchEvaluate;
  auto xData = x.getValBatch(begin, batchSize);
  auto meanData = mean.getValBatch(begin, batchSize);
  auto sigmaData = sigma.getValBatch(begin, batchSize);
  
  batchSize = findSize({ xData, meanData, sigmaData });
  auto output = _batchData.makeWritableBatchUnInit(begin, batchSize);
  
  const bool batchX = !xData.empty();
  const bool batchMean = !meanData.empty();
  const bool batchSigma = !sigmaData.empty();
  if (batchX && !batchMean && !batchSigma ) {
    compute(output, xData, BracketAdapter<RooRealProxy>(mean), BracketAdapter<RooRealProxy>(sigma));
  }
  else if (!batchX && batchMean && !batchSigma ) {
    compute(output, BracketAdapter<RooRealProxy>(x), meanData, BracketAdapter<RooRealProxy>(sigma));
  }
  else if (batchX && batchMean && !batchSigma ) {
    compute(output, xData, meanData, BracketAdapter<RooRealProxy>(sigma));
  }
  else if (!batchX && !batchMean && batchSigma ) {
    compute(output, BracketAdapter<RooRealProxy>(x), BracketAdapter<RooRealProxy>(mean), sigmaData);
  }
  else if (batchX && !batchMean && batchSigma ) {
    compute(output, xData, BracketAdapter<RooRealProxy>(mean), sigmaData);
  }
  else if (!batchX && batchMean && batchSigma ) {
    compute(output, BracketAdapter<RooRealProxy>(x), meanData, sigmaData);
  }
  else if (batchX && batchMean && batchSigma ) {
    compute(output, xData, meanData, sigmaData);
  }
  else{
    throw std::logic_error("Requested a batch computation, but no batch data available.");
  }

  return output;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooLandau::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

void RooLandau::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;
  Double_t xgen ;
  while(1) {
    xgen = RooRandom::randomGenerator()->Landau(mean,sigma);
    if (xgen<x.max() && xgen>x.min()) {
      x = xgen ;
      break;
    }
  }
  return;
}
