/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   RW, Ruddick William  UC Colorado        wor@slac.stanford.edu           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooBukinPdf
    \ingroup Roofit

The RooBukinPdf implements the NovosibirskA function. For the parameters, see
RooBukinPdf().

Credits:
May 26, 2003.
A.Bukin, Budker INP, Novosibirsk

\image html RooBukin.png
http://www.slac.stanford.edu/BFROOT/www/Organization/CollabMtgs/2003/detJuly2003/Tues3a/bukin.ps
**/

#include "RooBukinPdf.h"
#include "RooRealVar.h"
#include "BatchHelpers.h"
#include "RooVDTHeaders.h"

#include <cmath>

using namespace std;

ClassImp(RooBukinPdf);

////////////////////////////////////////////////////////////////////////////////
/// Construct a Bukin PDF.
/// \param name  The name of the PDF for RooFit's bookeeping.
/// \param title The title for e.g. plotting it.
/// \param _x    The variable.
/// \param _Xp   The peak position.
/// \param _sigp The peak width as FWHM divided by 2*sqrt(2*log(2))=2.35
/// \param _xi   Peak asymmetry. Use values around 0.
/// \param _rho1 Left tail. Use slightly negative starting values.
/// \param _rho2 Right tail. Use slightly positive starting values.
RooBukinPdf::RooBukinPdf(const char *name, const char *title,
          RooAbsReal& _x,    RooAbsReal& _Xp,
          RooAbsReal& _sigp, RooAbsReal& _xi,
          RooAbsReal& _rho1, RooAbsReal& _rho2) :
  // The two addresses refer to our first dependent variable and
  // parameter, respectively, as declared in the rdl file
  RooAbsPdf(name, title),
  x("x","x",this,_x),
  Xp("Xp","Xp",this,_Xp),
  sigp("sigp","sigp",this,_sigp),
  xi("xi","xi",this,_xi),
  rho1("rho1","rho1",this,_rho1),
  rho2("rho2","rho2",this,_rho2)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy a Bukin PDF.
RooBukinPdf::RooBukinPdf(const RooBukinPdf& other, const char *name):
  RooAbsPdf(other,name),
  x("x",this,other.x),
  Xp("Xp",this,other.Xp),
  sigp("sigp",this,other.sigp),
  xi("xi",this,other.xi),
  rho1("rho1",this,other.rho1),
  rho2("rho2",this,other.rho2)

{
}

////////////////////////////////////////////////////////////////////////////////
/// Implementation

Double_t RooBukinPdf::evaluate() const
{
  const double consts = 2*sqrt(2*log(2.0));
  double r1=0,r2=0,r3=0,r4=0,r5=0,hp=0;
  double x1 = 0,x2 = 0;
  double fit_result = 0;

  hp=sigp*consts;
  r3=log(2.);
  r4=sqrt(xi*xi+1);
  r1=xi/r4;

  if(fabs(xi) > exp(-6.)){
    r5=xi/log(r4+xi);
  }
  else
    r5=1;

  x1 = Xp + (hp / 2) * (r1-1);
  x2 = Xp + (hp / 2) * (r1+1);

  //--- Left Side
  if(x < x1){
    r2=rho1*(x-x1)*(x-x1)/(Xp-x1)/(Xp-x1)-r3 + 4 * r3 * (x-x1)/hp * r5 * r4/(r4-xi)/(r4-xi);
  }


  //--- Center
  else if(x < x2) {
    if(fabs(xi) > exp(-6.)) {
      r2=log(1 + 4 * xi * r4 * (x-Xp)/hp)/log(1+2*xi*(xi-r4));
      r2=-r3*r2*r2;
    }
    else{
      r2=-4*r3*(x-Xp)*(x-Xp)/hp/hp;
    }
  }


  //--- Right Side
  else {
    r2=rho2*(x-x2)*(x-x2)/(Xp-x2)/(Xp-x2)-r3 - 4 * r3 * (x-x2)/hp * r5 * r4/(r4+xi)/(r4+xi);
  }

  if(fabs(r2) > 100){
    fit_result = 0;
  }
  else{
    //---- Normalize the result
    fit_result = exp(r2);
  }

  return fit_result;
}

////////////////////////////////////////////////////////////////////////////////////////

namespace BukinBatchEvaluate {
//Author: Emmanouil Michalainas, CERN 26 JULY 2019  

template<class Tx, class TXp, class TSigp, class Txi, class Trho1, class Trho2>
void compute(  size_t batchSize,
               double * __restrict__ output,
               Tx X, TXp XP, TSigp SP, Txi XI, Trho1 R1, Trho2 R2)
{
  const double r3 = log(2.0);
  const double r6 = exp(-6.0);
  const double r7 = 2*sqrt(2*log(2.0));
  
  for (size_t i=0; i<batchSize; i++) {
    const double r1 = XI[i]/sqrt(XI[i]*XI[i]+1);
    const double r4 = sqrt(XI[i]*XI[i]+1);
    const double hp = 1 / (SP[i]*r7);
    const double x1 = XP[i] + 0.5*SP[i]*r7*(r1-1);
    const double x2 = XP[i] + 0.5*SP[i]*r7*(r1+1);
    
    double r5 = 1.0;
    if (XI[i]>r6 || XI[i]<-r6) r5 = XI[i]/log(r4+XI[i]);
    
    double factor=1, y=X[i]-x1, Yp=XP[i]-x1, yi=r4-XI[i], rho=R1[i];
    if (X[i]>=x2) {
      factor = -1;
      y = X[i]-x2;
      Yp = XP[i]-x2;
      yi = r4+XI[i];
      rho = R2[i];
    }
    
    output[i] = rho*y*y/Yp/Yp -r3 + factor*4*r3*y*hp*r5*r4/yi/yi;
    if (X[i]>=x1 && X[i]<x2) {
      output[i] = vdt::fast_log(1 + 4*XI[i]*r4*(X[i]-XP[i])*hp) / vdt::fast_log(1 +2*XI[i]*( XI[i]-r4 ));
      output[i] *= -output[i]*r3;
    }
    if (X[i]>=x1 && X[i]<x2 && XI[i]<r6 && XI[i]>-r6) {
      output[i] = -4*r3*(X[i]-XP[i])*(X[i]-XP[i])*hp*hp;
    }
  }
  for (size_t i=0; i<batchSize; i++) {
    output[i] = vdt::fast_exp(output[i]);
  }
}
};


RooSpan<double> RooBukinPdf::evaluateBatch(std::size_t begin, std::size_t batchSize) const {
  using namespace BatchHelpers;
  using namespace BukinBatchEvaluate;
    
  EvaluateInfo info = getInfo( {&x,&Xp,&sigp,&xi,&rho1,&rho2}, begin, batchSize );
  auto output = _batchData.makeWritableBatchUnInit(begin, info.size);
  auto xData = x.getValBatch(begin, info.size);
  if (info.nBatches == 0) {
    throw std::logic_error("Requested a batch computation, but no batch data available.");
  }
  else if (info.nBatches==1 && !xData.empty()) {
    compute(info.size, output.data(), xData.data(), 
    BracketAdapter<double> (Xp), 
    BracketAdapter<double> (sigp), 
    BracketAdapter<double> (xi), 
    BracketAdapter<double> (rho1), 
    BracketAdapter<double> (rho2));
  }
  else {
    compute(info.size, output.data(), 
    BracketAdapterWithMask (x,x.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (Xp,Xp.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (sigp,sigp.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (xi,xi.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (rho1,rho1.getValBatch(begin,batchSize)), 
    BracketAdapterWithMask (rho2,rho2.getValBatch(begin,batchSize)));
  }
  return output;
}
