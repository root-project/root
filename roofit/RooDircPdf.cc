/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooDircPdf.hh"
#include "RooFitCore/RooAbsReal.hh"

ClassImp(RooDircPdf)

RooDircPdf::RooDircPdf(const char *name, const char *title, 		       
		       RooAbsReal& _drcMtm,      RooAbsReal& _thetaC,
		       RooAbsReal& _refraction,  RooAbsReal& _mass, 
		       RooAbsReal& _otherMass,   RooAbsReal& _coreMean,  RooAbsReal& _coreSigma, 
		       RooAbsReal& _tailMean,    RooAbsReal& _tailSigma, RooAbsReal& _relNorm,
		       Bool_t milliRadians) :
  RooAbsPdf(name,title), _milliRadians(milliRadians),
  drcMtm    ("drcMtm"    , "DIRC momentum"         , this, _drcMtm),
  thetaC    ("thetaC"    , "DIRC thetaC"           , this, _thetaC),
  refraction("refraction", "Refraction"            , this, _refraction),
  mass      ("mass"      , "Mass"                  , this, _mass),
  otherMass ("otherMass" , "Other Mass"            , this, _otherMass),
  coreMean  ("coreMean"  , "Core Mean"             , this, _coreMean),
  coreSigma ("coreSigma" , "Core Sigma"            , this, _coreSigma),
  tailMean  ("tailMean"  , "Tail Mean"             , this, _tailMean),
  tailSigma ("tailSigma" , "Tails Sigma"           , this, _tailSigma),
  relNorm   ("relNorm"   , "Relative Normalization", this, _relNorm)
{
}


RooDircPdf::RooDircPdf(const RooDircPdf& other, const char* name) : 
  RooAbsPdf(other,name), _milliRadians(other._milliRadians),
  drcMtm    ("drcMtm"    , this, other.drcMtm),
  thetaC    ("thetaC"    , this, other.thetaC),
  refraction("refraction", this, other.refraction),
  mass      ("mass"      , this, other.mass),
  otherMass ("otherMass" , this, other.otherMass),
  coreMean  ("coreMean"  , this, other.coreMean),
  coreSigma ("coreSigma" , this, other.coreSigma),
  tailMean  ("tailMean"  , this, other.tailMean),
  tailSigma ("tailSigma" , this, other.tailSigma),
  relNorm   ("relNorm"   , this, other.relNorm)
{
}


// WVE - these PDFs need to be expressed as function 
// of (cosTheta,drcMtm) in their constructors
//
//   Double_t cosTheta = cos(trkTheta);
//   coreMean = _coreMeanFun.Eval(cosTheta, drcMtm);
//   coreSigma = _coreSigmaFun.Eval(cosTheta, drcMtm);
//   tailMean = _tailMeanFun.Eval(cosTheta, drcMtm);  
//   tailSigma = _tailSigmaFun.Eval(cosTheta, drcMtm);
//   relNorm = _relNormFun.Eval(cosTheta, drcMtm); // (core area)/(core + tail areas)
//
//   NB: This PDF is not normalized! This is going to give problems.
//       RooDircPDF uses (via composition) at least three dependents
//
//       (trkTheta,drcMtm,thetaC), so integration is non-trivial
// 
//       Unfortunately it wouldn't work with any RFC technology,
//       3D numerical integration is not implemented (yet) and
//       analytical integration is not possible because the above
//       functions cannot be LValues (>1 dependent)
//   
//       We currently have no mechanism to bypass the normalization requirement...


Double_t RooDircPdf::evaluate() const
{
  static Double_t root2 = sqrt(2);  
  static Double_t rootpiby2 = sqrt(atan2(0.0,-1.0)/2.0) ;

  Double_t coreMeanTmp  = coreMean,
           tailMeanTmp  = tailMean ;
   
  Double_t drcMtmSq = drcMtm*drcMtm;
  Double_t value = refraction*drcMtm;

  Double_t cosCoreAngle(0.0), cosTailAngle(0.0);
  Double_t expectedCore(0.0), expectedTail(0.0);

  Double_t milliFactor(1.0);
  if (_milliRadians == kTRUE) {
    milliFactor = 1000.0;
  }

  if (value != 0) { 
    cosCoreAngle = sqrt(drcMtmSq + mass*mass)/value;
    if (fabs(cosCoreAngle) <= 1.0) {
      expectedCore = milliFactor*acos(cosCoreAngle);
    }

    cosTailAngle = sqrt(drcMtmSq + otherMass*otherMass)/value;
    if (fabs(cosTailAngle) <= 1.0) {
      expectedTail = milliFactor*acos(cosTailAngle);
    }
  }

  coreMeanTmp += expectedCore;
  tailMeanTmp += expectedTail;

  Double_t coreScale = root2*coreSigma;
  Double_t coreNorm(0.0), corePart(0.0);
  Double_t relNorm(0.0);
    
  if (coreSigma > 0.0) {
    Double_t erf1 = erf((thetaC.max() - coreMeanTmp)/coreScale);
    Double_t erf2 = erf((thetaC.min() - coreMeanTmp)/coreScale);
    coreNorm = rootpiby2*coreSigma*(erf1-erf2);

    if (coreNorm > 0.0) {
      coreNorm = relNorm/coreNorm;
    }

    Double_t coreArg = thetaC - coreMeanTmp;
    corePart = exp(-0.5*(coreArg*coreArg)/(coreSigma*coreSigma))*coreNorm;
  }

  Double_t tailScale = root2*tailSigma;
  Double_t tailNorm(0.0), tailPart(0.0);

  if (tailSigma > 0.0) {
    Double_t erf1 = erf((thetaC.max() - tailMeanTmp)/tailScale);
    Double_t erf2 = erf((thetaC.min() - tailMeanTmp)/tailScale);
    tailNorm = rootpiby2*tailSigma*(erf1-erf2);

    if (tailNorm > 0.0) {
      tailNorm = (1.0 - relNorm)/tailNorm;
    }

    Double_t tailArg = thetaC - tailMeanTmp;
    tailPart = exp(-0.5*tailArg*tailArg/(tailSigma*tailSigma))*tailNorm;
  }

  Double_t result = corePart + tailPart;
  return result;


}



