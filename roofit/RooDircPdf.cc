/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooDircPdf.cc,v 1.11 2002/05/31 01:07:41 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   Lei Zhang, University of Colorado, zhanglei@slac.stanford.edu
 * History:
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *   17-Apr-2002 LZ Implement event generator for RooDircPdf
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

//#include "BaBar/BaBar.hh"
#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooDircPdf.hh"
#include "RooFitCore/RooAbsReal.hh"

ClassImp(RooDircPdf)
  ;

RooDircPdf::RooDircPdf(const char *name, const char *title, 		       
		       RooAbsReal& _drcMtm, RooAbsReal& _thetaC,
		       RooAbsReal& _refraction,
		       RooAbsReal& _mass, RooAbsReal& _otherMass,
		       RooAbsReal& _coreMean, RooAbsReal& _coreSigma, 
		       RooAbsReal& _tailMean, RooAbsReal& _tailSigma,
		       RooAbsReal& _relNorm,
		       Double_t minMtmVal, Double_t minThetaCVal,
		       Bool_t milliRadians,
		       RooAbsPdf *_pThetaPdf):
  RooAbsPdf(name,title), _minMtmVal(minMtmVal), _minThetaCVal(minThetaCVal), 
  _milliRadians(milliRadians), pThetaPdf(_pThetaPdf), pThetaCache(0),
  cachePtr(0), _MaxP(0), _selfGen(kFALSE),
  drcMtm    ("drcMtm"    , "DIRC momentum"         , this, _drcMtm),
  thetaC    ("thetaC"    , "DIRC thetaC"           , this, _thetaC),
  theta     ("thetaC"    , "DIRC thetaC"           , this, _thetaC),
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

RooDircPdf::RooDircPdf(const char *name, const char *title, 		       
		       RooAbsReal& _drcMtm, RooAbsReal& _thetaC,
		       RooAbsReal& _theta,
		       RooAbsReal& _refraction,
		       RooAbsReal& _mass, RooAbsReal& _otherMass,
		       RooAbsReal& _coreMean, RooAbsReal& _coreSigma, 
		       RooAbsReal& _tailMean, RooAbsReal& _tailSigma,
		       RooAbsReal& _relNorm,
		       Double_t minMtmVal, Double_t minThetaCVal,
		       Bool_t milliRadians,
		       RooAbsPdf *_pThetaPdf):
  RooAbsPdf(name,title), _minMtmVal(minMtmVal), _minThetaCVal(minThetaCVal), 
  _milliRadians(milliRadians), pThetaPdf(_pThetaPdf), pThetaCache(0),
  cachePtr(0), _MaxP(0), _selfGen(kFALSE),
  drcMtm    ("drcMtm"    , "DIRC momentum"         , this, _drcMtm),
  thetaC    ("thetaC"    , "DIRC thetaC"           , this, _thetaC),
  theta     ("theta",      "Polar angle"           , this, _theta),
  refraction("refraction", "Refraction"            , this, _refraction),
  mass      ("mass"      , "Mass"                  , this, _mass),
  otherMass ("otherMass" , "Other Mass"            , this, _otherMass),
  coreMean  ("coreMean"  , "Core Mean"             , this, _coreMean),
  coreSigma ("coreSigma" , "Core Sigma"            , this, _coreSigma),
  tailMean  ("tailMean"  , "Tail Mean"             , this, _tailMean),
  tailSigma ("tailSigma" , "Tails Sigma"           , this, _tailSigma),
  relNorm   ("relNorm"   , "Relative Normalization", this, _relNorm)
{
  if (pThetaPdf) _selfGen=kTRUE;
}


RooDircPdf::RooDircPdf(const RooDircPdf& other, const char* name) : 
  RooAbsPdf(other,name), _minMtmVal(other._minMtmVal), 
  _minThetaCVal(other._minThetaCVal), _milliRadians(other._milliRadians),
  pThetaPdf(other.pThetaPdf),
  pThetaCache(0), cachePtr(0), _MaxP(other._MaxP), _selfGen(other._selfGen),
  drcMtm    ("drcMtm"    , this, other.drcMtm),
  thetaC    ("thetaC"    , this, other.thetaC),
  theta     ("theta"     , this, other.theta),
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

RooDircPdf::~RooDircPdf() {
  if (pThetaCache) delete pThetaCache;
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

Int_t RooDircPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const
{
  if (matchArgs(allVars, analVars, thetaC)) return 1;
  return 0;
}


Double_t RooDircPdf::analyticalIntegral(Int_t code) const
{
  assert(code==1);
  return 1;
}


Double_t RooDircPdf::evaluate() const
{
 
  // First test if we have a valid mtm range and Cerenkov angle values...
  if (drcMtm < _minMtmVal) {
    return 1.0;
  }

  if (thetaC < _minThetaCVal) {
    return 1.0;
  }

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

Bool_t RooDircPdf::isDirectGenSafe(const RooAbsArg& arg) const
{
  const char* eins=arg.GetName();
  const char* zwei=thetaC.GetName();
  if (strcmp(eins,zwei)==0) return kTRUE;
  if (_selfGen) return _selfGen;
  return RooAbsPdf::isDirectGenSafe(arg);
}

void RooDircPdf::initGenerator(Int_t code) {
  
  // find the _MaxP
  cout<<"Begin initGenerator"<<endl;
  for(Int_t i=0; i<minInitTrial; i++) {
    generateEvent(code);
  }
  cout<<"End initGenerator"<<endl;
}
  

Int_t RooDircPdf::getGenerator(const RooArgSet& directVars,
			       RooArgSet &generateVars, Bool_t staticInitOK)const
{
  //cout <<"we are in RooDircPdf::getGenerator line 1"<<endl;
  Int_t haveGen=0;
  if (matchArgs(directVars, generateVars, drcMtm)) haveGen=2;
  if (matchArgs(directVars, generateVars, thetaC)) haveGen=3;
  if (matchArgs(directVars, generateVars, theta)) haveGen=4;
  if ((!_selfGen)&&(haveGen!=3)) haveGen=0;
  return haveGen;
}

void RooDircPdf::generateEvent(Int_t code)
{
  if (code!=3) {
    if (!_selfGen) return;
    if ((!pThetaCache||cachePtr>=nCache) && pThetaPdf) {
      cachePtr=0;
      if (pThetaCache) {
	delete pThetaCache;
	pThetaCache=0;
      }
      pThetaCache=pThetaPdf->
	generate(RooArgSet(drcMtm.arg(), theta.arg()), nCache);
    }
    if (pThetaCache) {
      const RooArgSet *argset=pThetaCache->get(cachePtr++);
      if(argset) {
	drcMtm=((RooAbsReal*)(argset->find(drcMtm.arg().GetName())))->getVal();
	theta=((RooAbsReal*)(argset->find(theta.arg().GetName())))->getVal();
      }
    }
  }  

  // finally calculate thetaC
  Int_t trials(minTrial);
  while (trials--) {
    thetaC=(thetaC.max()-thetaC.min())*RooRandom::uniform()+thetaC.min();
    Double_t prob=evaluate();
    if (prob>_MaxP) {
      cout<<"Increasing "<<fName<<" maxProb from "<<_MaxP<<" to ";
      _MaxP=1.1*prob;
      cout<<_MaxP<<endl;
    }
    // accept or reject
    Double_t r=RooRandom::uniform();
    if(_MaxP*r < prob) break;
  }
  if (0==trials) {
    cout<<"Failed to find values after "<<minTrial<<" trials."
	<<" But still use the value of last trial, "<<thetaC.arg().getVal()
	<<endl;
  }
  
  return;
}
