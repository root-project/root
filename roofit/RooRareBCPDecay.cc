/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: 
 * Authors:
 *   PB, Paul Bloom, CU Boulder, bloom@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// Implement standard CP physics model using |lambda|^2 and Im(lambda).
// These variables are better suited to unbiased transformation to C and S
// via Im(lambda) = S/(1+C), |lambda|^2 = (1-c)/(1+C)

#include <iostream.h>
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRandom.hh"
#include "RooFitModels/RooRareBCPDecay.hh"

ClassImp(RooRareBCPDecay) 
;

RooRareBCPDecay::RooRareBCPDecay(const char *name, const char *title, 
			       RooRealVar& t, RooAbsCategory& tag,
			       RooAbsReal& tau, RooAbsReal& dm,
			       RooAbsReal& avgMistag, RooAbsReal& CPeigenval,
			       RooAbsReal& a, RooAbsReal& b,
			       RooAbsReal& effRatio, RooAbsReal& delMistag,
			       const RooResolutionModel& model, DecayType type) :
  RooConvolutedPdf(name,title,model,t), 
  _absLambdaSq("absLambdaSq","Magnitude squared of lambda",this,a),
  _imLambda("imLambda","Imaginary part of lambda",this,b),
  _CPeigenval("CPeigenval","CP eigen value",this,CPeigenval),
  _effRatio("effRatio","B0/B0bar efficiency ratio",this,effRatio),
  _avgMistag("avgMistag","Average mistag rate",this,avgMistag),
  _delMistag("delMistag","Delta mistag rate",this,delMistag),  
  _tag("tag","CP state",this,tag),
  _tau("tau","decay time",this,tau),
  _dm("dm","mixing frequency",this,dm),
  _t("t","time",this,t),
  _type(type),
  _genB0Frac(0)
{
  // Constructor
  switch(type) {
  case SingleSided:
    _basisExp = declareBasis("exp(-@0/@1)",RooArgList(tau,dm)) ;
    _basisSin = declareBasis("exp(-@0/@1)*sin(@0*@2)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case Flipped:
    _basisExp = declareBasis("exp(@0)/@1)",RooArgList(tau,dm)) ;
    _basisSin = declareBasis("exp(@0/@1)*sin(@0*@2)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case DoubleSided:
    _basisExp = declareBasis("exp(-abs(@0)/@1)",RooArgList(tau,dm)) ;
    _basisSin = declareBasis("exp(-abs(@0)/@1)*sin(@0*@2)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-abs(@0)/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  }
}


RooRareBCPDecay::RooRareBCPDecay(const RooRareBCPDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _absLambdaSq("absLambdaSq",this,other._absLambdaSq),
  _imLambda("imLambda",this,other._imLambda),
  _CPeigenval("CPeigenval",this,other._CPeigenval),
  _effRatio("effRatio",this,other._effRatio),
  _avgMistag("avgMistag",this,other._avgMistag),
  _delMistag("delMistag",this,other._delMistag),
  _tag("tag",this,other._tag),
  _tau("tau",this,other._tau),
  _dm("dm",this,other._dm),
  _t("t",this,other._t),
  _type(other._type),
  _basisExp(other._basisExp),
  _basisSin(other._basisSin),
  _basisCos(other._basisCos),
  _genB0Frac(other._genB0Frac)
{
  // Copy constructor
}



RooRareBCPDecay::~RooRareBCPDecay()
{
  // Destructor
}


Double_t RooRareBCPDecay::coefficient(Int_t basisIndex) const 
{
  // B0    : _tag = +1 
  // B0bar : _tag = -1 

  if (basisIndex==_basisExp) {
    //exp term: (1 -/+ dw)(1+a^2)/2 
    return (1 - _tag*_delMistag)*(1+_absLambdaSq)/2 ;
    // =    1 + _tag*deltaDil/2
  }

  if (basisIndex==_basisSin) {
    //sin term: +/- (1-2w)*ImLambda
    return -1*_tag*(1-2*_avgMistag)*_imLambda ;
    // =   _tag*avgDil * ...
  }
  
  if (basisIndex==_basisCos) {
    //cos term: +/- (1-2w)*(1-a^2)/2
    return -1*_tag*(1-2*_avgMistag)*(1-_absLambdaSq)/2 ;
    // =   -_tag*avgDil * ...
  } 
  
  return 0 ;
}



Int_t RooRareBCPDecay::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,_tag)) return 1 ;
  return 0 ;
}



Double_t RooRareBCPDecay::coefAnalyticalIntegral(Int_t basisIndex, Int_t code) const 
{
  switch(code) {
    // No integration
  case 0: return coefficient(basisIndex) ;

    // Integration over 'tag'
  case 1:
    if (basisIndex==_basisExp) {
      return (1+_absLambdaSq) ;
    }
    
    if (basisIndex==_basisSin || basisIndex==_basisCos) {
      return 0 ;
    }
  default:
    assert(0) ;
  }
    
  return 0 ;
}



Int_t RooRareBCPDecay::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars) const
{
  if (matchArgs(directVars,generateVars,_t,_tag)) return 2 ;  
  if (matchArgs(directVars,generateVars,_t)) return 1 ;  
  return 0 ;
}



void RooRareBCPDecay::initGenerator(Int_t code)
{
  if (code==2) {
    // Calculate the fraction of mixed events to generate
    Double_t sumInt = RooRealIntegral("sumInt","sum integral",*this,RooArgSet(_t.arg(),_tag.arg())).getVal() ;
    _tag = 1 ;
    Double_t b0Int = RooRealIntegral("mixInt","mix integral",*this,RooArgSet(_t.arg())).getVal() ;
    _genB0Frac = b0Int/sumInt ;
  }  
}



void RooRareBCPDecay::generateEvent(Int_t code)
{
  // Generate mix-state dependent
  if (code==2) {
    Double_t rand = RooRandom::uniform() ;
    _tag = (rand<=_genB0Frac) ? 1 : -1 ;
  }

  // Generate delta-t dependent
  while(1) {
    Double_t rand = RooRandom::uniform() ;
    Double_t tval(0) ;

    switch(_type) {
    case SingleSided:
      tval = -_tau*log(rand);
      break ;
    case Flipped:
      tval= +_tau*log(rand);
      break ;
    case DoubleSided:
      tval = (rand<=0.5) ? -_tau*log(2*rand) : +_tau*log(2*(rand-0.5)) ;
      break ;
    }

    // Accept event if T is in generated range
    Double_t maxDil = 1.0 ;
    Double_t al2 = _absLambdaSq ;
    Double_t maxAcceptProb = (1+al2) + fabs(maxDil*_imLambda) + fabs(maxDil*(1-al2)/2);        
    Double_t acceptProb    = (1+al2)/2*(1-_tag*_delMistag) 
                           - (_tag*(1-2*_avgMistag))*(_CPeigenval*_imLambda)*sin(_dm*tval) 
                           - (_tag*(1-2*_avgMistag))*(1-al2)/2*cos(_dm*tval);

    Bool_t accept = maxAcceptProb*RooRandom::uniform() < acceptProb ? kTRUE : kFALSE ;
    
    if (tval<_t.max() && tval>_t.min() && accept) {
      _t = tval ;
      break ;
    }
  }
  
}

