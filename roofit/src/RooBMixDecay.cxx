/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooBMixDecay.cc,v 1.6 2001/10/30 07:38:53 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// 

#include <iostream.h>
#include "RooFitCore/RooRealVar.hh"
#include "RooFitModels/RooBMixDecay.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooBMixDecay) 
;


RooBMixDecay::RooBMixDecay(const char *name, const char *title, 
			   RooRealVar& t, RooAbsCategory& tag,
			   RooAbsReal& tau, RooAbsReal& dm,
			   RooAbsReal& mistag, const RooResolutionModel& model, 
			   DecayType type) :
  RooConvolutedPdf(name,title,model,t), 
  _mistag("mistag","Mistag rate",this,mistag),
  _tag("tag","Mixing state",this,tag),_type(type),
  _tau("tau","Mixing life time",this,tau),
  _dm("dm","Mixing frequency",this,dm),
  x("x","time",this,t)
{
  // Constructor
  switch(type) {
  case SingleSided:
    _basisExp = declareBasis("exp(-@0/@1)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case Flipped:
    _basisExp = declareBasis("exp(@0)/@1)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(@0/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  case DoubleSided:
    _basisExp = declareBasis("exp(-abs(@0)/@1)",RooArgList(tau,dm)) ;
    _basisCos = declareBasis("exp(-abs(@0)/@1)*cos(@0*@2)",RooArgList(tau,dm)) ;
    break ;
  }
}


RooBMixDecay::RooBMixDecay(const RooBMixDecay& other, const char* name) : 
  RooConvolutedPdf(other,name), 
  _mistag("mistag",this,other._mistag),
  _tag("tag",this,other._tag),
  _tau("tau",this,other._tau),
  _dm("dm",this,other._dm),
  x("x",this,other.x),
  _basisExp(other._basisExp),
  _basisCos(other._basisCos),
  _type(other._type)
{
  // Copy constructor
}



RooBMixDecay::~RooBMixDecay()
{
  // Destructor
}


Double_t RooBMixDecay::coefficient(Int_t basisIndex) const 
{
  if (basisIndex==_basisExp) {
    return 1 ;
  }

  if (basisIndex==_basisCos) {
    return _tag*(1-2*_mistag) ;
  }
  
  return 0 ;
}



Int_t RooBMixDecay::getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,_tag)) return 1 ;
  return 0 ;
}



Double_t RooBMixDecay::coefAnalyticalIntegral(Int_t basisIndex, Int_t code) const 
{
  switch(code) {
    // No integration
  case 0: return coefficient(basisIndex) ;

    // Integration over 'tag'
  case 1:
    if (basisIndex==_basisExp) {
      return 2.0 ;
    }    
    if (basisIndex==_basisCos) {
      return 0.0 ;
    }
  default:
    assert(0) ;
  }
    
  return 0 ;
}


Int_t RooBMixDecay::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars) const
{
  if (matchArgs(directVars,generateVars,x,_tag)) return 2 ;  
  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}




void RooBMixDecay::generateEvent(Int_t code)
{
  // Generate mix-state dependent
  if (code==2) {
    Double_t taudm2 = _tau*_tau*_dm*_dm ;
    Double_t mixedFrac = 0.5 * ( 1 - (1-2*_mistag)/(1+taudm2)) ;
    Double_t rand = RooRandom::uniform() ;
    _tag = (rand<=mixedFrac) ? -1 : 1 ;
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
    Double_t maxAcceptProb = 1 + fabs(1-2*_mistag) ;
    Double_t acceptProb = 1 + _tag*(1-2*_mistag)*cos(_dm*tval);
    Bool_t mixAccept = maxAcceptProb*RooRandom::uniform() < acceptProb ? kTRUE : kFALSE ;
    
    if (tval<x.max() && tval>x.min() && mixAccept) {
      x = tval ;
      break ;
    }
  }
  
}
