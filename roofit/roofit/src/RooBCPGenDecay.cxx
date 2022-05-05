/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   JGS, Jim Smith    , University of Colorado, jgsmith@pizero.colorado.edu *
 * History:
 *   15-Aug-2002 JGS Created initial version
 *   11-Sep-2002 JGS Mods to introduce mu (Mirna van Hoek, JGS, Nick Danielson)
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California,         *
 *                          University of Colorado                           *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooBCPGenDecay
    \ingroup Roofit

Implement standard CP physics model with S and C (no mention of lambda)
Suitably stolen and modified from RooBCPEffDecay
**/

#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooBCPGenDecay.h"
#include "RooRealIntegral.h"

using namespace std;

ClassImp(RooBCPGenDecay);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooBCPGenDecay::RooBCPGenDecay(const char *name, const char *title,
                RooRealVar& t, RooAbsCategory& tag,
                RooAbsReal& tau, RooAbsReal& dm,
                RooAbsReal& avgMistag,
                RooAbsReal& a, RooAbsReal& b,
                RooAbsReal& delMistag,
                               RooAbsReal& mu,
                const RooResolutionModel& model, DecayType type) :
  RooAbsAnaConvPdf(name,title,model,t),
  _avgC("C","Coefficient of cos term",this,a),
  _avgS("S","Coefficient of sin term",this,b),
  _avgMistag("avgMistag","Average mistag rate",this,avgMistag),
  _delMistag("delMistag","Delta mistag rate",this,delMistag),
  _mu("mu","Tagg efficiency difference",this,mu),
  _t("t","time",this,t),
  _tau("tau","decay time",this,tau),
  _dm("dm","mixing frequency",this,dm),
  _tag("tag","CP state",this,tag),
  _genB0Frac(0),
  _type(type)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooBCPGenDecay::RooBCPGenDecay(const RooBCPGenDecay& other, const char* name) :
  RooAbsAnaConvPdf(other,name),
  _avgC("C",this,other._avgC),
  _avgS("S",this,other._avgS),
  _avgMistag("avgMistag",this,other._avgMistag),
  _delMistag("delMistag",this,other._delMistag),
  _mu("mu",this,other._mu),
  _t("t",this,other._t),
  _tau("tau",this,other._tau),
  _dm("dm",this,other._dm),
  _tag("tag",this,other._tag),
  _genB0Frac(other._genB0Frac),
  _type(other._type),
  _basisExp(other._basisExp),
  _basisSin(other._basisSin),
  _basisCos(other._basisCos)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooBCPGenDecay::~RooBCPGenDecay()
{
}

////////////////////////////////////////////////////////////////////////////////
/// B0    : _tag = +1
/// B0bar : _tag = -1

Double_t RooBCPGenDecay::coefficient(Int_t basisIndex) const
{
  if (basisIndex==_basisExp) {
    //exp term: (1 -/+ dw + mu*_tag*w)
    return (1 - _tag*_delMistag + _mu*_tag*(1. - 2.*_avgMistag)) ;
    // =    1 + _tag*deltaDil/2 + _mu*avgDil
  }

  if (basisIndex==_basisSin) {
    //sin term: (+/- (1-2w) + _mu*(1 -/+ delw))*S
    return (_tag*(1-2*_avgMistag) + _mu*(1. - _tag*_delMistag))*_avgS ;
    // =   (_tag*avgDil + _mu*(1 + tag*deltaDil/2)) * S
  }

  if (basisIndex==_basisCos) {
    //cos term: -(+/- (1-2w) + _mu*(1 -/+ delw))*C
    return -1.*(_tag*(1-2*_avgMistag) + _mu*(1. - _tag*_delMistag))*_avgC ;
    // =   -(_tag*avgDil + _mu*(1 + _tag*deltaDil/2) )* C
  }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooBCPGenDecay::getCoefAnalyticalIntegral(Int_t /*code*/, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const
{
  if (rangeName) return 0 ;
  if (matchArgs(allVars,analVars,_tag)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Double_t RooBCPGenDecay::coefAnalyticalIntegral(Int_t basisIndex, Int_t code, const char* /*rangeName*/) const
{
  switch(code) {
    // No integration
  case 0: return coefficient(basisIndex) ;

    // Integration over 'tag'
  case 1:
    if (basisIndex==_basisExp) {
      return 2 ;
    }

    if (basisIndex==_basisSin) {
      return 2*_mu*_avgS ;
    }
    if (basisIndex==_basisCos) {
      return -2*_mu*_avgC ;
    }
    break ;

  default:
    assert(0) ;
  }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooBCPGenDecay::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK) const
{
  if (staticInitOK) {
    if (matchArgs(directVars,generateVars,_t,_tag)) return 2 ;
  }
  if (matchArgs(directVars,generateVars,_t)) return 1 ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

void RooBCPGenDecay::initGenerator(Int_t code)
{
  if (code==2) {
    // Calculate the fraction of mixed events to generate
    Double_t sumInt = RooRealIntegral("sumInt","sum integral",*this,RooArgSet(_t.arg(),_tag.arg())).getVal() ;
    _tag = 1 ;
    Double_t b0Int = RooRealIntegral("mixInt","mix integral",*this,RooArgSet(_t.arg())).getVal() ;
    _genB0Frac = b0Int/sumInt ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Generate mix-state dependent

void RooBCPGenDecay::generateEvent(Int_t code)
{
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
// 2 in next line is conservative and inefficient - allows for delMistag=1!
    Double_t maxAcceptProb = 2 + fabs(maxDil*_avgS) + fabs(maxDil*_avgC);
    Double_t acceptProb    = (1-_tag*_delMistag + _mu*_tag*(1. - 2.*_avgMistag))
                           + (_tag*(1-2*_avgMistag) + _mu*(1. - _tag*_delMistag))*_avgS*sin(_dm*tval)
                           - (_tag*(1-2*_avgMistag) + _mu*(1. - _tag*_delMistag))*_avgC*cos(_dm*tval);

    bool accept = maxAcceptProb*RooRandom::uniform() < acceptProb ? true : false ;

    if (tval<_t.max() && tval>_t.min() && accept) {
      _t = tval ;
      break ;
    }
  }

}
