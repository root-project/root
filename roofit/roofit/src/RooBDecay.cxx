/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   PL, Parker C Lund,   UC Irvine                                          *
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


/** \class RooBDecay
    \ingroup Roofit

Most general description of B decay time distribution with effects
of CP violation, mixing and life time differences. This function can
be analytically convolved with any RooResolutionModel implementation
**/

#include "Riostream.h"
#include "TMath.h"
#include "RooBDecay.h"
#include "RooRealVar.h"
#include "RooRandom.h"

#include "TError.h"

using namespace std;

ClassImp(RooBDecay);

////////////////////////////////////////////////////////////////////////////////

RooBDecay::RooBDecay(const char *name, const char* title,
          RooRealVar& t, RooAbsReal& tau, RooAbsReal& dgamma,
          RooAbsReal& f0, RooAbsReal& f1, RooAbsReal& f2, RooAbsReal& f3,
          RooAbsReal& dm, const RooResolutionModel& model, DecayType type) :
  RooAbsAnaConvPdf(name, title, model, t),
  _t("t", "time", this, t),
  _tau("tau", "Average Decay Time", this, tau),
  _dgamma("dgamma", "Delta Gamma", this, dgamma),
  _f0("f0", "Cosh Coefficient", this, f0),
  _f1("f1", "Sinh Coefficient", this, f1),
  _f2("f2", "Cos Coefficient", this, f2),
  _f3("f3", "Sin Coefficient", this, f3),
  _dm("dm", "Delta Mass", this, dm),
  _type(type)

{
  //Constructor
  switch(type)
    {
    case SingleSided:
      _basisCosh = declareBasis("exp(-@0/@1)*cosh(@0*@2/2)", RooArgList(tau,dgamma));
      _basisSinh = declareBasis("exp(-@0/@1)*sinh(@0*@2/2)", RooArgList(tau,dgamma));
      _basisCos = declareBasis("exp(-@0/@1)*cos(@0*@2)",RooArgList(tau, dm));
      _basisSin = declareBasis("exp(-@0/@1)*sin(@0*@2)",RooArgList(tau, dm));
      break;
    case Flipped:
      _basisCosh = declareBasis("exp(@0/@1)*cosh(@0*@2/2)", RooArgList(tau,dgamma));
      _basisSinh = declareBasis("exp(@0/@1)*sinh(@0*@2/2)", RooArgList(tau,dgamma));
      _basisCos = declareBasis("exp(@0/@1)*cos(@0*@2)",RooArgList(tau, dm));
      _basisSin = declareBasis("exp(@0/@1)*sin(@0*@2)",RooArgList(tau, dm));
      break;
    case DoubleSided:
      _basisCosh = declareBasis("exp(-abs(@0)/@1)*cosh(@0*@2/2)", RooArgList(tau,dgamma));
      _basisSinh = declareBasis("exp(-abs(@0)/@1)*sinh(@0*@2/2)", RooArgList(tau,dgamma));
      _basisCos = declareBasis("exp(-abs(@0)/@1)*cos(@0*@2)",RooArgList(tau, dm));
      _basisSin = declareBasis("exp(-abs(@0)/@1)*sin(@0*@2)",RooArgList(tau, dm));
      break;
    }
}

////////////////////////////////////////////////////////////////////////////////
///Copy constructor

RooBDecay::RooBDecay(const RooBDecay& other, const char* name) :
  RooAbsAnaConvPdf(other, name),
  _t("t", this, other._t),
  _tau("tau", this, other._tau),
  _dgamma("dgamma", this, other._dgamma),
  _f0("f0", this, other._f0),
  _f1("f1", this, other._f1),
  _f2("f2", this, other._f2),
  _f3("f3", this, other._f3),
  _dm("dm", this, other._dm),
  _basisCosh(other._basisCosh),
  _basisSinh(other._basisSinh),
  _basisCos(other._basisCos),
  _basisSin(other._basisSin),
  _type(other._type)
{
}

////////////////////////////////////////////////////////////////////////////////
///Destructor

RooBDecay::~RooBDecay()
{
}

////////////////////////////////////////////////////////////////////////////////

double RooBDecay::coefficient(Int_t basisIndex) const
{
  if(basisIndex == _basisCosh)
    {
      return _f0;
    }
  if(basisIndex == _basisSinh)
    {
      return _f1;
    }
  if(basisIndex == _basisCos)
    {
      return _f2;
    }
  if(basisIndex == _basisSin)
    {
      return _f3;
    }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

RooFit::OwningPtr<RooArgSet> RooBDecay::coefVars(Int_t basisIndex) const
{
  if(basisIndex == _basisCosh)
    {
      return _f0.arg().getVariables();
    }
  if(basisIndex == _basisSinh)
    {
      return _f1.arg().getVariables();
    }
  if(basisIndex == _basisCos)
    {
      return _f2.arg().getVariables();
    }
  if(basisIndex == _basisSin)
    {
      return _f3.arg().getVariables();
    }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooBDecay::getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const
{
  if(coef == _basisCosh)
    {
      return _f0.arg().getAnalyticalIntegral(allVars,analVars,rangeName) ;
    }
  if(coef == _basisSinh)
    {
      return _f1.arg().getAnalyticalIntegral(allVars,analVars,rangeName) ;
    }
  if(coef == _basisCos)
    {
      return _f2.arg().getAnalyticalIntegral(allVars,analVars,rangeName) ;
    }
  if(coef == _basisSin)
    {
      return _f3.arg().getAnalyticalIntegral(allVars,analVars,rangeName) ;
    }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

double RooBDecay::coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName) const
{
  if(coef == _basisCosh)
    {
      return _f0.arg().analyticalIntegral(code,rangeName) ;
    }
  if(coef == _basisSinh)
    {
      return _f1.arg().analyticalIntegral(code,rangeName) ;
    }
  if(coef == _basisCos)
    {
      return _f2.arg().analyticalIntegral(code,rangeName) ;
    }
  if(coef == _basisSin)
    {
      return _f3.arg().analyticalIntegral(code,rangeName) ;
    }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooBDecay::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  if (matchArgs(directVars, generateVars, _t)) return 1;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

void RooBDecay::generateEvent(Int_t code)
{
  R__ASSERT(code==1);
  double gammamin = 1/_tau-TMath::Abs(_dgamma)/2;
  while(1) {
    double t = -log(RooRandom::uniform())/gammamin;
    if (_type == Flipped || (_type == DoubleSided && RooRandom::uniform() <0.5) ) t *= -1;
    if ( t<_t.min() || t>_t.max() ) continue;

    double dgt = _dgamma*t/2;
    double dmt = _dm*t;
    double ft = std::abs(t);
    double f = exp(-ft/_tau)*(_f0*cosh(dgt)+_f1*sinh(dgt)+_f2*cos(dmt)+_f3*sin(dmt));
    if(f < 0) {
      cout << "RooBDecay::generateEvent(" << GetName() << ") ERROR: PDF value less than zero" << endl;
      ::abort();
    }
    double w = 1.001*exp(-ft*gammamin)*(TMath::Abs(_f0)+TMath::Abs(_f1)+sqrt(_f2*_f2+_f3*_f3));
    if(w < f) {
      cout << "RooBDecay::generateEvent(" << GetName() << ") ERROR: Envelope function less than p.d.f. " << endl;
      cout << _f0 << endl;
      cout << _f1 << endl;
      cout << _f2 << endl;
      cout << _f3 << endl;
      ::abort();
    }
    if(w*RooRandom::uniform() > f) continue;
    _t = t;
    break;
  }
}
