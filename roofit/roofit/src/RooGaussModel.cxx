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

/** \class RooGaussModel
    \ingroup Roofit

Class RooGaussModel implements a RooResolutionModel that models a Gaussian
distribution. Object of class RooGaussModel can be used
for analytical convolutions with classes inheriting from RooAbsAnaConvPdf
**/

#include "TMath.h"
#include "Riostream.h"
#include "RooGaussModel.h"
#include "RooMath.h"
#include "RooRealConstant.h"
#include "RooRandom.h"
#include "RooBatchCompute.h"

#include "TError.h"

#include <RooHeterogeneousMath.h>

#include <array>

namespace {

enum RooGaussBasis {
   noBasis = 0,
   expBasisMinus = 1,
   expBasisSum = 2,
   expBasisPlus = 3,
   sinBasisMinus = 11,
   sinBasisSum = 12,
   sinBasisPlus = 13,
   cosBasisMinus = 21,
   cosBasisSum = 22,
   cosBasisPlus = 23,
   linBasisPlus = 33,
   quadBasisPlus = 43,
   coshBasisMinus = 51,
   coshBasisSum = 52,
   coshBasisPlus = 53,
   sinhBasisMinus = 61,
   sinhBasisSum = 62,
   sinhBasisPlus = 63
};

enum BasisType {
   none = 0,
   expBasis = 1,
   sinBasis = 2,
   cosBasis = 3,
   linBasis = 4,
   quadBasis = 5,
   coshBasis = 6,
   sinhBasis = 7
};

enum BasisSign { Both = 0, Plus = +1, Minus = -1 };

BasisType getBasisType(int basisCode)
{
   return static_cast<BasisType>(basisCode == 0 ? 0 : (basisCode / 10) + 1);
}

} // namespace

using RooHeterogeneousMath::evalCerf;
using RooHeterogeneousMath::evalCerfApprox;


////////////////////////////////////////////////////////////////////////////////

RooGaussModel::RooGaussModel(const char *name, const char *title, RooAbsRealLValue &xIn, RooAbsReal &_mean,
                             RooAbsReal &_sigma)
   : RooGaussModel{name, title, xIn, _mean, _sigma, RooRealConstant::value(1), RooRealConstant::value(1)}
{
}

////////////////////////////////////////////////////////////////////////////////

RooGaussModel::RooGaussModel(const char *name, const char *title, RooAbsRealLValue &xIn, RooAbsReal &_mean,
                             RooAbsReal &_sigma, RooAbsReal &_msSF)
   : RooGaussModel{name, title, xIn, _mean, _sigma, _msSF, _msSF}
{
}

////////////////////////////////////////////////////////////////////////////////

RooGaussModel::RooGaussModel(const char *name, const char *title, RooAbsRealLValue& xIn,
              RooAbsReal& _mean, RooAbsReal& _sigma,
              RooAbsReal& _meanSF, RooAbsReal& _sigmaSF) :
  RooResolutionModel(name,title,xIn),
  _flatSFInt(false),
  _asympInt(false),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma),
  msf("msf","Mean Scale Factor",this,_meanSF),
  ssf("ssf","Sigma Scale Factor",this,_sigmaSF)
{
}

////////////////////////////////////////////////////////////////////////////////

RooGaussModel::RooGaussModel(const RooGaussModel& other, const char* name) :
  RooResolutionModel(other,name),
  _flatSFInt(other._flatSFInt),
  _asympInt(other._asympInt),
  mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma),
  msf("msf",this,other.msf),
  ssf("ssf",this,other.ssf)
{
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGaussModel::basisCode(const char* name) const
{
   std::string str = name;

   // Remove whitespaces from the input string
   str.erase(remove(str.begin(),str.end(),' '),str.end());

   if (str == "exp(-@0/@1)") return expBasisPlus ;
   if (str == "exp(@0/@1)") return expBasisMinus ;
   if (str == "exp(-abs(@0)/@1)") return expBasisSum ;
   if (str == "exp(-@0/@1)*sin(@0*@2)") return sinBasisPlus ;
   if (str == "exp(@0/@1)*sin(@0*@2)") return sinBasisMinus ;
   if (str == "exp(-abs(@0)/@1)*sin(@0*@2)") return sinBasisSum ;
   if (str == "exp(-@0/@1)*cos(@0*@2)") return cosBasisPlus ;
   if (str == "exp(@0/@1)*cos(@0*@2)") return cosBasisMinus ;
   if (str == "exp(-abs(@0)/@1)*cos(@0*@2)") return cosBasisSum ;
   if (str == "(@0/@1)*exp(-@0/@1)") return linBasisPlus ;
   if (str == "(@0/@1)*(@0/@1)*exp(-@0/@1)") return quadBasisPlus ;
   if (str == "exp(-@0/@1)*cosh(@0*@2/2)") return coshBasisPlus;
   if (str == "exp(@0/@1)*cosh(@0*@2/2)") return coshBasisMinus;
   if (str == "exp(-abs(@0)/@1)*cosh(@0*@2/2)") return coshBasisSum;
   if (str == "exp(-@0/@1)*sinh(@0*@2/2)") return sinhBasisPlus;
   if (str == "exp(@0/@1)*sinh(@0*@2/2)") return sinhBasisMinus;
   if (str == "exp(-abs(@0)/@1)*sinh(@0*@2/2)") return sinhBasisSum;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////

double RooGaussModel::evaluate() const
{
   auto arg1 = static_cast<RooAbsReal*>(basis().getParameter(1));
   auto arg2 = static_cast<RooAbsReal*>(basis().getParameter(2));
   double param1 = arg1 ? arg1->getVal() : 0.0;
   double param2 = arg2 ? arg2->getVal() : 0.0;
   return evaluate(x, mean * msf, sigma * ssf, param1, param2, _basisCode);
}

void RooGaussModel::doEval(RooFit::EvalContext &ctx) const
{
   std::span<double> output = ctx.output();
   std::size_t size = output.size();

   auto xVals = ctx.at(x);
   auto meanVals = ctx.at(mean);
   auto meanSfVals = ctx.at(msf);
   auto sigmaVals = ctx.at(sigma);
   auto sigmaSfVals = ctx.at(ssf);

   auto param1 = static_cast<RooAbsReal *>(basis().getParameter(1));
   auto param2 = static_cast<RooAbsReal *>(basis().getParameter(2));
   const double zeroVal = 0.0;
   auto param1Vals = param1 ? ctx.at(param1) : std::span<const double>{&zeroVal, 1};
   auto param2Vals = param2 ? ctx.at(param2) : std::span<const double>{&zeroVal, 1};

   BasisType basisType = getBasisType(_basisCode);
   double basisSign = _basisCode - 10 * (basisType - 1) - 2;

   // We have an implementation also for CUDA right now only for the most used
   // basis type, which is expBasis. If the need to support other basis types
   // arises, they can be implemented following this example. Remember to also
   // adapt RooGaussModel::canComputeBatchWithCuda().
   if (basisType == expBasis) {
      std::array<double, 1> extraArgs{basisSign};
      RooBatchCompute::compute(ctx.config(this), RooBatchCompute::GaussModelExpBasis, output,
                        {xVals, meanVals, meanSfVals, sigmaVals, sigmaSfVals, param1Vals}, extraArgs);
      return;
   }

   // For now, if the arrays don't have the expected input shape, fall back to the scalar mode
   if (xVals.size() != size || meanVals.size() != 1 || meanSfVals.size() != 1 || sigmaVals.size() != 1 ||
       sigmaSfVals.size() != 1 || param1Vals.size() != 1 || param2Vals.size() != 1) {
      return RooAbsPdf::doEval(ctx);
   }

   for (unsigned int i = 0; i < size; ++i) {
      output[i] = evaluate(xVals[i], meanVals[0] * meanSfVals[0], sigmaVals[0] * sigmaSfVals[0], param1Vals[0],
                           param2Vals[0], _basisCode);
   }
}

double RooGaussModel::evaluate(double x, double mean, double sigma, double param1, double param2, int basisCode)
{
  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  static double root2(std::sqrt(2.)) ;
  static double root2pi(std::sqrt(2.*std::atan2(0.,-1.))) ;
  static double rootpi(std::sqrt(std::atan2(0.,-1.))) ;

  BasisType basisType = getBasisType(basisCode);
  BasisSign basisSign = (BasisSign)( basisCode - 10*(basisType-1) - 2 ) ;

  double tau = (basisCode!=noBasis) ? param1 : 0.0;
  if (basisType == coshBasis && basisCode!=noBasis ) {
     double dGamma = param2;
     if (dGamma==0) basisType = expBasis;
  }

  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    double xprime = (x-mean)/sigma ;
    double result = std::exp(-0.5*xprime*xprime)/(sigma*root2pi) ;
    if (basisCode!=0 && basisSign==Both) result *= 2 ;
    return result ;
  }

  // *** 2nd form: 0, used for sinBasis, linBasis, and quadBasis with tau=0 ***
  if (tau==0) {
    return 0. ;
  }

  // *** 3nd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  double omega =  (basisType==sinBasis  || basisType==cosBasis)  ? param2 : 0 ;
  double dgamma = (basisType==sinhBasis || basisType==coshBasis) ? param2 : 0 ;
  double _x = omega *tau ;
  double _y = tau*dgamma/2;
  double xprime = (x-mean)/tau ;
  double c = sigma/(root2*tau) ;
  double u = xprime/(2*c) ;

  if (basisType==expBasis || (basisType==cosBasis && _x==0.)) {
    double result(0) ;
    if (basisSign!=Minus) result += evalCerf(0,-u,c).real();
    if (basisSign!=Plus)  result += evalCerf(0, u,c).real();
    return result ;
  }

  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  if (basisType==sinBasis) {
    double result(0) ;
    if (_x==0.) return result ;
    if (basisSign!=Minus) result += -evalCerf(-_x,-u,c).imag();
    if (basisSign!=Plus)  result += -evalCerf( _x, u,c).imag();
    return result ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    double result(0) ;
    if (basisSign!=Minus) result += evalCerf(-_x,-u,c).real();
    if (basisSign!=Plus)  result += evalCerf( _x, u,c).real();
    return result ;
  }

  // ***8th form: Convolution with exp(-|t|/tau)*cosh(dgamma*t/2), used for         coshBasisSum ***
  if (basisType==coshBasis || basisType ==sinhBasis) {
    double result(0);
    int sgn = ( basisType == coshBasis ? +1 : -1 );
    if (basisSign!=Minus) result += 0.5*(    evalCerf(0,-u,c*(1-_y)).real()+sgn*evalCerf(0,-u,c*(1+_y)).real()) ;
    if (basisSign!=Plus)  result += 0.5*(sgn*evalCerf(0, u,c*(1-_y)).real()+    evalCerf(0, u,c*(1+_y)).real()) ;
    return result ;
  }

  // *** 6th form: Convolution with (t/tau)*exp(-t/tau), used for linBasis ***
  if (basisType==linBasis) {
    R__ASSERT(basisSign==Plus);  // This should only be for positive times

    double f0 = std::exp(-xprime+c*c) * RooMath::erfc(-u+c);
    double f1 = std::exp(-u*u);
    return (xprime - 2*c*c)*f0 + (2*c/rootpi)*f1 ;
  }

  // *** 7th form: Convolution with (t/tau)^2*exp(-t/tau), used for quadBasis ***
  if (basisType==quadBasis) {
    R__ASSERT(basisSign==Plus);  // This should only be for positive times

    double f0 = std::exp(-xprime+c*c) * RooMath::erfc(-u+c);
    double f1 = std::exp(-u*u);
    double x2c2 = xprime - 2*c*c;
    return ( x2c2*x2c2*f0 + (2*c/rootpi)*x2c2*f1 + 2*c*c*f0 );
  }

  R__ASSERT(0) ;
  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGaussModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  switch(_basisCode) {

  // Analytical integration capability of raw PDF
  case noBasis:

    // Optionally advertise flat integral over sigma scale factor
    if (_flatSFInt) {
      if (matchArgs(allVars,analVars,RooArgSet(convVar(),ssf.arg()))) return 2 ;
    }

    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;

  // Analytical integration capability of convoluted PDF
  case expBasisPlus:
  case expBasisMinus:
  case expBasisSum:
  case sinBasisPlus:
  case sinBasisMinus:
  case sinBasisSum:
  case cosBasisPlus:
  case cosBasisMinus:
  case cosBasisSum:
  case linBasisPlus:
  case quadBasisPlus:
  case coshBasisMinus:
  case coshBasisPlus:
  case coshBasisSum:
  case sinhBasisMinus:
  case sinhBasisPlus:
  case sinhBasisSum:

    // Optionally advertise flat integral over sigma scale factor
    if (_flatSFInt) {

      if (matchArgs(allVars,analVars,RooArgSet(convVar(),ssf.arg()))) {
   return 2 ;
      }
    }

    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;
  }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////

double RooGaussModel::analyticalIntegral(Int_t code, const char* rangeName) const
{
  static const double root2 = std::sqrt(2.) ;
  //static double rootPiBy2 = std::sqrt(std::atan2(0.0,-1.0)/2.0);
  static const double rootpi = std::sqrt(std::atan2(0.0,-1.0));
  double ssfInt(1.0) ;

  // Code must be 1 or 2
  R__ASSERT(code==1||code==2) ;
  if (code==2) ssfInt = (ssf.max(rangeName)-ssf.min(rangeName)) ;

  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  // *** 1st form: Straight Gaussian, used for unconvoluted PDF or expBasis with 0 lifetime ***
  double tau = (_basisCode!=noBasis)?(static_cast<RooAbsReal*>(basis().getParameter(1)))->getVal():0 ;
  if (basisType == coshBasis && _basisCode!=noBasis ) {
     double dGamma = (static_cast<RooAbsReal*>(basis().getParameter(2)))->getVal();
     if (dGamma==0) basisType = expBasis;
  }
  if (basisType==none || ((basisType==expBasis || basisType==cosBasis) && tau==0.)) {
    double xscale = root2*(sigma*ssf);
    if (verboseEval()>0) std::cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 1st form" << std::endl ;

    double xpmin = (x.min(rangeName)-(mean*msf))/xscale ;
    double xpmax = (x.max(rangeName)-(mean*msf))/xscale ;

    double result ;
    if (_asympInt) { // modified FMV, 07/24/03
        result = 1.0 ;
    } else {
       result = 0.5*(RooMath::erf(xpmax)-RooMath::erf(xpmin)) ;
    }

    if (_basisCode!=0 && basisSign==Both) result *= 2 ;
    //cout << "Integral 1st form " << " result= " << result*ssfInt << std::endl;
    if (TMath::IsNaN(result)) { cxcoutE(Tracing) << "RooGaussModel::analyticalIntegral(" << GetName() << ") got nan during case 1 " << std::endl; }
    return result*ssfInt ;
  }


  double omega = ((basisType==sinBasis)||(basisType==cosBasis)) ?  (static_cast<RooAbsReal*>(basis().getParameter(2)))->getVal() : 0 ;
  double dgamma =((basisType==sinhBasis)||(basisType==coshBasis)) ?  (static_cast<RooAbsReal*>(basis().getParameter(2)))->getVal() : 0 ;

  // *** 2nd form: unity, used for sinBasis and linBasis with tau=0 (PDF is zero) ***
  if (tau==0) {
    if (verboseEval()>0) std::cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 2nd form" << std::endl ;
    return 0. ;
  }

  // *** 3rd form: Convolution with exp(-t/tau), used for expBasis and cosBasis(omega=0) ***
  double c = (sigma*ssf)/(root2*tau) ;
  double xpmin = (x.min(rangeName)-(mean*msf))/tau ;
  double xpmax = (x.max(rangeName)-(mean*msf))/tau ;
  double umin = xpmin/(2*c) ;
  double umax = xpmax/(2*c) ;

  if (basisType==expBasis || (basisType==cosBasis && omega==0.)) {
    if (verboseEval()>0) std::cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 3d form tau=" << tau << std::endl ;
    double result(0) ;
    if (basisSign!=Minus) result += evalCerfInt(+1,0,tau,-umin,-umax,c).real();
    if (basisSign!=Plus)  result += evalCerfInt(-1,0,tau, umin, umax,c).real();
    if (TMath::IsNaN(result)) { cxcoutE(Tracing) << "RooGaussModel::analyticalIntegral(" << GetName() << ") got nan during case 3 " << std::endl; }
    return result*ssfInt ;
  }

  // *** 4th form: Convolution with exp(-t/tau)*sin(omega*t), used for sinBasis(omega<>0,tau<>0) ***
  double _x = omega * tau ;
  double _y = tau*dgamma/2;

  if (basisType==sinBasis) {
    if (verboseEval()>0) std::cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 4th form omega = " << omega << ", tau = " << tau << std::endl ;
    double result(0) ;
    if (_x==0) return result*ssfInt ;
    if (basisSign!=Minus) result += -1*evalCerfInt(+1,-_x,tau,-umin,-umax,c).imag();
    if (basisSign!=Plus)  result += -1*evalCerfInt(-1, _x,tau, umin, umax,c).imag();
    if (TMath::IsNaN(result)) { cxcoutE(Tracing) << "RooGaussModel::analyticalIntegral(" << GetName() << ") got nan during case 4 " << std::endl; }
    return result*ssfInt ;
  }

  // *** 5th form: Convolution with exp(-t/tau)*cos(omega*t), used for cosBasis(omega<>0) ***
  if (basisType==cosBasis) {
    if (verboseEval()>0) std::cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 5th form omega = " << omega << ", tau = " << tau << std::endl ;
    double result(0) ;
    if (basisSign!=Minus) result += evalCerfInt(+1,-_x,tau,-umin,-umax,c).real();
    if (basisSign!=Plus)  result += evalCerfInt(-1, _x,tau, umin, umax,c).real();
    if (TMath::IsNaN(result)) { cxcoutE(Tracing) << "RooGaussModel::analyticalIntegral(" << GetName() << ") got nan during case 5 " << std::endl; }
    return result*ssfInt ;
  }

  // *** 8th form: Convolution with exp(-|t|/tau)*cosh(dgamma*t/2), used for coshBasis ***
  // *** 9th form: Convolution with exp(-|t|/tau)*sinh(dgamma*t/2), used for sinhBasis ***
  if (basisType==coshBasis || basisType == sinhBasis) {
    if (verboseEval()>0) {std::cout << "RooGaussModel::analyticalIntegral(" << GetName()                             << ") 8th form tau=" << tau << std::endl ; }
    double result(0) ;
    int sgn = ( basisType == coshBasis ? +1 : -1 );
    if (basisSign!=Minus) result += 0.5*(    evalCerfInt(+1,0,tau/(1-_y),-umin,-umax,c*(1-_y)).real()+ sgn*evalCerfInt(+1,0,tau/(1+_y),-umin,-umax,c*(1+_y)).real());
    if (basisSign!=Plus)  result += 0.5*(sgn*evalCerfInt(-1,0,tau/(1-_y), umin, umax,c*(1-_y)).real()+     evalCerfInt(-1,0,tau/(1+_y), umin, umax,c*(1+_y)).real());
    if (TMath::IsNaN(result)) { cxcoutE(Tracing) << "RooGaussModel::analyticalIntegral(" << GetName() << ") got nan during case 6 " << std::endl; }
    return result*ssfInt ;
  }

  // *** 6th form: Convolution with (t/tau)*exp(-t/tau), used for linBasis ***
  if (basisType==linBasis) {
    if (verboseEval()>0) std::cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 6th form tau=" << tau << std::endl ;

    double f0 = RooMath::erf(-umax) - RooMath::erf(-umin);
    double f1 = std::exp(-umax*umax) - std::exp(-umin*umin);

    double tmp1 = std::exp(-xpmax)*RooMath::erfc(-umax + c);
    double tmp2 = std::exp(-xpmin)*RooMath::erfc(-umin + c);

    double f2 = tmp1 - tmp2;
    double f3 = xpmax*tmp1 - xpmin*tmp2;

    double expc2 = std::exp(c*c);

    return -tau*(              f0 +
        (2*c/rootpi)*f1 +
        (1 - 2*c*c)*expc2*f2 +
          expc2*f3
      )*ssfInt;
  }

  // *** 7th form: Convolution with (t/tau)*(t/tau)*exp(-t/tau), used for quadBasis ***
  if (basisType==quadBasis) {
    if (verboseEval()>0) std::cout << "RooGaussModel::analyticalIntegral(" << GetName() << ") 7th form tau=" << tau << std::endl ;

    double f0 = RooMath::erf(-umax) - RooMath::erf(-umin);

    double tmpA1 = std::exp(-umax*umax);
    double tmpA2 = std::exp(-umin*umin);

    double f1 = tmpA1 - tmpA2;
    double f2 = umax*tmpA1 - umin*tmpA2;

    double tmpB1 = std::exp(-xpmax)*RooMath::erfc(-umax + c);
    double tmpB2 = std::exp(-xpmin)*RooMath::erfc(-umin + c);

    double f3 = tmpB1 - tmpB2;
    double f4 = xpmax*tmpB1 - xpmin*tmpB2;
    double f5 = xpmax*xpmax*tmpB1 - xpmin*xpmin*tmpB2;

    double expc2 = std::exp(c*c);

    return -tau*( 2*f0 +
        (4*c/rootpi)*((1-c*c)*f1 + c*f2) +
        (2*c*c*(2*c*c-1) + 2)*expc2*f3 - (4*c*c-2)*expc2*f4 + expc2*f5
                )*ssfInt;
  }
  R__ASSERT(0) ;
  return 0 ;
}


////////////////////////////////////////////////////////////////////////////////

std::complex<double> RooGaussModel::evalCerfInt(double sign, double _x, double tau, double umin, double umax, double c) const
{
  std::complex<double> diff(2., 0.);
  if (!_asympInt) {
    diff = evalCerf(_x,umin,c);
    diff -= evalCerf(_x,umax,c);
    diff += RooMath::erf(umin) - RooMath::erf(umax);
    diff *= sign;
  }
  diff *= std::complex<double>(1., _x);
  diff *= tau / (1.+_x*_x);
  return diff;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooGaussModel::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
   return matchArgs(directVars,generateVars,x) ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////

void RooGaussModel::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;
  double xmin = x.min();
  double xmax = x.max();
  TRandom *generator = RooRandom::randomGenerator();
  while(true) {
    double xgen = generator->Gaus(mean*msf,sigma*ssf);
    if (xgen<xmax && xgen>xmin) {
      x = xgen ;
      return ;
    }
  }
}

bool RooGaussModel::canComputeBatchWithCuda() const
{
   return getBasisType(_basisCode) == expBasis;
}
