/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

/**
\file RooTruthModel.cxx
\class RooTruthModel
\ingroup Roofitcore

Implements a RooResolution model that corresponds to a delta function.
The truth model supports <i>all</i> basis functions because it evaluates each basis function as
as a RooFormulaVar.  The 6 basis functions used in B mixing and decay and 2 basis
functions used in D mixing have been hand coded for increased execution speed.
**/

#include <RooTruthModel.h>

#include <RooAbsAnaConvPdf.h>
#include <RooBatchCompute.h>
#include <RooGenContext.h>

#include <RooFit/Detail/MathFuncs.h>

#include <Riostream.h>

#include <TError.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace {

enum RooTruthBasis {
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
   sinhBasisPlus = 63,
   genericBasis = 100
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

} // namespace


////////////////////////////////////////////////////////////////////////////////
/// Constructor of a truth resolution model, i.e. a delta function in observable 'xIn'

RooTruthModel::RooTruthModel(const char *name, const char *title, RooAbsRealLValue& xIn) :
  RooResolutionModel(name,title,xIn)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Return basis code for given basis definition string. Return special
/// codes for 'known' bases for which compiled definition exists. Return
/// generic bases code if implementation relies on TFormula interpretation
/// of basis name

Int_t RooTruthModel::basisCode(const char* name) const
{
   std::string str = name;

   // Remove whitespaces from the input string
   str.erase(remove(str.begin(),str.end(),' '),str.end());

   // Check for optimized basis functions
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

   // Truth model is delta function, i.e. convolution integral is basis
   // function, therefore we can handle any basis function
   return genericBasis ;
}



////////////////////////////////////////////////////////////////////////////////
/// Changes associated bases function to 'inBasis'

void RooTruthModel::changeBasis(RooFormulaVar* inBasis)
{
   // Remove client-server link to old basis
   if (_basis) {
      if (_basisCode == genericBasis) {
         // In the case of a generic basis, we evaluate it directly, so the
         // basis was a direct server.
         removeServer(*_basis);
      } else {
         for (RooAbsArg *basisServer : _basis->servers()) {
            removeServer(*basisServer);
         }
      }

      if (_ownBasis) {
         delete _basis;
      }
   }
   _ownBasis = false;

   _basisCode = inBasis ? basisCode(inBasis->GetTitle()) : 0;

   // Change basis pointer and update client-server link
   _basis = inBasis;
   if (_basis) {
      if (_basisCode == genericBasis) {
         // Since we actually evaluate the basis function object, we need to
         // adjust our client-server links to the basis function here
         addServer(*_basis, true, false);
      } else {
         for (RooAbsArg *basisServer : _basis->servers()) {
            addServer(*basisServer, true, false);
         }
      }
   }
}



////////////////////////////////////////////////////////////////////////////////
/// Evaluate the truth model: a delta function when used as PDF,
/// the basis function itself, when convoluted with a basis function.

double RooTruthModel::evaluate() const
{
  // No basis: delta function
  if (_basisCode == noBasis) {
    if (x==0) return 1 ;
    return 0 ;
  }

  // Generic basis: evaluate basis function object
  if (_basisCode == genericBasis) {
    return basis().getVal() ;
  }

  // Precompiled basis functions
  BasisType basisType = (BasisType)( (_basisCode == 0) ? 0 : (_basisCode/10) + 1 );
  BasisSign basisSign = (BasisSign)( _basisCode - 10*(basisType-1) - 2 ) ;

  // Enforce sign compatibility
  if ((basisSign==Minus && x>0) ||
      (basisSign==Plus  && x<0)) return 0 ;


  double tau = (static_cast<RooAbsReal*>(basis().getParameter(1)))->getVal() ;
  // Return desired basis function
  switch(basisType) {
  case expBasis: {
    return std::exp(-std::abs((double)x)/tau) ;
  }
  case sinBasis: {
    double dm = (static_cast<RooAbsReal*>(basis().getParameter(2)))->getVal() ;
    return std::exp(-std::abs((double)x)/tau)*std::sin(x*dm) ;
  }
  case cosBasis: {
    double dm = (static_cast<RooAbsReal*>(basis().getParameter(2)))->getVal() ;
    return std::exp(-std::abs((double)x)/tau)*std::cos(x*dm) ;
  }
  case linBasis: {
    double tscaled = std::abs((double)x)/tau;
    return std::exp(-tscaled)*tscaled ;
  }
  case quadBasis: {
    double tscaled = std::abs((double)x)/tau;
    return std::exp(-tscaled)*tscaled*tscaled;
  }
  case sinhBasis: {
    double dg = (static_cast<RooAbsReal*>(basis().getParameter(2)))->getVal() ;
    return std::exp(-std::abs((double)x)/tau)*std::sinh(x*dg/2) ;
  }
  case coshBasis: {
    double dg = (static_cast<RooAbsReal*>(basis().getParameter(2)))->getVal() ;
    return std::exp(-std::abs((double)x)/tau)*std::cosh(x*dg/2) ;
  }
  default:
    R__ASSERT(0) ;
  }

  return 0 ;
}


void RooTruthModel::doEval(RooFit::EvalContext &ctx) const
{
   auto config = ctx.config(this);
   auto xVals = ctx.at(x);

   // No basis: delta function
   if (_basisCode == noBasis) {
      RooBatchCompute::compute(config, RooBatchCompute::DeltaFunction, ctx.output(), {xVals});
      return;
   }

   // Generic basis: evaluate basis function object
   if (_basisCode == genericBasis) {
      RooBatchCompute::compute(config, RooBatchCompute::Identity, ctx.output(), {ctx.at(&basis())});
      return;
   }

   // Precompiled basis functions
   const BasisType basisType = static_cast<BasisType>((_basisCode == 0) ? 0 : (_basisCode / 10) + 1);

   // Cast the int from the enum to double because we can only pass doubles to
   // RooBatchCompute at this point.
   const double basisSign = static_cast<double>((BasisSign)(_basisCode - 10 * (basisType - 1) - 2));

   auto param1 = static_cast<RooAbsReal const *>(basis().getParameter(1));
   auto param2 = static_cast<RooAbsReal const *>(basis().getParameter(2));
   auto param1Vals = param1 ? ctx.at(param1) : std::span<const double>{};
   auto param2Vals = param2 ? ctx.at(param2) : std::span<const double>{};

   // Return desired basis function
   std::array<double, 1> extraArgs{basisSign};
   switch (basisType) {
   case expBasis: {
      RooBatchCompute::compute(config, RooBatchCompute::TruthModelExpBasis, ctx.output(), {xVals, param1Vals},
                               extraArgs);
      break;
   }
   case sinBasis: {
      RooBatchCompute::compute(config, RooBatchCompute::TruthModelSinBasis, ctx.output(),
                               {xVals, param1Vals, param2Vals}, extraArgs);
      break;
   }
   case cosBasis: {
      RooBatchCompute::compute(config, RooBatchCompute::TruthModelCosBasis, ctx.output(),
                               {xVals, param1Vals, param2Vals}, extraArgs);
      break;
   }
   case linBasis: {
      RooBatchCompute::compute(config, RooBatchCompute::TruthModelLinBasis, ctx.output(), {xVals, param1Vals},
                               extraArgs);
      break;
   }
   case quadBasis: {
      RooBatchCompute::compute(config, RooBatchCompute::TruthModelQuadBasis, ctx.output(), {xVals, param1Vals},
                               extraArgs);
      break;
   }
   case sinhBasis: {
      RooBatchCompute::compute(config, RooBatchCompute::TruthModelSinhBasis, ctx.output(),
                               {xVals, param1Vals, param2Vals}, extraArgs);
      break;
   }
   case coshBasis: {
      RooBatchCompute::compute(config, RooBatchCompute::TruthModelCoshBasis, ctx.output(),
                               {xVals, param1Vals, param2Vals}, extraArgs);
      break;
   }
   default: R__ASSERT(0);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Advertise analytical integrals for compiled basis functions and when used
/// as p.d.f without basis function.

Int_t RooTruthModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  switch(_basisCode) {

  // Analytical integration capability of raw PDF
  case noBasis:
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
  case sinhBasisPlus:
  case sinhBasisMinus:
  case sinhBasisSum:
  case coshBasisPlus:
  case coshBasisMinus:
  case coshBasisSum:
    if (matchArgs(allVars,analVars,convVar())) return 1 ;
    break ;
  }

  return 0 ;
}


namespace {

// From asking WolframAlpha: integrate exp(-x/tau) over x.
inline double indefiniteIntegralExpBasisPlus(double x, double tau, double /*dm*/)
{
   // Restrict to positive x
   x = std::max(x, 0.0);
   return -tau * std::exp(-x / tau);
}

// From asking WolframAlpha: integrate exp(-x/tau)* x / tau over x.
inline double indefiniteIntegralLinBasisPlus(double x, double tau, double /*dm*/)
{
   // Restrict to positive x
   x = std::max(x, 0.0);
   return -(tau + x) * std::exp(-x / tau);
}

// From asking WolframAlpha: integrate exp(-x/tau) * (x / tau)^2 over x.
inline double indefiniteIntegralQuadBasisPlus(double x, double tau, double /*dm*/)
{
   // Restrict to positive x
   x = std::max(x, 0.0);
   return -(std::exp(-x / tau) * (2 * tau * tau + x * x + 2 * tau * x)) / tau;
}

// A common factor that appears in the integrals of the trigonometric
// function bases (sin and cos).
inline double commonFactorPlus(double x, double tau, double dm)
{
   const double num = tau * std::exp(-x / tau);
   const double den = dm * dm * tau * tau + 1.0;
   return num / den;
}

// A common factor that appears in the integrals of the hyperbolic
// trigonometric function bases (sinh and cosh).
inline double commonFactorHyperbolicPlus(double x, double tau, double dm)
{
   const double num = 2 * tau * std::exp(-x / tau);
   const double den = dm * dm * tau * tau - 4.0;
   return num / den;
}

// From asking WolframAlpha: integrate exp(-x/tau)*sin(x*m) over x.
inline double indefiniteIntegralSinBasisPlus(double x, double tau, double dm)
{
   // Restrict to positive x
   x = std::max(x, 0.0);
   const double fac = commonFactorPlus(x, tau, dm);
   // Only multiply with the sine term if the coefficient is non zero,
   // i.e. if x was not infinity. Otherwise, we are evaluating the
   // sine of infinity, which is NAN!
   return fac != 0.0 ? fac * (-tau * dm * std::cos(dm * x) - std::sin(dm * x)) : 0.0;
}

// From asking WolframAlpha: integrate exp(-x/tau)*cos(x*m) over x.
inline double indefiniteIntegralCosBasisPlus(double x, double tau, double dm)
{
   // Restrict to positive x
   x = std::max(x, 0.0);
   const double fac = commonFactorPlus(x, tau, dm);
   return fac != 0.0 ? fac * (tau * dm * std::sin(dm * x) - std::cos(dm * x)) : 0.0;
}

// From asking WolframAlpha: integrate exp(-x/tau)*sinh(x*m/2) over x.
inline double indefiniteIntegralSinhBasisPlus(double x, double tau, double dm)
{
   // Restrict to positive x
   x = std::max(x, 0.0);
   const double fac = commonFactorHyperbolicPlus(x, tau, dm);
   const double arg = 0.5 * dm * x;
   return fac != 0.0 ? fac * (tau * dm * std::cosh(arg) - 2. * std::sinh(arg)) : 0.0;
}

// From asking WolframAlpha: integrate exp(-x/tau)*cosh(x*m/2) over x.
inline double indefiniteIntegralCoshBasisPlus(double x, double tau, double dm)
{
   // Restrict to positive x
   x = std::max(x, 0.0);
   const double fac = commonFactorHyperbolicPlus(x, tau, dm);
   const double arg = 0.5 * dm * x;
   return fac != 0.0 ? fac * (tau * dm * std::sinh(arg) + 2. * std::cosh(arg)) : 0.0;
}

// Integrate one of the basis functions. Takes a function that represents the
// indefinite integral, some parameters, and a flag that indicates whether the
// basis function is symmetric or antisymmetric. This information is used to
// evaluate the integrals for the "Minus" and "Sum" cases.
template <class Function>
double definiteIntegral(Function indefiniteIntegral, double xmin, double xmax, double tau, double dm,
                        BasisSign basisSign, bool isSymmetric)
{
   // Note: isSymmetric == false implies antisymmetric
   if (tau == 0.0)
      return isSymmetric ? 1.0 : 0.0;
   double result = 0.0;
   if (basisSign != Minus) {
      result += indefiniteIntegral(xmax, tau, dm) - indefiniteIntegral(xmin, tau, dm);
   }
   if (basisSign != Plus) {
      const double resultMinus = indefiniteIntegral(-xmax, tau, dm) - indefiniteIntegral(-xmin, tau, dm);
      result += isSymmetric ? -resultMinus : resultMinus;
   }
   return result;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrals when used as p.d.f and for compiled
/// basis functions.

double RooTruthModel::analyticalIntegral(Int_t code, const char *rangeName) const
{
   // Code must be 1
   R__ASSERT(code == 1);

   // Unconvoluted PDF
   if (_basisCode == noBasis)
      return 1;

   // Precompiled basis functions
   BasisType basisType = (BasisType)((_basisCode == 0) ? 0 : (_basisCode / 10) + 1);
   BasisSign basisSign = (BasisSign)(_basisCode - 10 * (basisType - 1) - 2);

   const bool needsDm =
      basisType == sinBasis || basisType == cosBasis || basisType == sinhBasis || basisType == coshBasis;

   const double tau = (static_cast<RooAbsReal *>(basis().getParameter(1)))->getVal();
   const double dm =
      needsDm ? (static_cast<RooAbsReal *>(basis().getParameter(2)))->getVal() : std::numeric_limits<Double_t>::quiet_NaN();

   const double xmin = x.min(rangeName);
   const double xmax = x.max(rangeName);

   auto integrate = [&](auto indefiniteIntegral, bool isSymmetric) {
      return definiteIntegral(indefiniteIntegral, xmin, xmax, tau, dm, basisSign, isSymmetric);
   };

   switch (basisType) {
   case expBasis: return integrate(indefiniteIntegralExpBasisPlus, /*isSymmetric=*/true);
   case sinBasis: return integrate(indefiniteIntegralSinBasisPlus, /*isSymmetric=*/false);
   case cosBasis: return integrate(indefiniteIntegralCosBasisPlus, /*isSymmetric=*/true);
   case linBasis: return integrate(indefiniteIntegralLinBasisPlus, /*isSymmetric=*/false);
   case quadBasis: return integrate(indefiniteIntegralQuadBasisPlus, /*isSymmetric=*/true);
   case sinhBasis: return integrate(indefiniteIntegralSinhBasisPlus, /*isSymmetric=*/false);
   case coshBasis: return integrate(indefiniteIntegralCoshBasisPlus, /*isSymmetric=*/true);
   default: R__ASSERT(0);
   }

   R__ASSERT(0);
   return 0;
}


////////////////////////////////////////////////////////////////////////////////

RooAbsGenContext* RooTruthModel::modelGenContext
(const RooAbsAnaConvPdf& convPdf, const RooArgSet &vars, const RooDataSet *prototype,
 const RooArgSet* auxProto, bool verbose) const
{
   RooArgSet forceDirect(convVar()) ;
   return new RooGenContext(convPdf, vars, prototype, auxProto, verbose, &forceDirect);
}



////////////////////////////////////////////////////////////////////////////////
/// Advertise internal generator for observable x

Int_t RooTruthModel::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implement internal generator for observable x,
/// x=0 for all events following definition
/// of delta function

void RooTruthModel::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;
  double zero(0.) ;
  x = zero ;
  return;
}
