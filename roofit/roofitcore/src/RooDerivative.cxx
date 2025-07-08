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
\file RooDerivative.cxx
\class RooDerivative
\ingroup Roofitcore

Represents the first, second, or third order derivative
of any RooAbsReal as calculated (numerically) by the MathCore Richardson
derivator class.
**/

#include "Riostream.h"
#include <cmath>

#include "RooDerivative.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooFunctor.h"

#include "Math/WrappedFunction.h"
#include "Math/RichardsonDerivator.h"




////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooDerivative::RooDerivative() = default;

////////////////////////////////////////////////////////////////////////////////

RooDerivative::RooDerivative(const char* name, const char* title, RooAbsReal& func, RooRealVar& x, Int_t orderIn, double epsIn) :
  RooAbsReal(name, title),
  _order(orderIn),
  _eps(epsIn),
  _nset("nset","nset",this,false,false),
  _func("function","function",this,func),
  _x("x","x",this,x)
{
  if (_order<0 || _order>3 ) {
    throw std::runtime_error(Form("RooDerivative::ctor(%s) ERROR, derivation order must be 1,2 or 3",name)) ;
  }
}

////////////////////////////////////////////////////////////////////////////////

RooDerivative::RooDerivative(const char *name, const char *title, RooAbsReal &func, RooRealVar &x,
                             const RooArgSet &nset, Int_t orderIn, double epsIn)
   : RooDerivative(name, title, func, x, orderIn, epsIn)
{
   _nset.add(nset);
}

////////////////////////////////////////////////////////////////////////////////

RooDerivative::RooDerivative(const RooDerivative& other, const char* name) :
  RooAbsReal(other, name),
  _order(other._order),
  _eps(other._eps),
  _nset("nset",this,other._nset),
  _func("function",this,other._func),
  _x("x",this,other._x)
{
}

RooDerivative::~RooDerivative() = default;

////////////////////////////////////////////////////////////////////////////////
/// Calculate value

double RooDerivative::evaluate() const
{
   if (!_ftor) {
      _ftor = std::unique_ptr<RooFunctor>{_func.arg().functor(_x.arg(), RooArgSet(), _nset)};
      ROOT::Math::WrappedFunction<RooFunctor &> wf(*_ftor);
      _rd = std::make_unique<ROOT::Math::RichardsonDerivator>(wf, _eps, true);
   }

   // Figure out if we are close to the variable boundaries
   double val = _x;
   auto &xVar = static_cast<RooRealVar &>(*_x);
   double valMin = xVar.getMin();
   double valMax = xVar.getMax();
   bool isCloseLo = val - valMin < _eps;
   bool isCloseHi = valMax - val < _eps;

   // If we hit the boundary left and right, there is obviously a mistake when setting epsilon
   if (isCloseLo && isCloseHi) {
      std::stringstream errMsg;
      errMsg << "error in numerical derivator: 2 * epsilon is larger than the variable range!";
      coutE(Eval) << errMsg.str() << std::endl;
      throw std::runtime_error(errMsg.str());
   }

   // If we are close to the variable boundary on either side
   if (isCloseLo || isCloseHi) {
      // For first-order derivatives we are good: the RichardsonDerivator can
      // also calculate the derivative using only forward or backward
      // variations:
      if (_order == 1) {
         return isCloseLo ? _rd->DerivativeForward(val) : _rd->DerivativeBackward(val);
      }
      // If the function is constant within floating point precision anyway,
      // we don't have a problem.
      const double eps = std::numeric_limits<double>::epsilon();
      const double yval1 = _ftor->eval(val);
      const double yval2 = isCloseLo ? _ftor->eval(val + _eps) : _ftor->eval(val - _eps);
      if (std::abs(yval2 - yval1) <= eps) {
         return 0.0;
      }

      // Give up
      std::stringstream errMsg;
      errMsg << "error in numerical derivator: variable value is to close to limits to compute finite differences";
      coutE(Eval) << errMsg.str() << std::endl;
      throw std::runtime_error(errMsg.str());
   }

   switch (_order) {
   case 1: return _rd->Derivative1(val);
   case 2: return _rd->Derivative2(val);
   case 3: return _rd->Derivative3(val);
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Zap functor and derivator ;

bool RooDerivative::redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive)
{
  _ftor = nullptr ;
  _rd = nullptr ;
  return RooAbsReal::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursive);
}
