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

//////////////////////////////////////////////////////////////////////////////
///
/// Class RooRealSumFunc implements a PDF constructed from a sum of
/// functions:
/// ```
///                 Sum(i=1,n-1) coef_i * func_i(x) + [ 1 - (Sum(i=1,n-1) coef_i ] * func_n(x)
///   pdf(x) =    ------------------------------------------------------------------------------
///             Sum(i=1,n-1) coef_i * Int(func_i)dx + [ 1 - (Sum(i=1,n-1) coef_i ] * Int(func_n)dx
///
/// ```
/// where coef_i and func_i are RooAbsReal objects, and x is the collection of dependents.
/// In the present version coef_i may not depend on x, but this limitation may be removed in the future
///
/// ### Difference between RooAddPdf / RooRealSum{Func|Pdf}
/// - RooAddPdf is a PDF of PDFs, *i.e.* its components need to be normalised and non-negative.
/// - RooRealSumPdf is a PDF of functions, *i.e.*, its components can be negative, but their sum cannot be. The normalisation
///   is computed automatically, unless the PDF is extended (see above).
/// - RooRealSumFunc is a sum of functions. It is neither normalised, nor need it be positive.

#include "RooRealSumFunc.h"

#include "RooRealVar.h"
#include "RooAddGenContext.h"
#include "RooRealConstant.h"
#include "RooRealIntegral.h"
#include "RooMsgService.h"
#include "RooNameReg.h"
#include "RooTrace.h"

#include "Riostream.h"

#include <memory>

ClassImp(RooRealSumFunc);

bool RooRealSumFunc::_doFloorGlobal = false;

//_____________________________________________________________________________
RooRealSumFunc::RooRealSumFunc() : _normIntMgr(this, 10)
{
   // Default constructor
   // coverity[UNINIT_CTOR]
   TRACE_CREATE
}

//_____________________________________________________________________________
RooRealSumFunc::RooRealSumFunc(const char *name, const char *title)
   : RooAbsReal(name, title), _normIntMgr(this, 10), _haveLastCoef(false),
     _funcList("!funcList", "List of functions", this), _coefList("!coefList", "List of coefficients", this)
{
   // Constructor with name and title
   TRACE_CREATE
}

//_____________________________________________________________________________
RooRealSumFunc::RooRealSumFunc(const char *name, const char *title, RooAbsReal &func1, RooAbsReal &func2,
                               RooAbsReal &coef1)
   : RooAbsReal(name, title), _normIntMgr(this, 10), _haveLastCoef(false),
     _funcList("!funcList", "List of functions", this), _coefList("!coefList", "List of coefficients", this)
{
   // Construct p.d.f consisting of coef1*func1 + (1-coef1)*func2
   // The input coefficients and functions are allowed to be negative
   // but the resulting sum is not, which is enforced at runtime

   // Special constructor with two functions and one coefficient

   _funcList.add(func1);
   _funcList.add(func2);
   _coefList.add(coef1);
   TRACE_CREATE
}

//_____________________________________________________________________________
RooRealSumFunc::RooRealSumFunc(const char *name, const char *title, const RooArgList &inFuncList,
                               const RooArgList &inCoefList)
   : RooAbsReal(name, title), _normIntMgr(this, 10), _haveLastCoef(false),
     _funcList("!funcList", "List of functions", this), _coefList("!coefList", "List of coefficients", this)
{
   // Constructor p.d.f implementing sum_i [ coef_i * func_i ], if N_coef==N_func
   // or sum_i [ coef_i * func_i ] + (1 - sum_i [ coef_i ] )* func_N if Ncoef==N_func-1
   //
   // All coefficients and functions are allowed to be negative
   // but the sum is not, which is enforced at runtime.

   const std::string ownName(GetName() ? GetName() : "");
   if (!(inFuncList.getSize() == inCoefList.getSize() + 1 || inFuncList.getSize() == inCoefList.getSize())) {
      coutE(InputArguments) << "RooRealSumFunc::RooRealSumFunc(" << ownName
                            << ") number of pdfs and coefficients inconsistent, must have Nfunc=Ncoef or Nfunc=Ncoef+1"
                            << "\n";
      assert(0);
   }

   // Constructor with N functions and N or N-1 coefs

   std::string funcName;
   for (unsigned int i = 0; i < inCoefList.size(); ++i) {
      const auto& func = inFuncList[i];
      const auto& coef = inCoefList[i];

      if (!dynamic_cast<RooAbsReal const*>(&coef)) {
         const std::string coefName(coef.GetName() ? coef.GetName() : "");
         coutW(InputArguments) << "RooRealSumFunc::RooRealSumFunc(" << ownName << ") coefficient " << coefName
                               << " is not of type RooAbsReal, ignored"
                               << "\n";
         continue;
      }
      if (!dynamic_cast<RooAbsReal const*>(&func)) {
         funcName = (func.GetName() ? func.GetName() : "");
         coutW(InputArguments) << "RooRealSumFunc::RooRealSumFunc(" << ownName << ") func " << funcName
                               << " is not of type RooAbsReal, ignored"
                               << "\n";
         continue;
      }
      _funcList.add(func);
      _coefList.add(coef);
   }

   if (inFuncList.size() == inCoefList.size() + 1) {
      const auto& func = inFuncList[inFuncList.size()-1];
      if (!dynamic_cast<RooAbsReal const*>(&func)) {
         funcName = (func.GetName() ? func.GetName() : "");
         coutE(InputArguments) << "RooRealSumFunc::RooRealSumFunc(" << ownName << ") last func " << funcName
                               << " is not of type RooAbsReal, fatal error\n";
         throw std::invalid_argument("RooRealSumFunc: Function passed as is not of type RooAbsReal.");
      }
      _funcList.add(func);
   } else {
      _haveLastCoef = true;
   }

   TRACE_CREATE
}

//_____________________________________________________________________________
RooRealSumFunc::RooRealSumFunc(const RooRealSumFunc &other, const char *name)
   : RooAbsReal(other, name), _normIntMgr(other._normIntMgr, this), _haveLastCoef(other._haveLastCoef),
     _funcList("!funcList", this, other._funcList), _coefList("!coefList", this, other._coefList),
     _doFloor(other._doFloor)
{
   // Copy constructor

   TRACE_CREATE
}

//_____________________________________________________________________________
RooRealSumFunc::~RooRealSumFunc()
{
   TRACE_DESTROY
}

//_____________________________________________________________________________
double RooRealSumFunc::evaluate() const
{
  return RooRealSumPdf::evaluate(*this, _funcList, _coefList, _doFloor || _doFloorGlobal, _haveWarned);
}


//_____________________________________________________________________________
bool RooRealSumFunc::checkObservables(const RooArgSet *nset) const
{
  return RooRealSumPdf::checkObservables(*this, nset, _funcList, _coefList);
}

//_____________________________________________________________________________
Int_t RooRealSumFunc::getAnalyticalIntegralWN(RooArgSet &allVars, RooArgSet &analVars, const RooArgSet *normSet2,
                                              const char *rangeName) const
{
   return RooRealSumPdf::getAnalyticalIntegralWN(*this, _normIntMgr, _funcList, _coefList, allVars, analVars, normSet2, rangeName);
}

//_____________________________________________________________________________
double RooRealSumFunc::analyticalIntegralWN(Int_t code, const RooArgSet *normSet2, const char *rangeName) const
{
   return RooRealSumPdf::analyticalIntegralWN(*this, _normIntMgr, _funcList, _coefList, code, normSet2, rangeName, _haveWarned);
}

//_____________________________________________________________________________
std::list<double> *RooRealSumFunc::binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   return RooRealSumPdf::binBoundaries(_funcList, obs, xlo, xhi);
}

//_____________________________________________________________________________B
bool RooRealSumFunc::isBinnedDistribution(const RooArgSet &obs) const
{
   // If all components that depend on obs are binned that so is the product

   RooFIter iter = _funcList.fwdIterator();
   RooAbsReal *func;
   while ((func = (RooAbsReal *)iter.next())) {
      if (func->dependsOn(obs) && !func->isBinnedDistribution(obs)) {
         return false;
      }
   }

   return true;
}

//_____________________________________________________________________________
std::list<double> *RooRealSumFunc::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   return RooRealSumPdf::plotSamplingHint(_funcList, obs, xlo, xhi);
}

//_____________________________________________________________________________
void RooRealSumFunc::setCacheAndTrackHints(RooArgSet &trackNodes)
{
   // Label OK'ed components of a RooRealSumFunc with cache-and-track
   RooFIter siter = funcList().fwdIterator();
   RooAbsArg *sarg;
   while ((sarg = siter.next())) {
      if (sarg->canNodeBeCached() == Always) {
         trackNodes.add(*sarg);
         // cout << "tracking node RealSumFunc component " << sarg->ClassName() << "::" << sarg->GetName() << endl
         // ;
      }
   }
}

/// Customized printing of arguments of a RooRealSumFunc to more intuitively
/// reflect the contents of the product operator construction.

void RooRealSumFunc::printMetaArgs(std::ostream &os) const
{
   RooRealSumPdf::printMetaArgs(_funcList, _coefList, os);
}
