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
#include "RooRealSumPdf.h"
#include "RooTrace.h"

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
   : RooRealSumFunc{name, title}
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
   : RooRealSumFunc{name, title}
{
   // Constructor p.d.f implementing sum_i [ coef_i * func_i ], if N_coef==N_func
   // or sum_i [ coef_i * func_i ] + (1 - sum_i [ coef_i ] )* func_N if Ncoef==N_func-1
   //
   // All coefficients and functions are allowed to be negative
   // but the sum is not, which is enforced at runtime.

   RooRealSumPdf::initializeFuncsAndCoefs(*this, inFuncList, inCoefList, _funcList, _coefList);

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
   return RooRealSumPdf::isBinnedDistribution(_funcList, obs);
}

//_____________________________________________________________________________
std::list<double> *RooRealSumFunc::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   return RooRealSumPdf::plotSamplingHint(_funcList, obs, xlo, xhi);
}

//_____________________________________________________________________________
void RooRealSumFunc::setCacheAndTrackHints(RooArgSet &trackNodes)
{
   RooRealSumPdf::setCacheAndTrackHints(_funcList, trackNodes);
}

/// Customized printing of arguments of a RooRealSumFunc to more intuitively
/// reflect the contents of the product operator construction.

void RooRealSumFunc::printMetaArgs(std::ostream &os) const
{
   RooRealSumPdf::printMetaArgs(_funcList, _coefList, os);
}
