/// \cond ROOFIT_INTERNAL

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl   *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/// \class RooMinimizerFcn
/// RooMinimizerFcn is an interface to the ROOT::Math::IBaseFunctionMultiDim,
/// a function that ROOT's minimisers use to carry out minimisations.
///

#include "RooMinimizerFcn.h"

#include "RooAbsArg.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooMsgService.h"
#include "RooMinimizer.h"
#include "RooNaNPacker.h"
#include "RooCategory.h"

#include "Math/Functor.h"
#include "TMatrixDSym.h"

#include <fstream>
#include <iomanip>

using std::setprecision;

namespace {

// Helper function that wraps RooAbsArg::getParameters and directly returns the
// output RooArgSet. To be used in the initializer list of the RooMinimizerFcn
// constructor. In the case of figuring out all parameters for the minimizer,
// we don't want to strip disconnected parameters, becuase which parameters are
// disconnected can change between minimization runs.
RooArgSet getAllParameters(RooAbsReal const &funct)
{
   RooArgSet out;
   funct.getParameters(nullptr, out, /*stripDisconnected*/ false);
   return out;
}

} // namespace

// use reference wrapper for the Functor, such that the functor points to this RooMinimizerFcn by reference.
RooMinimizerFcn::RooMinimizerFcn(RooAbsReal *funct, RooMinimizer *context)
   : RooAbsMinimizerFcn(getAllParameters(*funct), context), _funct(funct)
{
   unsigned int nDim = getNDim();

   if (context->_cfg.useGradient && funct->hasGradient()) {
      _gradientOutput.resize(_allParams.size());
      _multiGenFcn = std::make_unique<ROOT::Math::GradFunctor>(this, &RooMinimizerFcn::operator(),
                                                               &RooMinimizerFcn::evaluateGradient, nDim);
   } else {
      _multiGenFcn = std::make_unique<ROOT::Math::Functor>(std::cref(*this), getNDim());
   }
   if (context->_cfg.useHessian) {
      _hessianOutput.resize(_allParams.size() * _allParams.size());
   }
}

void RooMinimizerFcn::setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt)
{
   _funct->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
}

/// Evaluate function given the parameters in `x`.
double RooMinimizerFcn::operator()(const double *x) const
{
   // Set the parameter values for this iteration
   for (unsigned index = 0; index < getNDim(); index++) {
      if (_logfile)
         (*_logfile) << x[index] << " ";
      SetPdfParamVal(index, x[index]);
   }

   // Calculate the function for these parameters
   RooAbsReal::setHideOffset(false);
   double fvalue = _funct->getVal();
   RooAbsReal::setHideOffset(true);

   fvalue = applyEvalErrorHandling(fvalue);

   // Optional logging
   if (_logfile)
      (*_logfile) << setprecision(15) << fvalue << setprecision(4) << std::endl;
   if (cfg().verbose) {
      std::cout << "\nprevFCN" << (_funct->isOffsetting() ? "-offset" : "") << " = " << setprecision(10) << fvalue
                << setprecision(4) << "  ";
      std::cout.flush();
   }

   finishDoEval();

   return fvalue;
}

void RooMinimizerFcn::evaluateGradient(const double *x, double *out) const
{
   // Set the parameter values for this iteration
   for (unsigned index = 0; index < getNDim(); index++) {
      if (_logfile)
         (*_logfile) << x[index] << " ";
      SetPdfParamVal(index, x[index]);
   }

   _funct->gradient(_gradientOutput.data());

   std::size_t iAll = 0;
   std::size_t iFloating = 0;
   for (RooAbsArg *param : _allParamsInit) {
      if (!treatAsConstant(*param)) {
         out[iFloating] = _gradientOutput[iAll];
         ++iFloating;
      }
      ++iAll;
   }

   // Optional logging
   if (cfg().verbose) {
      std::cout << "\n    gradient = ";
      for (std::size_t i = 0; i < getNDim(); ++i) {
         std::cout << out[i] << ", ";
      }
   }
}

std::string RooMinimizerFcn::getFunctionName() const
{
   return _funct->GetName();
}

std::string RooMinimizerFcn::getFunctionTitle() const
{
   return _funct->GetTitle();
}

void RooMinimizerFcn::setOffsetting(bool flag)
{
   _funct->enableOffsetting(flag);
}

RooArgSet RooMinimizerFcn::freezeDisconnectedParameters() const
{

   RooArgSet paramsDisconnected;
   RooArgSet paramsConnected;

   _funct->getParameters(nullptr, paramsDisconnected, /*stripDisconnected*/ false);
   _funct->getParameters(nullptr, paramsConnected, /*stripDisconnected*/ true);

   paramsDisconnected.remove(paramsConnected, true, true);

   RooArgSet changedSet;

   for (RooAbsArg *a : paramsDisconnected) {
      auto *v = dynamic_cast<RooRealVar *>(a);
      auto *cv = dynamic_cast<RooCategory *>(a);
      if (v && !v->isConstant()) {
         v->setConstant();
         changedSet.add(*v);
      } else if (cv && !cv->isConstant()) {
         cv->setConstant();
         changedSet.add(*cv);
      }
   }

   return changedSet;
}

bool RooMinimizerFcn::evaluateHessian(std::span<const double> x, double *out) const
{
   // Set the parameter values for this iteration
   for (unsigned index = 0; index < getNDim(); index++) {
      if (_logfile)
         (*_logfile) << x[index] << " ";
      SetPdfParamVal(index, x[index]);
   }

   _funct->hessian(_hessianOutput.data());

   std::size_t m = _allParamsInit.size();
   std::size_t n = getNDim();
   std::size_t iAll = 0;
   std::size_t iFloating = 0;
   for (RooAbsArg *param_i : _allParamsInit) {
      if (!treatAsConstant(*param_i)) {
         std::size_t jAll = 0;
         std::size_t jFloating = 0;
         for (RooAbsArg *param_j : _allParamsInit) {
            if (!treatAsConstant(*param_j)) {
               out[iFloating * n + jFloating] = _hessianOutput[iAll * m + jAll];
               ++jFloating;
            }
            ++jAll;
         }
         ++iFloating;
      }
      ++iAll;
   }

   // Optional logging
   if (cfg().verbose) {
      std::cout << "\n    hessian = " << std::endl;
      for (std::size_t i = 0; i < getNDim(); ++i) {
         for (std::size_t j = 0; j < getNDim(); ++j) {
            std::cout << out[i * n + j] << ", ";
         }
         std::cout << std::endl;
      }
   }
   return true;
}

void RooMinimizerFcn::initMinimizer(ROOT::Math::Minimizer &minim, RooMinimizer *context)
{
   minim.SetFunction(*_multiGenFcn);
   if (context->_cfg.useHessian && _funct->hasHessian()) {
      minim.SetHessianFunction(
         std::bind(&RooMinimizerFcn::evaluateHessian, this, std::placeholders::_1, std::placeholders::_2));
   }
}

/// \endcond
