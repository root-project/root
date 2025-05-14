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
#include "RooFit/VariableGroups.h"
#include "RooMinimizer.h"
#include "RooMsgService.h"
#include "RooNaNPacker.h"
#include "RooRealVar.h"

#include "Math/Functor.h"
#include "TMatrixDSym.h"

#include <fstream>
#include <iomanip>

using std::setprecision;

namespace {

template <class InputIt1, class InputIt2>
bool intersect(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
{
   while (first1 != last1 && first2 != last2) {
      if (*first1 < *first2) {
         ++first1;
         continue;
      }
      if (*first2 < *first1) {
         ++first2;
         continue;
      }
      return true;
   }
   return false;
}

// Helper function that wraps RooAbsArg::getParameters and directly returns the
// output RooArgSet. To be used in the initializer list of the RooMinimizerFcn
// constructor.
RooArgSet getParameters(RooAbsReal const &funct)
{
   RooArgSet out;
   funct.getParameters(nullptr, out);
   return out;
}

} // namespace

// use reference wrapper for the Functor, such that the functor points to this RooMinimizerFcn by reference.
RooMinimizerFcn::RooMinimizerFcn(RooAbsReal *funct, RooMinimizer *context)
   : RooAbsMinimizerFcn(getParameters(*funct), context), _funct(funct)
{
   RooFit::VariableGroups groups;
   funct->fillVariableGroups(groups);

   RooArgList parameters;
   for (std::size_t i = 0; i < getNDim(); ++i) {
      parameters.add(floatableParam(i));
   }

   std::size_t nParams = parameters.size();

   _secondDerivMask.resize(nParams * nParams);
   for (std::size_t i = 0; i < nParams; ++i) {
      _secondDerivMask[nParams * i + i] = 1;
      for (std::size_t j = 0; j < i; ++j) {
         // std::cout << parameters[i].GetName() << "   " << parameters[j].GetName() << std::endl;
         auto const &gr1 = groups.groups.at(parameters[i].namePtr());
         auto const &gr2 = groups.groups.at(parameters[j].namePtr());
         _secondDerivMask[nParams * i + j] = intersect(gr1.begin(), gr1.end(), gr2.begin(), gr2.end());
         _secondDerivMask[nParams * j + i] = _secondDerivMask[nParams * i + j];
      }
   }

   if (context->_cfg.useGradient && funct->hasGradient()) {
      _gradientOutput.resize(_allParams.size());
      auto functor = std::make_unique<ROOT::Math::GradFunctor>(this, &RooMinimizerFcn::operator(),
                                                               &RooMinimizerFcn::evaluateGradient, getNDim());
      functor->SetVanishingSecondDerivativeFunc([this](int i, int j) { return this->vanishingSecondDerivative(i, j); });
      _multiGenFcn = std::move(functor);
   } else {
      auto functor = std::make_unique<ROOT::Math::Functor>(std::cref(*this), getNDim());
      functor->SetVanishingSecondDerivativeFunc([this](int i, int j) { return this->vanishingSecondDerivative(i, j); });
      _multiGenFcn = std::move(functor);
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

bool RooMinimizerFcn::vanishingSecondDerivative(int i, int j) const
{
   return _secondDerivMask[getNDim() * i + j] == 0;
}

/// \endcond
