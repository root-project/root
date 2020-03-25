/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "TestStatistics/MinuitFcnGrad.h"
#include "../../inc/TestStatistics/MinuitFcnGrad.h"

namespace RooFit {
namespace TestStatistics {

MinuitFcnGrad::MinuitFcnGrad(const MinuitFcnGrad &other)
   : ROOT::Math::IMultiGradFunction(other), _evalCounter(other._evalCounter), _maxFCN(other._maxFCN),
     _numBadNLL(other._numBadNLL), _printEvalErrors(other._printEvalErrors), _doEvalErrorWall(other._doEvalErrorWall),
     _nDim(other._nDim), _floatParamVec(other._floatParamVec), _context(other._context), _verbose(other._verbose)
{
   _floatParamList = new RooArgList(*other._floatParamList);
   _constParamList = new RooArgList(*other._constParamList);
   _initFloatParamList = (RooArgList *)other._initFloatParamList->snapshot(kFALSE);
   _initConstParamList = (RooArgList *)other._initConstParamList->snapshot(kFALSE);
}

// IMultiGradFunction overrides necessary for Minuit: DoEval, Gradient, G2ndDerivative and GStepSize
// The likelihood and gradient wrappers do the actual calculations.

double MinuitFcnGrad::DoEval(const double *x) const
{
   _evalCounter++;
   return likelihood->get_value(x);
}

void MinuitFcnGrad::Gradient(const double *x, double *grad) const
{
   gradient->fill_gradient(x, grad);
}

void MinuitFcnGrad::G2ndDerivative(const double *x, double *g2) const
{
   gradient->fill_second_derivative(x, g2);
}

void MinuitFcnGrad::GStepSize(const double *x, double *gstep) const
{
   gradient->fill_step_size(x, gstep);
}

} // namespace TestStatistics
} // namespace RooFit
