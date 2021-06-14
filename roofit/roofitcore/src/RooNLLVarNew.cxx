/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *   Emmanouil Michalainas, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooNLLVarNew.h"
#include "RooBatchCompute.h"

#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ROOT::Experimental;

RooNLLVarNew::RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf)
   : RooAbsReal(name, title), _pdf{"pdf", "pdf", this, pdf}
{
}

RooNLLVarNew::RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooAbsReal &weight)
   : RooAbsReal(name, title), _pdf{"pdf", "pdf", this, pdf}, _weight{"_weight", "_weight", this, weight}
{
}

RooNLLVarNew::RooNLLVarNew(const RooNLLVarNew &other, const char *name)
   : RooAbsReal(other, name), _pdf{"pdf", this, other._pdf}, _weight{"_weight", this, other._weight}
{
}

void RooNLLVarNew::computeBatch(double *output, size_t nEvents, RooBatchCompute::DataMap &dataMap) const
{
   RooBatchCompute::VarVector vars = {&*_pdf};
   if (_weight.operator->())
      vars.push_back(&*_weight);
   RooBatchCompute::ArgVector args = {static_cast<double>(vars.size() - 1)};
   RooBatchCompute::dispatch->compute(RooBatchCompute::NegativeLogarithms, output, nEvents, dataMap, vars, args);
}

double RooNLLVarNew::reduce(const double *input, size_t nEvents) const
{
   return RooBatchCompute::dispatch->sumReduce(input, nEvents);
}

double RooNLLVarNew::getValV(const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::getValV was called directly which should not happen!");
}

double RooNLLVarNew::evaluate() const
{
   throw std::runtime_error("RooNLLVarNew::evaluate was called directly which should not happen!");
}

RooSpan<double> RooNLLVarNew::evaluateSpan(RooBatchCompute::RunContext &, const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::evaluatSpan was called directly which should not happen!");
}

RooSpan<const double> RooNLLVarNew::getValues(RooBatchCompute::RunContext &, const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::getValues was called directly which should not happen!");
}

bool RooNLLVarNew::getParameters(const RooArgSet *depList, RooArgSet &outSet, bool stripDisconnected) const
{
   return _pdf->getParameters(depList, outSet, stripDisconnected);
}
