// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <TestStatistics/RooRealL.h>
#include <TestStatistics/RooAbsL.h>

namespace RooFit {
namespace TestStatistics {

/** \class RooRealL
 * \ingroup Roofitcore
 *
 * \brief RooAbsReal that wraps RooAbsL likelihoods for use in RooFit outside of the RooMinimizer context
 *
 * This class provides a simple wrapper to evaluate RooAbsL derived likelihood objects like a regular RooFit real value.
 * Whereas the RooAbsL objects are meant to be used within the context of minimization, RooRealL can be used in any
 * RooFit context, like plotting. The value can be accessed through getVal(), like with other RooFit real variables.
 **/

RooRealL::RooRealL(const char *name, const char *title, std::shared_ptr<RooAbsL> likelihood)
   : RooAbsReal(name, title), likelihood_(std::move(likelihood)),
     vars_proxy_("varsProxy", "proxy set of parameters", this)
{
   vars_proxy_.add(*likelihood_->getParameters());
}

RooRealL::RooRealL(const RooRealL &other, const char *name)
   : RooAbsReal(other, name), likelihood_(other.likelihood_), vars_proxy_("varsProxy", "proxy set of parameters", this)
{
   vars_proxy_.add(*likelihood_->getParameters());
}

Double_t RooRealL::evaluate() const
{
   // Evaluate as straight FUNC
   std::size_t last_component = likelihood_->getNComponents();

   Double_t ret = likelihood_->evaluatePartition({0, 1}, 0, last_component);

   const Double_t norm = globalNormalization();
   ret /= norm;
   eval_carry = likelihood_->getCarry() / norm;

   return ret;
}

} // namespace TestStatistics
} // namespace RooFit
