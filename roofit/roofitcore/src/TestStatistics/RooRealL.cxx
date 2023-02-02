/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/TestStatistics/RooRealL.h>
#include <RooFit/TestStatistics/RooAbsL.h>
#include <RooRealVar.h>

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
   vars_obs_.add(*likelihood_->getParameters());
   vars_proxy_.add(*likelihood_->getParameters());
}

RooRealL::RooRealL(const RooRealL &other, const char *name)
   : RooAbsReal(other, name), likelihood_(other.likelihood_), vars_proxy_("varsProxy", this, other.vars_proxy_)
{
   vars_obs_.add(other.vars_obs_) ;
}

double RooRealL::evaluate() const
{
   // Transfer values from proxy variables to internal variables of likelihood
   if (!vars_proxy_.empty()) {
      for (auto i = 0u; i < vars_obs_.size(); ++i) {
         auto harg = vars_obs_[i];
         const auto parg = vars_proxy_[i];

         if (harg != parg) {
            ((RooAbsRealLValue*)harg)->setVal(((RooAbsReal*)parg)->getVal());
         }
      }
   }
   // Evaluate as straight FUNC
   std::size_t last_component = likelihood_->getNComponents();

   auto ret_kahan = likelihood_->evaluatePartition({0, 1}, 0, last_component);

   const double norm = globalNormalization();
   double ret = ret_kahan.Sum() / norm;
   eval_carry = ret_kahan.Carry() / norm;

   return ret;
}

} // namespace TestStatistics
} // namespace RooFit
