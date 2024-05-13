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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_likelihood_builders
#define ROOT_ROOFIT_TESTSTATISTICS_likelihood_builders

#include <RooGlobalFunc.h>
#include <RooFit/TestStatistics/RooAbsL.h>

#include <memory>

// forward declarations
class RooAbsPdf;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

class NLLFactory {
public:
   NLLFactory(RooAbsPdf &pdf, RooAbsData &data);
   std::unique_ptr<RooAbsL> build();

   NLLFactory &Extended(RooAbsL::Extended extended);
   NLLFactory &ConstrainedParameters(const RooArgSet &constrainedParameters);
   NLLFactory &ExternalConstraints(const RooArgSet &externalconstraints);
   NLLFactory &GlobalObservables(const RooArgSet &globalObservables);
   NLLFactory &GlobalObservablesTag(const char *globalObservablesTag);
   NLLFactory &EvalBackend(RooFit::EvalBackend evalBackend);

private:
   std::vector<std::unique_ptr<RooAbsL>> getSimultaneousComponents();

   RooAbsPdf &_pdf;
   RooAbsData &_data;

   RooAbsL::Extended _extended = RooAbsL::Extended::Auto;
   RooArgSet _constrainedParameters;
   RooArgSet _externalConstraints;
   RooArgSet _globalObservables;
   std::string _globalObservablesTag;
   RooFit::EvalBackend _evalBackend = RooFit::EvalBackend::Legacy();
};

/// Delegating function to build a likelihood without additional arguments.
inline std::unique_ptr<RooAbsL> buildLikelihood(RooAbsPdf *pdf, RooAbsData *data)
{
   return NLLFactory{*pdf, *data}.build();
}

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_likelihood_builders
