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

#include <RooFit/TestStatistics/RooAbsL.h>
#include <RooFit/TestStatistics/optional_parameter_types.h>

#include <memory>

// forward declarations
class RooAbsPdf;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

std::unique_ptr<RooAbsL>
buildLikelihood(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended = RooAbsL::Extended::Auto,
                ConstrainedParameters constrained_parameters = {}, ExternalConstraints external_constraints = {},
                GlobalObservables global_observables = {}, std::string global_observables_tag = {});

// delegating builder calls, for more convenient "optional" parameter passing
std::unique_ptr<RooAbsL>
buildLikelihood(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters);
std::unique_ptr<RooAbsL> buildLikelihood(RooAbsPdf* pdf, RooAbsData* data, ExternalConstraints external_constraints);
std::unique_ptr<RooAbsL> buildLikelihood(RooAbsPdf* pdf, RooAbsData* data, GlobalObservables global_observables);
std::unique_ptr<RooAbsL> buildLikelihood(RooAbsPdf* pdf, RooAbsData* data, std::string global_observables_tag);
std::unique_ptr<RooAbsL>
buildLikelihood(RooAbsPdf *pdf, RooAbsData *data, ConstrainedParameters constrained_parameters, GlobalObservables global_observables);

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_likelihood_builders
