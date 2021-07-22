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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_likelihood_builders
#define ROOT_ROOFIT_TESTSTATISTICS_likelihood_builders

#include "TestStatistics/RooAbsL.h"
#include "TestStatistics/optional_parameter_types.h"

#include <memory>

// forward declarations
class RooAbsPdf;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended = RooAbsL::Extended::Auto,
                                                       ConstrainedParameters constrained_parameters = {}, ExternalConstraints external_constraints = {},
                                                       GlobalObservables global_observables = {}, std::string global_observables_tag = {});

// delegating builder calls, for more convenient "optional" parameter passing
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters);
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, ExternalConstraints external_constraints);
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, GlobalObservables global_observables);
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, std::string global_observables_tag);
std::shared_ptr<RooAbsL> buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters, GlobalObservables global_observables);

std::shared_ptr<RooAbsL> buildUnbinnedConstrainedLikelihood(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended = RooAbsL::Extended::Auto,
                                                               ConstrainedParameters constrained_parameters = {}, ExternalConstraints external_constraints = {},
                                                               GlobalObservables global_observables = {}, std::string global_observables_tag = {});

// delegating builder calls, for more convenient "optional" parameter passing
std::shared_ptr<RooAbsL>
buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters);
std::shared_ptr<RooAbsL> buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, ExternalConstraints external_constraints);
std::shared_ptr<RooAbsL> buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, GlobalObservables global_observables);
std::shared_ptr<RooAbsL> buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, std::string global_observables_tag);

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_likelihood_builders
