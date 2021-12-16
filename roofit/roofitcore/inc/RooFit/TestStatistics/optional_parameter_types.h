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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_optional_parameter_types
#define ROOT_ROOFIT_TESTSTATISTICS_optional_parameter_types

#include <RooArgSet.h>

namespace RooFit {
namespace TestStatistics {

// strongly named container types for use as optional parameters in the test statistics constructors

/// Optional parameter used in buildLikelihood(), see documentation there.
struct ConstrainedParameters {
   ConstrainedParameters();
   explicit ConstrainedParameters(const RooArgSet &parameters);
   RooArgSet set;
};

/// Optional parameter used in buildLikelihood(), see documentation there.
struct ExternalConstraints {
   ExternalConstraints();
   explicit ExternalConstraints(const RooArgSet &constraints);
   RooArgSet set;
};

/// Optional parameter used in buildLikelihood(), see documentation there.
struct GlobalObservables {
   GlobalObservables();
   explicit GlobalObservables(const RooArgSet &observables);
   RooArgSet set;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_optional_parameter_types
