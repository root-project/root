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
