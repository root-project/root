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

#include <TestStatistics/optional_parameter_types.h>

namespace RooFit {
namespace TestStatistics {

ConstrainedParameters::ConstrainedParameters(const RooArgSet &parameters) : set(parameters) {}
// N.B.: the default constructor must be _user-provided_ defaulted, otherwise aggregate initialization will be possible,
// which bypasses the explicit constructor and even leads to errors in some compilers; when initializing as
//   ConstrainedParameters({someRooArgSet})
// compilers can respond with
//   call of overloaded ‘GlobalObservables(<brace-enclosed initializer list>)’ is ambiguous
// This problem is well documented in http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1008r0.pdf. The solution
// used here is detailed in section 1.4 of that paper. Note that in C++17, the workaround is no longer necessary and
// the constructor can be _user-declared_ default (i.e. directly in the declaration above).
ConstrainedParameters::ConstrainedParameters() = default;

ExternalConstraints::ExternalConstraints(const RooArgSet &constraints) : set(constraints) {}
ExternalConstraints::ExternalConstraints() = default;

GlobalObservables::GlobalObservables(const RooArgSet &observables) : set(observables) {}
GlobalObservables::GlobalObservables() = default;

}
}