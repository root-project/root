/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROOT_ROOFIT_TESTSTATISTICS_GRAD_MINIMIZER
#define ROOT_ROOFIT_TESTSTATISTICS_GRAD_MINIMIZER

#include <TestStatistics/Minimizer.h>

namespace RooFit {
namespace TestStatistics {

// Same as in Minimizer, GradMinimizerFcn must be forward declared here
// to avoid circular dependency problems. The GradMinimizerFcn.h include
// must then be done below this, otherwise GradMinimizerFcn.h has no
// definition of GradMinimizer when needed.
class GradMinimizerFcn;
using GradMinimizer = MinimizerTemplate<GradMinimizerFcn, MinimizerType::Minuit2>;

} // namespace TestStatistics
} // namespace RooFit

#include <TestStatistics/GradMinimizerFcn.h>

#endif // ROOT_ROOFIT_TESTSTATISTICS_GRAD_MINIMIZER
