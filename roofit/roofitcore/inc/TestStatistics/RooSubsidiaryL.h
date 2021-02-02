/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL
#define ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL

#include <TestStatistics/RooAbsL.h>

namespace RooFit {
namespace TestStatistics {

/// Gathers all subsidiary PDF terms from the component PDFs of RooSimultaneousL likelihoods.
/// These are summed separately for increased numerical stability, since these terms are often
/// small and cause numerical variances in their original PDFs, whereas by summing as one
/// separate subsidiary collective term, it is numerically very stable.
/// Note that when a subsidiary PDF is part of multiple component PDFs, it will only be summed
/// once in this class! This doesn't change the derivative of the log likelihood (which is what
/// matters in fitting the likelihood), but does change the value of the (log-)likelihood itself.
class RooSubsidiaryL : RooAbsL {
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL
