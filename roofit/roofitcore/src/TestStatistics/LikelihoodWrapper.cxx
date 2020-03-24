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
#include <TestStatistics/LikelihoodWrapper.h>

namespace RooFit {
namespace TestStatistics {

LikelihoodWrapper::LikelihoodWrapper(std::shared_ptr<RooAbsL> _likelihood) : likelihood(std::move(_likelihood)) {}

void LikelihoodWrapper::synchronize_with_minimizer(const ROOT::Math::MinimizerOptions & options) {}

}
}
