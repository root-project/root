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
#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooSimultaneousL
#define ROOT_ROOFIT_TESTSTATISTICS_RooSimultaneousL

#include <TestStatistics/RooAbsL.h>

namespace RooFit {
namespace TestStatistics {

class RooSimultaneousL : public RooAbsL {
public:
   RooSimultaneousL(RooAbsPdf* pdf, RooAbsData* data, RooAbsL::Extended extended = RooAbsL::Extended::Auto);

   double evaluate_partition(Section events, std::size_t components_begin,
                             std::size_t components_end) override;

private:
   bool processEmptyDataSets() const;

   std::vector<std::unique_ptr<RooAbsL>> components_;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooSimultaneousL
