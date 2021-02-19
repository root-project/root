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
#include <TestStatistics/optional_parameter_types.h>

namespace RooFit {
namespace TestStatistics {

class RooSimultaneousL : public RooAbsL {
public:
   // main constructor
   RooSimultaneousL(RooAbsPdf* pdf, RooAbsData* data, RooAbsL::Extended extended = RooAbsL::Extended::Auto,
                    ConstrainedParameters constrained_parameters = {}, ExternalConstraints external_constraints = {},
                    GlobalObservables global_observables = {}, std::string global_observables_tag = {});

   // delegating constructors to main constructor, for more convenient "optional" parameter passing
   RooSimultaneousL(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters);
   RooSimultaneousL(RooAbsPdf* pdf, RooAbsData* data, ExternalConstraints external_constraints);
   RooSimultaneousL(RooAbsPdf* pdf, RooAbsData* data, GlobalObservables global_observables);
   RooSimultaneousL(RooAbsPdf* pdf, RooAbsData* data, std::string global_observables_tag);

   double evaluate_partition(Section events, std::size_t components_begin,
                             std::size_t components_end) override;

private:
   bool processEmptyDataSets() const;

   std::vector<std::unique_ptr<RooAbsL>> components_;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooSimultaneousL
