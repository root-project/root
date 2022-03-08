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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooBinnedL
#define ROOT_ROOFIT_TESTSTATISTICS_RooBinnedL

#include <RooFit/TestStatistics/RooAbsL.h>
#include "RooAbsReal.h"
#include <vector>

// forward declarations
class RooAbsPdf;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

class RooBinnedL : public RooAbsL {
public:
   RooBinnedL(RooAbsPdf *pdf, RooAbsData *data);
   ROOT::Math::KahanSum<double>
   evaluatePartition(Section bins, std::size_t components_begin, std::size_t components_end) override;

   virtual std::string GetClassName() const override { return "RooBinnedL"; };

private:
   mutable bool _first = true;        ///<!
   mutable std::vector<double> _binw; ///<!
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooBinnedL
