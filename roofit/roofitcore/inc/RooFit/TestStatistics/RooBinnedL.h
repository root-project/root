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

#include "Math/Util.h" // KahanSum

#include <vector>

// forward declarations
class RooAbsPdf;
class RooAbsData;
class RooChangeTracker;

namespace RooFit {
namespace TestStatistics {

class RooBinnedL : public RooAbsL {
public:
   RooBinnedL(RooAbsPdf *pdf, RooAbsData *data);
   ~RooBinnedL();
   ROOT::Math::KahanSum<double>
   evaluatePartition(Section bins, std::size_t components_begin, std::size_t components_end) override;

private:
   mutable bool _first = true;        ///<!
   mutable std::vector<double> _binw; ///<!
   std::unique_ptr<RooChangeTracker> paramTracker_;
   mutable ROOT::Math::KahanSum<double> cachedResult_ = 0;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooBinnedL
