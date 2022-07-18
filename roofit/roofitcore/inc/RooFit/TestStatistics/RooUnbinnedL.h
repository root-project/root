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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooUnbinnedL
#define ROOT_ROOFIT_TESTSTATISTICS_RooUnbinnedL

#include <RooFit/TestStatistics/RooAbsL.h>

#include "Math/Util.h" // KahanSum

// forward declarations
class RooAbsPdf;
class RooAbsData;
class RooArgSet;
namespace RooBatchCompute {
struct RunContext;
}
class RooChangeTracker;

namespace RooFit {
namespace TestStatistics {

class RooUnbinnedL : public RooAbsL {
public:
   RooUnbinnedL(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended = RooAbsL::Extended::Auto,
                bool useBatchedEvaluations = false);
   RooUnbinnedL(const RooUnbinnedL &other);
   ~RooUnbinnedL();
   bool setApplyWeightSquared(bool flag);

   ROOT::Math::KahanSum<double>
   evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end) override;

   void setUseBatchedEvaluations(bool flag);
private:
   bool apply_weight_squared = false;                              // Apply weights squared?
   mutable bool _first = true;                                     //!
   bool useBatchedEvaluations_ = false;
   std::unique_ptr<RooChangeTracker> paramTracker_;
   mutable ROOT::Math::KahanSum<double> cachedResult_ = 0;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooUnbinnedL
