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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooUnbinnedL
#define ROOT_ROOFIT_TESTSTATISTICS_RooUnbinnedL

#include <TestStatistics/RooAbsL.h>

// forward declarations
class RooAbsPdf;
class RooAbsData;
class RooArgSet;

namespace RooFit {
namespace TestStatistics {

class RooUnbinnedL :
   public RooAbsL {
public:
   RooUnbinnedL(RooAbsPdf* pdf, RooAbsData* data, RooAbsL::Extended extended = RooAbsL::Extended::Auto);
   RooUnbinnedL(const RooUnbinnedL &other);
   bool setApplyWeightSquared(bool flag);

   double evaluatePartition(Section events, std::size_t components_begin,
                             std::size_t components_end) override;

private:
   bool processEmptyDataSets() const;
   bool apply_weight_squared = false;            // Apply weights squared?
   mutable bool _first = true;       //!
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooUnbinnedL
