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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooBinnedL
#define ROOT_ROOFIT_TESTSTATISTICS_RooBinnedL

#include <TestStatistics/RooAbsL.h>
#include "RooAbsReal.h"
#include <vector>

// forward declarations
class RooAbsPdf;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

class RooBinnedL :
   public RooAbsL {
public:
   RooBinnedL(RooAbsPdf* pdf, RooAbsData* data);
   double evaluatePartition(Section bins, std::size_t components_begin,
                             std::size_t components_end) override;
private:
   mutable bool _first = true;       //!
   mutable std::vector<double> _binw; //!
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooBinnedL
