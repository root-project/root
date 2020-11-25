/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooUnbinnedL.h,v 1.10 2007/07/21 21:32:52 wouter Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 * Copyright (c) 2000-2020, Regents of the University of California          *
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
   RooUnbinnedL(RooAbsPdf* pdf, RooAbsData* data, bool do_offset, double offset, double offset_carry, RooAbsL::Extended extended = RooAbsL::Extended::Auto);
   RooUnbinnedL(const RooUnbinnedL &other);
   void set_apply_weight_squared(bool flag);

private:
   bool processEmptyDataSets() const;
   double evaluate_partition(std::size_t events_begin, std::size_t events_end, std::size_t components_begin,
                             std::size_t components_end) override;
   double get_carry() const override;
   bool apply_weight_squared = false;            // Apply weights squared?
   mutable bool _first = true;       //!
   double _offset_save_weight_squared = 0;      //!
   double _offset_carry_save_weight_squared = 0; //!
   mutable double _evalCarry = 0;   //! carry of Kahan sum in evaluatePartition
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooUnbinnedL
