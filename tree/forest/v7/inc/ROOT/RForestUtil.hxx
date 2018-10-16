/// \file ROOT/RForestUtil.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RForestUtil
#define ROOT7_RForestUtil

#include <cstdint>

namespace ROOT {
namespace Experimental {

/// Integer types long enough to hold the maximum number of entries in a tree
using TreeIndex_t = std::uint64_t;
using TreeOffset_t = std::int64_t;

} // namespace Experimental
} // namespace ROOT

#endif
