/// \file ROOT/RTreeUtil.hxx
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

#ifndef ROOT7_RTreeUtil
#define ROOT7_RTreeUtil

#include <cstdint>

namespace ROOT {
namespace Experimental {

using TreeIndex_t = std::uint64_t;
using TreeOffset_t = std::int64_t;

} // namespace Experimental
} // namespace ROOT

#endif
