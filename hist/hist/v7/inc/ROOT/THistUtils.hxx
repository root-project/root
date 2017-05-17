/// \file ROOT/THistData.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2016-06-01
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistUtils_h
#define ROOT7_THistUtils_h

#include <array>

namespace ROOT {
namespace Experimental {
namespace Hist {

template <int DIMENSIONS>
using CoordArray_t = std::array<double, DIMENSIONS>;


} // namespace Hist
} // namespace Experimental
} // namespace ROOT

#endif //ROOT7_THistUtils_h
