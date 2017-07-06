/// \file ROOT/TFit.h
/// \ingroup MathCore ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-06
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFit
#define ROOT_TFit

#include <array>
#include <functional>

#include "ROOT/RArrayView.h"

#include "ROOT/THist.hxx"

namespace ROOT {
namespace Experimental {

class TFitResult {
};

template <int DIMENSION>
class TFunction {
public:
   TFunction(std::function<double(const std::array<double, DIMENSION> &, const std::array_view<double> &par)> func) {}
};

template <int DIMENSIONS, class PRECISION, template <int D_, class P_, template <class P__> class S_> class... STAT>
TFitResult FitTo(const THist<DIMENSIONS, PRECISION, STAT...> &hist, const TFunction<DIMENSIONS> &func,
                 std::array_view<double> paramInit)
{
   return TFitResult();
}

} // namespace Experimental
} // namespace ROOT

#endif
