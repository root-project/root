/// \file TFit.h
/// \ingroup MathCore ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-06
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

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

#include "ROOT/THist.h"

namespace ROOT {
namespace Experimental {

class TFitResult {

};

template <int DIMENSION>
class TFunction {
public:
  TFunction(std::function<double (const std::array<double, DIMENSION>&,
                                  const std::array_view<double>& par)> func) {}
};

template <class DATA>
TFitResult FitTo(const THist<DATA>& hist,
               const TFunction<THist<DATA>::GetNDim()>& func,
               std::array_view<double> paramInit){
  return TFitResult();
}

} // namespace Experimental
} // namespace ROOT

#endif
