/// \file TFit.h
/// \ingroup MathCore
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-06

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

namespace ROOT {
class TFitResult {

};

template <int DIMENSION, class PRECISION> class THist;

template <int DIMENSION>
class TFunction {
public:
  TFunction(std::function<double (const std::array<double, DIMENSION>&,
                                  const std::array_view<double>& par)> func) {}
};

template <int DIMENSION, class PRECISION>
TFitResult FitTo(const THist<DIMENSION, PRECISION>& hist,
               const TFunction<DIMENSION>& func,
               std::array_view<double> paramInit){
  return TFitResult();
}

}

#endif
