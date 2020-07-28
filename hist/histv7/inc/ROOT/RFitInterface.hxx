/// \file ROOT/RFitInterface.hxx
/// \ingroup Hist ROOT7
/// \author Claire Guyot
/// \date 2020-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT7_RFitInterface
#define ROOT7_RFitInterface

#include "TFitResult.h"
#include "TFitResultPtr.h"

namespace ROOT {
namespace Experimental {
namespace RFit {

   template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
   TFitResultPtr Fit(RHist<DIMENSIONS, PRECISION, STAT...> & hist, TF1 *f1, const ROOT::Fit::DataOptions & fitOption, const ROOT::Fit::FitConfig & fitConfig);

}// namespace RFit
}// namespace Experimental
}// namespace ROOT

#endif
