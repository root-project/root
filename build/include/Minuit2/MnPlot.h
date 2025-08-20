// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnPlot
#define ROOT_Minuit2_MnPlot

#include "Minuit2/MnConfig.h"

#include <ROOT/RSpan.hxx>

#include <algorithm>
#include <vector>
#include <utility>

namespace ROOT {

namespace Minuit2 {

/** MnPlot produces a text-screen graphical output of (x,y) points, e.g.
    from Scan or Contours.
*/

class MnPlot {

public:
   MnPlot() = default;

   MnPlot(unsigned int width, unsigned int length)
      : fPageWidth(std::min(width, 120u)), fPageLength(std::min(length, 56u))
   {
   }

   void operator()(std::span<const std::pair<double, double>> ) const;
   void operator()(double, double, std::span<const std::pair<double, double>> ) const;

   unsigned int Width() const { return fPageWidth; }
   unsigned int Length() const { return fPageLength; }

private:
   unsigned int fPageWidth = 80;
   unsigned int fPageLength = 30;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnPlot
