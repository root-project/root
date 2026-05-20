// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnParabolaFactory
#define ROOT_Minuit2_MnParabolaFactory

#include "Minuit2/MnParabola.h"
#include "Minuit2/MnPoint.h"

namespace ROOT {

namespace Minuit2 {

class MnParabolaFactory {
public:
   MnParabola operator()(const MnPoint &, const MnPoint &, const MnPoint &) const;

   MnParabola operator()(const MnPoint &, double, const MnPoint &) const;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnParabolaFactory
