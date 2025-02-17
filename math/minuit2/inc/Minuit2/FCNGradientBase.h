// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNGradientBase
#define ROOT_Minuit2_FCNGradientBase

#include "Minuit2/FCNBase.h"

namespace ROOT {

namespace Minuit2 {

//________________________________________________________________________
/** Extension of the FCNBase for providing the analytical Gradient of the
    function.
    The size of the output Gradient vector must be equal to the size of the
    input Parameter vector.
 */

class FCNGradientBase : public FCNBase {
public:
   bool HasGradient() const final { return true; }
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_FCNGradientBase
