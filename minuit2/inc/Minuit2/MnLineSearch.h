// @(#)root/minuit2:$Name:  $:$Id: MnLineSearch.h,v 1.9.4.4 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnLineSearch
#define ROOT_Minuit2_MnLineSearch

#include "Minuit2/MnMatrix.h"

namespace ROOT {

   namespace Minuit2 {


class MnFcn;
class MinimumParameters;
class MnMachinePrecision;
class MnParabolaPoint;




/** 

Implements a 1-dimensional minimization along a given direction 
(i.e. quadratic interpolation) It is independent of the algorithm 
that generates the direction vector. It brackets the 1-dimensional 
Minimum and iterates to approach the real Minimum of the n-dimensional
function.


@author Fred James and Matthias Winkler; comments added by Andras Zsenei
and Lorenzo Moneta

@ingroup Minuit

*/




class MnLineSearch  {

public:

  MnLineSearch() {}

  ~MnLineSearch() {}

  MnParabolaPoint operator()(const MnFcn&, const MinimumParameters&, const MnAlgebraicVector&, double, const MnMachinePrecision&) const;

private:

};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnLineSearch
