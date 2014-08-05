// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnMatrix
#define ROOT_Minuit2_MnMatrix

//add MnConfig file to define before everything compiler
// dependent macros

#include "Minuit2/MnConfig.h"

// Removing this the following include will cause the library to fail
// to compile with gcc 4.0.0 under Red Hat Enterprise Linux 3.  That
// is, FumiliBuiilder.cpp will fail with message about ambigous enum.
// Putting an include <vector> before other includes in that file will
// Fix it, but then another file class will fail with the same
// message.  I don't understand it, but putting the include <vector>
// in this one spot, fixes the problem and does not require any other
// changes to the source code.
//
// Paul_Kunz@slac.stanford.edu  3 June 2005
//
#include <vector>

#include "Minuit2/LASymMatrix.h"
#include "Minuit2/LAVector.h"
#include "Minuit2/LaInverse.h"
#include "Minuit2/LaOuterProduct.h"

namespace ROOT {

   namespace Minuit2 {


typedef LASymMatrix MnAlgebraicSymMatrix;
typedef LAVector MnAlgebraicVector;

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnMatrix
