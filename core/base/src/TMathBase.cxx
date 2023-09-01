// @(#)root/base:$Id$
// Authors: Rene Brun 08/02/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** TMath Base functions
\ingroup Base

Define the functions Min, Max, Abs, Sign, Range for all types.
NB: These functions are unfortunately not available in a portable
way in std::.

More functions are defined in TMath.h. TMathBase.h is designed to be
a stable file and used in place of TMath.h in the ROOT miniCore.
*/

#include "TMathBase.h"
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
/// Return next prime number after x, unless x is a prime in which case
/// x is returned.

Long_t TMath::NextPrime(Long_t x)
{
   if (x <= 2)
      return 2;
   if (x == 3)
      return 3;

   if (x % 2 == 0)
      x++;

   Long_t sqr = (Long_t) sqrt((Double_t)x) + 1;

   for (;;) {
      Long_t n;
      for (n = 3; (n <= sqr) && ((x % n) != 0); n += 2)
         ;
      if (n > sqr)
         return x;
      x += 2;
   }
}
