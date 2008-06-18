// @(#)root/memstat:$Name$:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 09/05/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// STD
#include <iomanip>
#include <sstream>
// ROOT
#include "Rtypes.h"
// Memstat
#include "TMemStatHelpers.h"

using namespace std;

//______________________________________________________________________________
string Memstat::dig2bytes(Long64_t bytes)
{
   // This function creates a string representation of the number of bytes,
   // represented as a number in B, kB, MB or GB depending on the value.
   // The result is rounded to a sensible number of digits

   ostringstream ss;
   ss << fixed;

   if (bytes < 0) {
      ss << '-';
      bytes = -bytes;
   }

   static const long kB = 1024l;
   static const long MB = kB * kB;
   static const long GB = MB * kB;

   if (bytes < kB)
      ss << bytes << " B";
   else if (bytes < (10l * kB))
      ss << setprecision(2) << ((double)bytes / (float)kB) << " kB";
   else if (bytes < (100l * kB))
      ss << setprecision(1) << ((double)bytes / (float)kB) << " kB";
   else if (bytes < MB)
      ss << setprecision(0) << ((double)bytes / (float)kB) << " kB";
   else if (bytes < (10l * MB))
      ss << setprecision(2) << ((double)bytes / (float)MB) << " MB";
   else if (bytes < (100l * MB))
      ss << setprecision(1) << ((double)bytes / (float)MB) << " MB";
   else if (bytes < GB)
      ss << setprecision(0) << ((double)bytes / (float)MB) << " MB";
   else
      ss << setprecision(2) << ((double)bytes / (float)GB) << " GB";

   return ss.str();
}
