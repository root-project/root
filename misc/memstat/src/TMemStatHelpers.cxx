// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2008-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
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

   if(bytes < 0) {
      ss << '-';
      bytes = -bytes;
   }

   static const long kB = 1024L;
   static const long lMB = kB * kB;
   static const long lGB = lMB * kB;

   if(bytes < kB)
      ss << bytes << " B";
   else if(bytes < (10L * kB))
      ss << setprecision(2) << ((double)bytes / (float)kB) << " kB";
   else if(bytes < (100L * kB))
      ss << setprecision(1) << ((double)bytes / (float)kB) << " kB";
   else if(bytes < lMB)
      ss << setprecision(0) << ((double)bytes / (float)kB) << " kB";
   else if(bytes < (10L * lMB))
      ss << setprecision(2) << ((double)bytes / (float)lMB) << " MB";
   else if(bytes < (100L * lMB))
      ss << setprecision(1) << ((double)bytes / (float)lMB) << " MB";
   else if(bytes < lGB)
      ss << setprecision(0) << ((double)bytes / (float)lMB) << " MB";
   else
      ss << setprecision(2) << ((double)bytes / (float)lGB) << " GB";

   return ss.str();
}
