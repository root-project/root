// @(#)root/zip:$Id$
// Author: Sergey Linev   7 July 2014

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RZip.h"

#include "zlib.h"

ULong_t R__crc32(ULong_t crc, const UChar_t* buf, UInt_t len)
{
   return crc32(crc, buf, len);
}
