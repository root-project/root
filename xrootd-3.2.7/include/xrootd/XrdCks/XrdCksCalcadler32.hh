#ifndef __XRDCKSCALCADLER32_HH__
#define __XRDCKSCALCADLER32_HH__
/******************************************************************************/
/*                                                                            */
/*                  X r d C k s C a l c a d l e r 3 2 . h h                   */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>

#include "XrdCks/XrdCksCalc.hh"
#include "XrdSys/XrdSysPlatform.hh"

/* The following implementation of adler32 was derived from zlib and is
                   * Copyright (C) 1995-1998 Mark Adler
   Below are the zlib license terms for this implementation.
*/
  
/* zlib.h -- interface of the 'zlib' general purpose compression library
  version 1.1.4, March 11th, 2002

  Copyright (C) 1995-2002 Jean-loup Gailly and Mark Adler

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Jean-loup Gailly        Mark Adler
  jloup@gzip.org          madler@alumni.caltech.edu


  The data format used by the zlib library is described by RFCs (Request for
  Comments) 1950 to 1952 in the files ftp://ds.internic.net/rfc/rfc1950.txt
  (zlib format), rfc1951.txt (deflate format) and rfc1952.txt (gzip format).
*/

#define DO1(buf)  {unSum1 += *buf++; unSum2 += unSum1;}
#define DO2(buf)  DO1(buf); DO1(buf);
#define DO4(buf)  DO2(buf); DO2(buf);
#define DO8(buf)  DO4(buf); DO4(buf);
#define DO16(buf) DO8(buf); DO8(buf);

class XrdCksCalcadler32 : public XrdCksCalc
{
public:

char *Final()
            {AdlerValue = (unSum2 << 16) | unSum1;
#ifndef Xrd_Big_Endian
             AdlerValue = htonl(AdlerValue);
#endif
             return (char *)&AdlerValue;
            }

void        Init() {unSum1 = AdlerStart; unSum2 = 0;}

XrdCksCalc *New() {return (XrdCksCalc *)new XrdCksCalcadler32;}

void        Update(const char *Buff, int BLen)
                  {int k;
                   unsigned char *buff = (unsigned char *)Buff;
                   while(BLen > 0)
                        {k = (BLen < AdlerNMax ? BLen : AdlerNMax);
                         BLen -= k;
                         while(k >= 16) {DO16(buff); k -= 16;}
                         if (k != 0) do {DO1(buff);} while (--k);
                         unSum1 %= AdlerBase; unSum2 %= AdlerBase;
                        }
                  }

const char *Type(int &csSize) {csSize = sizeof(AdlerResult); return "adler32";}

            XrdCksCalcadler32() {Init();}
virtual    ~XrdCksCalcadler32() {}

private:

static const unsigned int AdlerBase  = 0xFFF1;
static const unsigned int AdlerStart = 0x0001;
static const          int AdlerNMax  = 5552;

/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

             unsigned int AdlerResult;
             unsigned int AdlerValue;
             unsigned int unSum1;
             unsigned int unSum2;
                      int n;
};
#endif
