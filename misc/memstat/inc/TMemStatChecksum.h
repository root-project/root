// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2008-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/
#ifndef ROOT_TMemStatChecksum
#define ROOT_TMemStatChecksum

#include <stdio.h>
#include "Rtypes.h"


//______________________________________________________________________________
class CCRC {
   UInt_t fValue;
public:
   static UInt_t fTable[256];
   static void InitTable();

   CCRC():  fValue(0xFFFFFFFF) {};
   void Init() {
      fValue = 0xFFFFFFFF;
   }
   void UpdateUChar(UChar_t v);
   void UpdateUShort(UShort_t v);
   void UpdateUInt(UInt_t v);
   void UpdateULong64(ULong64_t v);
   void Update(const void *data, size_t size);
   UInt_t GetDigest() const {
      return fValue ^ 0xFFFFFFFF;
   }
   static UInt_t CalculateDigest(const void *data, size_t size) {
      CCRC crc;
      crc.Update(data, size);
      return crc.GetDigest();
   }
   static bool VerifyDigest(UInt_t digest, const void *data, size_t size) {
      return (CalculateDigest(data, size) == digest);
   }
};

#endif
