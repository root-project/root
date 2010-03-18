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

//______________________________________________________________________________
typedef unsigned char Byte;
typedef short Int16;
typedef unsigned short UInt16;
typedef int Int32;
typedef unsigned int UInt32;
#ifdef _MSC_VER
typedef __int64 Int64;
typedef unsigned __int64 UInt64;
#else
typedef long long int Int64;
typedef unsigned long long int UInt64;
#endif

//______________________________________________________________________________
class CCRC {
   UInt32 _value;
public:
   static UInt32 Table[256];
   static void InitTable();

   CCRC():  _value(0xFFFFFFFF) {};
   void Init() {
      _value = 0xFFFFFFFF;
   }
   void UpdateByte(Byte v);
   void UpdateUInt16(UInt16 v);
   void UpdateUInt32(UInt32 v);
   void UpdateUInt64(UInt64 v);
   void Update(const void *data, size_t size);
   UInt32 GetDigest() const {
      return _value ^ 0xFFFFFFFF;
   }
   static UInt32 CalculateDigest(const void *data, size_t size) {
      CCRC crc;
      crc.Update(data, size);
      return crc.GetDigest();
   }
   static bool VerifyDigest(UInt32 digest, const void *data, size_t size) {
      return (CalculateDigest(data, size) == digest);
   }
};

#endif
