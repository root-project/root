// @(#)root/memstat:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 2008-03-02

/*************************************************************************
* Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/
#include "TMemStatChecksum.h"

//______________________________________________________________________________
static const UInt32 kCRCPoly = 0xEDB88320;

//______________________________________________________________________________
UInt32 CCRC::Table[256];

//______________________________________________________________________________
void CCRC::InitTable()
{
   for (UInt32 i = 0; i < 256; i++) {
      UInt32 r = i;
      for (int j = 0; j < 8; j++)
         if (r & 1)
            r = (r >> 1) ^ kCRCPoly;
         else
            r >>= 1;
      CCRC::Table[i] = r;
   }
}

//______________________________________________________________________________
class CCRCTableInit {
public:
   CCRCTableInit() {
      CCRC::InitTable();
   }
} g_CRCTableInit;

//______________________________________________________________________________
void CCRC::UpdateByte(Byte b)
{
   _value = Table[((Byte)(_value)) ^ b] ^(_value >> 8);
}

//______________________________________________________________________________
void CCRC::UpdateUInt16(UInt16 v)
{
   UpdateByte(Byte(v));
   UpdateByte(Byte(v >> 8));
}

//______________________________________________________________________________
void CCRC::UpdateUInt32(UInt32 v)
{
   for (int i = 0; i < 4; i++)
      UpdateByte((Byte)(v >> (8 * i)));
}

//______________________________________________________________________________
void CCRC::UpdateUInt64(UInt64 v)
{
   for (int i = 0; i < 8; i++)
      UpdateByte((Byte)(v >> (8 * i)));
}

//______________________________________________________________________________
void CCRC::Update(const void *data, size_t size)
{
   UInt32 v = _value;
   const Byte *p = (const Byte *)data;
   for (; size > 0 ; size--, p++)
      v = Table[((Byte)(v)) ^ *p] ^(v >> 8);
   _value = v;
}
