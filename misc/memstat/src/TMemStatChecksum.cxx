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
static const UInt_t kCRCPoly = 0xEDB88320;

//______________________________________________________________________________
UInt_t CCRC::fTable[256];

//______________________________________________________________________________
void CCRC::InitTable()
{
   for(UInt_t i = 0; i < 256; i++) {
      UInt_t r = i;
      for(int j = 0; j < 8; j++)
         if(r & 1)
            r = (r >> 1) ^ kCRCPoly;
         else
            r >>= 1;
      CCRC::fTable[i] = r;
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
void CCRC::UpdateUChar(UChar_t b)
{
   fValue = fTable[((UChar_t)(fValue)) ^ b] ^(fValue >> 8);
}

//______________________________________________________________________________
void CCRC::UpdateUShort(UShort_t v)
{
   UpdateUChar(UChar_t(v));
   UpdateUChar(UChar_t(v >> 8));
}

//______________________________________________________________________________
void CCRC::UpdateUInt(UInt_t v)
{
   for(int i = 0; i < 4; i++)
      UpdateUChar((UChar_t)(v >> (8 * i)));
}

//______________________________________________________________________________
void CCRC::UpdateULong64(ULong64_t v)
{
   for(int i = 0; i < 8; i++)
      UpdateUChar((UChar_t)(v >> (8 * i)));
}

//______________________________________________________________________________
void CCRC::Update(const void *data, size_t size)
{
   UInt_t v = fValue;
   const UChar_t *p = (const UChar_t *)data;
   for(; size > 0 ; size--, p++)
      v = fTable[((UChar_t)(v)) ^ *p] ^(v >> 8);
   fValue = v;
}
