// @(#)root/cont:$Id$
// Author: Fons Rademakers   20/11/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrayL64
#define ROOT_TArrayL64


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayL64                                                            //
//                                                                      //
// Array of long64s (64 bits per element)    .                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArray.h"


class TArrayL64 : public TArray {

public:
   Long64_t   *fArray;       //[fN] Array of fN long64s

   TArrayL64();
   TArrayL64(Int_t n);
   TArrayL64(Int_t n, const Long64_t *array);
   TArrayL64(const TArrayL64 &array);
   TArrayL64  &operator=(const TArrayL64 &rhs);
   virtual    ~TArrayL64();

   void            Adopt(Int_t n, Long64_t *array);
   void            AddAt(Long64_t c, Int_t i);
   Long64_t        At(Int_t i) const;
   void            Copy(TArrayL64 &array) const {array.Set(fN,fArray);}
   const Long64_t *GetArray() const { return fArray; }
   Long64_t       *GetArray() { return fArray; }
   Double_t        GetAt(Int_t i) const { return At(i); }
   Stat_t          GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void            Reset()           {memset(fArray,  0, fN*sizeof(Long64_t));}
   void            Reset(Long64_t val) {for (Int_t i=0;i<fN;i++) fArray[i] = val;}
   void            Set(Int_t n);
   void            Set(Int_t n, const Long64_t *array);
   void            SetAt(Double_t v, Int_t i) { AddAt((Long64_t)v, i); }
   Long64_t       &operator[](Int_t i);
   Long64_t        operator[](Int_t i) const;

   ClassDef(TArrayL64,1)  //Array of long64s
};



#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator>>(TBuffer &buf, TArrayL64 *&obj)
{
   // Read TArrayL64 object from buffer.

   obj = (TArrayL64 *) TArray::ReadArray(buf, TArrayL64::Class());
   return buf;
}

#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator<<(TBuffer &buf, const TArrayL64 *obj)
{
   // Write a TArrayL64 object into buffer.

   return buf << (const TArray*)obj;
}

inline Long64_t TArrayL64::At(Int_t i) const
{
   if (!BoundsOk("TArrayL64::At", i)) return 0;
   return fArray[i];
}

inline Long64_t &TArrayL64::operator[](Int_t i)
{
   if (!BoundsOk("TArrayL64::operator[]", i))
      i = 0;
   return fArray[i];
}

inline Long64_t TArrayL64::operator[](Int_t i) const
{
   if (!BoundsOk("TArrayL64::operator[]", i)) return 0;
   return fArray[i];
}

#endif
