// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrayI
#define ROOT_TArrayI


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayI                                                              //
//                                                                      //
// Array of integers (32 bits per element).                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TArray
#include "TArray.h"
#endif


class TArrayI : public TArray {

public:
   Int_t    *fArray;       //[fN] Array of fN 32 bit integers

   TArrayI();
   TArrayI(Int_t n);
   TArrayI(Int_t n, const Int_t *array);
   TArrayI(const TArrayI &array);
   TArrayI    &operator=(const TArrayI &rhs);
   virtual    ~TArrayI();

   void         Adopt(Int_t n, Int_t *array);
   void         AddAt(Int_t c, Int_t i);
   Int_t        At(Int_t i) const ;
   void         Copy(TArrayI &array) const {array.Set(fN,fArray);}
   const Int_t *GetArray() const { return fArray; }
   Int_t       *GetArray() { return fArray; }
   Double_t     GetAt(Int_t i) const { return At(i); }
   Stat_t       GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void         Reset()           {memset(fArray, 0, fN*sizeof(Int_t));}
   void         Reset(Int_t val)  {for (Int_t i=0;i<fN;i++) fArray[i] = val;}
   void         Set(Int_t n);
   void         Set(Int_t n, const Int_t *array);
   void         SetAt(Double_t v, Int_t i) { AddAt((Int_t)v, i); }
   Int_t       &operator[](Int_t i);
   Int_t        operator[](Int_t i) const;

   ClassDef(TArrayI,1)  //Array of ints
};


#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator>>(TBuffer &buf, TArrayI *&obj)
{
   // Read TArrayI object from buffer.

   obj = (TArrayI *) TArray::ReadArray(buf, TArrayI::Class());
   return buf;
}

#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator<<(TBuffer &buf, const TArrayI *obj)
{
   // Write a TArrayI object into buffer
   return buf << (const TArray*)obj;
}

inline Int_t TArrayI::At(Int_t i) const
{
   if (!BoundsOk("TArrayI::At", i)) return 0;
   return fArray[i];
}

inline Int_t &TArrayI::operator[](Int_t i)
{
   if (!BoundsOk("TArrayI::operator[]", i))
      i = 0;
   return fArray[i];
}

inline Int_t TArrayI::operator[](Int_t i) const
{
   if (!BoundsOk("TArrayI::operator[]", i)) return 0;
   return fArray[i];
}

#endif
