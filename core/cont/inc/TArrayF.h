// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrayF
#define ROOT_TArrayF


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayF                                                              //
//                                                                      //
// Array of floats (32 bits per element).                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArray.h"


class TArrayF : public TArray {

public:
   Float_t    *fArray;       //[fN] Array of fN floats

   TArrayF();
   TArrayF(Int_t n);
   TArrayF(Int_t n, const Float_t *array);
   TArrayF(const TArrayF &array);
   TArrayF    &operator=(const TArrayF &rhs);
   virtual    ~TArrayF();

   void           Adopt(Int_t n, Float_t *array);
   Float_t        At(Int_t i) const ;
   void           Copy(TArrayF &array) const {array.Set(fN,fArray);}
   const Float_t *GetArray() const { return fArray; }
   Float_t       *GetArray() { return fArray; }
   Double_t       GetAt(Int_t i) const { return At(i); }
   Stat_t         GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void           Reset()             {memset(fArray,  0, fN*sizeof(Float_t));}
   void           Reset(Float_t val)  {for (Int_t i=0;i<fN;i++) fArray[i] = val;}
   void           Set(Int_t n);
   void           Set(Int_t n, const Float_t *array);
   void           SetAt(Double_t v, Int_t i) { fArray[i] = (Float_t)v; }
   void           SetAt(Float_t v, Int_t i) { fArray[i] = v; }
   Float_t       &operator[](Int_t i);
   Float_t        operator[](Int_t i) const;

   ClassDef(TArrayF,1)  //Array of floats
};

#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator>>(TBuffer &buf, TArrayF *&obj)
{
   // Read TArrayF object from buffer.

   obj = (TArrayF *) TArray::ReadArray(buf, TArrayF::Class());
   return buf;
}

#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator<<(TBuffer &buf, const TArrayF *obj)
{
   // Write a TArrayF object into buffer
   return buf << (const TArray*)obj;
}

inline Float_t TArrayF::At(Int_t i) const
{
   if (!BoundsOk("TArrayF::At", i)) return 0;
   return fArray[i];
}

inline Float_t &TArrayF::operator[](Int_t i)
{
   if (!BoundsOk("TArrayF::operator[]", i))
      i = 0;
   return fArray[i];
}

inline Float_t TArrayF::operator[](Int_t i) const
{
   if (!BoundsOk("TArrayF::operator[]", i)) return 0;
   return fArray[i];
}

#endif
