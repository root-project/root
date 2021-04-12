// @(#)root/cont:$Id$
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrayD
#define ROOT_TArrayD


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayD                                                              //
//                                                                      //
// Array of doubles (64 bits per element).                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TArray.h"


class TArrayD : public TArray {

public:
   Double_t    *fArray;       //[fN] Array of fN doubles

   TArrayD();
   TArrayD(Int_t n);
   TArrayD(Int_t n, const Double_t *array);
   TArrayD(const TArrayD &array);
   TArrayD    &operator=(const TArrayD &rhs);
   virtual    ~TArrayD();

   void            Adopt(Int_t n, Double_t *array);
   Double_t        At(Int_t i) const ;
   void            Copy(TArrayD &array) const {array.Set(fN,fArray);}
   const Double_t *GetArray() const { return fArray; }
   Double_t       *GetArray() { return fArray; }
   Double_t        GetAt(Int_t i) const { return At(i); }
   Stat_t          GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void            Reset()             {memset(fArray, 0, fN*sizeof(Double_t));}
   void            Reset(Double_t val) {for (Int_t i=0;i<fN;i++) fArray[i] = val;}
   void            Set(Int_t n);
   void            Set(Int_t n, const Double_t *array);
   void            SetAt(Double_t v, Int_t i) { fArray[i] = v; }
   Double_t       &operator[](Int_t i);
   Double_t        operator[](Int_t i) const;

   ClassDef(TArrayD,1)  //Array of doubles
};


#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator>>(TBuffer &buf, TArrayD *&obj)
{
   // Read TArrayD object from buffer.

   obj = (TArrayD *) TArray::ReadArray(buf, TArrayD::Class());
   return buf;
}

#if defined R__TEMPLATE_OVERLOAD_BUG
template <>
#endif
inline TBuffer &operator<<(TBuffer &buf, const TArrayD *obj)
{
   // Write a TArrayD object into buffer
   return buf << (const TArray*)obj;
}

inline Double_t TArrayD::At(Int_t i) const
{
   if (!BoundsOk("TArrayD::At", i)) return 0;
   return fArray[i];
}

inline Double_t &TArrayD::operator[](Int_t i)
{
   if (!BoundsOk("TArrayD::operator[]", i))
      i = 0;
   return fArray[i];
}

inline Double_t TArrayD::operator[](Int_t i) const
{
   if (!BoundsOk("TArrayD::operator[]", i)) return 0;
   return fArray[i];
}

#endif
