// @(#)root/cont:$Name:  $:$Id: TArrayD.h,v 1.5 2002/04/04 10:28:35 brun Exp $
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

#ifndef ROOT_TArray
#include "TArray.h"
#endif


class TArrayD : public TArray {

public:
   Double_t    *fArray;       //[fN] Array of fN doubles

   TArrayD();
   TArrayD(Int_t n);
   TArrayD(Int_t n, const Double_t *array);
   TArrayD(const TArrayD &array);
   TArrayD    &operator=(const TArrayD &rhs);
   virtual    ~TArrayD();

   void       Adopt(Int_t n, Double_t *array);
   void       AddAt(Double_t c, Int_t i);
   Double_t   At(Int_t i) const ;
   void       Copy(TArrayD &array) {array.Set(fN); for (Int_t i=0;i<fN;i++) array.fArray[i] = fArray[i];}
   Double_t  *GetArray() const { return fArray; }
   Stat_t     GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void       Reset(Double_t val=0)  {for (Int_t i=0;i<fN;i++) fArray[i] = val;}
   void       Set(Int_t n);
   void       Set(Int_t n, const Double_t *array);
   Double_t  &operator[](Int_t i);
   Double_t   operator[](Int_t i) const;

   ClassDef(TArrayD,1)  //Array of doubles
};

inline TBuffer &operator>>(TBuffer &buf, TArrayD *&obj)
{
   // Read TArrayD object from buffer.

   obj = (TArrayD *) TArray::ReadArray(buf, TArrayD::Class());
   return buf;
}

inline Double_t TArrayD::At(Int_t i) const
{
   if (!BoundsOk("TArrayD::At", i))
      i = 0;
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
   if (!BoundsOk("TArrayD::operator[]", i))
      i = 0;
   return fArray[i];
}

#endif
