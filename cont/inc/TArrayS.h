// @(#)root/cont:$Name:  $:$Id: TArrayS.h,v 1.5 2002/04/04 10:28:35 brun Exp $
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrayS
#define ROOT_TArrayS


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayS                                                              //
//                                                                      //
// Array of shorts (16 bits per element).                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TArray
#include "TArray.h"
#endif


class TArrayS : public TArray {

public:
   Short_t    *fArray;       //[fN] Array of fN shorts

   TArrayS();
   TArrayS(Int_t n);
   TArrayS(Int_t n, const Short_t *array);
   TArrayS(const TArrayS &array);
   TArrayS    &operator=(const TArrayS &rhs);
   virtual    ~TArrayS();

   void       Adopt(Int_t n, Short_t *array);
   void       AddAt(Short_t c, Int_t idx);
   Short_t    At(Int_t i) const ;
   void       Copy(TArrayS &array) {array.Set(fN); for (Int_t i=0;i<fN;i++) array.fArray[i] = fArray[i];}
   Short_t   *GetArray() const { return fArray; }
   Stat_t     GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void       Reset(Short_t val=0)  {for (Int_t i=0;i<fN;i++) fArray[i] = val;}
   void       Set(Int_t n);
   void       Set(Int_t n, const Short_t *array);
   Short_t   &operator[](Int_t i);
   Short_t    operator[](Int_t i) const;

   ClassDef(TArrayS,1)  //Array of shorts
};

inline TBuffer &operator>>(TBuffer &buf, TArrayS *&obj)
{
   // Read TArrayS object from buffer.

   obj = (TArrayS *) TArray::ReadArray(buf, TArrayS::Class());
   return buf;
}

inline Short_t TArrayS::At(Int_t i) const
{
   if (!BoundsOk("TArrayS::At", i))
      i = 0;
   return fArray[i];
}

inline Short_t &TArrayS::operator[](Int_t i)
{
   if (!BoundsOk("TArrayS::operator[]", i))
      i = 0;
   return fArray[i];
}

inline Short_t TArrayS::operator[](Int_t i) const
{
   if (!BoundsOk("TArrayS::operator[]", i))
      i = 0;
   return fArray[i];
}

#endif
