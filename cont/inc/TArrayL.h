// @(#)root/cont:$Name:  $:$Id: TArrayL.h,v 1.3 2001/02/08 15:31:13 brun Exp $
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrayL
#define ROOT_TArrayL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayL                                                              //
//                                                                      //
// Array of longs (32 or 64 bits per element).                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TArray
#include "TArray.h"
#endif


class TArrayL : public TArray {

public:
   Long_t    *fArray;       //[fN] Array of fN longs

   TArrayL();
   TArrayL(Int_t n);
   TArrayL(Int_t n, const Long_t *array);
   TArrayL(const TArrayL &array);
   TArrayL    &operator=(const TArrayL &rhs);
   virtual    ~TArrayL();

   void       Adopt(Int_t n, Long_t *array);
   void       AddAt(Long_t c, Int_t i);
   Long_t     At(Int_t i) const;
   void       Copy(TArrayL &array) {array.Set(fN); for (Int_t i=0;i<fN;i++) array.fArray[i] = fArray[i];}
   Long_t    *GetArray() const { return fArray; }
   Stat_t     GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void       Reset(Long_t val=0) {for (Int_t i=0;i<fN;i++) fArray[i] = val;}
   void       Set(Int_t n);
   void       Set(Int_t n, const Long_t *array);
   Long_t    &operator[](Int_t i);

   ClassDef(TArrayL,1)  //Array of longs
};

inline Long_t TArrayL::At(Int_t i) const
{
   if (!BoundsOk("TArrayL::At", i))
      i = 0;
   return fArray[i];
}

inline Long_t &TArrayL::operator[](Int_t i)
{
   if (!BoundsOk("TArrayL::operator[]", i))
      i = 0;
   return fArray[i];
}

#endif
