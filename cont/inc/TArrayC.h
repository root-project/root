// @(#)root/cont:$Name:  $:$Id: TArrayC.h,v 1.1.1.1 2000/05/16 17:00:40 rdm Exp $
// Author: Rene Brun   06/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArrayC
#define ROOT_TArrayC


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArrayC                                                              //
//                                                                      //
// Array of chars or bytes (8 bits per element).                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TArray
#include "TArray.h"
#endif


class TArrayC : public TArray {

public:
   Char_t    *fArray;       //[fN] Array of fN chars

   TArrayC();
   TArrayC(Int_t n);
   TArrayC(Int_t n, Char_t *array);
   TArrayC(const TArrayC &array);
   TArrayC    &operator=(const TArrayC &rhs);
   virtual    ~TArrayC();

   void       Adopt(Int_t n, Char_t *array);
   void       AddAt(Char_t c, Int_t i);
   Char_t     At(Int_t i) const ;
   void       Copy(TArrayC &array) {array.Set(fN); for (Int_t i=0;i<fN;i++) array.fArray[i] = fArray[i];}
   Char_t    *GetArray() const { return fArray; }
   Stat_t     GetSum() const {Stat_t sum=0; for (Int_t i=0;i<fN;i++) sum+=fArray[i]; return sum;}
   void       Reset()  {memset(fArray,0,fN*sizeof(Char_t));}
   void       Set(Int_t n);
   void       Set(Int_t n, Char_t *array);
   Char_t    &operator[](Int_t i);

   ClassDef(TArrayC,1)  //Array of chars
};

inline Char_t TArrayC::At(Int_t i) const
{
   if (!BoundsOk("TArrayC::At", i))
      i = 0;
   return fArray[i];
}

inline Char_t &TArrayC::operator[](Int_t i)
{
   if (!BoundsOk("TArrayC::operator[]", i))
      i = 0;
   return fArray[i];
}

#endif
