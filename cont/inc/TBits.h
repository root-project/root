// @(#)root/base:$Name:  $:$Id: TBits.h,v 1.1 2001/01/12 11:24:36 brun Exp $
// Author: Philippe Canal 05/02/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBits
#define ROOT_TBits

//------------------------------------------------------------------------------
// Copyright(c) 2001,.Canal (FNAL)
//
// Permission to use, copy, modify and distribute this software and its
// documentation for non-commercial purposes is hereby granted without fee,
// provided that the above copyright notice appears in all copies and
// that both the copyright notice and this permission notice appear in
// the supporting documentation. The authors make no claims about the
// suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//------------------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBits                                                                //
//                                                                      //
// Container of bits.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TBits : public TObject {

protected:

   Int_t    fNbits;         // Number of bits (around fNbytes*8)
   Int_t    fNbytes;        // Number of UChars in fAllBits
   UChar_t *fAllBits;       //[fNbytes] array of UChars 

public:
   TBits(Int_t nbits = 8);
   TBits(const TBits&);
   TBits& operator=(const TBits&);
   virtual ~TBits();

   //----- bit manipulation
   //----- (note the difference with TObject's bit manipulations)
   void   ResetAllBits(Int_t value=kFALSE);  // if value=1 set all bits to 1
   void   ResetBitNumber(Int_t bitnumber) { SetBitNumber(bitnumber,kFALSE); }
   void   SetBitNumber(Int_t bitnumber, Bool_t value = kTRUE);
   Bool_t TestBitNumber(Int_t bitnumber);

   //----- Utilities
   void   Compact();                         // Reduce the space used.
   Int_t  CountBits();                       // return number of bits set to 1
   Int_t  GetNbits() { return fNbits; }
   Int_t  GetNbytes() { return fNbytes; }

   void   Paint(Option_t *option="");        // to visualize the bits array as an histogram, etc
   void   Print(Option_t *option="");        // to show the list of active bits


   ClassDef(TBits,1)        // Bit container
};

#endif
