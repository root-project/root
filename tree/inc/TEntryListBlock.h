// @(#)root/tree:$Name:  $:$Id: TEntryListBlock.h,v 1.1 2006/10/27 09:58:02 brun Exp $
// Author: Anna Kreshuk 27/10/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
// TEntryListBlock
//
// Used internally in TEntryList to store the entry numbers. TEntryListBlock
// can have two representations - stored as bits and stored as a list.
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TEntryListBlock
#define ROOT_TEntryListBlock

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TEntryListBlock:public TObject 
{
 protected:
   Int_t    fNPassed;    //number of entries in the entry list 
   Int_t    fN;          //size of fIndices for I/O  =fNPassed for list, fBlockSize for bits
   UShort_t *fIndices;   //[fN]
   Int_t    fType;       //0 - bits, 1 - list
   Bool_t   fPassing;    //1 - stores entries that belong to the list
                         //0 - stores entries that don't belong to the list (not there yet)
   UShort_t fCurrent;    //! to fasten Enter() and Contains() in list mode
   Int_t  fLastIndexQueried; //! to optimize GetEntry() in a loop
   Int_t  fLastIndexReturned; //! to optimize GetEntry() in a loop

   void Transform(Bool_t dir, UShort_t *indexnew);

 public:

   enum { kBlockSize = 4000 }; //size of the block, 4000 UShort_ts
   TEntryListBlock();
   TEntryListBlock(const TEntryListBlock &eblock);
   ~TEntryListBlock();

   Bool_t  Enter(Int_t entry);
   Bool_t  Remove(Int_t entry);
   Int_t   Contains(Int_t entry);
   void    OptimizeStorage();
   Int_t   Merge(TEntryListBlock *block);
   Int_t   Next();
   Int_t   GetEntry(Int_t entry);
   void    ResetIndices() {fLastIndexQueried = -1, fLastIndexReturned = -1;}
   Int_t   GetType() { return fType; }
   Int_t   GetNPassed() { return fNPassed; }
   virtual void Print(const Option_t *option = "") const;
   void    PrintWithShift(Int_t shift) const;

   ClassDef(TEntryListBlock, 1) //Used internally in TEntryList to store the entry numbers

};

#endif
