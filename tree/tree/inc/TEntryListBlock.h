// @(#)root/tree:$Id$
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
// Used internally in TEntryList to store the entry numbers. 
//
// There are 2 ways to represent entry numbers in a TEntryListBlock:
// 1) as bits, where passing entry numbers are assigned 1, not passing - 0
// 2) as a simple array of entry numbers
// In both cases, a UShort_t* is used. The second option is better in case
// less than 1/16 of entries passes the selection, and the representation can be
// changed by calling OptimizeStorage() function. 
// When the block is being filled, it's always stored as bits, and the OptimizeStorage()
// function is called by TEntryList when it starts filling the next block. If
// Enter() or Remove() is called after OptimizeStorage(), representation is 
// again changed to 1).
//
// Operations on blocks (see also function comments):
// - Merge() - adds all entries from one block to the other. If the first block 
//             uses array representation, it's changed to bits representation only
//             if the total number of passing entries is still less than kBlockSize
// - GetEntry(n) - returns n-th non-zero entry.
// - Next()      - return next non-zero entry. In case of representation 1), Next()
//                 is faster than GetEntry()
//
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TEntryListBlock
#define ROOT_TEntryListBlock

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TEntryListBlock:public TObject 
{
 protected:
   Int_t    fNPassed;    //number of entries in the entry list (if fPassing=0 - number of entries
                         //not in the entry list
   Int_t    fN;          //size of fIndices for I/O  =fNPassed for list, fBlockSize for bits
   UShort_t *fIndices;   //[fN]
   Int_t    fType;       //0 - bits, 1 - list
   Bool_t   fPassing;    //1 - stores entries that belong to the list
                         //0 - stores entries that don't belong to the list
   UShort_t fCurrent;    //! to fasten  Contains() in list mode
   Int_t    fLastIndexQueried; //! to optimize GetEntry() in a loop
   Int_t    fLastIndexReturned; //! to optimize GetEntry() in a loop

   void Transform(Bool_t dir, UShort_t *indexnew);

 public:

   enum { kBlockSize = 4000 }; //size of the block, 4000 UShort_ts
   TEntryListBlock();
   TEntryListBlock(const TEntryListBlock &eblock);
   ~TEntryListBlock();
   TEntryListBlock &operator=(const TEntryListBlock &rhs);

   Bool_t  Enter(Int_t entry);
   Bool_t  Remove(Int_t entry);
   Int_t   Contains(Int_t entry);
   void    OptimizeStorage();
   Int_t   Merge(TEntryListBlock *block);
   Int_t   Next();
   Int_t   GetEntry(Int_t entry);
   void    ResetIndices() {fLastIndexQueried = -1, fLastIndexReturned = -1;}
   Int_t   GetType() { return fType; }
   Int_t   GetNPassed();
   virtual void Print(const Option_t *option = "") const;
   void    PrintWithShift(Int_t shift) const;

   ClassDef(TEntryListBlock, 1) //Used internally in TEntryList to store the entry numbers

};

#endif
