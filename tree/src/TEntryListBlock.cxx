// @(#)root/tree:$Name:  $:$Id: TEntryListBlock.cxx,v 1.3 2006/10/31 15:18:34 brun Exp $
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


#include "TEntryListBlock.h"
#include "TString.h"

ClassImp(TEntryListBlock)

//______________________________________________________________________________
TEntryListBlock::TEntryListBlock()
{
   //default c-tor

   fIndices = 0;
   fN = kBlockSize;
   fNPassed = 0;
   fType = -1;
   fPassing = 1;
   fCurrent = 0;
   fLastIndexReturned = -1;
   fLastIndexQueried = -1;
}

//______________________________________________________________________________
TEntryListBlock::TEntryListBlock(const TEntryListBlock &eblock) : TObject(eblock)
{
   //copy c-tor

   Int_t i;
   if (eblock.fIndices){
      fIndices = new UShort_t[eblock.fN];
      for (i=0; i<eblock.fN; i++)
         fIndices[i] = eblock.fIndices[i];
   }
   fN = eblock.fN;
   fNPassed = eblock.fNPassed;
   fType = eblock.fType;
   fPassing = eblock.fPassing;
   fCurrent = eblock.fCurrent;
   fLastIndexReturned = -1;
   fLastIndexQueried = -1;
}


//______________________________________________________________________________
TEntryListBlock::~TEntryListBlock()
{
   //destructor

   if (fIndices)
      delete [] fIndices;
   fIndices = 0;
}

//______________________________________________________________________________
Bool_t TEntryListBlock::Enter(Int_t entry)
{
   //If the block has already been optimized and the entries
   //are stored as a list and not as bits, trying to enter a new entry
   //will make the block switch to bits representation

   if (entry > kBlockSize*16) {
      Error("Enter", "illegal entry value!");
      return 0;
   }
   if (!fIndices){
      fIndices = new UShort_t[kBlockSize] ;
      for (Int_t i=0; i<kBlockSize; i++)
         fIndices[i] = 0;
      fType = 0; //start in bits
   }
   if (fType==0){
      //bits
      Int_t i = entry>>4;
      Int_t j = entry & 15;
      if ((fIndices[i] & (1<<j))==0){
         fIndices[i] |= 1<<j;
         fNPassed++;
         return 1;
      } else {
         return 0;
      }
   }
   //list
   //change to bits
   UShort_t *bits = new UShort_t[kBlockSize];
   Transform(1, bits);
   Enter(entry);
   return 0;
}

//______________________________________________________________________________
Bool_t TEntryListBlock::Remove(Int_t entry)
{
//Remove entry #entry

   if (entry > kBlockSize*16) {
      printf("illegal entry value!\n");
      return 0;
   }
   if (fType==0){
      Int_t i = entry>>4;
      Int_t j = entry & 15;
      if ((fIndices[i] & (1<<j))!=0){
         fIndices[i] &= (0xFFFF^(1<<j));
         fNPassed--;
         return 1;
      } else { 
         printf("not entered\n");
         return 0;
      }
   }
   //list
   //change to bits
   UShort_t *bits = new UShort_t[kBlockSize];
   Transform(1, bits);
   Remove(entry);
   return 0;
}
//______________________________________________________________________________
Int_t TEntryListBlock::Contains(Int_t entry)
{
//true if the block contains entry #entry

   if (entry > kBlockSize*16) {
      printf("illegal entry value!\n");
      return 0;
   }
   if (!fIndices)
      return 0;
   if (fType==0){
      //bits
      Int_t i = entry>>4;
      Int_t j = entry & 15;
      Bool_t result = (fIndices[i] & (1<<j))!=0;
      return result;
   }
   //list
   if (entry < fCurrent) fCurrent = 0;
  
   for (Int_t i = fCurrent; i<fNPassed; i++){
      if (fIndices[i]==entry){
         fCurrent = i;
         return kTRUE;
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TEntryListBlock::Merge(TEntryListBlock *block)
{
   //Merge with the other block

   Int_t i;
   if (fType==0){
      //stored as bits
      if (block->fType == 0){
         for (i=0; i<kBlockSize*16; i++){
            if (block->Contains(i))
               Enter(i);
         }
      } else {
         for (i=0; i<block->fNPassed; i++){
            Enter(block->fIndices[i]);
         }
      }
   } else {
      //stored as a list
      if (fNPassed + block->fNPassed > kBlockSize){
         //change to bits
         UShort_t *bits = new UShort_t[kBlockSize];
         Transform(1, bits);
         Merge(block);
      } else {
         if (block->fType==1){
            //second block stored as a list
            //make a bigger list
            Int_t en = block->fNPassed;
            Int_t newsize = fNPassed + en;
            UShort_t *newlist = new UShort_t[newsize];
            UShort_t *elst = block->fIndices;
            Int_t newpos, elpos;
            newpos = elpos = 0;
            for (i=0; i<fNPassed; i++) {
               while (elpos < en && fIndices[i] > elst[elpos]) {
                  newlist[newpos] = elst[elpos];
                  newpos++;
                  elpos++;
               }
               if (fIndices[i] == elst[elpos]) elpos++;
               newlist[newpos] = fIndices[i];
               newpos++;
            }
            while (elpos < en) {
               newlist[newpos] = elst[elpos];
               newpos++;
               elpos++;
            }
            delete [] fIndices;
            fIndices = newlist;
            fNPassed = newpos;
            fN = fNPassed;
         } else {
            //second block is stored as bits

            Int_t en = block->fNPassed;
            Int_t newsize = fNPassed + en;
            UShort_t *newlist = new UShort_t[newsize];
            Int_t newpos, current;
            newpos = current = 0;
            for (i=0; i<kBlockSize*16; i++){
               if (!block->Contains(i)) continue;
               while(current < fNPassed && fIndices[current]<i){
                  newlist[newpos] = fIndices[current];
                  current++;
                  newpos++;
               }
               if (fIndices[current]==i) current++;
               newlist[newpos] = i;
               newpos++;
            }
            while(current<fNPassed){
               newlist[newpos] = fIndices[current];
               newpos++;
               current++;
            }
            delete [] fIndices;
            fIndices = newlist;
            fNPassed = newpos;
            fN = fNPassed;
         }
      }
   }
   fLastIndexQueried = -1;
   fLastIndexReturned = -1;
   return fNPassed;
}

//______________________________________________________________________________
Int_t TEntryListBlock::GetEntry(Int_t entry)
{
//Return entry #entry
//See also Next()

   if (entry>kBlockSize*16) return -1;
   if (entry>fNPassed) return -1;
   if (entry==fLastIndexQueried+1) return Next();
   else {
      if (fType==0){
         Int_t entries_found = 0;
         Int_t i=0; 
         Int_t j=0;
         if ((fIndices[i] & (1<<j))!=0)
            entries_found++;
         while (entries_found<entry+1){
            if (j==15){i++; j=0;}
            else j++;
            if ((fIndices[i] & (1<<j))!=0)
               entries_found++;
         }
         fLastIndexQueried = entry;
         fLastIndexReturned = i*16+j;
         return fLastIndexReturned;
      }
      if (fType==1){
         fLastIndexQueried = entry;
         fLastIndexReturned = fIndices[entry];
         return fIndices[entry];
      }
      return -1;
   }
}

//______________________________________________________________________________
Int_t TEntryListBlock::Next()
{
//Return the next non-zero entry
//Faster than GetEntry() function

   if (fLastIndexQueried==fNPassed-1){
      fLastIndexQueried=-1;
      fLastIndexReturned = -1;
      return -1;
   }

   if (fType==0) {
      //bits
      Int_t i=0;
      Int_t j=0;
      fLastIndexReturned++;
      i = fLastIndexReturned>>4;
      j = fLastIndexReturned & 15;
      Bool_t result=(fIndices[i] & (1<<j))!=0;
      while (result==0){
         if (j==15) {j=0; i++;}
         else j++;
         result = (fIndices[i] & (1<<j))!=0;
      }
      fLastIndexReturned = i*16+j;
      fLastIndexQueried++;
      return fLastIndexReturned;

   } 
   if (fType==1) {
      fLastIndexQueried++;

      if (fLastIndexQueried==kBlockSize) {
         fLastIndexQueried = -1;
         fLastIndexReturned = -1;
         return -1;
      }
      else {
         return fIndices[fLastIndexQueried];
         fLastIndexReturned = fIndices[fLastIndexQueried];
      }
   }
   return -1;
}

//______________________________________________________________________________
void TEntryListBlock::Print(const Option_t *option) const
{
//Print the entries in this block

   TString opt = option;
   opt.ToUpper();
   if (opt.Contains("A")){
      if (fType==0){
         Int_t ibit, ibite;
         Bool_t result;
         for (Int_t i=0; i<kBlockSize; i++){
            ibite = i>>4;
            ibit = i & 15;
            result = (fIndices[ibite] & (1<<ibit))!=0;
            if (result)
               printf("%d\n", i);
         }
      } else {
         for (Int_t i=0; i<fNPassed; i++){
            printf("%d\n", fIndices[i]);
         }
      }
   }
}

//______________________________________________________________________________
void TEntryListBlock::PrintWithShift(Int_t shift) const
{
   //print the indices of this block + shift (used from TEntryList::Print()) to 
   //print the corrent values

   Int_t i;
   if (fType==0){
      Int_t ibit, ibite;
      Bool_t result;
      for (i=0; i<kBlockSize*16; i++){
         ibite = i>>4;
         ibit = i & 15;
         result = (fIndices[ibite] & (1<<ibit))!=0;
         if (result)
            printf("%d\n", i+shift);
      }
   } else {
      for (i=0; i<fNPassed; i++){
         printf("%d\n", fIndices[i]+shift);
      }
   }
}


//______________________________________________________________________________
void TEntryListBlock::OptimizeStorage()
{
   //if there are < kBlockSize entries, change to an array representation

   if (fType!=0) return;
   if (fNPassed<kBlockSize){
      //less than 4000 entries passing, makes sense to change from bits to list
      UShort_t *indexnew = new UShort_t[fNPassed];
      Transform(0, indexnew);
   }
}


//______________________________________________________________________________
void TEntryListBlock::Transform(Bool_t dir, UShort_t *indexnew)
{
   //Transform the existing fIndices
   //dir=0 - transform from bits to a list
   //dir=1 - tranform from a list to bits

   Int_t i=0;
   Int_t ilist = 0;
   Int_t ibite, ibit;
   if (!dir) {
      for (i=0; i<kBlockSize; i++){
         ibite = i >> 4;
         ibit = i & 15;
         Bool_t result = (fIndices[ibite] & (1<<ibit))!=0;
         if (result){
            indexnew[ilist] = i;
            ilist++;
         }
      }
      delete [] fIndices;
      fIndices = indexnew;
      fType = 1;
      fN = fNPassed;
      return;
   }

   for (i=0; i<kBlockSize; i++)
      indexnew[i] = 0;
   for (i=0; i<fNPassed; i++){
      ibite = fIndices[i]>>4;
      ibit = fIndices[i] & 15;
      indexnew[ibite] |= 1<<ibit;
   }
   delete [] fIndices;
   fIndices = indexnew;
   fType = 0;
   fN = kBlockSize;
   return;
}
