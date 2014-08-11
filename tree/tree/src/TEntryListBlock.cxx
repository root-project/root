// @(#)root/tree:$Id$
// Author: Anna Kreshuk 27/10/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
/* Begin_Html
<center><h2>TEntryListBlock: Used by TEntryList to store the entry numbers</h2></center>
 There are 2 ways to represent entry numbers in a TEntryListBlock:
<ol>
 <li> as bits, where passing entry numbers are assigned 1, not passing - 0
 <li> as a simple array of entry numbers
<ul>
<li> storing the numbers of entries that pass
<li> storing the numbers of entries that don't pass
</ul>
 </ol>
 In both cases, a UShort_t* is used. The second option is better in case
 less than 1/16 or more than 15/16 of entries pass the selection, and the representation can be
 changed by calling OptimizeStorage() function.
 When the block is being filled, it's always stored as bits, and the OptimizeStorage()
 function is called by TEntryList when it starts filling the next block. If
 Enter() or Remove() is called after OptimizeStorage(), representation is
 again changed to 1).
End_Html
Begin_Macro(source)
entrylistblock_figure1.C
End_Macro

Begin_Html
 <h4>Operations on blocks (see also function comments)</h4>
<ul>
 <li> <b>Merge</b>() - adds all entries from one block to the other. If the first block
             uses array representation, it's changed to bits representation only
             if the total number of passing entries is still less than kBlockSize
 <li> <b>GetEntry(n)</b> - returns n-th non-zero entry.
 <li> <b>Next</b>()      - return next non-zero entry. In case of representation 1), Next()
                 is faster than GetEntry()
</ul>
End_Html */


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

   fN = eblock.fN;
   if (eblock.fIndices){
      fIndices = new UShort_t[fN];
      for (Int_t i=0; i<fN; i++)
         fIndices[i] = eblock.fIndices[i];
   } else {
      fIndices = 0;
   }
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
TEntryListBlock &TEntryListBlock::operator=(const TEntryListBlock &eblock)
{
   if (this != &eblock) {
      if (fIndices)
         delete [] fIndices;

      fN = eblock.fN;
      if (eblock.fIndices){
         fIndices = new UShort_t[fN];
         for (Int_t i=0; i<fN; i++)
            fIndices[i] = eblock.fIndices[i];
      } else {
         fIndices = 0;
      }
      fNPassed = eblock.fNPassed;
      fType = eblock.fType;
      fPassing = eblock.fPassing;
      fCurrent = eblock.fCurrent;
      fLastIndexReturned = -1;
      fLastIndexQueried = -1;
   }
   return *this;
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
//If the block has already been optimized and the entries
//are stored as a list and not as bits, trying to remove a new entry
//will make the block switch to bits representation

   if (entry > kBlockSize*16) {
      Error("Remove", "Illegal entry value!\n");
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
         return 0;
      }
   }
   //list
   //change to bits
   UShort_t *bits = new UShort_t[kBlockSize];
   Transform(1, bits);
   return Remove(entry);
   //return 0;
}
//______________________________________________________________________________
Int_t TEntryListBlock::Contains(Int_t entry)
{
//true if the block contains entry #entry

   if (entry > kBlockSize*16) {
      Error("Contains", "Illegal entry value!\n");
      return 0;
   }
   if (!fIndices && fPassing)
      return 0;
   if (fType==0 && fIndices){
      //bits
      Int_t i = entry>>4;
      Int_t j = entry & 15;
      Bool_t result = (fIndices[i] & (1<<j))!=0;
      return result;
   }
   //list
   if (entry < fCurrent) fCurrent = 0;
   if (fPassing && fIndices){
      for (Int_t i = fCurrent; i<fNPassed; i++){
         if (fIndices[i]==entry){
            fCurrent = i;
            return kTRUE;
         }
      }
   } else {
      if (!fIndices || fNPassed==0){
         //all entries pass
         return kTRUE;
      }
      if (entry > fIndices[fNPassed-1])
         return kTRUE;
      for (Int_t i= fCurrent; i<fNPassed; i++){
         if (fIndices[i]==entry){
            fCurrent = i;
            return kFALSE;
         }
         if (fIndices[i]>entry){
            fCurrent = i;
            return kTRUE;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TEntryListBlock::Merge(TEntryListBlock *block)
{
   //Merge with the other block
   //Returns the resulting number of entries in the block

   Int_t i, j;
   if (block->GetNPassed() == 0) return GetNPassed();
   if (GetNPassed() == 0){
      //this block is empty
      fN = block->fN;
      fIndices = new UShort_t[fN];
      for (i=0; i<fN; i++)
         fIndices[i] = block->fIndices[i];
      fNPassed = block->fNPassed;
      fType = block->fType;
      fPassing = block->fPassing;
      fCurrent = block->fCurrent;
      fLastIndexReturned = -1;
      fLastIndexQueried = -1;
      return fNPassed;
   }
   if (fType==0){
      //stored as bits
      if (block->fType == 0){
         for (i=0; i<kBlockSize*16; i++){
            if (block->Contains(i))
               Enter(i);
         }
      } else {
         if (block->fPassing){
            //the other block stores entries that pass
            for (i=0; i<block->fNPassed; i++){
               Enter(block->fIndices[i]);
            }
         } else {
            //the other block stores entries that don't pass
            if (block->fNPassed==0){
               for (i=0; i<kBlockSize*16; i++){
                  Enter(i);
               }
            }
            for (j=0; j<block->fIndices[0]; j++)
               Enter(j);
            for (i=0; i<block->fNPassed-1; i++){
               for (j=block->fIndices[i]+1; j<block->fIndices[i+1]; j++){
                  Enter(j);
               }
            }
            for (j=block->fIndices[block->fNPassed-1]+1; j<kBlockSize*16; j++)
               Enter(j);
         }
      }
   } else {
      //stored as a list
      if (GetNPassed() + block->GetNPassed() > kBlockSize){
         //change to bits
         UShort_t *bits = new UShort_t[kBlockSize];
         Transform(1, bits);
         Merge(block);
      } else {
         //this is only possible if fPassing=1 in both blocks
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
   OptimizeStorage();
   return GetNPassed();
}

//______________________________________________________________________________
Int_t TEntryListBlock::GetNPassed()
{
//Returns the number of entries, passing the selection.
//In case, when the block stores entries that pass (fPassing=1) returns fNPassed

   if (fPassing)
      return fNPassed;
   else
      return kBlockSize*16-fNPassed;
}

//______________________________________________________________________________
Int_t TEntryListBlock::GetEntry(Int_t entry)
{
//Return entry #entry
//See also Next()

   if (entry > kBlockSize*16) return -1;
   if (entry > GetNPassed()) return -1;
   if (entry == fLastIndexQueried+1) return Next();
   else {
      Int_t i=0; Int_t j=0; Int_t entries_found=0;
      if (fType==0){
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
         if (fPassing){
            fLastIndexQueried = entry;
            fLastIndexReturned = fIndices[entry];
            return fIndices[entry];
         } else {
            fLastIndexQueried = entry;
            if (!fIndices || fNPassed==0){
               //all entries pass
               fLastIndexReturned = entry;
               return fLastIndexReturned;
            }
            for (i=0; i<fIndices[0]; i++){
               entries_found++;
               if (entries_found==entry+1){
                  fLastIndexReturned = i;
                  return fLastIndexReturned;
               }
            }
            for (i=0; i<fNPassed-1; i++){
               for (j=fIndices[i]+1; j<fIndices[i+1]; j++){
                  entries_found++;
                  if (entries_found==entry+1){
                     fLastIndexReturned = j;
                     return fLastIndexReturned;
                  }
               }
            }
            for (j=fIndices[fNPassed-1]+1; j<kBlockSize*16; j++){
               entries_found++;
               if (entries_found==entry+1){
                  fLastIndexReturned = j;
                  return fLastIndexReturned;
               }
            }
         }
      }
      return -1;
   }
}

//______________________________________________________________________________
Int_t TEntryListBlock::Next()
{
//Return the next non-zero entry
//Faster than GetEntry() function

   if (fLastIndexQueried==GetNPassed()-1){
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
      if (fPassing){
         fLastIndexReturned = fIndices[fLastIndexQueried];
         return fIndices[fLastIndexQueried];
      } else {
         fLastIndexReturned++;
         while (!Contains(fLastIndexReturned)){
            fLastIndexReturned++;
            }
         return fLastIndexReturned;
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
   if (opt.Contains("A")) PrintWithShift(0);
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
      if (fPassing){
         for (i=0; i<fNPassed; i++){
            printf("%d\n", fIndices[i]+shift);
         }
      } else {
         if (fNPassed==0){
            for (i=0; i<kBlockSize*16; i++)
               printf("%d\n", i+shift);
            return;
         }
         for (i=0; i<fIndices[0]; i++){
            printf("%d\n", i+shift);
         }
         for (i=0; i<fNPassed-1; i++){
            for (Int_t j=fIndices[i]+1; j<fIndices[i+1]; j++){
               printf("%d\n", j+shift);
            }
         }
         for (Int_t j=fIndices[fNPassed-1]+1; j<kBlockSize*16; j++){
            printf("%d\n", j+shift);
         }
      }
   }
}


//______________________________________________________________________________
void TEntryListBlock::OptimizeStorage()
{
   //if there are < kBlockSize or >kBlockSize*15 entries, change to an array representation

   if (fType!=0) return;
   if (fNPassed > kBlockSize*15)
      fPassing = 0;
   if (fNPassed<kBlockSize || !fPassing){
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
         for (i=0; i<kBlockSize*16; i++){
            ibite = i >> 4;
            ibit = i & 15;
            Bool_t result = (fIndices[ibite] & (1<<ibit))!=0;
            if (result && fPassing){
               //fill with the entries that pass
               indexnew[ilist] = i;
               ilist++;
            }
            else if (!result && !fPassing){
               //fill with the entries that don't pass
               indexnew[ilist] = i;
               ilist++;
            }
         }
      if (fIndices)
         delete [] fIndices;
      fIndices = indexnew;
      fType = 1;
      if (!fPassing)
         fNPassed = kBlockSize*16-fNPassed;
      fN = fNPassed;
      return;
   }


   if (fPassing){
      for (i=0; i<kBlockSize; i++)
         indexnew[i] = 0;
      for (i=0; i<fNPassed; i++){
         ibite = fIndices[i]>>4;
         ibit = fIndices[i] & 15;
         indexnew[ibite] |= 1<<ibit;
      }
   } else {
      for (i=0; i<kBlockSize; i++)
         indexnew[i] = 65535;
      for (i=0; i<fNPassed; i++){
         ibite = fIndices[i]>>4;
         ibit = fIndices[i] & 15;
         indexnew[ibite] ^= 1<<ibit;
      }
      fNPassed = kBlockSize*16-fNPassed;
   }
   if (fIndices)
      delete [] fIndices;
   fIndices = indexnew;
   fType = 0;
   fN = kBlockSize;
   fPassing = 1;
   return;
}
