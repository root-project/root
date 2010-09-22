// @(#)root/new:$Id$
// Author: D.Bertini and M.Ivanov   10/08/2000

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//****************************************************************************//
//
//
// MemCheck is used to check the memory in ROOT based applications.
//
// Principe:
//  A memory leak often arises whenever memory allocated through
//  (new, new[]) is never returned via a corresponding (delete, delete[]).
//  Redefining a special version of (operator new, operator delete) will
//  allow the bookkeeping of created pointers to chunks of memory and, at
//  the same time, pointers to the current function (and all of its callers
//  scanning the stack) in order to trace back where the memory is not freed.
//  This specific bookkeeping of pointers will be done by a kind of
//  "memory info" container class TMemHashTable that will be used via
//  the ROOT memory management defined inside the NewDelete.cxx file.
//
//  To activate the memory checker you have to set in the .rootrc file
//  the resource Root.MemCheck to 1 (e.g.: Root.MemCheck: 1) and you
//  have to link with libNew.so (e.g. use root-config --new --libs) or
//  use rootn.exe. When all this is the case you will find at the end
//  of the program execution a file "memcheck.out" in the directory
//  where you started your ROOT program. Alternatively you can set
//  the resource Root.MemCheckFile to the name of a file to which
//  the leak information will be written. The contents of this
//  "memcheck.out" file can be analyzed and transformed into printable
//  text via the memprobe program (in $ROOTSYS/bin).
//
// (c) 2000 : Gesellschaft fuer Schwerionenforschung GmbH
//            Planckstrasse, 1
//            D-64291 Darmstadt
//            Germany
//
// Created 10/08/2000 by: D.Bertini and M.Ivanov.
// Based on ideas from LeakTracer by Erwin Andreasen.
//
// - Updated:
//    Date: 12/02/2001 Adapt script to new GDB 5.0, new glibc2.2.x and gcc 2.96.
//    Date: 23/10/2000 (hash mechanism speeding up the bookkeeping)
//
// - Documentation:
//
//    http://www-hades.gsi.de/~dbertini/mem.html
//
//****************************************************************************//

#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#include "MemCheck.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TError.h"

#define stack_history_size 20


static TMemHashTable gMemHashTable;


//****************************************************************************//
//                                 Storage of Stack information
//****************************************************************************//

//______________________________________________________________________________
void TStackInfo::Init(int stacksize, void **stackptrs)
{
   //Initialize the stack
   fSize = stacksize;
   memcpy(&(this[1]), stackptrs, stacksize * sizeof(void *));
   fTotalAllocCount = fTotalAllocSize = fAllocCount = fAllocSize = 0;
}

//______________________________________________________________________________
ULong_t TStackInfo::HashStack(unsigned int size, void **ptr)
{
   // Hash stack information.

   ULong_t hash = 0;
   for (unsigned int i = 0; i < size; i++)
      hash ^= TString::Hash(&ptr[i], sizeof(void*));
   return hash;
}

//______________________________________________________________________________
int TStackInfo::IsEqual(unsigned int size, void **ptr)
{
   // Return 0 if stack information not equal otherwise return 1.

   if (size != fSize)
      return 0;
   void **stptr = (void **) &(this[1]);
   for (unsigned int i = 0; i < size; i++)
      if (ptr[i] != stptr[i])
         return 0;
   return 1;
}


//****************************************************************************//
//                                   Global Stack Table
//****************************************************************************//

//______________________________________________________________________________
void TStackTable::Init()
{
   //Initialize table.

   fSize = 65536;
   fCount = 0;
   fTable = (char *) malloc(fSize);
   if (!fTable)
      _exit(1);
   memset(fTable, 0, fSize);
   fNext = fTable;
   //initialize hash table
   fHashSize = 65536;
   fHashTable = (TStackInfo **) malloc(sizeof(TStackInfo *) * fHashSize);
   memset(fHashTable, 0, sizeof(TStackInfo *) * fHashSize);
}

//______________________________________________________________________________
void TStackTable::Expand(int newsize)
{
   // Expand stack buffer to the new size.

   char *tableold = fTable;
   fTable = (char *) realloc(fTable, newsize);
   fSize = newsize;
   int nextindex = (char *) fNext - tableold;
   memset(&fTable[nextindex], 0, fSize - nextindex);
   fNext = (char *) (&fTable[nextindex]);
   //
   //update list
   TStackInfo *info = (TStackInfo *) fTable;
   while (((char *) info->Next() - fTable) <= nextindex) {
      if (info->fNextHash != 0)
         info->fNextHash = (TStackInfo *)
             & fTable[(char *) info->fNextHash - tableold];
      info = info->Next();
   }
   //
   //update hash table
   for (int i = 0; i < fHashSize; i++)
      if (fHashTable[i] != 0)
         fHashTable[i] =
             (TStackInfo *) & fTable[((char *) fHashTable[i]) - tableold];
   //  printf("new table %p\n",fTable);
}

//______________________________________________________________________________
TStackInfo *TStackTable::AddInfo(int size, void **stackptrs)
{
   // Add stack information to table.

   // add next stack to table
   TStackInfo *info = (TStackInfo *) fNext;
   if (((char *) info + size * sizeof(void *)
        + sizeof(TStackInfo) - fTable) > fSize) {
      //need expand
      Expand(2 * fSize);
      info = (TStackInfo *) fNext;
   }
   info->Init(size, stackptrs);
   info->fNextHash = 0;
   fNext = (char *) info->Next();

   //add info to hash table
   int hash = int(info->Hash() % fHashSize);
   TStackInfo *info2 = fHashTable[hash];
   if (info2 == 0) {
      fHashTable[hash] = info;
   } else {
      while (info2->fNextHash)
         info2 = info2->fNextHash;
      info2->fNextHash = info;
   }
   fCount++;
   return info;
}

//______________________________________________________________________________
TStackInfo *TStackTable::FindInfo(int size, void **stackptrs)
{
   // Try to find stack info in hash table if doesn't find it will add it.

   int hash = int(TStackInfo::HashStack(size, (void **) stackptrs) % fHashSize);
   TStackInfo *info = fHashTable[hash];
   if (info == 0) {
      info = AddInfo(size, stackptrs);
      //printf("f0 %p    - %d\n",info,(char*)info-fTable);
      return info;
   }
   while (info->IsEqual(size, stackptrs) == 0) {
      if (info->fNextHash == 0) {
         info = AddInfo(size, stackptrs);
         //  printf("f1 %p    - %d\n",info,(char*)info-fTable);
         return info;
      } else
         info = info->fNextHash;
   }
   //printf("f2  %p   - %d\n",info,(char*)info-fTable);
   return info;
};

//______________________________________________________________________________
int TStackTable::GetIndex(TStackInfo * info)
{
   //return index of info
   return (char *) info - fTable;
}

//______________________________________________________________________________
TStackInfo *TStackTable::GetInfo(int index)
{
   //return TStackInfo class corresponding to index
   return (TStackInfo *) & fTable[index];
}


Int_t        TMemHashTable::fgSize = 0;
Int_t        TMemHashTable::fgAllocCount = 0;
TMemTable  **TMemHashTable::fgLeak = 0;
TDeleteTable TMemHashTable::fgMultDeleteTable;
TStackTable  TMemHashTable::fgStackTable;


static void *get_stack_pointer(int level);

//______________________________________________________________________________
void TMemHashTable::Init()
{
   //Initialize the hash table
   fgStackTable.Init();
   fgSize = 65536;
   fgAllocCount = 0;
   fgLeak = (TMemTable **) malloc(sizeof(void *) * fgSize);
   fgMultDeleteTable.fLeaks = 0;
   fgMultDeleteTable.fAllocCount = 0;
   fgMultDeleteTable.fTableSize = 0;

   for (int i = 0; i < fgSize; i++) {
      fgLeak[i] = (TMemTable *) malloc(sizeof(TMemTable));
      fgLeak[i]->fAllocCount = 0;
      fgLeak[i]->fMemSize = 0;
      fgLeak[i]->fFirstFreeSpot = 0;
      fgLeak[i]->fTableSize = 0;
      fgLeak[i]->fLeaks = 0;
   }
}

//______________________________________________________________________________
void TMemHashTable::RehashLeak(int newSize)
{
   // Rehash leak pointers.

   if (newSize <= fgSize)
      return;
   TMemTable **newLeak = (TMemTable **) malloc(sizeof(void *) * newSize);
   for (int i = 0; i < newSize; i++) {
      //build new branches
      newLeak[i] = (TMemTable *) malloc(sizeof(TMemTable));
      newLeak[i]->fAllocCount = 0;
      newLeak[i]->fMemSize = 0;
      newLeak[i]->fFirstFreeSpot = 0;
      newLeak[i]->fTableSize = 0;
      newLeak[i]->fLeaks = 0;
   }
   for (int ib = 0; ib < fgSize; ib++) {
      TMemTable *branch = fgLeak[ib];
      for (int i = 0; i < branch->fTableSize; i++)
         if (branch->fLeaks[i].fAddress != 0) {
            int hash = int(TString::Hash(&branch->fLeaks[i].fAddress, sizeof(void*)) % newSize);
            TMemTable *newbranch = newLeak[hash];
            if (newbranch->fAllocCount >= newbranch->fTableSize) {
               int newTableSize =
                   newbranch->fTableSize ==
                   0 ? 16 : newbranch->fTableSize * 2;
               newbranch->fLeaks =
                   (TMemInfo *) realloc(newbranch->fLeaks,
                                        sizeof(TMemInfo) * newTableSize);
               if (!newbranch->fLeaks) {
                  Error("TMemHashTable::AddPointer", "realloc failure");
                  _exit(1);
               }
               memset(newbranch->fLeaks + newbranch->fTableSize, 0,
                      sizeof(TMemInfo) * (newTableSize -
                                          newbranch->fTableSize));
               newbranch->fTableSize = newTableSize;
            }
            memcpy(&newbranch->fLeaks[newbranch->fAllocCount],
                   &branch->fLeaks[i], sizeof(TMemInfo));
            newbranch->fAllocCount++;
            newbranch->fMemSize += branch->fLeaks[i].fSize;
         }
      free(branch->fLeaks);
      free(branch);
   }                 //loop over all old branches and rehash information
   free(fgLeak);
   fgLeak = newLeak;
   fgSize = newSize;
}

//______________________________________________________________________________
void *TMemHashTable::AddPointer(size_t size, void *ptr)
{
   // Add pointer to table.

   void *p = 0;

   if (ptr == 0) {
      p = malloc(size);
      if (!p) {
         Error("TMemHashTable::AddPointer", "malloc failure");
         _exit(1);
      }
   } else {
      p = realloc((char *) ptr, size);
      if (!p) {
         Error("TMemHashTable::AddPointer", "realloc failure");
         _exit(1);
      }
      return p;
   }

   if (!fgSize)
      Init();
   fgAllocCount++;
   if ((fgAllocCount / fgSize) > 128)
      RehashLeak(fgSize * 2);
   int hash = int(TString::Hash(&p, sizeof(void*)) % fgSize);
   TMemTable *branch = fgLeak[hash];
   branch->fAllocCount++;
   branch->fMemSize += size;
   for (;;) {
      for (int i = branch->fFirstFreeSpot; i < branch->fTableSize; i++)
         if (branch->fLeaks[i].fAddress == 0) {
            branch->fLeaks[i].fAddress = p;
            branch->fLeaks[i].fSize = size;
            void *sp = 0;
            int j = 0;
            void *stptr[stack_history_size + 1];
            for (j = 0; (j < stack_history_size); j++) {
               sp = get_stack_pointer(j + 1);
               if (sp == 0)
                  break;
               stptr[j] = sp;
            }
            TStackInfo *info = fgStackTable.FindInfo(j, stptr);
            info->Inc(size);
            branch->fLeaks[i].fStackIndex = fgStackTable.GetIndex(info);
            branch->fFirstFreeSpot = i + 1;
            return p;
         }

      int newTableSize =
          branch->fTableSize == 0 ? 16 : branch->fTableSize * 2;
      branch->fLeaks =
          (TMemInfo *) realloc(branch->fLeaks,
                               sizeof(TMemInfo) * newTableSize);
      if (!branch->fLeaks) {
         Error("TMemHashTable::AddPointer", "realloc failure (2)");
         _exit(1);
      }
      memset(branch->fLeaks + branch->fTableSize, 0, sizeof(TMemInfo) *
             (newTableSize - branch->fTableSize));
      branch->fTableSize = newTableSize;
   }
}

//______________________________________________________________________________
void TMemHashTable::FreePointer(void *p)
{
   // Free pointer.

   if (p == 0)
      return;
   int hash = int(TString::Hash(&p, sizeof(void*)) % fgSize);
   fgAllocCount--;
   TMemTable *branch = fgLeak[hash];
   for (int i = 0; i < branch->fTableSize; i++) {
      if (branch->fLeaks[i].fAddress == p) {
         branch->fLeaks[i].fAddress = 0;
         branch->fMemSize -= branch->fLeaks[i].fSize;
         if (i < branch->fFirstFreeSpot)
            branch->fFirstFreeSpot = i;
         free(p);
         TStackInfo *info =
             fgStackTable.GetInfo(branch->fLeaks[i].fStackIndex);
         info->Dec(branch->fLeaks[i].fSize);
         branch->fAllocCount--;
         return;
      }
   }
   //
   //if try to delete non existing pointer
   //printf("***TMemHashTable::FreePointer: Multiple deletion %8p ** ?  \n",p);
   //  printf("-+-+%8p  \n",p);
   //free(p);
   if (fgMultDeleteTable.fTableSize + 1 > fgMultDeleteTable.fAllocCount) {
      int newTableSize =
          fgMultDeleteTable.fTableSize ==
          0 ? 16 : fgMultDeleteTable.fTableSize * 2;
      fgMultDeleteTable.fLeaks =
          (TMemInfo *) realloc(fgMultDeleteTable.fLeaks,
                               sizeof(TMemInfo) * newTableSize);
      fgMultDeleteTable.fAllocCount = newTableSize;
   }

   fgMultDeleteTable.fLeaks[fgMultDeleteTable.fTableSize].fAddress = 0;
   void *sp = 0;
   void *stptr[stack_history_size + 1];
   int j;
   for (j = 0; (j < stack_history_size); j++) {
      sp = get_stack_pointer(j + 1);
      if (sp == 0)
         break;
      stptr[j] = sp;
   }
   TStackInfo *info = fgStackTable.FindInfo(j, stptr);
   info->Dec(0);
   fgMultDeleteTable.fLeaks[fgMultDeleteTable.fTableSize].fStackIndex =
       fgStackTable.GetIndex(info);
   fgMultDeleteTable.fTableSize++;
}

//______________________________________________________________________________
void TMemHashTable::Dump()
{
   // Print memory check information.

   const char *filename;
   if (gEnv)
      filename = gEnv->GetValue("Root.MemCheckFile", "memcheck.out");
   else
      filename = "memcheck.out";

   char *fn = 0;
   if (gSystem)
      fn = gSystem->ExpandPathName(filename);

   FILE *fp;
   if (!(fp = fn ? fopen(fn, "w") : fopen(filename, "w")))
      Error("TMenHashTable::Dump", "could not open %s", filename);
   else {
      /*
         for (int i = 0; i <  fgMultDeleteTable.fTableSize; i++){
         fprintf(fp, "size %9ld  ",(long)0);
         fprintf(fp, "stack:");
         TStackInfo *info = fgStackTable.GetInfo(fgMultDeleteTable.fLeaks[i].fStackIndex);
         for (int j=0; info->StackAt(j);j++)
         fprintf(fp, "%8p  ", info->StackAt(j));
         fprintf(fp, "\n");
         }
       */
      TStackInfo *info = fgStackTable.First();
      while (info->fSize) {
         fprintf(fp, "size %d:%d:%d:%d  ",
                 info->fTotalAllocCount, info->fTotalAllocSize,
                 info->fAllocCount, info->fAllocSize);
         fprintf(fp, "stack:");
         for (int j = 0; info->StackAt(j); j++)
            fprintf(fp, "%8p  ", info->StackAt(j));
         fprintf(fp, "\n");
         info = info->Next();
      }
      fclose(fp);
   }
   delete [] fn;
}

//______________________________________________________________________________
static void *get_stack_pointer(int level)
{
   // These special __builtin calls are supported by gcc "only".
   // For other compiler one will need to implement this again !

   void *p = 0;
#if defined(R__GNU) && (defined(R__LINUX) || defined(R__HURD)) && \
   !defined(__alpha__)
   switch (level) {
   case 0:
      if (__builtin_frame_address(1))
         p = __builtin_return_address(1);
      break;
   case 1:
      if (__builtin_frame_address(2))
         p = __builtin_return_address(2);
      break;
   case 2:
      if (__builtin_frame_address(3))
         p = __builtin_return_address(3);
      break;
   case 3:
      if (__builtin_frame_address(4))
         p = __builtin_return_address(4);
      break;
   case 4:
      if (__builtin_frame_address(5))
         p = __builtin_return_address(5);
      break;
   case 5:
      if (__builtin_frame_address(6))
         p = __builtin_return_address(6);
      break;
   case 6:
      if (__builtin_frame_address(7))
         p = __builtin_return_address(7);
      break;
   case 7:
      if (__builtin_frame_address(8))
         p = __builtin_return_address(8);
      break;
   case 8:
      if (__builtin_frame_address(9))
         p = __builtin_return_address(9);
      break;
   case 9:
      if (__builtin_frame_address(10))
         p = __builtin_return_address(10);
      break;
   case 10:
      if (__builtin_frame_address(11))
         p = __builtin_return_address(11);
      break;
   case 11:
      if (__builtin_frame_address(12))
         p = __builtin_return_address(12);
      break;
   case 12:
      if (__builtin_frame_address(13))
         p = __builtin_return_address(13);
      break;
   case 13:
      if (__builtin_frame_address(14))
         p = __builtin_return_address(14);
      break;
   case 14:
      if (__builtin_frame_address(15))
         p = __builtin_return_address(15);
      break;
   case 15:
      if (__builtin_frame_address(16))
         p = __builtin_return_address(16);
      break;
   case 16:
      if (__builtin_frame_address(17))
         p = __builtin_return_address(17);
      break;
   case 17:
      if (__builtin_frame_address(18))
         p = __builtin_return_address(18);
      break;
   case 18:
      if (__builtin_frame_address(19))
         p = __builtin_return_address(19);
      break;
   case 19:
      if (__builtin_frame_address(20))
         p = __builtin_return_address(20);
      break;

   default:
      p = 0;
   }
#else
   if (level) { }
#endif
   return p;
}
