// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveChunkManager.h"

//______________________________________________________________________________
//
// Vector-like container with chunked memory allocation.
//
// Allocation chunk can accommodate fN atoms of byte-size fS each.
// The chunks themselves are TArrayCs and are stored in a std::vector<TArrayC*>.
// Holes in the structure are not supported, neither is removal of atoms.
// The structure can be Refit() to occupy a single contiguous array.
//

ClassImp(TEveChunkManager);
ClassImp(TEveChunkManager::iterator);

//______________________________________________________________________________
void TEveChunkManager::ReleaseChunks()
{
   // Release all memory chunks.

   for (Int_t i=0; i<fVecSize; ++i)
      delete fChunks[i];
   fChunks.clear();
}

//______________________________________________________________________________
TEveChunkManager::TEveChunkManager() :
   fS(0), fN(0),
   fSize(0), fVecSize(0), fCapacity(0)
{
   // Default constructor.
   // Call reset for initialization.
}

//______________________________________________________________________________
TEveChunkManager::TEveChunkManager(Int_t atom_size, Int_t chunk_size) :
   fS(atom_size), fN(chunk_size),
   fSize(0), fVecSize(0), fCapacity(0)
{
   // Constructor.
}

//______________________________________________________________________________
TEveChunkManager::~TEveChunkManager()
{
   // Destructor.

   ReleaseChunks();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveChunkManager::Reset(Int_t atom_size, Int_t chunk_size)
{
   // Empty the container and reset it with given atom and chunk sizes.

   ReleaseChunks();
   fS = atom_size;
   fN = chunk_size;
   fSize = fVecSize = fCapacity = 0;
}

//______________________________________________________________________________
void TEveChunkManager::Refit()
{
   // Refit the container so that all current data fits into a single
   // chunk.

   if (fSize == 0 || (fVecSize == 1 && fSize == fCapacity))
      return;

   TArrayC* one = new TArrayC(fS*fSize);
   Char_t*  pos = one->fArray;
   for (Int_t i=0; i<fVecSize; ++i)
   {
      Int_t size = fS * NAtoms(i);
      memcpy(pos, fChunks[i]->fArray, size);
      pos += size;
   }
   ReleaseChunks();
   fN = fCapacity = fSize;
   fVecSize = 1;
   fChunks.push_back(one);
}

/******************************************************************************/

//______________________________________________________________________________
Char_t* TEveChunkManager::NewChunk()
{
   // Allocate a new memory chunk and register it.

   fChunks.push_back(new TArrayC(fS*fN));
   ++fVecSize;
   fCapacity += fN;
   return fChunks.back()->fArray;
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveChunkManager::iterator::next()
{
   // Go to next atom.

   if (fSelection == 0)
   {
      if (fAtomsToGo <= 0)
      {
         if (fNextChunk < fPlex->VecSize())
         {
            fCurrent   = fPlex->Chunk(fNextChunk);
            fAtomsToGo = fPlex->NAtoms(fNextChunk);
            ++fNextChunk;
         }
         else
         {
            return kFALSE;
         }
      }
      else
      {
         fCurrent += fPlex->S();
      }
      ++fAtomIndex;
      --fAtomsToGo;
      return kTRUE;
   }
   else
   {
      if (fAtomIndex == -1)
         fSelectionIterator = fSelection->begin();
      else
         ++fSelectionIterator;

      if (fSelectionIterator != fSelection->end())
      {
         fAtomIndex = *fSelectionIterator;
         fCurrent   =  fPlex->Atom(fAtomIndex);
         return kTRUE;
      }
      else
      {
         return kFALSE;
      }
   }
}
