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

/** \class TEveChunkManager
\ingroup TEve
Vector-like container with chunked memory allocation.

Allocation chunk can accommodate fN atoms of byte-size fS each.
The chunks themselves are TArrayCs and are stored in a std::vector<TArrayC*>.
Holes in the structure are not supported, neither is removal of atoms.
The structure can be Refit() to occupy a single contiguous array.
*/

ClassImp(TEveChunkManager);
ClassImp(TEveChunkManager::iterator);

////////////////////////////////////////////////////////////////////////////////
/// Release all memory chunks.

void TEveChunkManager::ReleaseChunks()
{
   for (Int_t i=0; i<fVecSize; ++i)
      delete fChunks[i];
   fChunks.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.
/// Call reset for initialization.

TEveChunkManager::TEveChunkManager() :
   fS(0), fN(0),
   fSize(0), fVecSize(0), fCapacity(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveChunkManager::TEveChunkManager(Int_t atom_size, Int_t chunk_size) :
   fS(atom_size), fN(chunk_size),
   fSize(0), fVecSize(0), fCapacity(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveChunkManager::~TEveChunkManager()
{
   ReleaseChunks();
}

////////////////////////////////////////////////////////////////////////////////
/// Empty the container and reset it with given atom and chunk sizes.

void TEveChunkManager::Reset(Int_t atom_size, Int_t chunk_size)
{
   ReleaseChunks();
   fS = atom_size;
   fN = chunk_size;
   fSize = fVecSize = fCapacity = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Refit the container so that all current data fits into a single
/// chunk.

void TEveChunkManager::Refit()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Allocate a new memory chunk and register it.

Char_t* TEveChunkManager::NewChunk()
{
   fChunks.push_back(new TArrayC(fS*fN));
   ++fVecSize;
   fCapacity += fN;
   return fChunks.back()->fArray;
}

////////////////////////////////////////////////////////////////////////////////
/// Go to next atom.

Bool_t TEveChunkManager::iterator::next()
{
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
