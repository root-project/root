// @(#)root/io:$Id$
// Author: Elvin Sindrilaru   19/05/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TFPBlock TFPBlock.cxx
\ingroup IO

This class represents the encapsulation of a block request.
It contains the chunks to be prefetched and also serves as a
container for the information read.
These blocks are prefetch in a special reader thread by the
TFilePrefetch class.
*/

#include "TFPBlock.h"
#include "TStorage.h"
#include <cstdlib>

using std::calloc;
using std::free;
using std::realloc;

ClassImp(TFPBlock);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFPBlock::TFPBlock(Long64_t* offset, Int_t* length, Int_t nb)
{
   Long64_t aux = 0;

   fNblock = nb;
   fPos = new Long64_t[nb];
   fRelOffset = new Long64_t[nb];
   fLen = new Int_t[nb];

   for (Int_t i=0; i < nb; i++){
      fPos[i] = offset[i];
      fLen[i] = length[i];
      fRelOffset[i] = aux;
      aux += length[i];
   }

   fCapacity = aux;
   fDataSize = aux;
   fBuffer = (char*) calloc(fCapacity, sizeof(char));
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TFPBlock::~TFPBlock()
{
   delete[] fPos;
   delete[] fLen;
   delete[] fRelOffset;
   free(fBuffer);
}


////////////////////////////////////////////////////////////////////////////////
/// Set pos value for index idx.

void TFPBlock::SetPos(Int_t idx, Long64_t value)
{
   fPos[idx] = value;
}


////////////////////////////////////////////////////////////////////////////////
/// Set block buffer.

void TFPBlock::SetBuffer(char* buf)
{
   if ( fBuffer ) {
     free(fBuffer);
   }
   fBuffer = buf;

}

////////////////////////////////////////////////////////////////////////////////
/// Reallocate the block's buffer based on the length
/// of the elements it will contain.

void TFPBlock::ReallocBlock(Long64_t* offset, Int_t* length, Int_t nb)
{
   Long64_t newSize = 0;

   fPos = (Long64_t*) TStorage::ReAlloc(fPos, nb * sizeof(Long64_t), fNblock * sizeof(Long64_t));
   fRelOffset = (Long64_t*) TStorage::ReAlloc(fRelOffset, nb * sizeof(Long64_t), fNblock * sizeof(Long64_t));
   fLen = TStorage::ReAllocInt(fLen, nb, fNblock);
   fNblock = nb;

   for(Int_t i=0; i < nb; i++){
      fPos[i] = offset[i];
      fLen[i] = length[i];
      fRelOffset[i] = newSize;
      newSize += fLen[i];
   }

   if (newSize > fCapacity) {
     fCapacity = newSize;
     fBuffer = (char*) realloc(fBuffer, fCapacity);
   }

   fDataSize = newSize;
}
