// @(#)root/io:$Id$
// Author: Elvin Sindrilaru   19/05/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFPBlock                                                             //
//                                                                      //
// This class represents the encapsulation of a block request.          //
// It contains the chunks to be prefetched and also serves as a         //
// container for the information read.                                  //
// These blocks are prefetch in a special reader thread by the          //
// TFilePrefetch class.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFPBlock.h"
#include "TStorage.h"
#include <cstdlib>


ClassImp(TFPBlock)

//__________________________________________________________________
TFPBlock::TFPBlock(Long64_t* offset, Int_t* length, Int_t nb)
{
   // Constructor.

   Int_t aux = 0;

   fNblock = nb;
   fPos = new Long64_t[nb];
   fLen = new Int_t[nb];

   for (Int_t i=0; i < nb; i++){
      fPos[i] = offset[i];
      fLen[i] = length[i];
      aux += length[i];
   }
   fFullSize = aux;
   fBuffer = new char[fFullSize];
}

//__________________________________________________________________
TFPBlock::~TFPBlock()
{
   // Destructor.

   delete[] fPos;
   delete[] fLen;
   delete[] fBuffer;
}

//__________________________________________________________________
Long64_t* TFPBlock::GetPos() const
{
   // Get pointer to the array of postions.

   return fPos;
}

//__________________________________________________________________
Int_t* TFPBlock::GetLen() const
{
   // Get pointer to the array of lengths.

   return fLen;
}

//__________________________________________________________________
Int_t TFPBlock::GetFullSize() const
{
   // Return size of the block.

   return fFullSize;
}

//__________________________________________________________________
Int_t TFPBlock::GetNoElem() const
{
   // Return number of elements in the block.

   return fNblock;
}

//__________________________________________________________________
Long64_t TFPBlock::GetPos(Int_t i) const
{
   // Get position of the element at index i.

   return fPos[i];
}

//__________________________________________________________________
Int_t TFPBlock::GetLen(Int_t i) const
{
  // Get length of the element at index i.

   return fLen[i];
}

//__________________________________________________________________
char* TFPBlock::GetBuffer() const
{
   // Get block buffer.

   return fBuffer;
}


//__________________________________________________________________
void TFPBlock::SetPos(Int_t idx, Long64_t value)
{
   // Set pos value for index idx.

   fPos[idx] = value;
}


//__________________________________________________________________
void TFPBlock::SetBuffer(char* buf)
{
   // Set block buffer.

   fBuffer = buf;
}

//__________________________________________________________________
void TFPBlock::ReallocBlock(Long64_t* offset, Int_t* length, Int_t nb)
{
   // Reallocate the block's buffer based on the length
   // of the elements it will contain.

   Int_t aux = 0;

   fPos = (Long64_t*) TStorage::ReAlloc(fPos, nb * sizeof(Long64_t), fNblock * sizeof(Long64_t));
   fLen = TStorage::ReAllocInt(fLen, nb, fNblock);
   fNblock = nb;

   for(Int_t i=0; i < nb; i++){

      fPos[i] = offset[i];
      fLen[i] = length[i];
      aux += fLen[i];
   }

   fBuffer = TStorage::ReAllocChar(fBuffer, aux, fFullSize);
   fFullSize = aux;
}
