// @(#)root/io:$Id$
// Author: Elvin Sindrilaru   19/05/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFPBlock
#define ROOT_TFPBlock

#include "TObject.h"

/**
\class TFPBlock
\ingroup IO
*/

class TFPBlock : public TObject{

private:
   char     *fBuffer;       ///< Content of the block
   Int_t     fNblock;       ///< Number of segment in the block
   Long64_t  fDataSize;     ///< Total size of useful data in the block
   Long64_t  fCapacity;     ///< Capacity of the buffer
   Int_t    *fLen;          ///< Array of lengths of each segment
   Long64_t *fPos;          ///< Array of positions of each segment
   Long64_t *fRelOffset;    ///< Relative offset of piece in the buffer

   TFPBlock(const TFPBlock&) = delete;            // Not implemented.
   TFPBlock &operator=(const TFPBlock&) = delete; // Not implemented.

public:

   TFPBlock(Long64_t*, Int_t*, Int_t);
   virtual ~TFPBlock();

   Long64_t  GetPos(Int_t) const;
   Int_t     GetLen(Int_t) const;

   Long64_t *GetPos() const;
   Int_t    *GetLen() const;
   Long64_t  GetDataSize() const;
   Long64_t  GetCapacity() const;
   Int_t     GetNoElem() const;
   char     *GetBuffer() const;
   char     *GetPtrToPiece(Int_t index) const;

   void      SetBuffer(char*);
   void      SetPos(Int_t, Long64_t);
   void      ReallocBlock(Long64_t*, Int_t*, Int_t);

   ClassDefOverride(TFPBlock, 0);  // File prefetch block
};

/// Get pointer to the array of postions.
inline Long64_t* TFPBlock::GetPos() const
{
   return fPos;
}

/// Get pointer to the array of lengths.
inline Int_t* TFPBlock::GetLen() const
{
   return fLen;
}

/// Return size of the data in the block.
inline Long64_t TFPBlock::GetDataSize() const
{
   return fDataSize;
}

/// Return capacity of the block.
inline Long64_t TFPBlock::GetCapacity() const
{
   return fCapacity;
}

/// Return number of elements in the block.
inline Int_t TFPBlock::GetNoElem() const
{
   return fNblock;
}

/// Get position of the element at index i.
inline Long64_t TFPBlock::GetPos(Int_t i) const
{
   return fPos[i];
}

/// Get length of the element at index i.
inline Int_t TFPBlock::GetLen(Int_t i) const
{
   return fLen[i];
}

/// Get block buffer.
inline char* TFPBlock::GetBuffer() const
{
   return fBuffer;
}

/// Get block buffer.
inline char* TFPBlock::GetPtrToPiece(Int_t index) const
{
  return (fBuffer + fRelOffset[index]);
}

#endif
