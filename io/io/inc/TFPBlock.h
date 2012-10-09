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

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TFPBlock : public TObject{

private:
   char     *fBuffer;       // content of the block
   Int_t     fNblock;       // number of segment in the block
   Long64_t  fDataSize;     // total size of useful data in the block
   Long64_t  fCapacity;     // capacity of the buffer
   Int_t    *fLen;          // array of lengths of each segment
   Long64_t *fPos;          // array of positions of each segment
   Long64_t *fRelOffset;    // relative offset of piece in the buffer

   TFPBlock(const TFPBlock&);            // Not implemented.
   TFPBlock &operator=(const TFPBlock&); // Not implemented.

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

   void SetBuffer(char*);
   void SetPos(Int_t, Long64_t);
   void ReallocBlock(Long64_t*, Int_t*, Int_t);

   ClassDef(TFPBlock, 0);  // File prefetch block
};

//__________________________________________________________________
inline Long64_t* TFPBlock::GetPos() const
{
   // Get pointer to the array of postions.

   return fPos;
}

//__________________________________________________________________
inline Int_t* TFPBlock::GetLen() const
{
   // Get pointer to the array of lengths.

   return fLen;
}

//__________________________________________________________________
inline Long64_t TFPBlock::GetDataSize() const
{
   // Return size of the data in the block.

   return fDataSize;
}

//__________________________________________________________________
inline Long64_t TFPBlock::GetCapacity() const
{
   // Return capacity of the block.

   return fCapacity;
}

//__________________________________________________________________
inline Int_t TFPBlock::GetNoElem() const
{
   // Return number of elements in the block.

   return fNblock;
}

//__________________________________________________________________
inline Long64_t TFPBlock::GetPos(Int_t i) const
{
   // Get position of the element at index i.

   return fPos[i];
}

//__________________________________________________________________
inline Int_t TFPBlock::GetLen(Int_t i) const
{
  // Get length of the element at index i.

   return fLen[i];
}

//__________________________________________________________________
inline char* TFPBlock::GetBuffer() const
{
   // Get block buffer.

   return fBuffer;
}

//__________________________________________________________________
inline char* TFPBlock::GetPtrToPiece(Int_t index) const
{
   // Get block buffer.
 
  return (fBuffer + fRelOffset[index]);
}

#endif
