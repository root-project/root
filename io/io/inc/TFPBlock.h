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
   Int_t     fFullSize;     // total size of segments that make up the block
   Int_t    *fLen;          // array of lengths of each segment
   Long64_t *fPos;          // array of positions of each segment

public:
   TFPBlock(Long64_t*, Int_t*, Int_t);
   virtual ~TFPBlock();

   Long64_t  GetPos(Int_t) const;
   Int_t     GetLen(Int_t) const;

   Long64_t *GetPos() const;
   Int_t    *GetLen() const;
   Int_t     GetFullSize() const;
   Int_t     GetNoElem() const;
   char     *GetBuffer() const;

   void SetBuffer(char*);
   void ReallocBlock(Long64_t*, Int_t*, Int_t);

   ClassDef(TFPBlock, 0);  // File prefetch block
};

#endif
