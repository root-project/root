// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLSelectBuffer.h"
#include "TGLSelectRecord.h"
#include <TMath.h>

#include <algorithm>

/** \class TGLSelectBuffer
\ingroup opengl
Encapsulates OpenGL select buffer.
Provides sorting of the results based on z-coordinate of the
selection hit and can fill the TGLSelectRecordBase records.
*/

Int_t TGLSelectBuffer::fgMaxBufSize = 1 << 20; // 1MByte

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGLSelectBuffer::TGLSelectBuffer() :
   fBufSize  (1024),
   fBuf      (new UInt_t [fBufSize]),
   fNRecords (-1)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLSelectBuffer::~TGLSelectBuffer()
{
   delete [] fBuf;
}

////////////////////////////////////////////////////////////////////////////////
///static: return true if current buffer is smaller than the max buffer size

Bool_t TGLSelectBuffer::CanGrow()
{
   return 2 * fBufSize < fgMaxBufSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Increase size of the select buffer.

void TGLSelectBuffer::Grow()
{
   delete [] fBuf;
   fBufSize = TMath::Min(2*fBufSize, fgMaxBufSize);
   fBuf = new UInt_t[fBufSize];
}

////////////////////////////////////////////////////////////////////////////////
/// Process result of GL-selection: sort the hits by their minimum
/// z-coordinate.

void TGLSelectBuffer::ProcessResult(Int_t glResult)
{
   // The '-1' case should be handled on the caller side.
   // Here we just assume no hits were recorded.

   if (glResult < 0)
      glResult = 0;

   fNRecords = glResult;
   fSortedRecords.resize(fNRecords);

   if (fNRecords > 0)
   {
      Int_t  i;
      UInt_t* buf = fBuf;
      for (i = 0; i < fNRecords; ++i)
      {
         fSortedRecords[i].first  = buf[1]; // minimum depth
         fSortedRecords[i].second = buf;    // record address
         buf += 3 + buf[0];
      }
      std::sort(fSortedRecords.begin(), fSortedRecords.end());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill select record rec with data on (sorted) position i.
/// Returns depth of name-stack for this record.

Int_t TGLSelectBuffer::SelectRecord(TGLSelectRecordBase& rec, Int_t i)
{
   rec.Set(fSortedRecords[i].second);
   return rec.GetN();
}
