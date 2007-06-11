// @(#)root/gl:$Name$:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLSelectBuffer.h"
#include <TObject.h>
#include <TMath.h>

#include <algorithm>

/**************************************************************************/
/**************************************************************************/
// TGLSelectBuffer
/**************************************************************************/
/**************************************************************************/

Int_t TGLSelectBuffer::fgMaxBufSize = 1 << 20; // 1MByte

//______________________________________________________________________________
TGLSelectBuffer::TGLSelectBuffer() :
   fBufSize  (1024),
   fBuf      (new UInt_t [fBufSize]),
   fNRecords (-1)
{
   // Constructor.
}

//______________________________________________________________________________
TGLSelectBuffer::~TGLSelectBuffer()
{
   // Destructor.

   delete [] fBuf;
}

//______________________________________________________________________________
void TGLSelectBuffer::Grow()
{
   // Increase size of the select buffer.

   delete [] fBuf;
   fBufSize = TMath::Min(2*fBufSize, fgMaxBufSize);
   fBuf = new UInt_t[fBufSize];
}

//______________________________________________________________________________
void TGLSelectBuffer::ProcessResult(Int_t glResult)
{
   // Process result of GL-selection: sort the hits by their minimum
   // z-coordinate.

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

//______________________________________________________________________________
void TGLSelectBuffer::SelectRecord(TGLSelectRecord& rec, Int_t i)
{
   // Return select record on (sorted) position i.

   rec.Set(fSortedRecords[i].second);
}


/**************************************************************************/
/**************************************************************************/
// TGLSelectRecord
/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________________
TGLSelectRecord::TGLSelectRecord() :
   fN     (0),
   fItems (0),
   fMinZ  (0),
   fMaxZ  (0),

   fTransparent (kFALSE),
   fSceneInfo   (0),
   fPhysShape   (0),
   fObject      (0),
   fSpecific    (0)
{
   // Default constructor.
}

//______________________________________________________________________________
TGLSelectRecord::TGLSelectRecord(UInt_t* data) :
   fN     (data[0]),
   fItems (0),
   fMinZ  ((Float_t)data[1] / 0x7fffffff),
   fMaxZ  ((Float_t)data[2] / 0x7fffffff),

   fTransparent (kFALSE),
   fSceneInfo   (0),
   fPhysShape   (0),
   fObject      (0),
   fSpecific    (0)
{
   // Constructor from raw GL-select record.

   CopyItems(&data[3]);
}

//______________________________________________________________________________
TGLSelectRecord::TGLSelectRecord(const TGLSelectRecord& rec) :
   fN     (rec.fN),
   fItems (0),
   fMinZ  (rec.fMinZ),
   fMaxZ  (rec.fMaxZ),

   fTransparent (rec.fTransparent),
   fSceneInfo   (rec.fSceneInfo),
   fPhysShape   (rec.fPhysShape),
   fObject      (rec.fObject),
   fSpecific    (rec.fSpecific)
{
   // Copy constructor.

   CopyItems(rec.fItems);
}

//______________________________________________________________________________
TGLSelectRecord::~TGLSelectRecord()
{
   // Destructor.

   delete [] fItems;
}

//______________________________________________________________________________
TGLSelectRecord& TGLSelectRecord::operator=(const TGLSelectRecord & rec)
{
   // Copy operator.

   Set(rec);
   return *this;
}

//______________________________________________________________________________
void TGLSelectRecord::CopyItems(UInt_t* items)
{
   // Copy data from names. fN must already be set.

   delete [] fItems;
   if (fN > 0) {
      fItems = new UInt_t[fN];
      memcpy(fItems, items, fN*sizeof(UInt_t));
   } else {
      fItems = 0;
   }
}

//______________________________________________________________________________
void TGLSelectRecord::Set(const TGLSelectRecord& rec)
{
   // Setup the record from another.

   if (&rec == this) return;

   fN      = rec.fN;
   fMinZ   = rec.fMinZ;
   fMaxZ   = rec.fMaxZ;
   CopyItems(rec.fItems);

   fTransparent = rec.fTransparent;
   fSceneInfo   = rec.fSceneInfo;
   fPhysShape   = rec.fPhysShape;
   fObject      = rec.fObject;
   fSpecific    = rec.fSpecific;
}
//______________________________________________________________________________
void TGLSelectRecord::Set(UInt_t* data)
{
   // Setup the record from raw buffer.

   fN     = data[0];
   fMinZ  = (Float_t)data[1] / 0x7fffffff;
   fMaxZ  = (Float_t)data[2] / 0x7fffffff;
   CopyItems(&data[3]);

   fTransparent = kFALSE;
   fSceneInfo   = 0;
   fPhysShape   = 0;
   fObject      = 0;
   fSpecific    = 0;
}

//______________________________________________________________________________
void TGLSelectRecord::SetRawOnly(UInt_t* data)
{
   // Setup the record from raw buffer.

   fN     = data[0];
   fMinZ  = (Float_t)data[1] / 0x7fffffff;
   fMaxZ  = (Float_t)data[2] / 0x7fffffff;
   CopyItems(&data[3]);
}

//______________________________________________________________________________
void TGLSelectRecord::Reset()
{
   // Reinitalize all data to null values.

   delete [] fItems;
   fN     = 0;
   fItems = 0;
   fMinZ  = 0;
   fMaxZ  = 0;

   fTransparent = kFALSE;
   fSceneInfo   = 0;
   fPhysShape   = 0;
   fObject      = 0;
   fSpecific    = 0;
}

//______________________________________________________________________________
void TGLSelectRecord::Print()
{
   // Print contents of the select record to stdout.

   printf("SelectRecord   N=%d, miZ=%.4f, maxZ=%.4f\n"
          "    sceneinfo=%p, pshp=%p, transp=%d,\n"
          "    tobj=%p (name='%s'), spec=%p\n",
          fN, fMinZ, fMaxZ,
          fSceneInfo,  fPhysShape,  fTransparent,
          fObject,   fObject ? fObject->GetName() : "",
          fSpecific);
}

//______________________________________________________________________________
Bool_t TGLSelectRecord::AreSameSelectionWise(const TGLSelectRecord& r1,
                                             const TGLSelectRecord& r2)
{
   // Check if the records imply the same selection result, that is,
   // their secondary members are all equal.

   return r1.fSceneInfo == r2.fSceneInfo && r1.fPhysShape == r2.fPhysShape &&
          r1.fObject    == r2.fObject    && r1.fSpecific  == r2.fSpecific;
}
