// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLSelectRecord.h"
#include <TObject.h>

#include <string.h>

//==============================================================================
// TGLSelectRecordBase
//==============================================================================

//______________________________________________________________________
//
// Base class for select records.
// Supports initialization from a raw GL record (UInt_t*) and
// copies the name-data into internal array.
//

ClassImp(TGLSelectRecordBase);

//______________________________________________________________________________
TGLSelectRecordBase::TGLSelectRecordBase() :
   fN     (0),
   fItems (0),
   fMinZ  (0),
   fMaxZ  (0),
   fPos   (0)
{
   // Default constructor.
}

//______________________________________________________________________________
TGLSelectRecordBase::TGLSelectRecordBase(UInt_t* data) :
   fN     (data[0]),
   fItems (0),
   fMinZ  ((Float_t)data[1] / 0x7fffffff),
   fMaxZ  ((Float_t)data[2] / 0x7fffffff),
   fPos   (0)
{
   // Constructor from raw GL-select record.

   CopyItems(&data[3]);
}

//______________________________________________________________________________
TGLSelectRecordBase::TGLSelectRecordBase(const TGLSelectRecordBase& rec) :
   fN     (rec.fN),
   fItems (0),
   fMinZ  (rec.fMinZ),
   fMaxZ  (rec.fMaxZ),
   fPos   (rec.fPos)
{
   // Copy constructor.

   CopyItems(rec.fItems);
}

//______________________________________________________________________________
TGLSelectRecordBase::~TGLSelectRecordBase()
{
   // Destructor.

   delete [] fItems;
}

//______________________________________________________________________________
TGLSelectRecordBase& TGLSelectRecordBase::operator=(const TGLSelectRecordBase& rec)
{
   // Copy operator.

   if (this != &rec)
   {
      fN      = rec.fN;
      fMinZ   = rec.fMinZ;
      fMaxZ   = rec.fMaxZ;
      fPos    = rec.fPos;
      CopyItems(rec.fItems);
   }
   return *this;
}

//______________________________________________________________________________
void TGLSelectRecordBase::CopyItems(UInt_t* items)
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
void TGLSelectRecordBase::SetRawOnly(UInt_t* data)
{
   // Setup the record from raw buffer.

   fN     = data[0];
   fMinZ  = (Float_t)data[1] / 0x7fffffff;
   fMaxZ  = (Float_t)data[2] / 0x7fffffff;
   CopyItems(&data[3]);
}

//______________________________________________________________________________
void TGLSelectRecordBase::Set(UInt_t* data)
{
   // Setup the record from raw buffer.

   fN     = data[0];
   fMinZ  = (Float_t)data[1] / 0x7fffffff;
   fMaxZ  = (Float_t)data[2] / 0x7fffffff;
   fPos   = 0;
   CopyItems(&data[3]);
}

//______________________________________________________________________________
void TGLSelectRecordBase::Reset()
{
   // Reinitalize all data to null values.

   delete [] fItems;
   fN     = 0;
   fItems = 0;
   fMinZ  = 0;
   fMaxZ  = 0;
   fPos   = 0;
}


//==============================================================================
// TGLSelectRecord
//==============================================================================

//______________________________________________________________________
//
// Standard selection record including information about containing
// scene and details ob out selected object (TGLPhysicalShape*,
// TObject* or simply a void* for foreign scenes).
//

ClassImp(TGLSelectRecord);

//______________________________________________________________________________
TGLSelectRecord::TGLSelectRecord() :
   TGLSelectRecordBase(),
   fTransparent (kFALSE),
   fSceneInfo   (0),
   fPhysShape   (0),
   fLogShape    (0),
   fObject      (0),
   fSpecific    (0),
   fMultiple    (kFALSE),
   fHighlight   (kFALSE),
   fSecSelRes   (kNone)
{
   // Default constructor.
}

//______________________________________________________________________________
TGLSelectRecord::TGLSelectRecord(UInt_t* data) :
   TGLSelectRecordBase(data),
   fTransparent (kFALSE),
   fSceneInfo   (0),
   fPhysShape   (0),
   fLogShape    (0),
   fObject      (0),
   fSpecific    (0),
   fMultiple    (kFALSE),
   fHighlight   (kFALSE),
   fSecSelRes   (kNone)
{
   // Constructor from raw GL-select record.
}

//______________________________________________________________________________
TGLSelectRecord::TGLSelectRecord(const TGLSelectRecord& rec) :
   TGLSelectRecordBase(rec),
   fTransparent (rec.fTransparent),
   fSceneInfo   (rec.fSceneInfo),
   fPhysShape   (rec.fPhysShape),
   fLogShape    (rec.fLogShape),
   fObject      (rec.fObject),
   fSpecific    (rec.fSpecific),
   fMultiple    (rec.fMultiple),
   fHighlight   (rec.fHighlight),
   fSecSelRes   (kNone)
{
   // Copy constructor.
}

//______________________________________________________________________________
TGLSelectRecord::~TGLSelectRecord()
{
   // Destructor.
}

//______________________________________________________________________________
TGLSelectRecord& TGLSelectRecord::operator=(const TGLSelectRecord& rec)
{
   // Copy operator.

   if (this != &rec)
   {
      TGLSelectRecordBase::operator=(rec);
      fTransparent = rec.fTransparent;
      fSceneInfo   = rec.fSceneInfo;
      fPhysShape   = rec.fPhysShape;
      fLogShape    = rec.fLogShape;
      fObject      = rec.fObject;
      fSpecific    = rec.fSpecific;
      fMultiple    = rec.fMultiple;
      fHighlight   = rec.fHighlight;
      fSecSelRes   = rec.fSecSelRes;
   }
   return *this;
}

//______________________________________________________________________________
void TGLSelectRecord::Set(UInt_t* data)
{
   // Setup the record from raw buffer.
   // Non-core members are reset.

   TGLSelectRecordBase::Set(data);
   fTransparent = kFALSE;
   fSceneInfo   = 0;
   fPhysShape   = 0;
   fLogShape    = 0;
   fObject      = 0;
   fSpecific    = 0;
   fMultiple    = kFALSE;
   fHighlight   = kFALSE;
   fSecSelRes   = kNone;
}

//______________________________________________________________________________
void TGLSelectRecord::Reset()
{
   // Reinitalize all data to null values.

   TGLSelectRecordBase::Reset();
   fTransparent = kFALSE;
   fSceneInfo   = 0;
   fPhysShape   = 0;
   fLogShape    = 0;
   fObject      = 0;
   fSpecific    = 0;
   fMultiple    = kFALSE;
   fHighlight   = kFALSE;
   fSecSelRes   = kNone;
}

//______________________________________________________________________________
void TGLSelectRecord::Print()
{
   // Print contents of the select record to stdout.

   printf("SelectRecord   N=%d, miZ=%.4f, maxZ=%.4f\n"
          "    sceneinfo=%p, pshp=%p, transp=%d, mult=%d, hilite=%d\n"
          "    tobj=%p (name='%s'), spec=%p\n",
          fN, fMinZ, fMaxZ,
          fSceneInfo,  fPhysShape,  fTransparent, fMultiple, fHighlight,
          fObject, fObject ? fObject->GetName() : "",
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


//==============================================================================
// TGLOvlSelectRecord
//==============================================================================

//______________________________________________________________________
//
// Selection record for overlay objects.
//

ClassImp(TGLOvlSelectRecord);

//______________________________________________________________________________
TGLOvlSelectRecord::TGLOvlSelectRecord() :
   TGLSelectRecordBase(),
   fOvlElement (0)
{
   // Default constructor.
}

//______________________________________________________________________________
TGLOvlSelectRecord::TGLOvlSelectRecord(UInt_t* data) :
   TGLSelectRecordBase(data),
   fOvlElement (0)
{
   // Constructor from raw GL-select record.
}

//______________________________________________________________________________
TGLOvlSelectRecord::TGLOvlSelectRecord(const TGLOvlSelectRecord& rec) :
   TGLSelectRecordBase(rec),
   fOvlElement (rec.fOvlElement)
{
   // Copy constructor.
}

//______________________________________________________________________________
TGLOvlSelectRecord::~TGLOvlSelectRecord()
{
   // Destructor.
}

//______________________________________________________________________________
TGLOvlSelectRecord& TGLOvlSelectRecord::operator=(const TGLOvlSelectRecord& rec)
{
   // Copy operator.

   if (this != &rec)
   {
      TGLSelectRecordBase::operator=(rec);
      fOvlElement = rec.fOvlElement;
   }
   return *this;
}

//______________________________________________________________________________
void TGLOvlSelectRecord::Set(UInt_t* data)
{
   // Setup the record from raw buffer.
   // Non-core members are reset.

   TGLSelectRecordBase::Set(data);
   fOvlElement = 0;
}

//______________________________________________________________________________
void TGLOvlSelectRecord::Reset()
{
   // Reinitalize all data to null values.

   TGLSelectRecordBase::Reset();
   fOvlElement = 0;
}

