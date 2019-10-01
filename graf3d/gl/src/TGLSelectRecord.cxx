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

/** \class TGLSelectRecordBase
\ingroup opengl TGLSelectRecordBase
Base class for select records.
Supports initialization from a raw GL record (UInt_t*) and
copies the name-data into internal array.
*/

ClassImp(TGLSelectRecordBase);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TGLSelectRecordBase::TGLSelectRecordBase() :
   fN     (0),
   fItems (0),
   fMinZ  (0),
   fMaxZ  (0),
   fPos   (0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from raw GL-select record.

TGLSelectRecordBase::TGLSelectRecordBase(UInt_t* data) :
   fN     (data[0]),
   fItems (0),
   fMinZ  ((Float_t)data[1] / (Float_t)0x7fffffff),
   fMaxZ  ((Float_t)data[2] / (Float_t)0x7fffffff),
   fPos   (0)
{
   CopyItems(&data[3]);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TGLSelectRecordBase::TGLSelectRecordBase(const TGLSelectRecordBase& rec) :
   fN     (rec.fN),
   fItems (0),
   fMinZ  (rec.fMinZ),
   fMaxZ  (rec.fMaxZ),
   fPos   (rec.fPos)
{
   CopyItems(rec.fItems);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLSelectRecordBase::~TGLSelectRecordBase()
{
   delete [] fItems;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy operator.

TGLSelectRecordBase& TGLSelectRecordBase::operator=(const TGLSelectRecordBase& rec)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Copy data from names. fN must already be set.

void TGLSelectRecordBase::CopyItems(UInt_t* items)
{
   delete [] fItems;
   if (fN > 0) {
      fItems = new UInt_t[fN];
      memcpy(fItems, items, fN*sizeof(UInt_t));
   } else {
      fItems = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the record from raw buffer.

void TGLSelectRecordBase::SetRawOnly(UInt_t* data)
{
   fN     = data[0];
   fMinZ  = (Float_t)data[1] / (Float_t)0x7fffffff;
   fMaxZ  = (Float_t)data[2] / (Float_t)0x7fffffff;
   CopyItems(&data[3]);
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the record from raw buffer.

void TGLSelectRecordBase::Set(UInt_t* data)
{
   fN     = data[0];
   fMinZ  = (Float_t)data[1] / (Float_t)0x7fffffff;
   fMaxZ  = (Float_t)data[2] / (Float_t)0x7fffffff;
   fPos   = 0;
   CopyItems(&data[3]);
}

////////////////////////////////////////////////////////////////////////////////
/// Reinitialise all data to null values.

void TGLSelectRecordBase::Reset()
{
   delete [] fItems;
   fN     = 0;
   fItems = 0;
   fMinZ  = 0;
   fMaxZ  = 0;
   fPos   = 0;
}


/** \class TGLSelectRecord
\ingroup opengl
Standard selection record including information about containing
scene and details ob out selected object (TGLPhysicalShape*,
TObject* or simply a void* for foreign scenes).
*/

ClassImp(TGLSelectRecord);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

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
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from raw GL-select record.

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
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

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
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLSelectRecord::~TGLSelectRecord()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy operator.

TGLSelectRecord& TGLSelectRecord::operator=(const TGLSelectRecord& rec)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Setup the record from raw buffer.
/// Non-core members are reset.

void TGLSelectRecord::Set(UInt_t* data)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Reinitialise all data to null values.

void TGLSelectRecord::Reset()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Print contents of the select record to stdout.

void TGLSelectRecord::Print()
{
   printf("SelectRecord   N=%d, miZ=%.4f, maxZ=%.4f\n"
          "    sceneinfo=%p, pshp=%p, transp=%d, mult=%d, hilite=%d\n"
          "    tobj=%p (name='%s'), spec=%p\n",
          fN, fMinZ, fMaxZ,
          fSceneInfo,  fPhysShape,  fTransparent, fMultiple, fHighlight,
          fObject, fObject ? fObject->GetName() : "",
          fSpecific);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the records imply the same selection result, that is,
/// their secondary members are all equal.

Bool_t TGLSelectRecord::AreSameSelectionWise(const TGLSelectRecord& r1,
                                             const TGLSelectRecord& r2)
{
   return r1.fSceneInfo == r2.fSceneInfo && r1.fPhysShape == r2.fPhysShape &&
          r1.fObject    == r2.fObject    && r1.fSpecific  == r2.fSpecific;
}


/** \class TGLOvlSelectRecord
\ingroup opengl
Selection record for overlay objects.
*/

ClassImp(TGLOvlSelectRecord);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TGLOvlSelectRecord::TGLOvlSelectRecord() :
   TGLSelectRecordBase(),
   fOvlElement (0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from raw GL-select record.

TGLOvlSelectRecord::TGLOvlSelectRecord(UInt_t* data) :
   TGLSelectRecordBase(data),
   fOvlElement (0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TGLOvlSelectRecord::TGLOvlSelectRecord(const TGLOvlSelectRecord& rec) :
   TGLSelectRecordBase(rec),
   fOvlElement (rec.fOvlElement)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLOvlSelectRecord::~TGLOvlSelectRecord()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy operator.

TGLOvlSelectRecord& TGLOvlSelectRecord::operator=(const TGLOvlSelectRecord& rec)
{
   if (this != &rec)
   {
      TGLSelectRecordBase::operator=(rec);
      fOvlElement = rec.fOvlElement;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Setup the record from raw buffer.
/// Non-core members are reset.

void TGLOvlSelectRecord::Set(UInt_t* data)
{
   TGLSelectRecordBase::Set(data);
   fOvlElement = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Reinitialise all data to null values.

void TGLOvlSelectRecord::Reset()
{
   TGLSelectRecordBase::Reset();
   fOvlElement = 0;
}

