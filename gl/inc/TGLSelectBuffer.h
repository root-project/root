// @(#)root/gl:$Name$:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSelectBuffer_H
#define ROOT_TGLSelectBuffer_H

#include <Rtypes.h>

class TGLSceneInfo;
class TGLPhysicalShape;
class TGLObject;

#include <vector>

class TGLSelectRecord;

/**************************************************************************/
// TGLSelectBuffer
/**************************************************************************/

class TGLSelectBuffer
{
protected:
   Int_t   fBufSize;  // Size of buffer.
   UInt_t* fBuf;      // Actual buffer.

   Int_t   fNRecords; // Number of records as returned by glRenderMode.

   typedef std::pair<UInt_t, UInt_t*>  RawRecord_t;
   typedef std::vector<RawRecord_t>   vRawRecord_t;

   vRawRecord_t fSortedRecords;

   static Int_t fgMaxBufSize;

public:
   TGLSelectBuffer();
   virtual ~TGLSelectBuffer();

   Int_t   GetBufSize()  const { return fBufSize; }
   UInt_t* GetBuf()      const { return fBuf; }
   Int_t   GetNRecords() const { return fNRecords; }

   Bool_t CanGrow() { return fBufSize < fgMaxBufSize; }
   void   Grow();

   void ProcessResult(Int_t glResult);

   UInt_t* RawRecord(Int_t i) { return fSortedRecords[i].second; }

   void SelectRecord(TGLSelectRecord& rec, Int_t i);

   ClassDef(TGLSelectBuffer, 0) // OpenGL select buffer with depth sorting.
};

/**************************************************************************/
// TGLSelectRecord
/**************************************************************************/

class TGLSelectRecord
{
protected:
   // Primary data - coming from GL.
   Int_t    fN;
   UInt_t  *fItems;
   Float_t  fMinZ;
   Float_t  fMaxZ;

   // Secondary data (scene dependent) - use
   // TGLSceneBase::ResolveSelectRecord to fill.
   Bool_t            fTransparent;
   TGLSceneInfo     *fSceneInfo; // SceneInfo
   TGLPhysicalShape *fPhysShape; // PhysicalShape, if applicable
   TObject          *fObject;    // Master TObject, if applicable
   void             *fSpecific;  // Scene specific, if applicable

   void CopyItems(UInt_t* items);

public:
   TGLSelectRecord();
   TGLSelectRecord(UInt_t* data);
   TGLSelectRecord(const TGLSelectRecord& rec);
   virtual ~TGLSelectRecord();

   TGLSelectRecord & operator=(const TGLSelectRecord& rec);

   void Set(const TGLSelectRecord& rec);
   void Set(UInt_t* data);
   void SetRawOnly(UInt_t* data);
   void Reset();

   Int_t   GetN()           const { return fN; }
   UInt_t* GetItems()       const { return fItems; }
   UInt_t  GetItem(Int_t i) const { return fItems[i]; }
   Float_t GetMinZ()        const { return fMinZ; }
   Float_t GetMaxZ()        const { return fMaxZ; }

   Bool_t             GetTransparent() const { return fTransparent; }
   TGLSceneInfo     * GetSceneInfo()   const { return fSceneInfo; }
   TGLPhysicalShape * GetPhysShape()   const { return fPhysShape; }
   TObject          * GetObject()      const { return fObject; }
   void             * GetSpecific()    const { return fSpecific; }

   void SetTransparent(Bool_t t)               { fTransparent = t; }
   void SetSceneInfo  (TGLSceneInfo* si)       { fSceneInfo = si; }
   void SetPhysShape  (TGLPhysicalShape* pshp) { fPhysShape = pshp; }
   void SetObject     (TObject* obj)           { fObject = obj; }
   void SetSpecific   (void* spec)             { fSpecific = spec; }

   void Print();

   static Bool_t AreSameSelectionWise(const TGLSelectRecord& r1,
                                      const TGLSelectRecord& r2);

   ClassDef(TGLSelectRecord, 0) // One record in OpenGL selection.
};



#endif
