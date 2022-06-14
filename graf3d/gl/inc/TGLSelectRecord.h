// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSelectRecord
#define ROOT_TGLSelectRecord

#include <Rtypes.h>

class TObject;
class TGLSceneInfo;
class TGLPhysicalShape;
class TGLLogicalShape;
class TGLOverlayElement;

/**************************************************************************/
// TGLSelectRecordBase
/**************************************************************************/

class TGLSelectRecordBase
{
protected:
   // Primary data - coming from GL.
   Int_t    fN;
   UInt_t  *fItems;
   Float_t  fMinZ;
   Float_t  fMaxZ;

   // Current position (for name resolutin in hierachies of unknown depth).
   Int_t    fPos;

   void CopyItems(UInt_t* items);

public:
   TGLSelectRecordBase();
   TGLSelectRecordBase(UInt_t* data);
   TGLSelectRecordBase(const TGLSelectRecordBase& rec);
   virtual ~TGLSelectRecordBase();

   TGLSelectRecordBase& operator=(const TGLSelectRecordBase& rec);

   void SetRawOnly(UInt_t* data);

   virtual void Set(UInt_t* data);
   virtual void Reset();

   Int_t   GetN()           const { return fN; }
   UInt_t* GetItems()       const { return fItems; }
   UInt_t  GetItem(Int_t i) const { return fItems[i]; }
   Float_t GetMinZ()        const { return fMinZ; }
   Float_t GetMaxZ()        const { return fMaxZ; }

   UInt_t  GetCurrItem() const { return fPos < fN ? fItems[fPos] : 0; }
   Int_t   GetNLeft()    const { return fN - fPos; }
   void    NextPos()           { ++fPos; }
   void    PrevPos()           { --fPos; }
   void    ResetPos()          { fPos = 0; }

   ClassDef(TGLSelectRecordBase, 0) // Base class for GL selection records.
};


/**************************************************************************/
// TGLSelectRecord
/**************************************************************************/

class TGLSelectRecord : public TGLSelectRecordBase
{
public:
   enum ESecSelResult { kNone, kEnteringSelection, kLeavingSelection, kModifyingInternalSelection };

protected:
   // Secondary data (scene dependent) - use
   // TGLSceneBase::ResolveSelectRecord() to fill.
   Bool_t            fTransparent;
   TGLSceneInfo     *fSceneInfo; // SceneInfo
   TGLPhysicalShape *fPhysShape; // PhysicalShape, if applicable
   TGLLogicalShape  *fLogShape;  // LogicalShape, if applicable
   TObject          *fObject;    // Master TObject, if applicable
   void             *fSpecific;  // Scene specific, if applicable
   Bool_t            fMultiple;  // Mutliple selection requested (set by event-handler).
   Bool_t            fHighlight; // Requested for highlight (set by event-handler).

   ESecSelResult     fSecSelRes; // Result of ProcessSelection;

public:
   TGLSelectRecord();
   TGLSelectRecord(UInt_t* data);
   TGLSelectRecord(const TGLSelectRecord& rec);
   virtual ~TGLSelectRecord();

   TGLSelectRecord& operator=(const TGLSelectRecord& rec);

   virtual void Set(UInt_t* data);
   virtual void Reset();

   Bool_t             GetTransparent() const { return fTransparent; }
   TGLSceneInfo     * GetSceneInfo()   const { return fSceneInfo; }
   TGLPhysicalShape * GetPhysShape()   const { return fPhysShape; }
   TGLLogicalShape  * GetLogShape()    const { return fLogShape; }
   TObject          * GetObject()      const { return fObject; }
   void             * GetSpecific()    const { return fSpecific; }
   Bool_t             GetMultiple()    const { return fMultiple; }
   Bool_t             GetHighlight()   const { return fHighlight; }

   ESecSelResult      GetSecSelResult() const { return fSecSelRes; }

   void SetTransparent(Bool_t t)               { fTransparent = t; }
   void SetSceneInfo  (TGLSceneInfo* si)       { fSceneInfo = si; }
   void SetPhysShape  (TGLPhysicalShape* pshp) { fPhysShape = pshp; }
   void SetLogShape   (TGLLogicalShape* lshp)  { fLogShape = lshp; }
   void SetObject     (TObject* obj)           { fObject = obj; }
   void SetSpecific   (void* spec)             { fSpecific = spec; }
   void SetMultiple   (Bool_t multi)           { fMultiple = multi; }
   void SetHighlight  (Bool_t hlt)             { fHighlight = hlt; }

   void SetSecSelResult(ESecSelResult r)       { fSecSelRes = r; }

   void Print();

   static Bool_t AreSameSelectionWise(const TGLSelectRecord& r1,
                                      const TGLSelectRecord& r2);

   ClassDef(TGLSelectRecord, 0) // Standard GL selection record.
};


/**************************************************************************/
// TGLOvlSelectRecord
/**************************************************************************/

class TGLOvlSelectRecord : public TGLSelectRecordBase
{
protected:
   // Secondary data (overlay dependent).
   TGLOverlayElement* fOvlElement;

public:
   TGLOvlSelectRecord();
   TGLOvlSelectRecord(UInt_t* data);
   TGLOvlSelectRecord(const TGLOvlSelectRecord& rec);
   virtual ~TGLOvlSelectRecord();

   TGLOvlSelectRecord& operator=(const TGLOvlSelectRecord& rec);

   virtual void Set(UInt_t* data);
   virtual void Reset();

   TGLOverlayElement* GetOvlElement() const { return fOvlElement; }
   void SetOvlElement(TGLOverlayElement* e) { fOvlElement = e; }

   ClassDef(TGLOvlSelectRecord, 0) // Standard GL overlay-selection record.
};

#endif
