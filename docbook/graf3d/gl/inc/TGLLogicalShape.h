// @(#)root/gl:$Id$
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLLogicalShape
#define ROOT_TGLLogicalShape

#ifndef ROOT_TGLBoundingBox
#include "TGLBoundingBox.h"
#endif

class TBuffer3D;
class TObject;
class TContextMenu;

class TGLPhysicalShape;
class TGLRnrCtx;
class TGLSelectRecord;
class TGLViewer;
class TGLSceneBase;
class TGLScene;


class TGLLogicalShape
{
   friend class TGLScene;

private:
   TGLLogicalShape(const TGLLogicalShape&);            // Not implemented.
   TGLLogicalShape& operator=(const TGLLogicalShape&); // Not implemented.

public:
   enum ELODAxes  { kLODAxesNone = 0,  // LOD will be set to high or pixel.
                    kLODAxesX    = 1 << 0,
                    kLODAxesY    = 1 << 1,
                    kLODAxesZ    = 1 << 2,
                    kLODAxesAll  = kLODAxesX | kLODAxesY | kLODAxesZ
                  };

protected:
   mutable UInt_t             fRef;           //! physical instance ref counting
   mutable TGLPhysicalShape  *fFirstPhysical; //! first replica

   TObject           *fExternalObj; //! Also plays the role of ID.
   TGLBoundingBox     fBoundingBox; //! Shape's bounding box.
   mutable TGLScene  *fScene;       //! scene where object is stored (can be zero!)
   mutable UInt_t     fDLBase;      //! display-list id base
   mutable Int_t      fDLSize;      //! display-list size for different LODs
   mutable UShort_t   fDLValid;     //! display-list validity bit-field
   mutable Bool_t     fDLCache;     //! use display list caching
   mutable Bool_t     fRefStrong;   //! Strong ref (delete on 0 ref); not in scene
   mutable Bool_t     fOwnExtObj;   //! External object is a fake

   void PurgeDLRange(UInt_t base, Int_t size) const;

public:
   TGLLogicalShape();
   TGLLogicalShape(TObject* obj);
   TGLLogicalShape(const TBuffer3D & buffer);
   virtual ~TGLLogicalShape();

   // Physical shape reference-counting, replica management
   UInt_t Ref() const { return fRef; }
   void   AddRef(TGLPhysicalShape* phys) const;
   void   SubRef(TGLPhysicalShape* phys) const;
   void   StrongRef(Bool_t strong) const { fRefStrong = strong; }
   void   DestroyPhysicals();
   UInt_t UnrefFirstPhysical();

   const TGLPhysicalShape* GetFirstPhysical() const { return fFirstPhysical; }

   TObject*  ID()          const { return fExternalObj; }
   TObject*  GetExternal() const { return fExternalObj; }
   TGLScene* GetScene()    const { return fScene; }

   const TGLBoundingBox& BoundingBox() const { return fBoundingBox; }
   virtual void          UpdateBoundingBox() {}
   void                  UpdateBoundingBoxesOfPhysicals();

   // Display List Caching
           Bool_t SetDLCache(Bool_t cached);
   virtual Bool_t ShouldDLCache(const TGLRnrCtx & rnrCtx) const;
   virtual UInt_t DLOffset(Short_t /*lod*/) const { return 0; }
   virtual void   DLCacheClear();
   virtual void   DLCacheDrop();
   virtual void   DLCachePurge();

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesNone; }
   virtual Short_t  QuantizeShapeLOD(Short_t shapeLOD, Short_t combiLOD) const;
   virtual void     Draw(TGLRnrCtx& rnrCtx) const;
   virtual void     DirectDraw(TGLRnrCtx& rnrCtx) const = 0; // Actual draw method (non DL cached)

   virtual void     DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t lvl=-1) const;

   virtual Bool_t IgnoreSizeForOfInterest() const { return kFALSE; }

   // Override in sub-classes that do direct object rendering (e.g. TGLObject).
   virtual Bool_t KeepDuringSmartRefresh() const { return kFALSE; }
   // Override in sub-classes that support secondary selection (e.g. TPointSet3DGL).
   virtual Bool_t SupportsSecondarySelect() const { return kFALSE; }
   virtual Bool_t AlwaysSecondarySelect()   const { return kFALSE; }
   virtual void   ProcessSelection(TGLRnrCtx& rnrCtx, TGLSelectRecord& rec);

   void InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const;

   ClassDef(TGLLogicalShape,0) // a logical (non-placed, local frame) drawable object
};


#endif // ROOT_TGLLogicalShape
