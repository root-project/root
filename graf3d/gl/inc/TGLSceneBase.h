// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSceneBase_H
#define ROOT_TGLSceneBase_H

#include "TGLLockable.h"
#include "TGLBoundingBox.h"

#include <TString.h>

#include <list>

class TGLViewerBase;
class TGLSceneInfo;
class TGLClip;
class TGLRnrCtx;
class TGLLogicalShape;
class TGLSelectRecord;

// Avoid TObject inheritance due to clash with TObject::Draw as well
// as possible inheritance of TGLPadScene from VierualViewer3D.

class TGLSceneBase : public TGLLockable // : public TObject / TNamed
{
private:
   TGLSceneBase(const TGLSceneBase&);            // Not implemented
   TGLSceneBase& operator=(const TGLSceneBase&); // Not implemented

   static UInt_t fgSceneIDSrc;

protected:
   UInt_t             fSceneID;     // Unique scene id.
   TString            fName;        // Object identifier.
   TString            fTitle;       // Object title.

   UInt_t             fTimeStamp;   // Counter increased on every update.
   UInt_t             fMinorStamp;  // Counter increased on minimal update.
   Short_t            fLOD;         // Scene-lod.
   Short_t            fStyle;       // Scene-style.
   Float_t            fWFLineW;     // Scene wire-frame line-width.
   Float_t            fOLLineW;     // Scene outline line-width.
   TGLClip          * fClip;        // Scene clipping-plane.
   Bool_t             fSelectable;  // Objects in the scene are selectable.

   // BoundingBox
   mutable TGLBoundingBox fBoundingBox;      // bounding box for scene (axis aligned) - lazy update - use BoundingBox() to access
   mutable Bool_t         fBoundingBoxValid; // bounding box valid?

   Bool_t  fDoFrustumCheck;  // Perform global frustum-check in UpdateSceneInfo()
   Bool_t  fDoClipCheck;     // Perform global clip-plane-check in UpdateSceneInfo()

   // Interface to other components
   typedef std::list<TGLViewerBase*>           ViewerList_t;
   typedef std::list<TGLViewerBase*>::iterator ViewerList_i;

   ViewerList_t       fViewers;
   Bool_t             fAutoDestruct;

   // Possible future extensions
   // TGLMatrix fGlobalTrans;

public:
   TGLSceneBase();
   virtual ~TGLSceneBase();

   void AddViewer(TGLViewerBase* viewer);
   void RemoveViewer(TGLViewerBase* viewer);
   void TagViewersChanged();

   virtual const char* LockIdStr() const;

   virtual const char  *GetName()  const { return fName; }
   virtual const char  *GetTitle() const { return fTitle; }
   virtual void  SetName (const char *name)  { fName = name; }
   virtual void  SetTitle(const char *title) { fTitle = title; }
   virtual void  SetNameTitle(const char *name, const char *title) { SetName(name); SetTitle(title); }

   virtual TGLSceneInfo* CreateSceneInfo(TGLViewerBase* view);
   virtual void          RebuildSceneInfo(TGLRnrCtx& ctx);
   virtual void          UpdateSceneInfo(TGLRnrCtx& ctx);
   virtual void          LodifySceneInfo(TGLRnrCtx& ctx);

   // Rendering
   virtual void PreDraw   (TGLRnrCtx& rnrCtx);
   virtual void PreRender (TGLRnrCtx& rnrCtx);
   virtual void Render    (TGLRnrCtx& rnrCtx);
   virtual void RenderOpaque   (TGLRnrCtx& rnrCtx);
   virtual void RenderTransp   (TGLRnrCtx& rnrCtx);
   virtual void RenderSelOpaque(TGLRnrCtx& rnrCtx);
   virtual void RenderSelTransp(TGLRnrCtx& rnrCtx);
   virtual void RenderSelOpaqueForHighlight(TGLRnrCtx& rnrCtx);
   virtual void RenderSelTranspForHighlight(TGLRnrCtx& rnrCtx);
   virtual void PostRender(TGLRnrCtx& rnrCtx);
   virtual void PostDraw  (TGLRnrCtx& rnrCtx);

   virtual TGLLogicalShape* FindLogical(TObject*) const { return nullptr; }

   // Selection interface
   virtual Bool_t ResolveSelectRecord(TGLSelectRecord& rec, Int_t curIdx);


   // Getters & setters

   UInt_t GetTimeStamp() const { return fTimeStamp; }
   void   IncTimeStamp()       { ++fTimeStamp; fMinorStamp = 1; }

   UInt_t GetMinorStamp() const { return fMinorStamp; }
   void   IncMinorStamp()       { ++fMinorStamp;      }

   Short_t  LOD()          const { return fLOD; }
   void     SetLOD(Short_t lod)  { fLOD = lod;  }

   Short_t  Style()        const { return fStyle; }
   void     SetStyle(Short_t st) { fStyle = st;   }

   TGLClip* Clip()         const { return fClip; }
   void     SetClip(TGLClip *p)  { fClip = p;    }

   Bool_t   GetSelectable()   const { return fSelectable; }
   void     SetSelectable(Bool_t a) { fSelectable = a;    }

   Bool_t   GetAutoDestruct()   const { return fAutoDestruct; }
   void     SetAutoDestruct(Bool_t a) { fAutoDestruct = a;    }

   // BoundingBox

   virtual void CalcBoundingBox() const = 0;
   void         InvalidateBoundingBox() { fBoundingBoxValid = kFALSE; }
   const TGLBoundingBox& BoundingBox() const
   { if (!fBoundingBoxValid) CalcBoundingBox(); return fBoundingBox; }


   ClassDef(TGLSceneBase, 0) // Base-class for OpenGL scenes.
}; // endclass TGLSceneBase


#endif
