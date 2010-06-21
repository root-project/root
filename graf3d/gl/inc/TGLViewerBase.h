// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLViewerBase
#define ROOT_TGLViewerBase

#include <TObject.h>

#include "TGLLockable.h"
#include <TGLBoundingBox.h>
#include "TGLOverlay.h"

#include <list>
#include <vector>

class TGLSceneBase;
class TGLSceneInfo;
class TGLCamera;
class TGLClip;
class TGLRnrCtx;
class TGLLogicalShape;
class TGLSelectRecord;
class TGLOvlSelectRecord;
class TGLOverlayElement;

// Avoid TObject inheritance due to clash with TVirtualViewer3D.

class TGLViewerBase : public TGLLockable // : public TObject
{
private:
   TGLViewerBase(const TGLViewerBase&);            // Not implemented
   TGLViewerBase& operator=(const TGLViewerBase&); // Not implemented

protected:
   typedef std::list<TGLSceneInfo*>             SceneInfoList_t;
   typedef SceneInfoList_t::iterator            SceneInfoList_i;

   typedef std::vector<TGLSceneInfo*>           SceneInfoVec_t;
   typedef SceneInfoVec_t::iterator             SceneInfoVec_i;

   typedef std::vector<TGLOverlayElement*>      OverlayElmVec_t;
   typedef OverlayElmVec_t::iterator            OverlayElmVec_i;

   SceneInfoList_i FindScene(TGLSceneBase* scene);

   typedef void (TGLSceneBase::* SubRender_foo) (TGLRnrCtx &);

   void SubRenderScenes(SubRender_foo render_foo);

   virtual void SetupClipObject();

   // Members

   TGLRnrCtx         *fRnrCtx;

   TGLCamera         *fCamera;      // Camera for rendering.
   TGLClip           *fClip;        // Viewer clipping-plane.
   Short_t            fLOD;         // Viewer-lod for rendering.
   Short_t            fStyle;       // Viewer-style for rendering.
   Float_t            fWFLineW;     // Optional override of scene wire-frame line-width
   Float_t            fOLLineW;     // Optional override of scene outline line-width

   Bool_t             fResetSceneInfosOnRender; // Request rebuild of view-specific scene data.
   Bool_t             fChanged;                 // Change requiring redraw is pending.

   SceneInfoList_t    fScenes;                  // Registered scenes.
   SceneInfoVec_t     fVisScenes;               // Visible scenes.

   TGLBoundingBox     fOverallBoundingBox;      // Axis-aligned union of scene bboxes.

   OverlayElmVec_t    fOverlay;

   // ================================================================

public:

   TGLViewerBase();
   virtual ~TGLViewerBase();

   virtual const char* LockIdStr() const;

   TGLSceneInfo* AddScene(TGLSceneBase* scene);
   void          RemoveScene(TGLSceneBase* scene);
   void          RemoveAllScenes();
   void          SceneDestructing(TGLSceneBase* scene);

   TGLSceneInfo* GetSceneInfo(TGLSceneBase* scene);

   TGLLogicalShape* FindLogicalInScenes(TObject* id);

   virtual void AddOverlayElement(TGLOverlayElement* el);
   virtual void RemoveOverlayElement(TGLOverlayElement* el);
   virtual void DeleteOverlayAnnotations();
   virtual void DeleteOverlayElements(TGLOverlayElement::ERole r);

   TGLClip* Clip()         const { return fClip; }
   void     SetClip(TGLClip *p)  { fClip = p;    }

   Short_t  LOD()          const { return fLOD; }
   void     SetLOD(Short_t lod)  { fLOD = lod;  }

   Short_t  Style()        const { return fStyle; }
   void     SetStyle(Short_t st) { fStyle = st;   }

   Float_t  WFLineW()       const { return fWFLineW; }
   void     SetWFLineW(Float_t w) { fWFLineW = w;    }
   Float_t  OLLineW()       const { return fOLLineW; }
   void     SetOLLineW(Float_t w) { fOLLineW = w;    }

   // ================================================================

   virtual void   ResetSceneInfos();
   virtual void   Changed() { fChanged = kTRUE; }
   virtual Bool_t IsChanged() const { return fChanged; }

   virtual void   MergeSceneBBoxes(TGLBoundingBox& bbox);

   // ================================================================

   // Low-level methods
   virtual void PreRender();
   virtual void Render();
   virtual void RenderNonSelected();
   virtual void RenderSelected();
   virtual void RenderOverlay(Int_t state, Bool_t selection);
   virtual void PostRender();

   virtual void PreRenderOverlaySelection();
   virtual void PostRenderOverlaySelection();

   // High-level methods
   // virtual void Draw();
   // virtual void Select();
   // virtual void SelectOverlay();

   // Demangle select buffer
   Bool_t ResolveSelectRecord(TGLSelectRecord& rec, Int_t recIdx);
   // Slightly higher-level search in select-buffer
   Bool_t FindClosestRecord      (TGLSelectRecord& rec, Int_t& recIdx);
   Bool_t FindClosestOpaqueRecord(TGLSelectRecord& rec, Int_t& recIdx);

   // Demangle overlay select buffer
   Bool_t FindClosestOverlayRecord(TGLOvlSelectRecord& rec, Int_t& recIdx);

   TGLRnrCtx* GetRnrCtx() const { return  fRnrCtx; }
   TGLRnrCtx& RnrCtx() const    { return *fRnrCtx; }

   ClassDef(TGLViewerBase, 0); // GL Viewer base-class.
};


#endif
