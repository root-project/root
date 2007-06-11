// @(#)root/gl:$Name$:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLViewerBase_H
#define ROOT_TGLViewerBase_H

#include <TObject.h>

#include "TGLLockable.h"
#include <TGLBoundingBox.h>

#include <list>
#include <vector>

class TGLSceneBase;
class TGLSceneInfo;
class TGLCamera;
class TGLClip;
class TGLRnrCtx;
class TGLSelectRecord;
class TGLOverlayElement;

// Avoid TObject inheritance due to clash with TVirtualViewer3D.

class TGLViewerBase : public TGLLockable // : public TObject
{
private:
   TGLViewerBase(const TGLViewerBase&);            // Not implemented
   TGLViewerBase& operator=(const TGLViewerBase&); // Not implemented

protected:
   typedef std::list<TGLSceneInfo*>             SceneInfoList_t;
   typedef std::list<TGLSceneInfo*>::iterator   SceneInfoList_i;

   typedef std::vector<TGLSceneInfo*>           SceneInfoVec_t;

   typedef std::vector<TGLOverlayElement*>      OverlayElmVec_t;

   SceneInfoList_i FindScene(TGLSceneBase* scene);

   // Members

   TGLRnrCtx         *fRnrCtx;

   TGLCamera         *fCamera;      // Camera for rendering.
   TGLClip           *fClip;        // Viewer clipping-plane.
   Short_t            fLOD;         // Viewer-lod for rendering.
   Short_t            fStyle;       // Viewer-style for rendering.

   Bool_t             fResetSceneInfosOnRender; // Request rebuild of view-specific scene data.

   SceneInfoList_t    fScenes;                  // Registered scenes.
   SceneInfoVec_t     fVisScenes;               // Visible scenes.

   TGLBoundingBox     fOverallBoundingBox;      // Axis-aligned union of scene bboxes.

   OverlayElmVec_t    fOverlay;

   // ================================================================

public:

   TGLViewerBase();
   virtual ~TGLViewerBase();

   virtual const char* LockIdStr() const;

   void AddScene(TGLSceneBase* scene);
   void RemoveScene(TGLSceneBase* scene);

   TGLSceneInfo* GetSceneInfo(TGLSceneBase* scene);

   TGLClip* Clip()         const { return fClip; }
   void     SetClip(TGLClip *p)  { fClip = p;    }

   Short_t  LOD()          const { return fLOD; }
   void     SetLOD(Short_t lod)  { fLOD = lod;  }

   Short_t  Style()        const { return fStyle; }
   void     SetStyle(Short_t st) { fStyle = st;   }

   // ================================================================

   virtual void ResetSceneInfos();

   // ================================================================

   // Low-level methods
   virtual void PreRender();
   virtual void Render();
   virtual void RenderOverlay();
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

   ClassDef(TGLViewerBase, 0) // GL Viewer base-class.
}; // endclass TGLViewerBase


#endif
