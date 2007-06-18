// @(#)root/gl:$Name:  $:$Id: TGLRnrCtx.h,v 1.1 2007/06/11 19:56:33 brun Exp $
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLRnrCtx
#define ROOT_TGLRnrCtx

#include <Rtypes.h>

#include <list>
#include <vector>

class TGLViewerBase;
class TGLCamera;
class TGLSceneBase;
class TGLSceneInfo;

class TGLContextIdentity;

class TGLClip;
class TGLSelectBuffer;

class TGLRect;

class GLUquadric;

/**************************************************************************/
// TGLRnrCtx
/**************************************************************************/

class TGLRnrCtx
{
public:
   enum EStyle {
      kStyleUndef     =  -1,
      kFill,
      kOutline,
      kWireFrame
   };
   static const char* StyleName(Short_t style);

   enum EPass {
      kPassUndef      =  -1,
      kPassFill,
      kPassOutlineFill,
      kPassOutlineLine,
      kPassWireFrame
   };
   enum ELODPresets {
      kLODUndef       =  -1,
      kLODPixel       =   0, // Projected size pixel or less
      kLODLow         =  20,
      kLODMed         =  50,
      kLODHigh        = 100
   };

private:
   TGLRnrCtx(const TGLRnrCtx&);            // Not implemented
   TGLRnrCtx& operator=(const TGLRnrCtx&); // Not implemented

protected:
   TGLViewerBase  *fViewer;
   TGLCamera      *fCamera;
   TGLSceneInfo   *fSceneInfo;

   Short_t         fViewerLOD;
   Short_t         fSceneLOD;
   Short_t         fCombiLOD; // Combined viewer/scene lod
   Short_t         fShapeLOD;

   Short_t         fViewerStyle;
   Short_t         fSceneStyle;

   TGLClip        *fViewerClip;
   TGLClip        *fSceneClip;
   TGLClip        *fClip;

   Short_t         fDrawPass;

   Double_t        fRenderTimeout;

   // Selection stuff
   Bool_t          fSelection;
   Bool_t          fSecSelection;
   TGLRect        *fPickRectangle;
   TGLSelectBuffer*fSelectBuffer;

   UInt_t          fEventKeySym;

   // GL state
   Bool_t              fDLCaptureOpen; //! DL-capture currently open
   TGLContextIdentity *fGLCtxIdentity; //! Current GL context identity

   GLUquadric         *fQuadric;

public:
   TGLRnrCtx(TGLViewerBase* viewer);
   virtual ~TGLRnrCtx();

   // Central objects
   TGLViewerBase * GetViewer() { return  fViewer; }
   TGLViewerBase & RefViewer() { return *fViewer; }
   TGLCamera     * GetCamera() { return  fCamera; }
   TGLCamera     & RefCamera() { return *fCamera; }
   TGLSceneInfo  * GetSceneInfo()  { return  fSceneInfo; }
   TGLSceneInfo  & RefSceneInfo()  { return *fSceneInfo; }
   TGLSceneBase  * GetScene();
   TGLSceneBase  & RefScene();

   // void SetViewer   (TGLViewerBase* v) { fViewer = v; }
   void SetCamera   (TGLCamera*     c) { fCamera = c; }
   void SetSceneInfo(TGLSceneInfo* si) { fSceneInfo = si; }


   // Draw LOD, style, clip, rnr-pass
   Short_t ViewerLOD()   const         { return fViewerLOD; }
   void    SetViewerLOD(Short_t LOD)   { fViewerLOD = LOD;  }
   Short_t SceneLOD()    const         { return fSceneLOD; }
   void    SetSceneLOD(Short_t LOD)    { fSceneLOD = LOD;  }
   Short_t CombiLOD()    const         { return fCombiLOD; }
   void    SetCombiLOD(Short_t LOD)    { fCombiLOD = LOD;  }
   Short_t ShapeLOD()    const         { return fShapeLOD; }
   void    SetShapeLOD(Short_t LOD)    { fShapeLOD = LOD;  }

   Short_t ViewerStyle() const         { return fViewerStyle; }
   void    SetViewerStyle(Short_t sty) { fViewerStyle = sty;  }
   Short_t SceneStyle()  const         { return fSceneStyle; }
   void    SetSceneStyle(Short_t sty)  { fSceneStyle = sty;  }

   TGLClip* ViewerClip()         const { return fViewerClip; }
   void     SetViewerClip(TGLClip *p)  { fViewerClip = p;    }
   TGLClip* SceneClip()          const { return fSceneClip;  }
   void     SetSceneClip(TGLClip *p)   { fSceneClip = p;     }
   TGLClip* Clip()               const { return  fClip;      }
   void     SetClip(TGLClip *p)        { fClip = p;          }
   Bool_t   HasClip()            const { return  fClip != 0; }

   Short_t DrawPass()    const         { return fDrawPass;  }
   void    SetDrawPass(Short_t dpass)  { fDrawPass = dpass; }
   Bool_t  IsDrawPassFilled() const;

   // Render time-out
   Double_t RenderTimeout()           const { return fRenderTimeout; }
   void     SetRenderTimeout(Double_t tout) { fRenderTimeout = tout; }

   // Selection stuff
   Bool_t  Selection()    const           { return fSelection;      }
   void    SetSelection(Bool_t sel)       { fSelection = sel;       }
   Bool_t  SecSelection() const           { return fSecSelection;   }
   void    SetSecSelection(Bool_t secSel) { fSecSelection = secSel; }
   // Low-level getters
   TGLRect         * GetPickRectangle();
   Int_t             GetPickRadius();
   TGLSelectBuffer * GetSelectBuffer() const { return fSelectBuffer; }
   // Composed operations
   void      BeginSelection(Int_t x, Int_t y, Int_t r=3);
   void      EndSelection  (Int_t glResult);

   UInt_t GetEventKeySym()   const { return fEventKeySym; }
   void   SetEventKeySym(UInt_t k) { fEventKeySym = k; }

   Bool_t IsDLCaptureOpen() const { return fDLCaptureOpen; }
   void   OpenDLCapture();
   void   CloseDLCapture();

   TGLContextIdentity* GetGLCtxIdentity()   const { return fGLCtxIdentity; }
   void SetGLCtxIdentity(TGLContextIdentity* cid) { fGLCtxIdentity = cid; }

   GLUquadric * GetGluQuadric() { return  fQuadric; }

   ClassDef(TGLRnrCtx, 0) // Collection of objects and data passes along all rendering calls.
}; // endclass TGLRnrCtx


#endif
