// @(#)root/gl:$Id$
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

#include "Rtypes.h"
#include "TGLStopwatch.h"

class TGLViewerBase;
class TGLCamera;
class TGLSceneBase;
class TGLSceneInfo;

class TGLColorSet;
class TGLFont;
class TGLContextIdentity;

class TGLClip;
class TGLSelectBuffer;

class TGLRect;

class GLUquadric;

namespace std
{
   template<typename _Tp> class allocator;
   template<typename _Tp, typename _Alc> class list;
}

/**************************************************************************/
// TGLRnrCtx
/**************************************************************************/

class TGLRnrCtx
{
public:
   enum EStyle
   {
      kStyleUndef     =  -1,
      kFill,
      kOutline,
      kWireFrame
   };
   static const char* StyleName(Short_t style);

   enum EPass
   {
      kPassUndef      =  -1,
      kPassFill,
      kPassOutlineFill,
      kPassOutlineLine,
      kPassWireFrame
   };

   enum ELODPresets
   {
      kLODUndef       =  -1,
      kLODPixel       =   0, // Projected size pixel or less
      kLODLow         =  20,
      kLODMed         =  50,
      kLODHigh        = 100
   };

   enum EShapeSelectionLevel
   {
      kSSLNotSelected,
      kSSLSelected,
      kSSLImpliedSelected,
      kSSLHighlighted,
      kSSLImpliedHighlighted,
      kSSLEnd
   };

private:
   TGLRnrCtx(const TGLRnrCtx&);            // Not implemented
   TGLRnrCtx& operator=(const TGLRnrCtx&); // Not implemented

   typedef std::list<TGLColorSet*, std::allocator<TGLColorSet*> > lpTGLColorSet_t;

protected:
   TGLViewerBase  *fViewer;
   TGLCamera      *fCamera;
   TGLSceneInfo   *fSceneInfo;

   Short_t         fViewerLOD;
   Short_t         fSceneLOD;
   Short_t         fCombiLOD;     // Combined viewer/scene lod.
   Short_t         fShapeLOD;     // LOD calculated for current shape.
   Float_t         fShapePixSize; // Only relevant when not using display lists.

   Short_t         fViewerStyle;
   Short_t         fSceneStyle;

   Float_t         fViewerWFLineW;
   Float_t         fSceneWFLineW;
   Float_t         fViewerOLLineW;
   Float_t         fSceneOLLineW;

   TGLClip        *fViewerClip;
   TGLClip        *fSceneClip;
   TGLClip        *fClip;

   Short_t         fDrawPass;

   TGLStopwatch    fStopwatch;
   Double_t        fRenderTimeOut;
   Bool_t          fIsRunning;
   Bool_t          fHasTimedOut;

   // Highlight / Selection stuff
   Bool_t          fHighlight;        // True when in highlight.
   Bool_t          fHighlightOutline; // True when in highlight-outline.
   Bool_t          fSelection;
   Bool_t          fSecSelection;
   Int_t           fPickRadius;
   TGLRect        *fPickRectangle;
   TGLSelectBuffer*fSelectBuffer;

   lpTGLColorSet_t*fColorSetStack;
   Float_t         fRenderScale;

   UInt_t          fEventKeySym;

   // GL state
   Bool_t              fDLCaptureOpen; //! DL-capture currently open
   TGLContextIdentity *fGLCtxIdentity; //! Current GL context identity

   GLUquadric         *fQuadric;

   // Picture grabbing
   Bool_t           fGrabImage;    // Set to true to store the image.
   Int_t            fGrabBuffer;   // Which buffer to grab after render.
   UChar_t         *fGrabbedImage; // Buffer where image was stored after rendering.

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

   const TGLCamera & RefCamera() const { return *fCamera; }
   const TGLCamera * GetCamera() const { return  fCamera; }

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
   Float_t ShapePixSize() const        { return fShapePixSize; }
   void    SetShapePixSize(Float_t ps) { fShapePixSize = ps; }

   Short_t ViewerStyle() const         { return fViewerStyle; }
   void    SetViewerStyle(Short_t sty) { fViewerStyle = sty;  }
   Short_t SceneStyle()  const         { return fSceneStyle; }
   void    SetSceneStyle(Short_t sty)  { fSceneStyle = sty;  }

   Float_t ViewerWFLineW()       const { return fViewerWFLineW; }
   void    SetViewerWFLineW(Float_t w) { fViewerWFLineW = w;    }
   Float_t SceneWFLineW()        const { return fSceneWFLineW;  }
   void    SetSceneWFLineW(Float_t w)  { fSceneWFLineW = w;     }
   Float_t ViewerOLLineW()       const { return fViewerOLLineW; }
   void    SetViewerOLLineW(Float_t w) { fViewerOLLineW = w;    }
   Float_t SceneOLLineW()        const { return fSceneOLLineW;  }
   void    SetSceneOLLineW(Float_t w)  { fSceneOLLineW = w;     }

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
   Bool_t  IsDrawPassOutlineLine() const { return fDrawPass == kPassOutlineLine; }

   // Render time-out
   Double_t RenderTimeOut()           const { return fRenderTimeOut; }
   void     SetRenderTimeOut(Double_t tout) { fRenderTimeOut = tout; }
   void     StartStopwatch();
   void     StopStopwatch();
   Bool_t   IsStopwatchRunning() const { return fIsRunning; }
   Bool_t   HasStopwatchTimedOut();

   // Highlight / Selection stuff
   Bool_t  Highlight()    const           { return fHighlight;      }
   void    SetHighlight(Bool_t hil)       { fHighlight = hil;       }
   Bool_t  HighlightOutline() const       { return fHighlightOutline; }
   void    SetHighlightOutline(Bool_t ho) { fHighlightOutline = ho;   }

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

   void         PushColorSet();
   TGLColorSet& ColorSet();
   void         PopColorSet();
   TGLColorSet* ChangeBaseColorSet(TGLColorSet* set);
   TGLColorSet* GetBaseColorSet();

   void         ColorOrForeground(Color_t col);

   Float_t GetRenderScale()    const { return fRenderScale; }
   void    SetRenderScale(Float_t s) { fRenderScale = s; }

   UInt_t GetEventKeySym()   const { return fEventKeySym; }
   void   SetEventKeySym(UInt_t k) { fEventKeySym = k; }

   Bool_t IsDLCaptureOpen() const  { return fDLCaptureOpen; }
   void   OpenDLCapture();
   void   CloseDLCapture();

   TGLContextIdentity* GetGLCtxIdentity()   const { return fGLCtxIdentity; }
   void SetGLCtxIdentity(TGLContextIdentity* cid) { fGLCtxIdentity = cid; }

   void  RegisterFont(Int_t size, Int_t file, Int_t mode, TGLFont& out);
   void  RegisterFont(Int_t size, const char* name, Int_t mode, TGLFont& out);
   void  RegisterFontNoScale(Int_t size, Int_t file, Int_t mode, TGLFont& out);
   void  RegisterFontNoScale(Int_t size, const char* name, Int_t mode, TGLFont& out);
   void  ReleaseFont(TGLFont& font);

   GLUquadric* GetGluQuadric() { return fQuadric; }

   // Picture grabbing
   void     SetGrabImage(Bool_t gi) { fGrabImage = gi;   }
   Bool_t   GetGrabImage()    const { return fGrabImage; }

   // Matrix manipulation helpers
   void ProjectionMatrixPushIdentity();
   void ProjectionMatrixPop();

   ClassDef(TGLRnrCtx, 0); // Collection of objects and data passes along all rendering calls.
};


#endif
