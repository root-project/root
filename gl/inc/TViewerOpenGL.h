// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.h,v 1.12 2004/09/15 14:26:58 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TViewerOpenGL
#define ROOT_TViewerOpenGL

#include <utility>

#ifndef ROOT_TVirtualViewer3D
#include "TVirtualViewer3D.h"
#endif
#ifndef ROOT_RQ_OBJECT
#include "RQ_OBJECT.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif
#ifndef ROOT_TGLRender
#include "TGLRender.h"
#endif

class TGLRenderArea;
class TContextMenu;
class TGLSelection;
class TGVSplitter;
class TGPopupMenu;
class TGLEditor;
class TGLCamera;
class TBuffer3D;
class TGMenuBar;
class TGCanvas;
class TArcBall;

class TViewerOpenGL : public TVirtualViewer3D, public TGMainFrame {
private:
   TGCompositeFrame  *fMainFrame;
   TGVerticalFrame   *fV1, *fV2;
   TGVSplitter       *fSplitter;
   TGCompositeFrame  *fEdFrame;
   TGLEditor         *fEditor;
   TGCanvas          *fCanvasWindow;
   TGLRenderArea     *fCanvasContainer;

   TGLayoutHints     *fL1, *fL2, *fL3, *fL4;
   TGLayoutHints     *fCanvasLayout;

   TGMenuBar         *fMenuBar;
   TGPopupMenu       *fFileMenu, *fModeMenu, *fViewMenu, *fHelpMenu;

   TGLayoutHints     *fMenuBarLayout;
   TGLayoutHints     *fMenuBarItemLayout;
   TGLayoutHints     *fMenuBarHelpLayout;

   TGLCamera         *fCamera[4];
   Double_t          fViewVolume[4];
   Double_t          fZoom[4];
   Int_t             fActiveViewport[4];

   typedef std::pair<Double_t, Double_t> PDD_t;
   PDD_t             fRangeX, fRangeY, fRangeZ, fLastPosRot;
   Double_t          fXc, fYc, fZc;
   Double_t          fRad;

   Bool_t            fPressed;
   TArcBall          *fArcBall;

   UInt_t            fNbShapes;
   TGLRender         fRender;
   TPoint            fLastPos;

   enum EMode{kNav, kPick};
   enum EViews{kXOY, kXOZ, kYOZ, kPERSP};

   EViews            fConf;
   EMode             fMode;

   TContextMenu      *fContextMenu;
   TGLSceneObject    *fSelectedObj;
   Color_t           fRGBA[4];

public:
   TViewerOpenGL(TVirtualPad * pad);
   ~TViewerOpenGL();
   //final overriders for TVirtualViewer3D
   void UpdateScene(Option_t *);
   void CreateScene(Option_t *);

   Bool_t HandleContainerButton(Event_t *ev);
   Bool_t HandleContainerConfigure(Event_t *ev);
   Bool_t HandleContainerKey(Event_t *ev);
   Bool_t HandleContainerMotion(Event_t *ev);
   Bool_t HandleContainerExpose(Event_t *ev);
   void ModifySelected();

private:
   void CreateViewer();
   void DrawObjects()const;
   void MakeCurrent()const;
   void SwapBuffers()const;
   void Show();
   TGLSelection *UpdateRange(const TBuffer3D *buff);
   TGLSceneObject *TestSelection(Event_t *);
   void CalculateViewports();
   void CalculateViewvolumes();
   void CreateCameras();
   // final overriders from TGMainFrame
   void CloseWindow();
   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   //non-copyable class
   TViewerOpenGL(const TViewerOpenGL &);
   TViewerOpenGL & operator = (const TViewerOpenGL &);

   ClassDef(TViewerOpenGL, 0)
};

#endif
