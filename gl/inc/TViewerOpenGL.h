// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.h,v 1.9 2004/09/03 12:52:42 brun Exp $
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

#include <TVirtualViewer3D.h>
#include <RQ_OBJECT.h>
#include <TGFrame.h>
#include <TPoint.h>
#include <Gtypes.h>

#include "TGLRender.h"

class TGLRenderArea;
class TGLSelection;
class TGVSplitter;
class TGPopupMenu;
class TGLEditor;
class TGLCamera;
class TGLRender;
class TBuffer3D;
class TGMenuBar;
class TGCanvas;
class TArcBall;

class TViewerOpenGL : public TVirtualViewer3D, public TGMainFrame {
private:
   typedef std::pair<Double_t, Double_t> PDD_t;
   enum EMode{kNav, kPick};
   enum EViews{kXOY, kXOZ, kYOZ, kPERSP};

   TGCanvas      *fCanvasWindow;
   TGLRenderArea *fCanvasContainer;
   TGLayoutHints *fCanvasLayout;
   
   TGMenuBar     *fMenuBar;
   TGPopupMenu   *fFileMenu, *fModeMenu, *fViewMenu, *fHelpMenu;
   TGLayoutHints *fMenuBarLayout;
   TGLayoutHints *fMenuBarItemLayout;
   TGLayoutHints *fMenuBarHelpLayout;

   TGLCamera     *fCamera[4];
   Double_t      fViewVolume[4];
   Double_t      fZoom[4];
   Int_t         fActiveViewport[4];

   Double_t      fXc, fYc, fZc;
   PDD_t         fRangeX, fRangeY, fRangeZ;
   Double_t      fRad;

   Bool_t        fPressed;
   TArcBall      *fArcBall;

   UInt_t        fSelected;
   UInt_t        fNbShapes;
   TGLRender     fRender;
   TPoint        fLastPos;

   EViews        fConf;
   EMode         fMode;
////////////XXX//////////////////////////////
   TGCompositeFrame *fMainFrame;
   TGCompositeFrame *fEdFrame;
   TGLEditor        *fEditor;

   TGLSceneObject *fSelectedObj;
   Color_t fRGBA[4];
   TGVerticalFrame *fV1;
   TGVerticalFrame *fV2;
   TGVSplitter *fSplitter;
   TGLayoutHints *fL1, *fL2, *fL3, *fL4;
/////////////////////////////////////////////
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
