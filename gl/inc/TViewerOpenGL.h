// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.h,v 1.25 2005/03/09 18:19:26 brun Exp $
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
#include <vector>

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
#ifndef ROOT_CsgOps
#include "CsgOps.h"
#endif

class TGLGeometryEditor;
class TGShutterItem;
class TGShutter;
class TGLRenderArea;
class TContextMenu;
class TGLSelection;
class TGVSplitter;
class TGPopupMenu;
class TGLColorEditor;
class TGLSceneEditor;
class TGLLightEditor;
class TGLCamera;
class TBuffer3D;
class TGMenuBar;
class TGCanvas;
class TArcBall;

class TViewerOpenGL : public TVirtualViewer3D, public TGMainFrame {
private:
   TGCompositeFrame  *fMainFrame;
   TGVerticalFrame   *fV1;
   TGVerticalFrame   *fV2;
   TGLColorEditor    *fColorEditor;
   TGLGeometryEditor *fGeomEditor;
   TGLSceneEditor    *fSceneEditor;
   TGLLightEditor    *fLightEditor;
   TGCanvas          *fCanvasWindow;
   TGLRenderArea     *fCanvasContainer;
   TGLRender         *fRender;
   TGShutter         *fShutter;
   TGShutterItem     *fShutItem1, *fShutItem2, *fShutItem3, *fShutItem4;

   TGLayoutHints     *fL1, *fL2, *fL3, *fL4;
   TGLayoutHints     *fCanvasLayout;

   TGMenuBar         *fMenuBar;
   TGPopupMenu       *fFileMenu, *fViewMenu, *fHelpMenu;

   TGLayoutHints     *fMenuBarLayout;
   TGLayoutHints     *fMenuBarItemLayout;
   TGLayoutHints     *fMenuBarHelpLayout;

   TGLCamera         *fCamera[4];
   Double_t          fViewVolume[4];
   Double_t          fZoom[4];
   Int_t             fActiveViewport[4];
   Int_t             fLightMask;

   typedef std::pair<Double_t, Double_t> PDD_t;
   PDD_t             fRangeX, fRangeY, fRangeZ, fLastPosRot;
   Double_t          fXc, fYc, fZc;
   Double_t          fRad;

   Bool_t            fPressed;
   TArcBall          *fArcBall;

   UInt_t            fNbShapes;
   TPoint            fLastPos;

   enum EViews{kXOY, kXOZ, kYOZ, kPERSP};
   EViews            fConf;

   TContextMenu      *fContextMenu;
   TGLSceneObject    *fSelectedObj;
   enum EAction{kNoAction, kRotating, kPicking, kZooming};
   EAction fAction;
   Bool_t            fBuildingScene;
   TVirtualPad      *fPad;
   Bool_t            fFirstScene;

   //CS specific
   typedef std::pair<UInt_t, RootCsg::BaseMesh *> CSPART_t;

   Bool_t                  fInsideComposite;
   UInt_t                  fCSLevel;
   std::vector<CSPART_t>   fCSTokens;
   TObject                 *fComposite;

   RootCsg::BaseMesh *BuildComposite();

public:
   TViewerOpenGL(TVirtualPad * pad);
   ~TViewerOpenGL();

   // TVirtualViewer3D interface
   virtual Bool_t PreferLocalFrame() const;
   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fBuildingScene; }
   virtual void   EndScene();
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Int_t  AddObject(UInt_t placedID, const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual void   OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual void   CloseComposite();
   virtual void   AddCompositeOp(UInt_t operation);

   Bool_t HandleContainerButton(Event_t *ev);
   Bool_t HandleContainerConfigure(Event_t *ev);
   Bool_t HandleContainerKey(Event_t *ev);
   Bool_t HandleContainerMotion(Event_t *ev);
   Bool_t HandleContainerExpose(Event_t *ev);
   void ModifyScene(Int_t id);

private:
   void AddValidatedObject(UInt_t placedID, const TBuffer3D & buffer, Bool_t * addChildren);
   void CreateViewer();
   void DrawObjects()const;
   void PrintObjects()const;
   void MakeCurrent()const;
   void SwapBuffers()const;
   void Show();
   void UpdateRange(const TGLSelection *box);
   TGLSceneObject *TestSelection(Event_t *);
   void CalculateViewports();
   void CalculateViewvolumes();
   void CreateCameras();
   //
   void MoveCenter(Int_t key);
   // final overriders from TGMainFrame
   void CloseWindow();
   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   static const Int_t fgInitX;
   static const Int_t fgInitY;
   static const Int_t fgInitW;
   static const Int_t fgInitH;
   //non-copyable class
   TViewerOpenGL(const TViewerOpenGL &);
   TViewerOpenGL & operator = (const TViewerOpenGL &);

   ClassDef(TViewerOpenGL, 0)
};

#endif
