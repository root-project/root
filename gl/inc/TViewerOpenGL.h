// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.h,v 1.32 2005/06/23 15:08:45 brun Exp $
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

#ifndef ROOT_TGLViewer
#include "TGLViewer.h"
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
class TGMenuBar;
class TGCanvas;

// Must derive from TGLViewer first, as this implements our
// TVirtualViewer3D interface, which we are cast to externally
class TViewerOpenGL : public TGLViewer, public TGMainFrame
{
private:
   // GUI components
   TGCompositeFrame  *fMainFrame;
   TGVerticalFrame   *fV1;
   TGVerticalFrame   *fV2;
   TGShutter         *fShutter;
   TGShutterItem     *fShutItem1, *fShutItem2, *fShutItem3, *fShutItem4;
   TGLayoutHints     *fL1, *fL2, *fL3, *fL4;
   TGLayoutHints     *fCanvasLayout;
   TGMenuBar         *fMenuBar;
   TGPopupMenu       *fFileMenu, *fViewMenu, *fHelpMenu;
   TGLayoutHints     *fMenuBarLayout;
   TGLayoutHints     *fMenuBarItemLayout;
   TGLayoutHints     *fMenuBarHelpLayout;
   TContextMenu      *fContextMenu;
   TGCanvas          *fCanvasWindow;
   TGLRenderArea     *fCanvasContainer;

   // Editors
   TGLColorEditor    *fColorEditor;
   TGLGeometryEditor *fGeomEditor;
   TGLSceneEditor    *fSceneEditor;
   TGLLightEditor    *fLightEditor;

   // Interaction
   enum EAction      { kNone, kRotate, kTruck, kDolly, kDrag };
   EAction           fAction;
   TPoint            fStartPos;
   TPoint            fLastPos;
   UInt_t            fActiveButtonID;

   // Lighting
   Int_t             fLightMask;

   // External handles
   TVirtualPad      *fPad;

   // Initial window positioning
   static const Int_t fgInitX;
   static const Int_t fgInitY;
   static const Int_t fgInitW;
   static const Int_t fgInitH;

public:
   TViewerOpenGL(TVirtualPad * pad);
   ~TViewerOpenGL();

   Bool_t HandleContainerEvent(Event_t *ev);
   Bool_t HandleContainerButton(Event_t *ev);
   Bool_t HandleContainerDoubleClick(Event_t *ev);
   Bool_t HandleContainerConfigure(Event_t *ev);
   Bool_t HandleContainerKey(Event_t *ev);
   Bool_t HandleContainerMotion(Event_t *ev);
   Bool_t HandleContainerExpose(Event_t *ev);
   void ModifyScene(Int_t id);

private:
   // Setup
   void CreateViewer();

   // TGLViewer overloads
   virtual void   InitGL();
   virtual void   Invalidate(UInt_t redrawLOD = kMed);
   virtual void   MakeCurrent() const;
   virtual void   SwapBuffers() const;
   virtual void   FillScene();

   void DoSelect(Event_t *event, Bool_t invokeContext);
   void DoRedraw(); // from TGMainFrame
   void Show();
   void CloseWindow();
   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void PrintObjects();

   //non-copyable class
   TViewerOpenGL(const TViewerOpenGL &);
   TViewerOpenGL & operator = (const TViewerOpenGL &);

   ClassDef(TViewerOpenGL, 0)
};

#endif
