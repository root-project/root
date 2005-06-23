// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.h,v 1.31 2005/06/21 16:54:17 brun Exp $
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
class TBuffer3D;
class TGMenuBar;
class TGCanvas;
class TGLFaceSet;

// TODO: Derv from TGLViewer or ag. as viewport?
class TViewerOpenGL : public TVirtualViewer3D, public TGMainFrame, public TGLViewer 
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

   // Scene management - TODO: Most of this can probably be moved down to
   // TGLViewer?
   Bool_t            fInternalRebuild;      //! internal scene rebuild in progress?
   Bool_t            fAcceptedAllPhysicals; //! did we take all physicals offered in AddObject()
   Bool_t            fInternalPIDs;         //! using internal physical IDs
   UInt_t            fNextInternalPID;      //! next internal physical ID (from 1 - 0 reserved)

   // Lighting
   Int_t             fLightMask;

   // External handles
   TVirtualPad      *fPad;

   // Composite Shape specific
   mutable TGLFaceSet     *fComposite; //! Paritally created composite
   typedef std::pair<UInt_t, RootCsg::BaseMesh *> CSPART_t;
   UInt_t                  fCSLevel;
   std::vector<CSPART_t>   fCSTokens;

   RootCsg::BaseMesh *BuildComposite();

   // Tracing for scene rebuilds
   UInt_t                  fAcceptedPhysicals;
   UInt_t                  fRejectedPhysicals;

   // Initial window positioning
   static const Int_t fgInitX;
   static const Int_t fgInitY;
   static const Int_t fgInitW;
   static const Int_t fgInitH;

public:
   TViewerOpenGL(TVirtualPad * pad);
   ~TViewerOpenGL();

   // TVirtualViewer3D interface
   virtual Bool_t PreferLocalFrame() const;
   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fScene.CurrentLock() == TGLScene::kModifyLock; }
   virtual void   EndScene();
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Int_t  AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual Bool_t OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren = 0);
   virtual void   CloseComposite();
   virtual void   AddCompositeOp(UInt_t operation);

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
   virtual Bool_t RebuildScene();

   // Scene Object Management
   Int_t              ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t logical) const;
   TGLLogicalShape *  CreateNewLogical(const TBuffer3D & buffer) const;
   TGLPhysicalShape * CreateNewPhysical(UInt_t physicalID, const TBuffer3D & buffer, 
                                        const TGLLogicalShape & logical) const;

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
