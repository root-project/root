// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.cxx,v 1.66 2005/06/23 15:08:45 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TViewerOpenGL.h"

#include "TPluginManager.h"
#include "TRootHelpDialog.h"
#include "TContextMenu.h"
#include "KeySymbols.h"
#include "TGShutter.h"
#include "TGLKernel.h"
#include "TVirtualGL.h"
#include "TGButton.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "HelpText.h"
#include "Buttons.h"
#include "TAtt3D.h"
#include "TGMenu.h"
#include "TColor.h"
#include "TMath.h"
#include "TSystem.h"

#include "TGLSceneObject.h"
#include "TGLRenderArea.h"
#include "TGLEditor.h"
#include "TGLCamera.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TVirtualPad.h"

#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLDisplayListCache.h"
#include "TGLStopwatch.h"
#include "TObject.h"

#include "gl2ps.h"

#include <assert.h>

const char gHelpViewerOpenGL[] = "\
     PRESS \n\
     \tw\t--- wireframe mode\n\
     \tr\t--- filled polygons mode\n\
     \tj\t--- zoom in\n\
     \tk\t--- zoom out\n\n\
	  \tArrow Keys\tpan (truck) across scene\n\
     You can ROTATE (ORBIT) the scene by holding the left \n\
     mouse button and moving the mouse (pespective camera only).\n\
     You can PAN (TRUCK) the camera using the middle mouse\n\
     button or arrow keys.\n\
     You can ZOOM (DOLLY) the camera by dragging side\n\
     to side holding the right mouse button or using the\n\
     mouse wheel.\n\
     RESET the camera by double clicking any button\n\
     SELECT an object with Shift+Left mouse button click.\n\
     MOVE the object using Shift+Mid mouse drag.\n\
     Invoked the CONTEXT menu with Shift+Right mouse click.\n\
     PROJECTIONS\n\n\
     You can select the different plane projections\n\
     in \"Projections\" menu.\n\n\
     COLOR\n\n\
     After you selected an object or a light source,\n\
     you can modify object's material and light\n\
     source color.\n\n\
     \tLIGHT SOURCES.\n\n\
     \tThere are two pickable light sources in\n\
     \tthe current implementation. They are shown as\n\
     \tspheres. Each light source has three light\n\
     \tcomponents : DIFFUSE, AMBIENT, SPECULAR.\n\
     \tEach of this components is defined by the\n\
     \tamounts of red, green and blue light it emits.\n\
     \tYou can EDIT this parameters:\n\
     \t1. Select light source sphere.\n" //hehe, too long string literal :)))
"    \t2. Select light component you want to modify\n\
     \t   by pressing one of radio buttons.\n\
     \t3. Change RGB by moving sliders\n\n\
     \tMATERIAL\n\n\
     \tObject's material is specified by the percentage\n\
     \tof red, green, blue light it reflects. A surface can\n\
     \treflect diffuse, ambient and specular light. \n\
     \tA surface has two additional parameters: EMISSION\n\
     \t- you can make surface self-luminous; SHININESS -\n\
     \tmodifying this parameter you can change surface\n\
     \thighlights.\n\
     \tSometimes changes are not visible, or light\n\
     \tsources seem not to work - you should understand\n\
     \tthe meaning of diffuse, ambient etc. light and material\n\
     \tcomponents. For example, if you define material, wich has\n\
     \tdiffuse component (1., 0., 0.) and you have a light source\n\
     \twith diffuse component (0., 1., 0.) - you surface does not\n\
     \treflect diffuse light from this source. For another example\n\
     \t- the color of highlight on the surface is specified by:\n\
     \tlight's specular component, material specular component.\n\
     \tAt the top of the color editor there is a small window\n\
     \twith sphere. When you are editing surface material,\n\
     \tyou can see this material applyed to sphere.\n\
     \tWhen edit light source, you see this light reflected\n\
     \tby sphere whith DIFFUSE and SPECULAR components\n\
     \t(1., 1., 1.).\n\n\
     OBJECT'S GEOMETRY\n\n\
     You can edit object's location and stretch it by entering\n\
     desired values in respective number entry controls.\n\n"
"    SCENE PROPERTIES\n\n\
     You can add clipping plane by clicking the checkbox and\n\
     specifying the plane's equation A*x+B*y+C*z+D=0.";

enum EGLViewerCommands {
   kGLHelpAbout,
   kGLHelpOnViewer,
   kGLXOY,
   kGLXOZ,
   kGLYOZ,
   kGLPersp,
   kGLPrintEPS_SIMPLE,
   kGLPrintEPS_BSP,
   kGLPrintPDF_SIMPLE,
   kGLPrintPDF_BSP,
   kGLExit
};

ClassImp(TViewerOpenGL)

const Int_t TViewerOpenGL::fgInitX = 0;
const Int_t TViewerOpenGL::fgInitY = 0;
const Int_t TViewerOpenGL::fgInitW = 780;
const Int_t TViewerOpenGL::fgInitH = 670;

int format = GL2PS_EPS;
int sortgl = GL2PS_BSP_SORT;

//______________________________________________________________________________
TViewerOpenGL::TViewerOpenGL(TVirtualPad * pad) :
   TGMainFrame(gClient->GetDefaultRoot(), fgInitW, fgInitH),
   fMainFrame(0), fV1(0), fV2(0), fShutter(0), fShutItem1(0), fShutItem2(0), 
   fShutItem3(0), fShutItem4(0), fL1(0), fL2(0), fL3(0), fL4(0),
   fCanvasLayout(0), fMenuBar(0), fFileMenu(0), fViewMenu(0), fHelpMenu(0),
   fMenuBarLayout(0), fMenuBarItemLayout(0), fMenuBarHelpLayout(0),
   fContextMenu(0), fCanvasWindow(0), fCanvasContainer(0),
   fColorEditor(0), fGeomEditor(0), fSceneEditor(0), fLightEditor(0),
   fAction(kNone), fStartPos(0,0), fLastPos(0,0), fActiveButtonID(0),
   fInternalRebuild(kFALSE), fAcceptedAllPhysicals(kTRUE),
   fInternalPIDs(kFALSE), fNextInternalPID(1), // 0 reserved
   fLightMask(0x1b), fPad(pad), fComposite(0), fCSLevel(0),
   fAcceptedPhysicals(0), fRejectedPhysicals(0)
{
   // Create OpenGL viewer.
   static Bool_t init = kFALSE;
   if (!init) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGLImp"))) {
         if (h->LoadPlugin() == -1)
            return;
         TVirtualGLImp * imp = (TVirtualGLImp *) h->ExecPlugin(0);
         new TGLKernel(imp);
      }
      init = kTRUE;
   }

   CreateViewer();
}

//______________________________________________________________________________
void TViewerOpenGL::CreateViewer()
{
   // Menus creation
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&Print EPS", kGLPrintEPS_SIMPLE);
   fFileMenu->AddEntry("&Print EPS (High quality)", kGLPrintEPS_BSP);
   fFileMenu->AddEntry("&Print PDF", kGLPrintPDF_SIMPLE);
   fFileMenu->AddEntry("&Print PDF (High quality)", kGLPrintPDF_BSP);
   fFileMenu->AddEntry("&Exit", kGLExit);
   fFileMenu->Associate(this);

   fViewMenu = new TGPopupMenu(fClient->GetRoot());
   fViewMenu->AddEntry("&XOY plane", kGLXOY);
   fViewMenu->AddEntry("XO&Z plane", kGLXOZ);
   fViewMenu->AddEntry("&YOZ plane", kGLYOZ);
   fViewMenu->AddEntry("&Perspective view", kGLPersp);
   fViewMenu->Associate(this);

   fHelpMenu = new TGPopupMenu(fClient->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...", kGLHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("Help on OpenGL Viewer...", kGLHelpOnViewer);
   fHelpMenu->Associate(this);

   // Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   // Create menubar
   fMenuBar = new TGMenuBar(this, 1, 1, kHorizontalFrame | kRaisedFrame);
   fMenuBar->AddPopup("&File", fFileMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Projections", fViewMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);
   AddFrame(fMenuBar, fMenuBarLayout);

   // Frames creation
   fMainFrame = new TGCompositeFrame(this, 100, 100, kHorizontalFrame | kRaisedFrame);
   fV1 = new TGVerticalFrame(fMainFrame, 150, 10, kSunkenFrame | kFixedWidth);
   fShutter = new TGShutter(fV1, kSunkenFrame | kFixedWidth);
   fShutItem1 = new TGShutterItem(fShutter, new TGHotString("Color"), 5001);
   fShutItem2 = new TGShutterItem(fShutter, new TGHotString("Object's geometry"), 5002);
   fShutItem3 = new TGShutterItem(fShutter, new TGHotString("Scene"), 5003);
   fShutItem4 = new TGShutterItem(fShutter, new TGHotString("Lights"), 5004);
   fShutter->AddItem(fShutItem1);
   fShutter->AddItem(fShutItem2);
   fShutter->AddItem(fShutItem3);
   fShutter->AddItem(fShutItem4);

   TGCompositeFrame *shutCont = (TGCompositeFrame *)fShutItem1->GetContainer();
   fColorEditor = new TGLColorEditor(shutCont, this);
   fL4 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY, 2, 5, 1, 2);
   shutCont->AddFrame(fColorEditor, fL4);
   fV1->AddFrame(fShutter, fL4);
   fL1 = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 2, 0, 2, 2);
   fMainFrame->AddFrame(fV1, fL1);

   shutCont = (TGCompositeFrame *)fShutItem2->GetContainer();
   fGeomEditor = new TGLGeometryEditor(shutCont, this);
   shutCont->AddFrame(fGeomEditor, fL4);

   shutCont = (TGCompositeFrame *)fShutItem3->GetContainer();
   fSceneEditor = new TGLSceneEditor(shutCont, this);
   shutCont->AddFrame(fSceneEditor, fL4);

   shutCont = (TGCompositeFrame *)fShutItem4->GetContainer();
   fLightEditor = new TGLLightEditor(shutCont, this);
   shutCont->AddFrame(fLightEditor, fL4);

   fV2 = new TGVerticalFrame(fMainFrame, 10, 10, kSunkenFrame);
   fL3 = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,0,2,2,2);
   fMainFrame->AddFrame(fV2, fL3);

   fCanvasWindow = new TGCanvas(fV2, 10, 10, kSunkenFrame | kDoubleBorder);
   fCanvasContainer = new TGLRenderArea(fCanvasWindow->GetViewPort()->GetId(), fCanvasWindow->GetViewPort());

   TGLWindow * glWin = fCanvasContainer->GetGLWindow();
   glWin->Connect("HandleButton(Event_t*)", "TViewerOpenGL", this, "HandleContainerButton(Event_t*)");
   glWin->Connect("HandleDoubleClick(Event_t*)", "TViewerOpenGL", this, "HandleContainerDoubleClick(Event_t*)");
   glWin->Connect("HandleKey(Event_t*)", "TViewerOpenGL", this, "HandleContainerKey(Event_t*)");
   glWin->Connect("HandleMotion(Event_t*)", "TViewerOpenGL", this, "HandleContainerMotion(Event_t*)");
   glWin->Connect("HandleExpose(Event_t*)", "TViewerOpenGL", this, "HandleContainerExpose(Event_t*)");
   glWin->Connect("HandleConfigureNotify(Event_t*)", "TViewerOpenGL", this, "HandleContainerConfigure(Event_t*)");

   fCanvasWindow->SetContainer(glWin);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fV2->AddFrame(fCanvasWindow, fCanvasLayout);
   AddFrame(fMainFrame, fCanvasLayout);

   SetWindowName("OpenGL experimental viewer");
   SetClassHints("GLViewer", "GLViewer");
   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);
   MapSubwindows();
   Resize(GetDefaultSize());
   MoveResize(fgInitX, fgInitY, fgInitW, fgInitH);
   SetWMPosition(fgInitX, fgInitY);
   Show();
}

//______________________________________________________________________________
TViewerOpenGL::~TViewerOpenGL()
{
   delete fFileMenu;
   delete fViewMenu;
   delete fHelpMenu;
   delete fMenuBar;
   delete fMenuBarLayout;
   delete fMenuBarHelpLayout;
   delete fMenuBarItemLayout;
   delete fCanvasContainer;
   delete fCanvasWindow;
   delete fCanvasLayout;
   delete fV1;
   delete fV2;
   delete fMainFrame;
   delete fL1;
   delete fL2;
   delete fL3;
   delete fL4;
   delete fContextMenu;
   delete fShutter;
   delete fShutItem1;
   delete fShutItem2;
   delete fShutItem3;
   delete fShutItem4;
}

//______________________________________________________________________________
void TViewerOpenGL::InitGL()
{
   // Actual GL window/context creation should have already been done in CreateViewer()
   assert(!fInitGL && fCanvasContainer && fCanvasContainer->GetGLWindow());

   // GL initialisation 
   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);
   glClearColor(0.0, 0.0, 0.0, 0.0);
   glClearDepth(1.0);

   glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
   Float_t lmodelAmb[] = {0.5f, 0.5f, 1.f, 1.f};
   glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodelAmb);
   
   // Calculate light source positions
   // Arrange round an expanded sphere of scene BB
   // TODO: These need to be positioned before each scene draw, after the camera MV translation so that they don't shift relative to objects.
   TGLBoundingBox box = fScene.BoundingBox();
   Double_t radius = box.Extents().Mag() * 4.0;
   
   // 0: Front
   // 1: Right
   // 2: Bottom
   // 3: Left
   // 4: Top   
   Float_t pos0[] = {0.0, 0.0, 0.0, 1.0};
   Float_t pos1[] = {box.Center().X() + radius, box.Center().Y()         , -box.Center().Z() - radius, 1.0};
   Float_t pos2[] = {box.Center().X()         , box.Center().Y() - radius, -box.Center().Z() - radius, 1.0};
   Float_t pos3[] = {box.Center().X() - radius, box.Center().Y()         , -box.Center().Z() - radius, 1.0};
   Float_t pos4[] = {box.Center().X()         , box.Center().Y() + radius, -box.Center().Z() - radius, 1.0};

   Float_t whiteCol[] = {0.7, 0.7, 0.7, 1.0};
   gVirtualGL->GLLight(kLIGHT0, kPOSITION, pos0);
   gVirtualGL->GLLight(kLIGHT0, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT1, kPOSITION, pos1);
   gVirtualGL->GLLight(kLIGHT1, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT2, kPOSITION, pos2);
   gVirtualGL->GLLight(kLIGHT2, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT3, kPOSITION, pos3);
   gVirtualGL->GLLight(kLIGHT3, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT4, kPOSITION, pos4);
   gVirtualGL->GLLight(kLIGHT4, kDIFFUSE, whiteCol);
   
   if (fLightMask & 1) gVirtualGL->EnableGL(kLIGHT4);
   if (fLightMask & 2) gVirtualGL->EnableGL(kLIGHT1);
   if (fLightMask & 4) gVirtualGL->EnableGL(kLIGHT2);
   if (fLightMask & 8) gVirtualGL->EnableGL(kLIGHT3);
   if (fLightMask & 16) gVirtualGL->EnableGL(kLIGHT0);

   TGLUtil::CheckError();
   fInitGL = kTRUE;
}

//______________________________________________________________________________
void TViewerOpenGL::Invalidate(UInt_t redrawLOD)
{
   if (fScene.IsLocked()) {
      Error("TViewerOpenGL::Invalidate", "scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      return;
   }

   TGLViewer::Invalidate(redrawLOD);
   
   // Mark the window as requiring a redraw - the GUI thread
   // will call our DoRedraw() method
   fClient->NeedRedraw(this);

   if (gDebug>3) {
      Info("TViewerOpenGL::Invalidate", "invalidated at %d LOD", fNextSceneLOD);
   }
}

//______________________________________________________________________________
void TViewerOpenGL::MakeCurrent() const
{
   fCanvasContainer->GetGLWindow()->MakeCurrent();
}

//______________________________________________________________________________
void TViewerOpenGL::SwapBuffers() const
{
   if (fScene.CurrentLock() != TGLScene::kDrawLock && 
      fScene.CurrentLock() != TGLScene::kSelectLock) {
      Error("TViewerOpenGL::MakeCurrent", "scene is %s", TGLScene::LockName(fScene.CurrentLock()));   
   }
   fCanvasContainer->GetGLWindow()->SwapBuffers();
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerEvent(Event_t *event)
{
   if (event->fType == kFocusIn) {
      assert(fAction == kNone);
      fAction = kNone;
   }
   if (event->fType == kFocusOut) {
      fAction = kNone;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerButton(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>2) {
         Info("TViewerOpenGL::HandleContainerButton", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   // Only process one action/button down/up pairing - block others
   if (fAction != kNone) {
      if (event->fType == kButtonPress ||
          (event->fType == kButtonRelease && event->fCode != fActiveButtonID)) {
         return kFALSE;
      }
   }
   
   // Button DOWN
   if (event->fType == kButtonPress) {
      Bool_t grabPointer = kFALSE;

      // Record active button for release
      fActiveButtonID = event->fCode;

      // Record mouse start
      fStartPos.fX = fLastPos.fX = event->fX;
      fStartPos.fY = fLastPos.fY = event->fY;
      
      switch(event->fCode) {
         // LEFT mouse button
         case(kButton1): {
            if (event->fState & kKeyShiftMask) {
               DoSelect(event, kFALSE); // without context menu

               // TODO: If no selection start a box select
            } else {
               fAction = kRotate;
               grabPointer = kTRUE;
            }
            break;
         }
         // MID mouse button
         case(kButton2): {
            if (event->fState & kKeyShiftMask) {
               DoSelect(event, kFALSE); // without context menu
               // Start object drag
               if (fScene.GetSelected()) {
                  fAction = kDrag;
                  grabPointer = kTRUE;
               }
            } else {
               fAction = kTruck;
               grabPointer = kTRUE;
            }
            break;
         }
         // RIGHT mouse button
         case(kButton3): {
            // Shift + Right mouse - select+context menu
            if (event->fState & kKeyShiftMask) {
               DoSelect(event, kTRUE); // with context menu
            } else {
               fAction = kDolly;
               grabPointer = kTRUE;
            }
            break;
         }
      }
   }
   // Button UP
   else if (event->fType == kButtonRelease) {
      // TODO: Check on Linux - on Win32 only see button release events
      // for mouse wheel
      switch(event->fCode) {
         // Buttons 4/5 are mouse wheel
         case(kButton4): {
            // Zoom out (adjust camera FOV)
            if (CurrentCamera().Zoom(-30, event->fState & kKeyControlMask, 
                                          event->fState & kKeyShiftMask)) { //TODO : val static const somewhere
               Invalidate();
            }
            break;
         }
         case(kButton5): {
            // Zoom in (adjust camera FOV)
            if (CurrentCamera().Zoom(+30, event->fState & kKeyControlMask, 
                                          event->fState & kKeyShiftMask)) { //TODO : val static const somewhere
               Invalidate();
            }
            break;
         }
      }
      fAction = kNone;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerDoubleClick(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TViewerOpenGL::HandleContainerDoubleClick", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   // Reset interactive camera mode on button double
   // click (unless mouse wheel)
   if (event->fCode != kButton4 && event->fCode != kButton5) {
      CurrentCamera().Reset();
      fStartPos.fX = fLastPos.fX = event->fX;
      fStartPos.fY = fLastPos.fY = event->fY;
      Invalidate();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerConfigure(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TViewerOpenGL::HandleContainerConfigure", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   if (event) {
      SetViewport(event->fX, event->fY, event->fWidth, event->fHeight);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerKey(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TViewerOpenGL::HandleContainerKey", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   char tmp[10] = {0};
   UInt_t keysym = 0;
   Float_t black[] = {0.f, 0.f, 0.f, 1.f};
   Float_t white[] = {1.f, 1.f, 1.f, 1.f};

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);
   
   Bool_t invalidate = kFALSE;

   switch (keysym) {
   case kKey_Plus:
   case kKey_J:
   case kKey_j:
      invalidate = CurrentCamera().Dolly(10, event->fState & kKeyControlMask, 
                                             event->fState & kKeyShiftMask); //TODO : val static const somewhere
      break;
   case kKey_Minus:
   case kKey_K:
   case kKey_k:
      invalidate = CurrentCamera().Dolly(-10, event->fState & kKeyControlMask, 
                                              event->fState & kKeyShiftMask); //TODO : val static const somewhere
      break;
   case kKey_R:
   case kKey_r:
      gVirtualGL->EnableGL(kLIGHTING);
      gVirtualGL->EnableGL(kCULL_FACE);
      gVirtualGL->PolygonGLMode(kFRONT, kFILL);
      gVirtualGL->ClearGLColor(black[0], black[1], black[2], black[3]);
      fScene.SetDrawMode(TGLScene::kFill);
      invalidate = kTRUE;
      break;
   case kKey_W:
   case kKey_w:
      gVirtualGL->DisableGL(kCULL_FACE);
      gVirtualGL->DisableGL(kLIGHTING);
      gVirtualGL->PolygonGLMode(kFRONT_AND_BACK, kLINE);
      gVirtualGL->ClearGLColor(black[0], black[1], black[2], black[3]);
      fScene.SetDrawMode(TGLScene::kWireFrame);
      invalidate = kTRUE;
      break;
   case kKey_T:
   case kKey_t:
      gVirtualGL->EnableGL(kLIGHTING);
      gVirtualGL->EnableGL(kCULL_FACE);
      gVirtualGL->PolygonGLMode(kFRONT, kFILL);
      gVirtualGL->ClearGLColor(white[0], white[1], white[2], white[3]);
      fScene.SetDrawMode(TGLScene::kOutline);
      invalidate = kTRUE;
      break;
   case kKey_Up:
      invalidate = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), 0, 5);
      break;
   case kKey_Down:
      invalidate = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), 0, -5);
      break;
   case kKey_Left:
      invalidate = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), -5, 0);
      break;
   case kKey_Right:
      invalidate = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), 5, 0);
      break;
   // Toggle debugging mode
   case kKey_D:
   case kKey_d:
      fDebugMode = !fDebugMode;
      invalidate = kTRUE;
      Info("OpenGL viewer debug mode : ", fDebugMode ? "ON" : "OFF");
      break;
   // Forced rebuild for debugging mode
   case kKey_Space:
      if (fDebugMode) {
         Info("OpenGL viewer FORCED rebuild", "");
         RebuildScene();
      }
   }

   if (invalidate) {
      Invalidate();
   }
   
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerMotion(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TViewerOpenGL::HandleContainerMotion", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   if (!event) {
      return kFALSE;
   }
   
   Bool_t invalidate = kFALSE;
   
   Int_t xDelta = event->fX - fLastPos.fX;
   Int_t yDelta = event->fY - fLastPos.fY;
   
   // Camera interface requires GL coords - Y inverted
   if (fAction == kRotate) {
      invalidate = CurrentCamera().Rotate(xDelta, -yDelta);
   } else if (fAction == kTruck) {
      invalidate = CurrentCamera().Truck(event->fX, fViewport.Y() - event->fY, xDelta, -yDelta);
   } else if (fAction == kDolly) {
      invalidate = CurrentCamera().Dolly(xDelta, event->fState & kKeyControlMask, 
                                                 event->fState & kKeyShiftMask);
   } else if (fAction == kDrag) {
      TGLPhysicalShape * selected = fScene.GetSelected();
      if (selected) {
         TGLVector3 shift = CurrentCamera().ProjectedShift(selected->BoundingBox().Center(), xDelta, -yDelta);
         selected->Shift(shift);
         fGeomEditor->SetCenter(selected->GetTranslation().CArr());
         fScene.SelectedModified();
         Invalidate();
      }
   }

   fLastPos.fX = event->fX;
   fLastPos.fY = event->fY;
   
   if (invalidate) {
      Invalidate();
   }
   
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerExpose(Event_t *)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TViewerOpenGL::HandleContainerExpose", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   Invalidate(kHigh);
   return kTRUE;
}

//______________________________________________________________________________
void TViewerOpenGL::DoSelect(Event_t *event, Bool_t invokeContext)
{
   // Take select lock on scene immediately we enter here - it is released
   // in the other (drawing) thread - see TGLViewer::Select()
   // Removed when gVirtualGL removed
   if (!fScene.TakeLock(TGLScene::kSelectLock)) {
      return;
   }

   // TODO: Check only the GUI thread ever enters here & DoSelect.
   // Then TVirtualGL and TGLKernel can be obsoleted.
   TGLRect selectRect(event->fX, event->fY, 3, 3); // TODO: Constant somewhere
   gVirtualGL->SelectViewer(this, &selectRect); 
      
   // Do this regardless of whether selection actually changed - safe and 
   // the context menu may need to be invoked anyway
   TGLPhysicalShape * selected = fScene.GetSelected();
   if (selected) {
      fColorEditor->SetRGBA(selected->GetColor());
      fGeomEditor->SetCenter(selected->GetTranslation().CArr());
      fGeomEditor->SetScale(selected->GetScale().CArr());
      if (invokeContext) {
         if (!fContextMenu) fContextMenu = new TContextMenu("glcm", "glcm");
         
         // Defer creating the context menu to the actual object
         selected->InvokeContextMenu(*fContextMenu, event->fXRoot, event->fYRoot);
      }
   } else { // No selection
      fColorEditor->Disable();
      fGeomEditor->Disable();
   }
}

//______________________________________________________________________________
void TViewerOpenGL::Show()
{
   if (fScene.IsLocked()) {
      Error("TViewerOpenGL::Show", "scene is %s", TGLScene::LockName(fScene.CurrentLock()));   
   }
   MapRaised();

   // Must NOT Invalidate() here as for some reason it throws the win32
   // GL kernel impl into a blank locked state? Poss related to having an
   // empty viewer - nothing drawn. TODO: Investigate why....
}

//______________________________________________________________________________
void TViewerOpenGL::CloseWindow() 
{
   fPad->ReleaseViewer3D();   
   TTimer::SingleShot(50, IsA()->GetName(), this, "ReallyDelete()");
}

//______________________________________________________________________________
void TViewerOpenGL::DoRedraw()
{
   // Take draw lock on scene immediately we enter here - it is released
   // in the other (drawing) drawing thread - see TGLViewer::Draw()
   // Removed when gVirtualGL removed
   if (!fScene.TakeLock(TGLScene::kDrawLock)) {
      // If taking drawlock fails the previous draw is still in progress
      // set timer to do this one later
      if (gDebug>3) {
         Info("TViewerOpenGL::DoRedraw", "scene drawlocked - requesting another draw");
      }
      fRedrawTimer->RequestDraw(100, fNextSceneLOD);
      return;
   }

   if (gDebug>3) {
      Info("TViewerOpenGL::DoRedraw", "request draw at %d LOD on this = %d", fNextSceneLOD, this);
   }

   // TODO: Check only the GUI thread ever enters here, DoSelect() and PrintObjects().
   // Then TVirtualGL and TGLKernel can be obsoleted and all GL context work done
   // in GUI thread.
   gVirtualGL->DrawViewer(this);
}

//______________________________________________________________________________
void TViewerOpenGL::PrintObjects()
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TViewerOpenGL::PrintObjects", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return;
   }

   // Generates a PostScript or PDF output of the OpenGL scene. They are vector
   // graphics files and can be huge and long to generate.
    TGLBoundingBox sceneBox = fScene.BoundingBox();
    gVirtualGL->PrintObjects(format, sortgl, this, fCanvasContainer->GetGLWindow(),
                             sceneBox.Extents().Mag(), sceneBox.Center().Y(), sceneBox.Center().Z());
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TViewerOpenGL::ProcessMessage", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   switch (GET_MSG(msg)) {
   case kC_COMMAND:
      switch (GET_SUBMSG(msg)) {
      case kCM_BUTTON:
	   case kCM_MENU:
	      switch (parm1) {
         case kGLHelpAbout: {
            char str[32];
            sprintf(str, "About ROOT %s...", gROOT->GetVersion());
            TRootHelpDialog * hd = new TRootHelpDialog(this, str, 600, 400);
            hd->SetText(gHelpAbout);
            hd->Popup();
            break;
         }
         case kGLHelpOnViewer: {
            TRootHelpDialog * hd = new TRootHelpDialog(this, "Help on GL Viewer...", 600, 400);
            hd->SetText(gHelpViewerOpenGL);
            hd->Popup();
            break;
         }
         case kGLPrintEPS_SIMPLE:
            format = GL2PS_EPS;
	         sortgl = GL2PS_SIMPLE_SORT;
            PrintObjects();
            break;
         case kGLPrintEPS_BSP:
            format = GL2PS_EPS;
	         sortgl = GL2PS_BSP_SORT;
            PrintObjects();
            break;
         case kGLPrintPDF_SIMPLE:
            format = GL2PS_PDF;
	         sortgl = GL2PS_SIMPLE_SORT;
            PrintObjects();
            break;
         case kGLPrintPDF_BSP:
            format = GL2PS_PDF;
	         sortgl = GL2PS_BSP_SORT;
            PrintObjects();
            break;
            case kGLXOY:
            SetCurrentCamera(kXOY);
            break;
         case kGLXOZ:
            SetCurrentCamera(kXOZ);
            break;
         case kGLYOZ:
            SetCurrentCamera(kYOZ);
            break;
        case kGLPersp:
           SetCurrentCamera(kPerspective);
           break;
         case kGLExit:
            CloseWindow();
            break;
	      default:
	         break;
	      }
	   default:
	      break;
      }
   default:
      break;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TViewerOpenGL::ModifyScene(Int_t wid)
{
   if (!fScene.TakeLock(TGLScene::kModifyLock)) {
      return;
   }

   MakeCurrent();

   TGLPhysicalShape * selected = fScene.GetSelected();
   switch (wid) {
   case kTBa:
      if (selected) {
         selected->SetColor(fColorEditor->GetRGBA());
      }
      break;
   case kTBaf:
      if (selected) {
         fScene.SetPhysicalsColorByLogical(selected->GetLogical().ID(), 
                                           fColorEditor->GetRGBA());
      }
      break;
   case kTBa1:
      {
         if (selected) {
            TGLVertex3 trans;
            TGLVector3 scale;
            fGeomEditor->GetObjectData(trans.Arr(), scale.Arr());
            selected->SetTranslation(trans);
            selected->SetScale(scale);
            fScene.SelectedModified();
         }
      }
      break;
   case kTBda:
      fDrawAxes = !fDrawAxes;
      break;
   case kTBcp:
      fUseClipPlane = !fUseClipPlane;
   case kTBcpm:
      {
         Double_t eqn[4] = {0.};
         fSceneEditor->GetPlaneEqn(eqn);
         fClipPlane.Set(eqn, kFALSE); // Don't normalise
         break;
      }
   case kTBTop:
      if ((fLightMask ^= 1) & 1) gVirtualGL->EnableGL(kLIGHT4);
      else gVirtualGL->DisableGL(kLIGHT4);
      break;
   case kTBRight:
      if ((fLightMask ^= 2) & 2) gVirtualGL->EnableGL(kLIGHT1);
      else gVirtualGL->DisableGL(kLIGHT1);
      break;
   case kTBBottom:
      if ((fLightMask ^= 4) & 4) gVirtualGL->EnableGL(kLIGHT2);
      else gVirtualGL->DisableGL(kLIGHT2);
      break;
   case kTBLeft:
      if ((fLightMask ^= 8) & 8) gVirtualGL->EnableGL(kLIGHT3);
      else gVirtualGL->DisableGL(kLIGHT3);
      break;
   case kTBFront:
      if ((fLightMask ^= 16) & 16) gVirtualGL->EnableGL(kLIGHT0);
      else gVirtualGL->DisableGL(kLIGHT0);
      break;
   }

   fScene.ReleaseLock(TGLScene::kModifyLock);

   Invalidate();
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::PreferLocalFrame() const
{
   return kTRUE;
}

//______________________________________________________________________________
void TViewerOpenGL::BeginScene()
{
   if (!fScene.TakeLock(TGLScene::kModifyLock)) {
      return;
   }

   UInt_t destroyedLogicals = 0;
   UInt_t destroyedPhysicals = 0;

   TGLStopwatch stopwatch;
   if (gDebug>2 || fDebugMode) {
      stopwatch.Start();
   }

   // External rebuild?
   if (!fInternalRebuild) 
   {
      // Potentially using external physical IDs
      fInternalPIDs = kFALSE;

      // Reset camera interest to ensure we respond to
      // new scene range
      CurrentCamera().ResetInterest();

      // External rebuilds could potentially invalidate all logical and
      // physical shapes - including any modified physicals
      // Physicals must be removed first
      destroyedPhysicals = fScene.DestroyPhysicals(kTRUE); // include modified
      destroyedLogicals = fScene.DestroyLogicals();

      // Purge out the DL cache - not required once shapes do this themselves properly
      TGLDisplayListCache::Instance().Purge();
   } else {
      // Internal rebuilds - destroy all non-modified physicals no longer of
      // interest to camera - retain logicals
      destroyedPhysicals = fScene.DestroyPhysicals(kFALSE, &CurrentCamera()); // excluded modified
   }

   // Reset internal physical ID counter
   fNextInternalPID = 1;
   
   // Potentially accepting all physicals from external client
   fAcceptedAllPhysicals = kTRUE;

  // Reset tracing info
   fAcceptedPhysicals = 0;
   fRejectedPhysicals = 0;

   if (gDebug>2 || fDebugMode) {
      Info("TViewerOpenGL::BeginScene", "destroyed %d physicals %d logicals in %f msec", 
            destroyedPhysicals, destroyedLogicals, stopwatch.End());
      fScene.Dump();
   }
}

//______________________________________________________________________________
void TViewerOpenGL::EndScene()
{
   fScene.ReleaseLock(TGLScene::kModifyLock);

   // External scene build
   if (!fInternalRebuild) {
      // Setup camera unless scene is empty
      if (!fScene.BoundingBox().IsEmpty()) {
         SetupCameras(fScene.BoundingBox());
      }
      Invalidate();
   } else if (fInternalRebuild) {
      fInternalRebuild = kFALSE;
   }      

   if (gDebug>2 || fDebugMode) {
      Info("TViewerOpenGL::EndScene", "Added %d, rejected %d physicals, accepted all:%s", fAcceptedPhysicals, 
                                       fRejectedPhysicals, fAcceptedAllPhysicals ? "Yes":"No");
      fScene.Dump();
   }
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::RebuildScene()
{
   // If we accepted all offered physicals into the scene no point in 
   // rebuilding it
   if (fAcceptedAllPhysicals) {
      if (gDebug>3 || fDebugMode) {
         Info("TViewerOpenGL::RebuildScene", "not required - all physicals previous accepted");
      }
      return kFALSE;   
   }
   // Update the camera interest (forced in debug mode) - if changed
   // scene should be rebuilt
   if (!CurrentCamera().UpdateInterest(fDebugMode)) {
      if (gDebug>3 || fDebugMode) {
         Info("TViewerOpenGL::RebuildScene", "not required - no camera interest change");
      }
      return kFALSE;
   }
   
   if (gDebug>3 || fDebugMode) {
      Info("TViewerOpenGL::RebuildScene", "required");
   }

   fInternalRebuild = kTRUE;
   
   TGLStopwatch timer;
   if (gDebug>2 || fDebugMode) {
      timer.Start();
   }
      
   // TODO: Just marking modified doesn't seem to result in pad repaint - need to check on
   //fPad->Modified();
   fPad->Paint();

   if (gDebug>2 || fDebugMode) {
      Info("TViewerOpenGL::RebuildScene", "rebuild complete in %f", timer.End());
   }

   // Need to invalidate/redraw via timer as under Win32 we are already inside the 
   // GUI(DoRedraw) thread - direct invalidation will be cleared when leaving
   fRedrawTimer->RequestDraw(20, kMed);

   return kTRUE;
}

//______________________________________________________________________________
Int_t TViewerOpenGL::AddObject(const TBuffer3D & buffer, Bool_t * addChildren)
{
   // Add an object to the viewer, using internal physical IDs

   // If this is called we are generating internal physical IDs
   fInternalPIDs = kTRUE;
   Int_t sections = AddObject(fNextInternalPID, buffer, addChildren);   
   return sections;
}

//______________________________________________________________________________
// TODO: Cleanup addChildren to UInt_t flag for full termination - how returned?
Int_t TViewerOpenGL::AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren)
{
   // Add an object to the viewer, using an external physical ID.

   // TODO: Break this up and make easier to understand. This is pretty convoluted
   // due to the large number of cases it has to deal with:
   // i) Exisiting physical and/or logical
   // ii) External provider can supply bounding box or not?
   // iii) Local/global reference frame
   // iv) Defered filling of some sections of the buffer
   // v) Internal or external physical IDs
   // vi) Composite components as special case
   //
   // The buffer filling means the function is re-entrant which adds to complication 

   if (physicalID == 0) {
      Error("TViewerOpenGL::AddObject", "0 physical ID reserved");
      return TBuffer3D::kNone;
   }

   // Internal and external physical IDs cannot be mixed in a scene build
   if (fInternalPIDs && physicalID != fNextInternalPID) {
      Error("TViewerOpenGL::AddObject", "invalid next physical ID - mix of internal + external IDs?");
      return TBuffer3D::kNone;
   }

   if (addChildren) {
      *addChildren = kFALSE;
   }
   
   // Scene should be modify locked
   if (fScene.CurrentLock() != TGLScene::kModifyLock) {
      Error("TViewerOpenGL::AddObject", "expected scene to be in mofifed locked");
      // TODO: For the moment live with this - DrawOverlap() problems to discuss with Andrei
      // Just reject as pad will redraw anyway
      // assert(kFALSE);
      return TBuffer3D::kNone;
   }
   
   // Note that 'object' here is really a physical/logical pair described
   // in buffer + physical ID.

   // If adding component to a current partial composite do this now
   if (fComposite) {
      RootCsg::BaseMesh *newMesh = RootCsg::ConvertToMesh(buffer);
      // Solaris CC can't create stl pair with enumerate type
      fCSTokens.push_back(std::make_pair(static_cast<UInt_t>(TBuffer3D::kCSNoOp), newMesh));
      return TBuffer3D::kNone;
   }

   // TODO: Could be static and save possible double lookup?
   TGLLogicalShape * logical = fScene.FindLogical(reinterpret_cast<ULong_t>(buffer.fID));
   TGLPhysicalShape * physical = fScene.FindPhysical(physicalID);

   // Function can be called twice if extra buffer filling for logical 
   // is required - record last physical ID to detect
   static UInt_t lastPID = 0;

   // First attempt to add this physical 
   if (physicalID != lastPID) {
      // Existing physical
      if (physical) {
         assert(logical); // Have physical - should have logical
         
         if (addChildren) {
            // For internal PID we request all children even if we will reject them.
            // This ensures PID always represent same external entity.
            if (fInternalPIDs) {
               *addChildren = kTRUE;
            } else 
            // For external PIDs we check child interest as we may have reject children previously
            // with a different camera configuration
            {
               *addChildren = CurrentCamera().OfInterest(physical->BoundingBox());
            }
         }
         
         // Always increment the internal physical ID so they
         // match external object sequence
         if (fInternalPIDs) {
            fNextInternalPID++;
         }

         // We don't need anything more for this object
         return TBuffer3D::kNone; 
      }
      // New physical 
      else {
         // First test interest in camera - requires a bounding box
         TGLBoundingBox box;
         
         // If already have logical use it's BB
         if (logical) {
            box = logical->BoundingBox();
            //assert(!box.IsEmpty());
         }
         // else if bounding box in buffer valid use this
         else if (buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
            box.Set(buffer.fBBVertex);
            //assert(!box.IsEmpty());

         // otherwise we need to use raw points to build a bounding box with
         // If raw sections not set it will be requested by ValidateObjectBuffer
         // below and we will re-enter here
         } else if (buffer.SectionsValid(TBuffer3D::kRaw)) {
            box.SetAligned(buffer.NbPnts(), buffer.fPnts);
            //assert(!box.IsEmpty());
         }
      
         // Box is valid?
         if (!box.IsEmpty()) {
            // Test transformed box with camera
            box.Transform(TGLMatrix(buffer.fLocalMaster));
            Bool_t ofInterest = CurrentCamera().OfInterest(box);
            if (addChildren) {
               // For internal PID we request all children even if we will reject them.
               // This ensures PID always represent same external entity.
               if (fInternalPIDs) {
                  *addChildren = kTRUE;
               } else 
               // For external PID request children if physical of interest
               {
                  *addChildren = ofInterest;
               }
            }            
            // Physical is of interest?
            if (!ofInterest) {
               ++fRejectedPhysicals;
               fAcceptedAllPhysicals = kFALSE;

               // Always increment the internal physical ID so they
               // match external object sequence
               if (fInternalPIDs) {
                  fNextInternalPID++;
               }
               return TBuffer3D::kNone;
            } 
         }
      }

      // Need any extra sections in buffer?
      Int_t extraSections = ValidateObjectBuffer(buffer, 
                                                 logical == 0); // Need logical?
      if (extraSections != TBuffer3D::kNone) {         
         return extraSections;
      } else {
         lastPID = physicalID; // Will not to re-test interest
      }
   }

   if(lastPID != physicalID)
   {
      assert(kFALSE);
   }
   // By now we should need to add a physical at least
   if (physical) {
      assert(kFALSE);
      return TBuffer3D::kNone; 
   }

   // Create logical if required
   if (!logical) {
      assert(ValidateObjectBuffer(buffer,true) == TBuffer3D::kNone); // Buffer should be ready
      logical = CreateNewLogical(buffer);
      if (!logical) { 
         assert(kFALSE);
         return TBuffer3D::kNone;
      }
      // Add logical to scene
      fScene.AdoptLogical(*logical);
   }

   // Finally create the physical, binding it to the logical, and add to scene
   physical = CreateNewPhysical(physicalID, buffer, *logical);

   if (physical) { 
      fScene.AdoptPhysical(*physical);
      ++fAcceptedPhysicals;
      if (gDebug>3 && fAcceptedPhysicals%1000 == 0) {
         Info("TViewerOpenGL::AddObject", "added %d physicals", fAcceptedPhysicals);
      }
   } else {
      assert(kFALSE);
   }

   // Always increment the internal physical ID so they
   // match external object sequence
   if (fInternalPIDs) {
      fNextInternalPID++;
   }

   // Reset last physical ID so can detect new one
   lastPID = 0;
   return TBuffer3D::kNone;
}

//______________________________________________________________________________
Int_t TViewerOpenGL::ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t logical) const
{
   // kCore: Should always be filled
   if (!buffer.SectionsValid(TBuffer3D::kCore)) {
      assert(kFALSE);
      return TBuffer3D::kNone;
   }

   // Currently all physical parts (kBoundingBox / kShapeSpecific) of buffer are 
   // filled automatically if producer can - no need to ask 
   if (!logical) {
      return TBuffer3D::kNone;
   }

   // kRawSizes / kRaw: These are on demand based on shape type
   Bool_t needRaw = kFALSE;

   // We need raw tesselation in these cases:
   //
   // 1. Shape type is NOT kSphere / kTube / kTubeSeg / kCutTube / kComposite
   if (buffer.Type() != TBuffer3DTypes::kSphere  &&
       buffer.Type() != TBuffer3DTypes::kTube    &&
       buffer.Type() != TBuffer3DTypes::kTubeSeg &&
       buffer.Type() != TBuffer3DTypes::kCutTube && 
       buffer.Type() != TBuffer3DTypes::kComposite) {
      needRaw = kTRUE;
   }
   // 2. Sphere type is kSPHE, but the sphere is hollow and/or cut - we
   //    do not support native drawing of these currently
   else if (buffer.Type() == TBuffer3DTypes::kSphere) {
      const TBuffer3DSphere * sphereBuffer = dynamic_cast<const TBuffer3DSphere *>(&buffer);
      if (sphereBuffer) {
         if (!sphereBuffer->IsSolidUncut()) {
            needRaw = kTRUE;
         }
      } else {
         assert(kFALSE);
         return TBuffer3D::kNone;
      }
   }
   // 3. kBoundingBox is not filled - we generate a bounding box from 
   else if (!buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
      needRaw = kTRUE;
   }
   // 3. kShapeSpecific is not filled - except in case of top level composite 
   else if (!buffer.SectionsValid(TBuffer3D::kShapeSpecific) && 
             buffer.Type() != TBuffer3DTypes::kComposite) {
      needRaw = kTRUE;
   }
   // 5. We are a component (not the top level) of a composite shape
   else if (fComposite) {
      needRaw = kTRUE;
   }

   if (needRaw && !buffer.SectionsValid(TBuffer3D::kRawSizes|TBuffer3D::kRaw)) {
      return TBuffer3D::kRawSizes|TBuffer3D::kRaw;
   } else {
      return TBuffer3D::kNone;
   }
}

//______________________________________________________________________________
TGLLogicalShape * TViewerOpenGL::CreateNewLogical(const TBuffer3D & buffer) const
{
   // Buffer should now be correctly filled
   assert(ValidateObjectBuffer(buffer,true) == TBuffer3D::kNone);

   TGLLogicalShape * newLogical = 0;

   switch (buffer.Type()) {
   case TBuffer3DTypes::kLine:
      newLogical = new TGLPolyLine(buffer, buffer.fID);
      break;
   case TBuffer3DTypes::kMarker:
      newLogical = new TGLPolyMarker(buffer, buffer.fID);
      break;
   case TBuffer3DTypes::kSphere: {
      const TBuffer3DSphere * sphereBuffer = dynamic_cast<const TBuffer3DSphere *>(&buffer);
      if (sphereBuffer) {
         // We can only draw solid uncut spheres natively at present
         if (sphereBuffer->IsSolidUncut()) {
            newLogical = new TGLSphere(*sphereBuffer, sphereBuffer->fID);
         } else {
            newLogical = new TGLFaceSet(buffer, buffer.fID);
         }
      }
      else {
         assert(kFALSE);
      }
      break;
   }
   case TBuffer3DTypes::kTube:
   case TBuffer3DTypes::kTubeSeg:
   case TBuffer3DTypes::kCutTube: {
      const TBuffer3DTube * tubeBuffer = dynamic_cast<const TBuffer3DTube *>(&buffer);
      if (tubeBuffer)
      {
         newLogical = new TGLCylinder(*tubeBuffer, tubeBuffer->fID);
      }
      else {
         assert(kFALSE);
      }
      break;
   }
   case TBuffer3DTypes::kComposite: {
      // Create empty faceset and record partial complete composite object
      // Will be populated with mesh in CloseComposite()
      assert(!fComposite);
      fComposite = new TGLFaceSet(buffer, buffer.fID);
      newLogical = fComposite;
      break;
   }
   default:
      newLogical = new TGLFaceSet(buffer, buffer.fID);
      break;
   }

   return newLogical;
}

//______________________________________________________________________________
TGLPhysicalShape * TViewerOpenGL::CreateNewPhysical(UInt_t ID, 
                                                    const TBuffer3D & buffer, 
                                                    const TGLLogicalShape & logical) const
{
   // Extract indexed color from buffer
   // TODO: Still required? Better use proper color triplet in buffer?
   Int_t colorIndex = buffer.fColor;
   if (colorIndex <= 1) colorIndex = 42; //temporary
   Float_t rgba[4] = { 0.0 };
   TColor *rcol = gROOT->GetColor(colorIndex);

   if (rcol) {
      rcol->GetRGB(rgba[0], rgba[1], rgba[2]);
   }
   
   // Extract transparency component - convert to opacity (alpha)
   rgba[3] = 1.f - buffer.fTransparency / 100.f;

   TGLPhysicalShape * newPhysical = new TGLPhysicalShape(ID, logical, buffer.fLocalMaster, 
                                                         buffer.fReflection, rgba);
   return newPhysical;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren)
{
   assert(!fComposite);
   UInt_t extraSections = AddObject(buffer, addChildren);
   assert(extraSections == TBuffer3D::kNone);
   
   // If composite was created it is of interest - we want the rest of the
   // child components   
   if (fComposite) {
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
void TViewerOpenGL::CloseComposite()
{
   // If we have a partially complete composite build it now
   if (fComposite) {
      // TODO: Why is this member and here - only used in BuildComposite()
      fCSLevel = 0;

      RootCsg::BaseMesh *resultMesh = BuildComposite();
      fComposite->SetFromMesh(resultMesh);
      delete resultMesh;
      for (UInt_t i = 0; i < fCSTokens.size(); ++i) delete fCSTokens[i].second;
      fCSTokens.clear();
      fComposite = 0;
   }
}

//______________________________________________________________________________
void TViewerOpenGL::AddCompositeOp(UInt_t operation)
{
   fCSTokens.push_back(std::make_pair(operation, (RootCsg::BaseMesh *)0));
}

//______________________________________________________________________________
RootCsg::BaseMesh *TViewerOpenGL::BuildComposite()
{
   const CSPART_t &currToken = fCSTokens[fCSLevel];
   UInt_t opCode = currToken.first;

   if (opCode != TBuffer3D::kCSNoOp) {
      ++fCSLevel;
      RootCsg::BaseMesh *left = BuildComposite();
      RootCsg::BaseMesh *right = BuildComposite();
      //RootCsg::BaseMesh *result = 0;
      switch (opCode) {
      case TBuffer3D::kCSUnion:
         return RootCsg::BuildUnion(left, right);
      case TBuffer3D::kCSIntersection:
         return RootCsg::BuildIntersection(left, right);
      case TBuffer3D::kCSDifference:
         return RootCsg::BuildDifference(left, right);
      default:
         Error("BuildComposite", "Wrong operation code %d\n", opCode);
         return 0;
      }
   } else return fCSTokens[fCSLevel++].second;
}
