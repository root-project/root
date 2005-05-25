// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.cxx,v 1.55 2005/04/06 09:43:39 brun Exp $
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
//#include "TGLRender.h"
#include "TGLCamera.h"
//#include "TArcBall.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TVirtualPad.h"

#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TObject.h"

#include "gl2ps.h"

#include "Riostream.h" // TODO: Remove

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
                   fInternalRebuild(kFALSE), fBuildingScene(kFALSE),
                   fPad(pad), fComposite(0)
{
   // Create OpenGL viewer.

   //good compiler (not VC 6.0) will initialize our
   //arrays in ctor-init-list with zeroes (default initialization)

   // TODO: Resort !!
   fMainFrame = 0;
   fV2 = 0;
   fV1 = 0;
   fColorEditor = 0;
   fGeomEditor = 0;
   fSceneEditor = 0;
   fCanvasWindow = 0;
   fCanvasContainer = 0;
   fL1 = fL2 = fL3 = fL4 = 0;
   fCanvasLayout = 0;
   fMenuBar = 0;
   fFileMenu = fViewMenu = fHelpMenu = 0;
   fMenuBarLayout = fMenuBarItemLayout = fMenuBarHelpLayout = 0;
   fLightMask = 0x1b;
   //fXc = fYc = fZc = fRad = 0.;
   fAction = kNone;
   fActiveButtonID = 0;
   //fConf = kPERSP;
   fContextMenu = 0;
   //fSelectedObj = 0;

   fNextPhysicalID = 1;

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
   assert(!fBuildingScene);

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
   TGLUtil::CheckError();
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
   //delete fArcBall;
   //delete fRender;
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
   TGLViewer::Invalidate(redrawLOD);
   
   // Mark the window as requiring a redraw - the GUI thread
   // will call our DoRedraw() method
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TViewerOpenGL::MakeCurrent() const
{
   assert(!fBuildingScene);
   fCanvasContainer->GetGLWindow()->MakeCurrent();
}

//______________________________________________________________________________
void TViewerOpenGL::SwapBuffers() const
{
   assert(!fBuildingScene);
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
            } else {
               fAction = kRotate;
               grabPointer = kTRUE;
            }
            break;
         }
         // MID mouse button
         case(kButton2): {
            if (!(event->fState & kKeyShiftMask)) {
               fAction = kTruck;
               grabPointer = kTRUE;
            }
            break;
         }
         // RIGHT mouse button
         case(kButton3): {
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
            // Dolly out
            if (CurrentCamera().Dolly(-30)) { //TODO : val static const somewhere
               Invalidate();
            }
            break;
         }
         case(kButton5): {
            // Dolly in
            if (CurrentCamera().Dolly(+30)) { //TODO : val static const somewhere
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
   if (event) {
      SetViewport(event->fX, event->fY, event->fWidth, event->fHeight);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerKey(Event_t *event)
{
   char tmp[10] = {0};
   UInt_t keysym = 0;

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);
   
   Bool_t invalidate = kFALSE;

   switch (keysym) {
   case kKey_Plus:
   case kKey_J:
   case kKey_j:
      invalidate = CurrentCamera().Dolly(10); //TODO : val static const somewhere
      break;
   case kKey_Minus:
   case kKey_K:
   case kKey_k:
      invalidate = CurrentCamera().Dolly(-10); //TODO : val static const somewhere
      break;
   case kKey_R:
   case kKey_r:
      gVirtualGL->PolygonGLMode(kFRONT, kFILL);
      gVirtualGL->EnableGL(kCULL_FACE);
      gVirtualGL->SetGLLineWidth(1.f);
      invalidate = kTRUE;
      break;
   case kKey_W:
   case kKey_w:
      gVirtualGL->DisableGL(kCULL_FACE);
      gVirtualGL->PolygonGLMode(kFRONT_AND_BACK, kLINE);
      gVirtualGL->SetGLLineWidth(1.5f);
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
   }

   if (invalidate) {
      Invalidate();
   }
   
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerMotion(Event_t *event)
{
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
      invalidate = CurrentCamera().Dolly(xDelta);
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
   Invalidate(kHigh);
   return kTRUE;
}

//______________________________________________________________________________
void TViewerOpenGL::DoSelect(Event_t *event, Bool_t invokeContext)
{
   // TODO: Check only the GUI thread ever enters here & DoSelect.
   // Then TVirtualGL and TGLKernel can be obsoleted.
   TGLRect selectRect(event->fX, event->fY, 3, 3); // TODO: Constant somewhere
   gVirtualGL->SelectViewer(this, &selectRect); 
      
   // Do this regardless of whether selection actually changed - safe and 
   // the context menu may need to be invoked anyway
   TGLPhysicalShape * selected = fScene.GetSelected();
   if (selected) {
      fColorEditor->SetRGBA(selected->GetColor());
      fGeomEditor->SetCenter(selected->BoundingBox().Center().CArr());
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
   assert(!fBuildingScene);
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
   //TGMainFrame::DoRedraw();
   // TODO: Check only the GUI thread ever enters here, DoSelect() and PrintObjects().
   // Then TVirtualGL and TGLKernel can be obsoleted and all GL context work done
   // in GUI thread.
   gVirtualGL->DrawViewer(this);
}

//______________________________________________________________________________
/*void TViewerOpenGL::DrawObjects()const
{
   assert(!fBuildingScene);
   MakeCurrent();
   gVirtualGL->NewMVGL();
   gVirtualGL->TraverseGraph((TGLRender *)fRender);
   SwapBuffers();
}*/

//______________________________________________________________________________
void TViewerOpenGL::PrintObjects()
{
   // Generates a PostScript or PDF output of the OpenGL scene. They are vector
   // graphics files and can be huge and long to generate.
    TGLBoundingBox sceneBox = fScene.BoundingBox();
    gVirtualGL->PrintObjects(format, sortgl, this, fCanvasContainer->GetGLWindow(),
                             sceneBox.Extents().Mag(), sceneBox.Center().Y(), sceneBox.Center().Z());
}

//______________________________________________________________________________
/*void TViewerOpenGL::UpdateRange(const TGLSelection *box)
{
   assert(fBuildingScene);
   const Double_t *X = box->GetRangeX();
   const Double_t *Y = box->GetRangeY();
   const Double_t *Z = box->GetRangeZ();

   if (!fRender->GetSize()) {
      fRangeX.first = X[0], fRangeX.second = X[1];
      fRangeY.first = Y[0], fRangeY.second = Y[1];
      fRangeZ.first = Z[0], fRangeZ.second = Z[1];
      return;
   }

   if (fRangeX.first > X[0])
      fRangeX.first = X[0];
   if (fRangeX.second < X[1])
      fRangeX.second = X[1];
   if (fRangeY.first > Y[0])
      fRangeY.first = Y[0];
   if (fRangeY.second < Y[1])
      fRangeY.second = Y[1];
   if (fRangeZ.first > Z[0])
      fRangeZ.first = Z[0];
   if (fRangeZ.second < Z[1])
      fRangeZ.second = Z[1];
}*/

//______________________________________________________________________________
Bool_t TViewerOpenGL::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
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
/*TGLSceneObject *TViewerOpenGL::TestSelection(Event_t *event)
{
   MakeCurrent();
   TGLSceneObject *obj = gVirtualGL->SelectObject(fRender, event->fX, event->fY, fConf);
   SwapBuffers();

   return obj;
}*/

//______________________________________________________________________________
/*void TViewerOpenGL::CalculateViewports()
{
   fActiveViewport[0] = 0;
   fActiveViewport[1] = 0;
   fActiveViewport[2] = fCanvasWindow->GetWidth();
   fActiveViewport[3] = fCanvasWindow->GetHeight();
}*/

//______________________________________________________________________________
/*void TViewerOpenGL::CalculateViewvolumes()
{
   if (fRender->GetSize()) {
      Double_t xdiff = fRangeX.second - fRangeX.first;
      Double_t ydiff = fRangeY.second - fRangeY.first;
      Double_t zdiff = fRangeZ.second - fRangeZ.first;
      Double_t max = xdiff > ydiff ? xdiff > zdiff ? xdiff : zdiff : ydiff > zdiff ? ydiff : zdiff;

      Int_t w = fCanvasWindow->GetWidth() / 2;
      Int_t h = (fCanvasWindow->GetHeight()) / 2;
      Double_t frx = 1., fry = 1.;

      if (w > h)
         frx = w / double(h);
      else if (w < h)
         fry = h / double(w);

      fViewVolume[0] = max / 1.9 * frx;
      fViewVolume[1] = max / 1.9 * fry;
      fViewVolume[2] = max * 0.707;
      fViewVolume[3] = 3 * max;
      fRad = max * 1.7;
   }
}*/

//______________________________________________________________________________
/*void TViewerOpenGL::CreateCameras()
{
   if (!fRender->GetSize())
      return;

   TGLSimpleTransform trXOY(gRotMatrixXOY, fRad, &fXc, &fYc, &fZc);
   TGLSimpleTransform trXOZ(gRotMatrixXOZ, fRad, &fXc, &fYc, &fZc);
   TGLSimpleTransform trYOZ(gRotMatrixYOZ, fRad, &fXc, &fYc, &fZc);
   TGLSimpleTransform trPersp(fArcBall->GetRotMatrix(), fRad, &fXc, &fYc, &fZc);

   fCamera[kXOY]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trXOY);
   fCamera[kXOZ]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trXOZ);
   fCamera[kYOZ]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trYOZ);
   fCamera[kPERSP] = new TGLPerspectiveCamera(fViewVolume, fActiveViewport, trPersp);

   fRender->AddNewCamera(fCamera[kXOY]);
   fRender->AddNewCamera(fCamera[kXOZ]);
   fRender->AddNewCamera(fCamera[kYOZ]);
   fRender->AddNewCamera(fCamera[kPERSP]);
}*/

//______________________________________________________________________________
void TViewerOpenGL::ModifyScene(Int_t wid)
{
   // TODO: Re-enable these bits
   MakeCurrent();
   switch (wid) {
   case kTBa:
      fScene.GetSelected()->SetColor(fColorEditor->GetRGBA());
      break;
   case kTBaf:
      //fRender->SetFamilyColor(fColorEditor->GetRGBA());
      break;
   case kTBa1:
      {
         Double_t c[3] = {0.};
         Double_t s[] = {1., 1., 1.};
         fGeomEditor->GetObjectData(c, s);
         //fSelectedObj->Stretch(s[0], s[1], s[2]);
         //fSelectedObj->Shift(c[0], c[1], c[2]);
      }
      break;
   case kTBda:
      //fRender->ResetAxes();
      break;
   case kTBcp:
      //if (fRender->ResetPlane()) gVirtualGL->EnableGL(kCLIP_PLANE0);
      //else gVirtualGL->DisableGL(kCLIP_PLANE0);
   case kTBcpm:
      {
         //Double_t eqn[4] = {0.};
         //fSceneEditor->GetPlaneEqn(eqn);
         //fRender->SetPlane(eqn);
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

   Invalidate();
}

//______________________________________________________________________________
/*void TViewerOpenGL::MoveCenter(Int_t key)
{
   fRender->SetNeedFrustum();
   Double_t shift[3] = {0.};
   Double_t steps[2] = {fViewVolume[0] * fZoom[0] / 40, fViewVolume[0] * fZoom[0] / 40};

   switch (key) {
   case kKey_Left:
      shift[0] = steps[0];
      break;
   case kKey_Right:
      shift[0] = -steps[0];
      break;
   case kKey_Up:
      shift[1] = -steps[1];
      break;
   case kKey_Down:
      shift[1] = steps[1];
      break;
   }

   const Double_t *rotM = fArcBall->GetRotMatrix();
   Double_t matrix[3][4] = {{rotM[0], -rotM[8], rotM[4], shift[0]},
                            {rotM[1], -rotM[9], rotM[5], shift[1]},
                            {rotM[2], -rotM[10], rotM[6], 0.}};

   TToySolver tr(*matrix);
   tr.GetSolution(shift);
   fXc += shift[0];
   fYc += shift[1];
   fZc += shift[2];

   DrawObjects();
}*/

//______________________________________________________________________________
Bool_t TViewerOpenGL::PreferLocalFrame() const
{
   return kTRUE;
}

//______________________________________________________________________________
void TViewerOpenGL::BeginScene()
{
   TGLUtil::CheckError();

   // Scene builds can't be nested
   if (fBuildingScene) {
      assert(kFALSE);
      return;
   }
   
   if (!fInternalRebuild) {
      CurrentCamera().ResetInterest();
      fScene.DestroyAllPhysicals();
   }
      
   fBuildingScene = kTRUE;

   TGLUtil::CheckError();
}

//______________________________________________________________________________
void TViewerOpenGL::EndScene()
{
   TGLUtil::CheckError();

   assert(fBuildingScene);
   fBuildingScene = kFALSE;
  
   if (!fInternalRebuild && !fScene.BoundingBox().IsEmpty()) {
      SetupCameras(fScene.BoundingBox());
   } else {
      fInternalRebuild = kFALSE;
   }      
    
   Invalidate();
   
   TGLUtil::CheckError();
}

//______________________________________________________________________________
void TViewerOpenGL::RebuildScene()
{   
   fInternalRebuild = kTRUE;
   
   // Destroy physicals no longer of interest to current camera
   // TODO: Only do this if above a certain number of physicals (or mem size)
   // otherwise why not just keep them all?
   //fScene.DestroyPhysicals(CurrentCamera());
   fScene.DestroyAllPhysicals();
   fNextPhysicalID = 1;
      
   // TODO: Just marking modified doesn't seem to result in pad repaint - need to check on
   //fPad->Modified();
   fPad->Paint();
}

//______________________________________________________________________________
Int_t TViewerOpenGL::AddObject(const TBuffer3D & buffer, Bool_t * addChildren)
{
   Int_t sections = AddObject(fNextPhysicalID, buffer, addChildren);   
   return sections;
}

//______________________________________________________________________________
// TODO: Cleanup addChildren to UInt_t flag for full termination - how returned?
Int_t TViewerOpenGL::AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren)
{
   if (addChildren) {
      *addChildren = kFALSE;
   }
   
   // 0 reserved for detecting re-entry
   assert(physicalID != 0);
   
   // Currently we protect against add objects being added outside a scene build 
   // as there are problems with mutliple 3D viewers on pad
   if (!fBuildingScene) {
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
   TGLLogicalShape * logical = fScene.FindLogical(reinterpret_cast<UInt_t>(buffer.fID));
   TGLPhysicalShape * physical = fScene.FindPhysical(physicalID);

   // Function can be called twice if extra buffer filling for logical 
   // is required. TODO: Could mess up as rebuild scene starting with last used physical...
   static UInt_t lastPhysicalID = 0;

   // First attempt to add this object
   if (physicalID != lastPhysicalID) {
      
      // If we already have physical we do not need this object
      if (physical) {
         assert(logical); // Have physical - should have logical
         
         // We still check child interest as we may have reject children previously
         // with a different camera configuration
         if (addChildren) {
            *addChildren = CurrentCamera().OfInterest(physical->BoundingBox());
         }
         
         // Either way we don't need anything more for this object
         return TBuffer3D::kNone; 
      }
      // New physical 
      else {
         // First test interest in camera - requires a bounding box
         TGLBoundingBox box;
         
         // If already have logical use it's BB
         if (logical) {
            box = logical->BoundingBox();
         }
         // else if bounding box in buffer valid use this
         else if (buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
            box.Set(buffer.fBBVertex);
         // otherwise we need to use raw points to build a bounding box with
         // If raw sections not set it will be requested by ValidateObjectBuffer
         // below and we will re-enter here
         } else if (buffer.SectionsValid(TBuffer3D::kRaw)) {
            box.SetAligned(buffer.NbPnts(), buffer.fPnts);
         }
      
         // Box is valid?
         if (!box.IsEmpty()) {
            // Test transformed box with camera
            box.Transform(TGLMatrix(buffer.fLocalMaster));
            Bool_t ofInterest = CurrentCamera().OfInterest(box);
            if (addChildren) {
               *addChildren = ofInterest;
            }            
            if (!ofInterest) {
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
         lastPhysicalID = physicalID;
      }  
   }

   assert(lastPhysicalID == physicalID);
   
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
   fNextPhysicalID++;
   
   if (physical) { 
      fScene.AdoptPhysical(*physical);
   } else {
      assert(kFALSE);
   }

   lastPhysicalID = 0;
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
   TGLPhysicalShape * newPhysical = new TGLPhysicalShape(ID, logical, buffer.fLocalMaster, buffer.fReflection);
   if (newPhysical) {
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

      // Setup the colors on new physical
      newPhysical->SetColor(rgba);
   }
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
