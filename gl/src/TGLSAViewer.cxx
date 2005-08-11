// @(#)root/gl:$Name:  $:$Id: TGLPerspectiveCamera.cxx,v 1.6 2005/07/08 15:39:29 brun Exp $
// Author:  Timur Pocheptsov / Richard Maunder

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLSAViewer.h"
#include "TGLSAFrame.h"
#include "TRootHelpDialog.h"
#include "TContextMenu.h"
#include "KeySymbols.h"
#include "TGShutter.h"
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

#include "TGLEditor.h"
#include "TGLOutput.h"

#include "TGLPhysicalShape.h"

#include <assert.h>

// Remove - replace with TGLManager
#include "TPluginManager.h"
#include "TGLKernel.h"
#include "TGLRenderArea.h"

const char * TGLSAViewer::fgHelpText = "\
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

ClassImp(TGLSAViewer)

const Int_t TGLSAViewer::fgInitX = 0;
const Int_t TGLSAViewer::fgInitY = 0;
const Int_t TGLSAViewer::fgInitW = 780;
const Int_t TGLSAViewer::fgInitH = 670;

//______________________________________________________________________________
TGLSAViewer::TGLSAViewer(TVirtualPad * pad) :
   TGLViewer(pad, fgInitX, fgInitY, fgInitW, fgInitH),
   fFrame(0),
   fCompositeFrame(0), fV1(0), fV2(0), fShutter(0), fShutItem1(0), fShutItem2(0),
   fShutItem3(0), fShutItem4(0), fL1(0), fL2(0), fL3(0), fL4(0),
   fCanvasLayout(0), fMenuBar(0), fFileMenu(0), fViewMenu(0), fHelpMenu(0),
   fMenuBarLayout(0), fMenuBarItemLayout(0), fMenuBarHelpLayout(0),
   fCanvasWindow(0),
   fColorEditor(0), fGeomEditor(0), fSceneEditor(0), fLightEditor(0)
{
   // First create gVirtualGL/kernel - to be replaced with TGLManager
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

   fFrame = new TGLSAFrame(*this);

   // Menus creation
   fFileMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
   fFileMenu->AddEntry("&Print EPS", kGLPrintEPS_SIMPLE);
   fFileMenu->AddEntry("&Print EPS (High quality)", kGLPrintEPS_BSP);
   fFileMenu->AddEntry("&Print PDF", kGLPrintPDF_SIMPLE);
   fFileMenu->AddEntry("&Print PDF (High quality)", kGLPrintPDF_BSP);
   fFileMenu->AddEntry("&Exit", kGLExit);
   fFileMenu->Associate(fFrame);

   fViewMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
   fViewMenu->AddEntry("&XOY plane", kGLXOY);
   fViewMenu->AddEntry("XO&Z plane", kGLXOZ);
   fViewMenu->AddEntry("&YOZ plane", kGLYOZ);
   fViewMenu->AddEntry("&Perspective view", kGLPersp);
   fViewMenu->Associate(fFrame);

   fHelpMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...", kGLHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("Help on OpenGL Viewer...", kGLHelpViewer);
   fHelpMenu->Associate(fFrame);

   // Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   // Create menubar
   fMenuBar = new TGMenuBar(fFrame, 1, 1, kHorizontalFrame | kRaisedFrame);
   fMenuBar->AddPopup("&File", fFileMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Projections", fViewMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);
   fFrame->AddFrame(fMenuBar, fMenuBarLayout);

   // Internal frames creation
   fCompositeFrame = new TGCompositeFrame(fFrame, 100, 100, kHorizontalFrame | kRaisedFrame);
   fV1 = new TGVerticalFrame(fCompositeFrame, 150, 10, kSunkenFrame | kFixedWidth);
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
   fCompositeFrame->AddFrame(fV1, fL1);

   shutCont = (TGCompositeFrame *)fShutItem2->GetContainer();
   fGeomEditor = new TGLGeometryEditor(shutCont, this);
   shutCont->AddFrame(fGeomEditor, fL4);

   shutCont = (TGCompositeFrame *)fShutItem3->GetContainer();
   fSceneEditor = new TGLSceneEditor(shutCont, this);
   shutCont->AddFrame(fSceneEditor, fL4);

   shutCont = (TGCompositeFrame *)fShutItem4->GetContainer();
   fLightEditor = new TGLLightEditor(shutCont, this);
   shutCont->AddFrame(fLightEditor, fL4);

   fV2 = new TGVerticalFrame(fCompositeFrame, 10, 10, kSunkenFrame);
   fL3 = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,0,2,2,2);
   fCompositeFrame->AddFrame(fV2, fL3);

   fCanvasWindow = new TGCanvas(fV2, 10, 10, kSunkenFrame | kDoubleBorder);
   fGLArea = new TGLRenderArea(fCanvasWindow->GetViewPort()->GetId(), fCanvasWindow->GetViewPort());
   fGLWindow = fGLArea->GetGLWindow();

   // Direct events from the TGWindow directly to the base viewer
   Bool_t ok = kTRUE;
   ok = ok && fGLWindow->Connect("ExecuteEvent(Int_t, Int_t, Int_t)", "TGLViewer", this, "ExecuteEvent(Int_t, Int_t, Int_t)");
   ok = ok && fGLWindow->Connect("HandleButton(Event_t*)", "TGLViewer", this, "HandleButton(Event_t*)");
   ok = ok && fGLWindow->Connect("HandleDoubleClick(Event_t*)", "TGLViewer", this, "HandleDoubleClick(Event_t*)");
   ok = ok && fGLWindow->Connect("HandleKey(Event_t*)", "TGLViewer", this, "HandleKey(Event_t*)");
   ok = ok && fGLWindow->Connect("HandleMotion(Event_t*)", "TGLViewer", this, "HandleMotion(Event_t*)");
   ok = ok && fGLWindow->Connect("HandleExpose(Event_t*)", "TGLViewer", this, "HandleExpose(Event_t*)");
   ok = ok && fGLWindow->Connect("HandleConfigureNotify(Event_t*)", "TGLViewer", this, "HandleConfigureNotify(Event_t*)");
   assert(ok);

   fCanvasWindow->SetContainer(fGLWindow);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fV2->AddFrame(fCanvasWindow, fCanvasLayout);
   fFrame->AddFrame(fCompositeFrame, fCanvasLayout);

   fFrame->SetWindowName("OpenGL experimental viewer");
   fFrame->SetClassHints("GLViewer", "GLViewer");
   fFrame->SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);
   fFrame->MapSubwindows();
   fFrame->Resize(fFrame->GetDefaultSize());
   fFrame->MoveResize(fgInitX, fgInitY, fgInitW, fgInitH);
   fFrame->SetWMPosition(fgInitX, fgInitY);

   Show();
}

//______________________________________________________________________________
TGLSAViewer::~TGLSAViewer()
{
   delete fFileMenu;
   delete fViewMenu;
   delete fHelpMenu;
   delete fMenuBar;
   delete fMenuBarLayout;
   delete fMenuBarHelpLayout;
   delete fMenuBarItemLayout;
   delete fGLArea;
   delete fCanvasWindow;
   delete fCanvasLayout;
   delete fV1;
   delete fV2;
   delete fCompositeFrame;
   delete fL1;
   delete fL2;
   delete fL3;
   delete fL4;
   delete fShutter;
   delete fShutItem1;
   delete fShutItem2;
   delete fShutItem3;
   delete fShutItem4;
   delete fFrame;
}

//______________________________________________________________________________
void TGLSAViewer::Show()
{
   fFrame->MapRaised();
   RequestDraw();
}

//______________________________________________________________________________
void TGLSAViewer::Close()
{
   // Commit suicide when contained GUI is closed
   delete this;
}

//______________________________________________________________________________
Bool_t TGLSAViewer::ProcessFrameMessage(Long_t msg, Long_t parm1, Long_t)
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
            TRootHelpDialog * hd = new TRootHelpDialog(fFrame, str, 600, 400);
            hd->SetText(gHelpAbout);
            hd->Popup();
            break;
         }
         case kGLHelpViewer: {
            TRootHelpDialog * hd = new TRootHelpDialog(fFrame, "Help on GL Viewer...", 600, 400);
            hd->SetText(fgHelpText);
            hd->Popup();
            break;
         }
         case kGLPrintEPS_SIMPLE:
            gVirtualGL->CaptureViewer(this, TGLOutput::kEPS_SIMPLE);
            break;
         case kGLPrintEPS_BSP:
            gVirtualGL->CaptureViewer(this, TGLOutput::kEPS_BSP);
            break;
         case kGLPrintPDF_SIMPLE:
            gVirtualGL->CaptureViewer(this, TGLOutput::kPDF_SIMPLE);
            break;
         case kGLPrintPDF_BSP:
            gVirtualGL->CaptureViewer(this, TGLOutput::kPDF_BSP);
            break;
         case kGLXOY:
            SetCurrentCamera(TGLViewer::kCameraXOY);
            break;
         case kGLXOZ:
            SetCurrentCamera(TGLViewer::kCameraXOZ);
            break;
         case kGLYOZ:
            SetCurrentCamera(TGLViewer::kCameraYOZ);
            break;
        case kGLPersp:
           SetCurrentCamera(TGLViewer::kCameraPerspective);
           break;
         case kGLExit:
            // Exit needs to be delayed to avoid bad drawable X ids - GUI
            // will all be changed in future anyway
            TTimer::SingleShot(50, "TGLSAFrame", fFrame, "SendCloseMessage()");
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
void TGLSAViewer::ProcessGUIEvent(Int_t wid)
{
   switch (wid) {
   case kTBa:
      SetSelectedColor(fColorEditor->GetRGBA());
      break;
   case kTBaf:
      SetColorOnSelectedFamily(fColorEditor->GetRGBA());
      break;
   case kTBa1: {
      TGLVertex3 trans;
      TGLVector3 scale;
      fGeomEditor->GetObjectData(trans.Arr(), scale.Arr());
      SetSelectedGeom(trans,scale);
      break;
   }
   case kTBda:
      ToggleAxes();
      break;
   case kTBcp:
      ToggleClip();
   case kTBcpm: {
      TGLPlane eqn;
      fSceneEditor->GetPlaneEqn(eqn.Arr());
      SetClipPlaneEq(eqn); // Don't normalise
      break;
   }
   case kTBFront:
      ToggleLight(TGLViewer::kLightFront);
      break;
   case kTBTop:
      ToggleLight(TGLViewer::kLightTop);
      break;
   case kTBBottom:
      ToggleLight(TGLViewer::kLightBottom);
      break;
   case kTBRight:
      ToggleLight(TGLViewer::kLightRight);
      break;
   case kTBLeft:
      ToggleLight(TGLViewer::kLightLeft);
      break;
   }
}

//______________________________________________________________________________
void TGLSAViewer::SelectionChanged()
{
   // Update GUI components for embedded viewer selection change

   const TGLPhysicalShape * selected = GetSelected();
   if (selected) {
      fColorEditor->SetRGBA(selected->GetColor());
      fGeomEditor->SetCenter(selected->GetTranslation().CArr());
      fGeomEditor->SetScale(selected->GetScale().CArr());
   } else { // No selection
      fColorEditor->Disable();
      fGeomEditor->Disable();
   }
}
