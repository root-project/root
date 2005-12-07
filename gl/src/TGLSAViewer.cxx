// @(#)root/gl:$Name:  $:$Id: TGLSAViewer.cxx,v 1.11 2005/12/05 17:34:45 brun Exp $
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
#include "TGTab.h"

#include "TGLEditor.h"
#include "TGLOutput.h"

#include "TGLPhysicalShape.h"
#include "TGLClip.h"

#ifdef WIN32
#include "TWin32SplashThread.h"
#endif


#include <assert.h>

// Remove - replace with TGLManager
#include "TPluginManager.h"
#include "TGLKernel.h"
#include "TGLRenderArea.h"

const char * TGLSAViewer::fgHelpText = "\
DIRECT SCENE INTERACTIONS\n\n\
   Press:\n\
   \tw          --- wireframe mode\n\
   \tr          --- filled polygons mode\n\
   \tt          --- outline mode\n\
   \tj          --- ZOOM in\n\
   \tk          --- ZOOM out\n\
   \tArrow Keys --- PAN (TRUCK) across scene\n\n\
   You can ROTATE (ORBIT) the scene by holding the left mouse button and moving\n\
   the mouse (perspective camera only).\n\n\
   You can PAN (TRUCK) the camera using the middle mouse button or arrow keys.\n\n\
   You can ZOOM (DOLLY) the camera by dragging side to side holding the right\n\
   mouse button or using the mouse wheel.\n\n\
   RESET the camera by double clicking any button.\n\n\
   SELECT a shape with Shift+Left mouse button click.\n\n\
   MOVE a selected shape using Shift+Mid mouse drag.\n\n\
   Invoke the CONTEXT menu with Shift+Right mouse click.\n\n\
CAMERA\n\n\
   The \"Camera\" menu is used to select the different projections from \n\
   the 3D world onto the 2D viewport. There are three perspective cameras:\n\n\
   \tPerspective (Floor XOZ)\n\
   \tPerspective (Floor YOZ)\n\
   \tPerspective (Floor XOY)\n\n\
   In each case the floor plane (defined by two axes) is kept level.\n\n\
   There are also three orthographic cameras:\n\n\
   \tOrthographic (XOY)\n\
   \tOrthographic (XOZ)\n\
   \tOrthographic (ZOY)\n\n\
   In each case the first axis is placed horizontal, the second vertical e.g.\n\
   XOY means X horizontal, Y vertical.\n\n\
SHAPES COLOR AND MATERIAL\n\n\
   The selected shape's color can be modified in the Shapes-Color tabs.\n\
   Shape's color is specified by the percentage of red, green, blue light\n\
   it reflects. A surface can reflect DIFFUSE, AMBIENT and SPECULAR light.\n\
   A surface can also emit light. The EMISSIVE parameter allows to define it.\n\
   The surface SHININESS can also be modified.\n\n\
SHAPES GEOMETRY\n\n\
   The selected shape's location and geometry can be modified in the Shapes-Geom\n\
   tabs by entering desired values in respective number entry controls.\n\n"
"  SCENE CLIPPING\n\n\
   In the Scene-Clipping tabs select a 'Clip Type': None, Plane, Box\n\n\
   For 'Plane' and 'Box' the lower pane shows the relevant parameters:\n\n\
\tPlane: Equation coefficients of form aX + bY + cZ + d = 0\n\
\tBox: Center X/Y/Z and Length X/Y/Z\n\n\
   For Box checking the 'Show / Edit' checkbox shows the clip box (in light blue)\n\
   in viewer. It also attaches the current manipulator to the box - enabling\n\
   direct editing in viewer.\n\n\
MANIPULATORS\n\n\
   A widget attached to the selected object - allowing direct manipulation\n\
   of the object with respect to its local axes.\n\
   There are three modes, toggled with keys:\n\
   \tMode\t\tWidget Component Style\t\tKey\n\
   \t----\t\t----------------------\t\t---\n\
   \tTranslation\tLocal axes with arrows\t\tv\n\
   \tScale\t\tLocal axes with boxes\t\tx\n\
   \tRotate\t\tLocal axes rings\t\tc\n\n\
   Each widget has three axis components - red (X), green (Y) and blue (Z).\n\
   The component turns yellow, indicating an active state, when the mouse is moved\n\
   over it. Left click and drag on the active component to adjust the objects\n\
   translation, scale or rotation.\n\
   Some objects do not support all manipulations (e.g. clipping planes cannot be \n\
   scaled). If a manipulation is not permitted the component it drawn in grey and \n\
   cannot be selected/dragged.\n";


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLSAViewer                                                          //
//                                                                      //
// The top level standalone viewer object - created via plugin manager. //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLSAViewer)

const Int_t TGLSAViewer::fgInitX = 0;
const Int_t TGLSAViewer::fgInitY = 0;
const Int_t TGLSAViewer::fgInitW = 780;
const Int_t TGLSAViewer::fgInitH = 670;

//______________________________________________________________________________
TGLSAViewer::TGLSAViewer(TVirtualPad * pad) :
   TGLViewer(pad, fgInitX, fgInitY, fgInitW, fgInitH),
   fFrame(0),
   fCompositeFrame(0), fV1(0), fV2(0), fL1(0), fL2(0), fL3(0),
   fCanvasLayout(0), fMenuBar(0), fFileMenu(0), fCameraMenu(0), fHelpMenu(0),
   fMenuBarLayout(0), fMenuBarItemLayout(0), fMenuBarHelpLayout(0),
   fCanvasWindow(0),
   fEditorTab(0), fShapesTab(0), fSceneTab(0),
   fColorEditor(0), fGeomEditor(0), fClipEditor(0), fLightEditor(0), fGuideEditor(0)
{
   // Construct a standalone viewer, bound to supplied 'pad'.
   
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
   fFileMenu->AddEntry("Print &EPS", kGLPrintEPS_SIMPLE);
   fFileMenu->AddEntry("Print EP&S (High quality)", kGLPrintEPS_BSP);
   fFileMenu->AddEntry("Print &PDF", kGLPrintPDF_SIMPLE);
   fFileMenu->AddEntry("Print P&DF (High quality)", kGLPrintPDF_BSP);
   fFileMenu->AddEntry("E&xit", kGLExit);
   fFileMenu->Associate(fFrame);

   fCameraMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
   fCameraMenu->AddEntry("Perspective (Floor XOZ)", kGLPerspXOZ);
   fCameraMenu->AddEntry("Perspective (Floor YOZ)", kGLPerspYOZ);
   fCameraMenu->AddEntry("Perspective (Floor XOY)", kGLPerspXOY);
   fCameraMenu->AddEntry("Orthographic (XOY)", kGLXOY);
   fCameraMenu->AddEntry("Orthographic (XOZ)", kGLXOZ);
   fCameraMenu->AddEntry("Orthographic (ZOY)", kGLZOY);
   fCameraMenu->Associate(fFrame);

   fHelpMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
   fHelpMenu->AddEntry("Help on GL Viewer...", kGLHelpViewer);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("&About ROOT...", kGLHelpAbout);
   fHelpMenu->Associate(fFrame);

   // Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   // Create menubar
   fMenuBar = new TGMenuBar(fFrame, 1, 1, kHorizontalFrame | kRaisedFrame);
   fMenuBar->AddPopup("&File", fFileMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Camera", fCameraMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);
   fFrame->AddFrame(fMenuBar, fMenuBarLayout);

   // Internal frames creation
   fCompositeFrame = new TGCompositeFrame(fFrame, 100, 100, kHorizontalFrame | kRaisedFrame);
   fV1 = new TGVerticalFrame(fCompositeFrame, 180, 10, /*kSunkenFrame |*/ kFixedWidth);
   fEditorTab = new TGTab(fV1, 180, 10);
   fL3 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY, 2, 2, 1, 2);
   fV1->AddFrame(fEditorTab, fL3);
   fL1 = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 2, 0, 2, 2);
   fCompositeFrame->AddFrame(fV1, fL1);
   
   // Scene main tab
   TGCompositeFrame *sceneTabFrame = fEditorTab->AddTab("Scene");
   fSceneTab = new TGTab(sceneTabFrame, 160, 10);
   sceneTabFrame->AddFrame(fSceneTab, fL3);

   // Scene / Clipping subtab
   TGCompositeFrame *tabCont = fSceneTab->AddTab("Clipping");
   fClipEditor = new TGLClipEditor(tabCont, this);
   tabCont->AddFrame(fClipEditor, fL3);

   // Scene / Lighting subtab
   tabCont = fSceneTab->AddTab("Lights");
   fLightEditor = new TGLLightEditor(tabCont, this);
   tabCont->AddFrame(fLightEditor, fL3);

   // Scene / Guides subtab
   tabCont = fSceneTab->AddTab("Guides");
   fGuideEditor = new TGLGuideEditor(tabCont, this);
   tabCont->AddFrame(fGuideEditor, fL3);

   // Shapes main tab
   TGCompositeFrame *objTabFrame = fEditorTab->AddTab("Shapes");
   fShapesTab = new TGTab(objTabFrame, 160, 10);
   objTabFrame->AddFrame(fShapesTab, fL3);
   
   // Shapes / Color subtab
   tabCont = fShapesTab->AddTab("Color");
   fColorEditor = new TGLColorEditor(tabCont, this);
   tabCont->AddFrame(fColorEditor, fL3);

   // Shapes / Geom subtab
   tabCont = fShapesTab->AddTab("Geom");
   fGeomEditor = new TGLGeometryEditor(tabCont, this);
   tabCont->AddFrame(fGeomEditor, fL3);

   fV2 = new TGVerticalFrame(fCompositeFrame, 10, 10, kSunkenFrame);
   fL2 = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,0,2,2,2);
   fCompositeFrame->AddFrame(fV2, fL2);

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

   fFrame->SetWindowName("ROOT's GL viewer");
   fFrame->SetClassHints("GLViewer", "GLViewer");
   fFrame->SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);
   fFrame->MapSubwindows();

   fFrame->Resize(fFrame->GetDefaultSize());
   fFrame->MoveResize(fgInitX, fgInitY, fgInitW, fgInitH);
   fFrame->SetWMPosition(fgInitX, fgInitY);

   // Defer until layout done
   fClipEditor->HideParts();

   Show();
}

//______________________________________________________________________________
TGLSAViewer::~TGLSAViewer()
{
   // Destroy standalone viewer object
   delete fFileMenu;
   delete fCameraMenu;
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
   delete fEditorTab;
   delete fSceneTab;
   delete fShapesTab;
   delete fFrame;
}

//______________________________________________________________________________
void TGLSAViewer::Show()
{
   // Show the viewer
   fFrame->MapRaised();
   RequestDraw();
}

//______________________________________________________________________________
void TGLSAViewer::Close()
{
   // Close the viewer - destructed
   
   // Commit suicide when contained GUI is closed
   delete this;
}

//______________________________________________________________________________
Bool_t TGLSAViewer::ProcessFrameMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process GUI message capture by the main GUI frame (TGLSAFrame)
   switch (GET_MSG(msg)) {
   case kC_COMMAND:
      switch (GET_SUBMSG(msg)) {
      case kCM_BUTTON:
      case kCM_MENU:
         switch (parm1) {
         case kGLHelpAbout: {
#ifdef R__UNIX
            TString rootx;
# ifdef ROOTBINDIR
            rootx = ROOTBINDIR;
# else
            rootx = gSystem->Getenv("ROOTSYS");
            if (!rootx.IsNull()) rootx += "/bin";
# endif
            rootx += "/root -a &";
            gSystem->Exec(rootx);
#else
#ifdef WIN32
            new TWin32SplashThread(kTRUE);
#else
            char str[32];
            sprintf(str, "About ROOT %s...", gROOT->GetVersion());
            hd = new TRootHelpDialog(this, str, 600, 400);
            hd->SetText(gHelpAbout);
            hd->Popup();
#endif
#endif
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
         case kGLZOY:
            SetCurrentCamera(TGLViewer::kCameraZOY);
            break;
         case kGLPerspYOZ:
            SetCurrentCamera(TGLViewer::kCameraPerspectiveYOZ);
            break;
         case kGLPerspXOZ:
            SetCurrentCamera(TGLViewer::kCameraPerspectiveXOZ);
            break;
         case kGLPerspXOY:
            SetCurrentCamera(TGLViewer::kCameraPerspectiveXOY);
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
   // Process GUI event generated by GUI components - TGLEditor derivived classes
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
   case kTBcpm: {
      if (!fClipEditor) {
         return;
      }
      // Sync clipping
      EClipType clipType;
      std::vector<Double_t> clipData;
      Bool_t  clipEdit;
      fClipEditor->GetCurrent(clipType, clipEdit);
      fClipEditor->GetState(clipType, clipData);
      SetClipState(clipType, clipData);
      SetCurrentClip(clipType, clipEdit);
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
   case kTBGuide:
      if (!fGuideEditor) {
         return;
      }
      EAxesType axesType;
      Bool_t referenceOn;
      TGLVertex3 referencePos;
      fGuideEditor->GetState(axesType, referenceOn, referencePos);
      SetGuideState(axesType, referenceOn, referencePos);
      break;
   }
}

//______________________________________________________________________________
void TGLSAViewer::SelectionChanged()
{
   // Update GUI components for embedded viewer selection change

   const TGLPhysicalShape * selected = GetSelected();
   if (selected) {
      fColorEditor->SetRGBA(selected->Color());
      fGeomEditor->SetCenter(selected->GetTranslation().CArr());
      fGeomEditor->SetScale(selected->GetScale().CArr());
   } else { // No selection
      fColorEditor->Disable();
      fGeomEditor->Disable();
   }
}

//______________________________________________________________________________
void TGLSAViewer::ClipChanged()
{
   // Update GUI components for embedded viewer clipping change

   EClipType type = GetCurrentClip();
   std::vector<Double_t> data;
   GetClipState(type, data);
   fClipEditor->SetState(type, data);
   fClipEditor->SetCurrent(type);
}

//______________________________________________________________________________
void TGLSAViewer::PostSceneBuildSetup()
{
   // Do setup work required after a scene build has completed.
   // Synconise the viewer GUI with new clips, guides etc
   
   // Do base work first
   TGLViewer::PostSceneBuildSetup();

   // Now synconise the GUI
   
   // Default clips
   std::vector<Double_t> data;
   GetClipState(kClipPlane, data);
   fClipEditor->SetState(kClipPlane, data);
   GetClipState(kClipBox, data);
   fClipEditor->SetState(kClipBox, data);
   fClipEditor->SetCurrent(kClipNone);

   // Guides
   EAxesType axesType;
   Bool_t referenceOn;
   TGLVertex3 referencePos;
   GetGuideState(axesType, referenceOn, referencePos);
   fGuideEditor->SetState(axesType, referenceOn, referencePos);
}

