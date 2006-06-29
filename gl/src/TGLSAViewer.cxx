// @(#)root/gl:$Name:  $:$Id: TGLSAViewer.cxx,v 1.18 2006/04/07 08:43:59 brun Exp $
// Author:  Timur Pocheptsov / Richard Maunder

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <memory>

#include "TRootHelpDialog.h"
#include "TPluginManager.h"
#include "TApplication.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "HelpText.h"
#include "GuiTypes.h"
#include "TG3DLine.h"
#include "TSystem.h"
#include "TGLabel.h"
#include "TGMenu.h"
#include "TGTab.h"
#include "TGSplitter.h"
#include "TColor.h"
#include "TString.h"
#include "TGFileDialog.h"
#include "TImage.h"

#include "TGLEditor.h"
#include "TGLOutput.h"

#include "TGLPhysicalShape.h"
#include "TGLClip.h"
#include "TROOT.h"

#ifdef WIN32
#include "TWin32SplashThread.h"
#endif

#include "TGLPhysicalShape.h"
#include "TGLViewerEditor.h"
#include "TGLRenderArea.h"
#include "TGLSAViewer.h"
#include "TGLSAFrame.h"
#include "TGLEditor.h"
#include "TGLOutput.h"
#include "TGLKernel.h"

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
   SELECT the viewer with Shift+Left mouse button click on a free space.\n\n\
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

//A lot of raw pointers/naked new-expressions - good way to discredit C++ (or C++ programmer :) ) :(
//ROOT has system to cleanup - I'll try to use it

const char *gGLSaveAsTypes[] = {
                                "Encapsulated PostScript", "*.eps",
                                "PDF",                     "*.pdf",
                                "GIF",                     "*.gif",
                                "JPEG",                    "*.jpg",
                                "PNG",                     "*.png",
                                0, 0
                               };

//______________________________________________________________________________
TGLSAViewer::TGLSAViewer(TVirtualPad * pad) 
               : TGLViewer(pad, fgInitX, fgInitY, fgInitW, fgInitH),
                 fFrame(0), 
                 fFileMenu(0),
                 fFileSaveMenu(0),
                 fCameraMenu(0), 
                 fHelpMenu(0), 
                 fGLArea(0),
                 fLeftVerticalFrame(0),
                 fEditorTab(0),
                 fGLEd(0),
                 fObjEdTab(0),
                 fColorEd(0),
                 fGeomEd(0),
                 fDirName("."),
                 fTypeIdx(0),
                 fOverwrite(kFALSE)
{
   // Construct a standalone viewer, bound to supplied 'pad'.
   // First create gVirtualGL/kernel - to be replaced with TGLManager
   if (!gVirtualGL) {
      if (TPluginHandler *h = gROOT->GetPluginManager()->FindHandler("TVirtualGLImp")) {
         if (h->LoadPlugin() == -1)
            return;// bad, must be exception
         TVirtualGLImp * imp = (TVirtualGLImp *) h->ExecPlugin(0);
         new TGLKernel(imp);
      }
   }

   fFrame = new TGLSAFrame(*this);
   fFrame->SetCleanup(kDeepCleanup);

   CreateMenus();
   CreateFrames();

   fFrame->SetWindowName("ROOT's GL viewer");
   fFrame->SetClassHints("GLViewer", "GLViewer");
   fFrame->SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);
   fFrame->MapSubwindows();

   fFrame->Resize(fFrame->GetDefaultSize());
   fFrame->MoveResize(fgInitX, fgInitY, fgInitW, fgInitH);
   fFrame->SetWMPosition(fgInitX, fgInitY);

   // Defer until layout done
   
   //fFrame->HideFrame(fEditorTab);
   fLeftVerticalFrame->HideFrame(fObjEdTab);
   fGLEd->HideClippingGUI();
   
   Show();
}

//______________________________________________________________________________
void TGLSAViewer::CreateMenus()
{
   //File/Camera/Help menus
   fFileMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
   fFileMenu->AddEntry("&Close Viewer", kGLCloseViewer);
   fFileMenu->AddSeparator();
   fFileSaveMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
   fFileSaveMenu->AddEntry("viewer.&eps", kGLSaveEPS);
   fFileSaveMenu->AddEntry("viewer.&pdf", kGLSavePDF);
   fFileSaveMenu->AddEntry("viewer.&gif", kGLSaveGIF);
   fFileSaveMenu->AddEntry("viewer.&jpg", kGLSaveJPG);
   fFileSaveMenu->AddEntry("viewer.p&ng", kGLSavePNG);
   fFileMenu->AddPopup("&Save", fFileSaveMenu);
   fFileMenu->AddEntry("Save &As...", kGLSaveAS);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Quit ROOT", kGLQuitROOT);
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

   // Create menubar
   TGMenuBar *menuBar = new TGMenuBar(fFrame, 1, 1, kHorizontalFrame | kRaisedFrame);
   menuBar->AddPopup("&File", fFileMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   menuBar->AddPopup("&Camera", fCameraMenu, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   menuBar->AddPopup("&Help",    fHelpMenu,    new TGLayoutHints(kLHintsTop | kLHintsRight));
   fFrame->AddFrame(menuBar, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1));

}

//______________________________________________________________________________
void TGLSAViewer::CreateFrames()
{
   // Internal frames creation
   TGCompositeFrame* compositeFrame = new TGCompositeFrame(fFrame, 100, 100, kHorizontalFrame | kRaisedFrame);
   fFrame->AddFrame(compositeFrame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   fLeftVerticalFrame = new TGVerticalFrame(compositeFrame, 180, 10, kFixedWidth);
   compositeFrame->AddFrame(fLeftVerticalFrame, new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 2, 0, 2, 2));

   fEditorTab = new TGTab(fLeftVerticalFrame, 180, 10);
   fLeftVerticalFrame->AddFrame(fEditorTab, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY , 2, 2, 1, 2));

   TGCompositeFrame *styleContainer = fEditorTab->AddTab("Style");
   TGCompositeFrame *styleFrame = new TGCompositeFrame(styleContainer, 110, 30, kVerticalFrame);
   TGCompositeFrame *nameBin = new TGCompositeFrame(styleFrame, 145, 10, kHorizontalFrame | kFixedWidth | kOwnBackground);

   nameBin->AddFrame(new TGLabel(nameBin,"Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 5, 0));
   nameBin->AddFrame(new TGHorizontal3DLine(nameBin), new TGLayoutHints(kLHintsExpandX, 5, 5, 12, 7));
   styleFrame->AddFrame(nameBin, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   TGLabel *nameLabel = new TGLabel(styleFrame, "TGLViewer::TGLViewer");
   Pixel_t color;
   gClient->GetColorByName("#ff0000", color);
   nameLabel->SetTextColor(color, kFALSE);
   styleFrame->AddFrame(nameLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   styleContainer->AddFrame(styleFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 5, 0, 2, 2));
   fGLEd = new TGLViewerEditor(styleFrame);
   styleFrame->AddFrame(fGLEd, new TGLayoutHints(kLHintsTop | kLHintsExpandX,0, 0, 2, 2));
//   fGLEd->DetachFromPad();
   fGLEd->SetModel(0, this, 0);
   
   //Shape's colour editor
   fObjEdTab = new TGTab(fLeftVerticalFrame, 180, 10);
   fLeftVerticalFrame->AddFrame(fObjEdTab, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY , 2, 2, 1, 2));
   TGCompositeFrame *colorCont = fObjEdTab->AddTab("Color");
   nameBin = new TGCompositeFrame(colorCont, 145, 10, kHorizontalFrame | kFixedWidth | kOwnBackground);
   nameBin->AddFrame(new TGLabel(nameBin,"Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 5, 0));
   nameBin->AddFrame(new TGHorizontal3DLine(nameBin), new TGLayoutHints(kLHintsExpandX, 5, 5, 12, 7));
   colorCont->AddFrame(nameBin, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   nameLabel = new TGLabel(colorCont, "TGLViewer::TGLViewer");
   nameLabel->SetTextColor(color, kFALSE);
   colorCont->AddFrame(nameLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   fColorEd = new TGLColorEditor(colorCont, this);
   colorCont->AddFrame(fColorEd, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY, 2, 2, 1, 2));
   //Geometry editor
   TGCompositeFrame *geomCont = fObjEdTab->AddTab("Geometry");
   nameBin = new TGCompositeFrame(geomCont, 145, 10, kHorizontalFrame | kFixedWidth | kOwnBackground);
   nameBin->AddFrame(new TGLabel(nameBin,"Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 5, 0));
   nameBin->AddFrame(new TGHorizontal3DLine(nameBin), new TGLayoutHints(kLHintsExpandX, 5, 5, 12, 7));
   geomCont->AddFrame(nameBin, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   nameLabel = new TGLabel(geomCont, "TGLViewer::TGLViewer");
   nameLabel->SetTextColor(color, kFALSE);
   geomCont->AddFrame(nameLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   fGeomEd = new TGLGeometryEditor(geomCont, this);
   geomCont->AddFrame(fGeomEd, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY, 2, 2, 1, 2));

   TGVerticalFrame *rightVerticalFrame = new TGVerticalFrame(compositeFrame, 10, 10, kSunkenFrame);
   compositeFrame->AddFrame(rightVerticalFrame, new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,0,2,2,2));

   TGCanvas *canvasWindow = new TGCanvas(rightVerticalFrame, 10, 10, kSunkenFrame | kDoubleBorder);
   fGLArea = new TGLRenderArea(canvasWindow->GetViewPort()->GetId(), canvasWindow->GetViewPort());
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

   canvasWindow->SetContainer(fGLWindow);
   rightVerticalFrame->AddFrame(canvasWindow, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
}

//______________________________________________________________________________
TGLSAViewer::TGLSAViewer(TGFrame * parent, TVirtualPad * pad) :
   TGLViewer(pad, fgInitX, fgInitY, fgInitW, fgInitH),
   fFrame(0),
   fFileMenu(0),
   fCameraMenu(0),
   fHelpMenu(0),
   fGLArea(0),
   fLeftVerticalFrame(0),
   fEditorTab(0),
   fGLEd(0),
   fObjEdTab(0)
{
   // Construct an embedded standalone viewer, bound to supplied 'pad'.
   //
   // Modified version of the previous constructor for embedding the
   // viewer into another frame (parent).
   
   // First create gVirtualGL/kernel - to be replaced with TGLManager
   if (!gVirtualGL) {
      if (TPluginHandler *h = gROOT->GetPluginManager()->FindHandler("TVirtualGLImp")) {
         if (h->LoadPlugin() == -1)
            return;// bad, must be exception
         TVirtualGLImp * imp = (TVirtualGLImp *) h->ExecPlugin(0);
         new TGLKernel(imp);
      }
   }

   fFrame = new TGLSAFrame(parent, *this);
   fFrame->SetCleanup(kDeepCleanup);

   CreateMenus();
   CreateFrames();

   fFrame->MapSubwindows();
   fFrame->Resize(fFrame->GetDefaultSize());
   fFrame->Resize(fgInitW, fgInitH);

   fLeftVerticalFrame->HideFrame(fObjEdTab);
   fGLEd->HideClippingGUI();
}

//______________________________________________________________________________
TGLSAViewer::~TGLSAViewer()
{
   // Destroy standalone viewer object
   delete fGLArea;
   delete fHelpMenu;
   delete fCameraMenu;
   delete fFileSaveMenu;
   delete fFileMenu;
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
#ifdef ROOTBINDIR
            rootx = ROOTBINDIR;
#else
            rootx = gSystem->Getenv("ROOTSYS");
            if (!rootx.IsNull()) rootx += "/bin";
#endif
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
         case kGLSaveEPS:
            gVirtualGL->CaptureViewer(this, TGLOutput::kEPS_BSP, "viewer.eps");
            break;
         case kGLSavePDF:
            gVirtualGL->CaptureViewer(this, TGLOutput::kPDF_BSP, "viewer.pdf");
            break;
         case kGLXOY:
            SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
            break;
         case kGLXOZ:
            SetCurrentCamera(TGLViewer::kCameraOrthoXOZ);
            break;
         case kGLZOY:
            SetCurrentCamera(TGLViewer::kCameraOrthoZOY);
            break;
         case kGLPerspYOZ:
            SetCurrentCamera(TGLViewer::kCameraPerspYOZ);
            break;
         case kGLPerspXOZ:
            SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
            break;
         case kGLPerspXOY:
            SetCurrentCamera(TGLViewer::kCameraPerspXOY);
            break;
         case kGLSaveGIF:
            SavePicture("viewer.gif");
            break;
         case kGLSaveJPG:
            SavePicture("viewer.jpg");
         case kGLSavePNG:
            SavePicture("viewer.png");
            break;
         case kGLSaveAS:
            {
               TGFileInfo fi;
               fi.fFileTypes   = gGLSaveAsTypes;
               fi.fIniDir      = StrDup(fDirName);
               fi.fFileTypeIdx = fTypeIdx;
               fi.fOverwrite   = fOverwrite;
               new TGFileDialog(gClient->GetDefaultRoot(), fFrame, kFDSave, &fi);
               if (!fi.fFilename) return kTRUE;
               TString fileName(fi.fFilename);
               TString ft(fi.fFileTypes[fi.fFileTypeIdx+1]);
               fDirName   = fi.fIniDir;
               fTypeIdx   = fi.fFileTypeIdx;
               fOverwrite = fi.fOverwrite;

               if (!fileName.EndsWith(".eps")  && !fileName.EndsWith(".pdf")  && 
                   !fileName.EndsWith(".jpg")  && !fileName.EndsWith(".gif")  && 
                   !fileName.EndsWith(".png"))
                  if (ft.Index(".") != kNPOS)
                     fileName += ft(ft.Index("."), ft.Length());
                  else {
                     Warning("ProcessMessage", "file %s cannot be saved with this extension", fi.fFilename);
                     return kTRUE;
                  }

               SavePicture(fileName);      
            }

            break;
         case kGLCloseViewer:
            // Exit needs to be delayed to avoid bad drawable X ids - GUI
            // will all be changed in future anyway
            TTimer::SingleShot(50, "TGLSAFrame", fFrame, "SendCloseMessage()");
            break;
         case kGLQuitROOT:
            if (!gApplication->ReturnFromRun())
               delete this;
            gApplication->Terminate(0);
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
      SetSelectedColor(fColorEd->GetRGBA());
      break;
   case kTBaf:
      SetColorOnSelectedFamily(fColorEd->GetRGBA());
      break;
   case kTBa1: {
      TGLVertex3 trans;
      TGLVector3 scale;
      fGeomEd->GetObjectData(trans.Arr(), scale.Arr());
      SetSelectedGeom(trans,scale);
      break;
   }
   }
}

//______________________________________________________________________________
void TGLSAViewer::SelectionChanged()
{
   // Update GUI components for embedded viewer selection change

   const TGLPhysicalShape * selected = GetSelected();
   if (selected) {
      fLeftVerticalFrame->HideFrame(fEditorTab);
      fLeftVerticalFrame->ShowFrame(fObjEdTab);
      fColorEd->SetRGBA(selected->Color());
      fGeomEd->SetCenter(selected->GetTranslation().CArr());
      fGeomEd->SetScale(selected->GetScale().CArr());
   } else { // No selection
      fLeftVerticalFrame->ShowFrame(fEditorTab);
      fLeftVerticalFrame->HideFrame(fObjEdTab);
      fColorEd->Disable();
      fGeomEd->Disable();
   }
}

//______________________________________________________________________________
void TGLSAViewer::ClipChanged()
{
   // Update GUI components for embedded viewer clipping change
   fGLEd->SetCurrentClip();
}

//______________________________________________________________________________
void TGLSAViewer::PostSceneBuildSetup()
{
   // Do setup work required after a scene build has completed.
   // Synconise the viewer GUI with new clips, guides etc
   
   // Do base work first
   TGLViewer::PostSceneBuildSetup();

   // Now synconise the GUI-removed
}

//______________________________________________________________________________
void TGLSAViewer::SavePicture(const TString &fileName)
{
   if (fileName.EndsWith(".eps"))
      gVirtualGL->CaptureViewer(this, TGLOutput::kEPS_BSP, fileName.Data());
   else if (fileName.EndsWith(".pdf"))
      gVirtualGL->CaptureViewer(this, TGLOutput::kPDF_BSP, fileName.Data());
   else if (fileName.EndsWith(".gif") || fileName.EndsWith(".jpg") || fileName.EndsWith(".png")) {
      std::auto_ptr<TImage>gif(TImage::Create());
      gif->FromWindow(fGLArea->GetGLWindow()->GetId());
      gif->WriteImage(fileName.Data());
   }
}
