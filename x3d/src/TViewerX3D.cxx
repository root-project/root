// @(#)root/x3d:$Name:  $:$Id: TViewerX3D.cxx,v 1.2 2000/10/13 19:04:40 rdm Exp $
// Author: Rene Brun   05/09/99
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewerX3D                                                           //
//                                                                      //
// C++ interface to the X3D viewer                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TViewerX3D.h"
#include "TAtt3D.h"
#include "X3DBuffer.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TMath.h"
#include "TROOT.h"
#include "TClass.h"

#include "TRootHelpDialog.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "TGMenu.h"
#include "TGWidget.h"
#include "TGMsgBox.h"
#include "TVirtualX.h"

#include "HelpText.h"

const char gHelpX3DViewer[] = "\
     PRESS \n\
     \tw\t--- wireframe mode\n\
     \te\t--- hidden line mode\n\
     \tr\t--- hidden surface mode\n\
     \tu\t--- move object down\n\
     \ti\t--- move object up\n\
     \to\t--- toggle controls style\n\
     \ts\t--- toggle stereo display\n\
     \td\t--- toggle blue stereo view\n\
     \tf\t--- toggle double buffer\n\
     \th\t--- move object right\n\
     \tj\t--- move object forward\n\
     \tk\t--- move object backward\n\
     \tl\t--- move object left\n\
     \tx a\t--- rotate about x\n\
     \ty b\t--- rotate about y\n\
     \tz c\t--- rotate about z\n\
     \t1 2 3\t--- autorotate about x\n\
     \t4 5 6\t--- autorotate about y\n\
     \t7 8 9\t--- autorotate about z\n\
     \t[ ] { }\t--- adjust focus\n\n\
     HOLD the left mouse button and MOVE mouse to ROTATE object\n\n\
";


extern "C" {
  Window_t x3d_main(Float_t *longitude, Float_t *latitude, Float_t *psi,
                    Option_t *option, Window_t parent);
  void     x3d_set_display(Display_t display);
  int      x3d_dispatch_event(Handle_t event);
  void     x3d_update();
  void     x3d_get_position(Float_t *longitude, Float_t *latitude, Float_t *psi);
  void     x3d_terminate();
}


// Canvas menu command ids
enum EX3DViewerCommands {
   kFileNewViewer,
   kFileSave,
   kFileSaveAs,
   kFilePrint,
   kFileCloseViewer,

   kHelpAbout,
   kHelpOnViewer
};

Bool_t TViewerX3D::fgActive = kFALSE;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TX3DContainer                                                        //
//                                                                      //
// Utility class used by TViewerX3D. The TX3DContainer is the frame     //
// embedded in the TGCanvas widget. The X3D graphics goes into this     //
// frame. This class is used to enable input events on this graphics    //
// frame and forward the events to X3D.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TX3DContainer : public TGCompositeFrame {
private:
   TViewerX3D  *fViewer;    // pointer back to viewer imp
public:
   TX3DContainer(TViewerX3D *c, Window_t id, const TGWindow *parent);

   Bool_t  HandleButton(Event_t *ev)
                { x3d_dispatch_event(gVirtualX->GetNativeEvent());
                  return fViewer->HandleContainerButton(ev); }
   Bool_t  HandleConfigureNotify(Event_t *ev)
                { TGFrame::HandleConfigureNotify(ev);
                  return x3d_dispatch_event(gVirtualX->GetNativeEvent()); }
   Bool_t  HandleKey(Event_t *ev)
                { return x3d_dispatch_event(gVirtualX->GetNativeEvent()); }
   Bool_t  HandleMotion(Event_t *ev)
                { return x3d_dispatch_event(gVirtualX->GetNativeEvent()); }
   Bool_t  HandleExpose(Event_t *ev)
                { return x3d_dispatch_event(gVirtualX->GetNativeEvent()); }
   Bool_t  HandleColormapChange(Event_t *ev)
                { return x3d_dispatch_event(gVirtualX->GetNativeEvent()); }
};

//______________________________________________________________________________
TX3DContainer::TX3DContainer(TViewerX3D *c, Window_t id, const TGWindow *p)
   : TGCompositeFrame(gClient, id, p)
{
   // Create a canvas container.

   fViewer = c;

   /* already done in x3d InitDisplay()
   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask,
                    kNone, kNone);

   gVirtualX->SelectInput(fId, kKeyPressMask | kExposureMask | kPointerMotionMask |
                     kStructureNotifyMask);
   */
}

ClassImp(TViewerX3D)

//______________________________________________________________________________
TViewerX3D::TViewerX3D(TVirtualPad *pad, Option_t *option, const char *title,
                       UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height)
{
   // Create ROOT X3D viewer.

   fPad = 0;
   if (fgActive) {
      Int_t retval;
      new TGMsgBox(gClient->GetRoot(), gClient->GetRoot(),
                   "X3D Viewer Warning", "Can have only one X3D viewer active",
                   kMBIconExclamation, kMBOk, &retval);
      return;
   }

   fPad    = pad;
   fOption = option;
   fX3DWin = 0;

   CreateViewer(title);
   if (!fX3DWin) return;

   Resize(width, height);

   x3d_update();

   fgActive = kTRUE;
}

//______________________________________________________________________________
TViewerX3D::TViewerX3D(TVirtualPad *pad, Option_t *option, const char *title,
                       Int_t x, Int_t y, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height)
{
   // Create ROOT X3D viewer.

   fPad = 0;
   if (fgActive) {
      Int_t retval;
      new TGMsgBox(gClient->GetRoot(), gClient->GetRoot(),
                   "X3D Viewer", "Can have only one X3D viewer active",
                   kMBIconExclamation, kMBOk, &retval);
      return;
   }

   fPad    = pad;
   fOption = option;
   fX3DWin = 0;

   CreateViewer(title);
   if (!fX3DWin) return;

   MoveResize(x, y, width, height);
   SetWMPosition(x, y);

   x3d_update();

   fgActive = kTRUE;
}

//______________________________________________________________________________
TViewerX3D::~TViewerX3D()
{
   // Delete ROOT X3D viewer.

   if (!fPad) return;

   DeleteX3DWindow();

   delete fContainer;
   delete fCanvas;
   delete fFileMenu;
   delete fHelpMenu;
   delete fMenuBar;
   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;
   delete fCanvasLayout;

   fgActive = kFALSE;
}

//______________________________________________________________________________
void TViewerX3D::CreateViewer(const char *name)
{
   // Create the actual canvas.

   // Create menus
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&New Viewer",         kFileNewViewer);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("Save",                kFileSave);
   fFileMenu->AddEntry("Save As...",          kFileSaveAs);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Print...",           kFilePrint);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Close Viewer",       kFileCloseViewer);

   //fFileMenu->DefaultEntry(kFileNewViewer);
   fFileMenu->DisableEntry(kFileNewViewer);
   fFileMenu->DisableEntry(kFileSave);
   fFileMenu->DisableEntry(kFileSaveAs);
   fFileMenu->DisableEntry(kFilePrint);

   fHelpMenu = new TGPopupMenu(fClient->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...",           kHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("Help On X3D Viewer...", kHelpOnViewer);

   // This main frame will process the menu commands
   fFileMenu->Associate(this);
   fHelpMenu->Associate(this);

   // Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   // Create menubar
   fMenuBar = new TGMenuBar(this, 1, 1, kHorizontalFrame);
   fMenuBar->AddPopup("&File",    fFileMenu,    fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);

   AddFrame(fMenuBar, fMenuBarLayout);

   // Create canvas and canvas container that will host the ROOT graphics
   fCanvas = new TGCanvas(this, GetWidth()+4, GetHeight()+4,
                          kSunkenFrame | kDoubleBorder);
   InitX3DWindow();
   if (!fX3DWin) {
      fContainer    = 0;
      fCanvasLayout = 0;
      return;
   }
   fContainer = new TX3DContainer(this, fX3DWin, fCanvas->GetViewPort());
   fCanvas->SetContainer(fContainer);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   AddFrame(fCanvas, fCanvasLayout);

   // Misc

   SetWindowName(name);
   SetIconName(name);
   SetClassHints("X3DViewer", "X3DViewer");

   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);

   MapSubwindows();

   // we need to use GetDefaultSize() to initialize the layout algorithm...
   Resize(GetDefaultSize());

   Show();
}

//______________________________________________________________________________
void TViewerX3D::InitX3DWindow()
{
   // Setup geometry and initialize X3D.

   TObject *obj;
   char x3dopt[32];

   TView *view = fPad->GetView();
   if (!view) {
      Error("InitX3DWindow", "view is not set");
      return;
   }

   gSize3D.numPoints = 0;
   gSize3D.numSegs   = 0;
   gSize3D.numPolys  = 0;

   TObjLink *lnk = fPad->GetListOfPrimitives()->FirstLink();
   while (lnk) {
      obj = lnk->GetObject();
      TAtt3D *att;
#ifdef R__RTTI
      if ((att = dynamic_cast<TAtt3D*>(obj)))
#else
      if ((att = (TAtt3D*)obj->IsA()->DynamicCast(TAtt3D::Class(), obj)))
#endif
         att->Sizeof3D();
      lnk = lnk->Next();
   }

   printf("Total size of x3d primitives:\n");
   printf("     gSize3D.numPoints= %d\n",gSize3D.numPoints);
   printf("     gSize3D.numSegs  = %d\n",gSize3D.numSegs);
   printf("     gSize3D.numPolys = %d\n",gSize3D.numPolys);

   if (!AllocateX3DBuffer()) {
      Error("InitX3DWindow", "x3d buffer allocation failure");
      return;
   }

   lnk = fPad->GetListOfPrimitives()->FirstLink();
   while (lnk) {
      obj = lnk->GetObject();
      if (obj->InheritsFrom(TAtt3D::Class())) {
         strcpy(x3dopt,"x3d");
         strcat(x3dopt,fOption.Data());
         obj->Paint(x3dopt);
      }
      lnk = lnk->Next();
   }

   const Float_t kPI = Float_t (TMath::Pi());

   Float_t longitude_rad = ( 90 + view->GetLongitude()) * kPI/180.0;
   Float_t  latitude_rad = (-90 + view->GetLatitude() ) * kPI/180.0;
   Float_t       psi_rad = (      view->GetPsi()      ) * kPI/180.0;

   // Call 'x3d' package
   x3d_set_display(gVirtualX->GetDisplay());
   fX3DWin = (Window_t) x3d_main(&longitude_rad, &latitude_rad, &psi_rad,
                                 fOption.Data(), fCanvas->GetViewPort()->GetId());
}

//______________________________________________________________________________
void TViewerX3D::DeleteX3DWindow()
{
   // Close X3D window.

   x3d_terminate();
}

//______________________________________________________________________________
void TViewerX3D::CloseWindow()
{
   // In case window is closed via WM we get here.
   // Forward message to central message handler as button event.

   SendMessage(this, MK_MSG(kC_COMMAND, kCM_BUTTON), kFileCloseViewer, 0);
}

//______________________________________________________________________________
void TViewerX3D::Update()
{
   // Update X3D viewer.

   x3d_update();
}

//______________________________________________________________________________
Bool_t TViewerX3D::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Handle menu and other command generated by the user.

   TRootHelpDialog *hd;

   switch (GET_MSG(msg)) {

      case kC_COMMAND:

         switch (GET_SUBMSG(msg)) {

            case kCM_BUTTON:
            case kCM_MENU:

               switch (parm1) {
                  // Handle File menu items...
                  case kFileNewViewer:
                     if (fPad) fPad->x3d();
                     break;
                  case kFileSave:
                  case kFileSaveAs:
                  case kFilePrint:
                     break;
                  case kFileCloseViewer:
                     delete this;
                     break;

                  // Handle Help menu items...
                  case kHelpAbout:
                     {
                        char str[32];
                        sprintf(str, "About ROOT %s...", gROOT->GetVersion());
                        hd = new TRootHelpDialog(this, str, 600, 400);
                        hd->SetText(gHelpAbout);
                        hd->Popup();
                     }
                     break;
                  case kHelpOnViewer:
                     hd = new TRootHelpDialog(this, "Help on X3D Viewer...", 600, 400);
                     hd->SetText(gHelpX3DViewer);
                     hd->Popup();
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
Bool_t TViewerX3D::HandleContainerButton(Event_t *ev)
{
   // After button release get current position and update associated pad.

   if (ev->fType == kButtonRelease) {
      Float_t longitude_rad;
      Float_t latitude_rad;
      Float_t psi_rad;
      const Float_t kPI = Float_t (TMath::Pi());

      x3d_get_position(&longitude_rad, &latitude_rad, &psi_rad);

      Int_t irep;

      Float_t longitude_deg = longitude_rad * 180.0/kPI - 90;
      Float_t  latitude_deg = latitude_rad  * 180.0/kPI + 90;
      Float_t       psi_deg = psi_rad       * 180.0/kPI;

      fPad->GetView()->SetView(longitude_deg, latitude_deg, psi_deg, irep);

      fPad->SetPhi(-90 - longitude_deg);
      fPad->SetTheta(90 - latitude_deg);

      fPad->Modified(kTRUE);
      fPad->Update();
   }
   return kTRUE;
}
