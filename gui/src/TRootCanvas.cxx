// @(#)root/gui:$Name:  $:$Id: TRootCanvas.cxx,v 1.3 2000/10/04 23:40:07 rdm Exp $
// Author: Fons Rademakers   15/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootCanvas                                                          //
//                                                                      //
// This class creates a main window with menubar, scrollbars and a      //
// drawing area. The widgets used are the new native ROOT GUI widgets.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRootCanvas.h"
#include "TRootApplication.h"
#include "TRootHelpDialog.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "TGMenu.h"
#include "TGWidget.h"
#include "TGFileDialog.h"
#include "TGStatusBar.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TBrowser.h"
#include "TClassTree.h"
#include "TMarker.h"
#include "TStyle.h"
#include "TVirtualX.h"
#include "TApplication.h"
#include "TFile.h"
#include "TInterpreter.h"

#include "HelpText.h"

#ifdef WIN32
#   undef SendMessage
#endif


// Canvas menu command ids
enum ERootCanvasCommands {
   kFileNewCanvas,
   kFileOpen,
   kFileSaveAs,
   kFileSaveAsRoot,
   kFileSaveAsC,
   kFileSaveAsPS,
   kFileSaveAsEPS,
   kFileSaveAsGIF,
   kFilePrint,
   kFileCloseCanvas,
   kFileQuit,

   kEditEditor,
   kEditUndo,
   kEditClearPad,
   kEditClearCanvas,

   kViewColors,
   kViewFonts,
   kViewMarkers,
   kViewIconify,
   kViewX3D,
   kViewOpenGL,
   kInterrupt,

   kOptionEventStatus,
   kOptionAutoExec,
   kOptionAutoResize,
   kOptionResizeCanvas,
   kOptionMoveOpaque,
   kOptionResizeOpaque,
   kOptionRefresh,
   kOptionStatistics,
   kOptionHistTitle,
   kOptionFitParams,
   kOptionCanEdit,

   kInspectRoot,
   kInspectBrowser,

   kClassesTree,

   kHelpAbout,
   kHelpOnCanvas,
   kHelpOnMenus,
   kHelpOnGraphicsEd,
   kHelpOnBrowser,
   kHelpOnObjects,
   kHelpOnPS
};

static const char *gOpenTypes[] = { "ROOT files",   "*.root",
                                    "All files",    "*",
                                    0,              0 };

static const char *gSaveAsTypes[] = { "PostScript",   "*.ps",
                                      "Encapsulated PostScript", "*.eps",
                                      "Gif files",    "*.gif",
                                      "Macro files",  "*.C",
                                      "ROOT files",   "*.root",
                                      "All files",    "*",
                                      0,              0 };


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootContainer                                                       //
//                                                                      //
// Utility class used by TRootCanvas. The TRootContainer is the frame   //
// embedded in the TGCanvas widget. The ROOT graphics goes into this    //
// frame. This class is used to enable input events on this graphics    //
// frame and forward the events to the TRootCanvas handlers.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TRootContainer : public TGCompositeFrame {
private:
   TRootCanvas  *fCanvas;    // pointer back to canvas imp
public:
   TRootContainer(TRootCanvas *c, Window_t id, const TGWindow *parent);

   Bool_t  HandleButton(Event_t *ev)
                { return fCanvas->HandleContainerButton(ev); }
   Bool_t  HandleDoubleClick(Event_t *ev)
                { return fCanvas->HandleContainerDoubleClick(ev); }
   Bool_t  HandleConfigureNotify(Event_t *ev)
                { TGFrame::HandleConfigureNotify(ev);
                  return fCanvas->HandleContainerConfigure(ev); }
   Bool_t  HandleKey(Event_t *ev)
                { return fCanvas->HandleContainerKey(ev); }
   Bool_t  HandleMotion(Event_t *ev)
                { return fCanvas->HandleContainerMotion(ev); }
   Bool_t  HandleExpose(Event_t *ev)
                { return fCanvas->HandleContainerExpose(ev); }
   Bool_t  HandleCrossing(Event_t *ev)
                { return fCanvas->HandleContainerCrossing(ev); }
};

//______________________________________________________________________________
TRootContainer::TRootContainer(TRootCanvas *c, Window_t id, const TGWindow *p)
   : TGCompositeFrame(gClient, id, p)
{
   // Create a canvas container.

   fCanvas = c;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask,
                    kNone, kNone);

   AddInput(kKeyPressMask | kKeyReleaseMask | kPointerMotionMask |
            kStructureNotifyMask | kLeaveWindowMask);
}



ClassImp(TRootCanvas)

//______________________________________________________________________________
TRootCanvas::TRootCanvas(TCanvas *c, const char *name, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height), TCanvasImp(c)
{
   // Create a basic ROOT canvas.

   CreateCanvas(name);

   Resize(width, height);
}

//______________________________________________________________________________
TRootCanvas::TRootCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height), TCanvasImp(c)
{
   // Create a basic ROOT canvas.

   CreateCanvas(name);

   MoveResize(x, y, width, height);
   SetWMPosition(x, y);
}

//______________________________________________________________________________
void TRootCanvas::CreateCanvas(const char *name)
{
   // Create the actual canvas.

   fButton  = 0;
   fCwidth  = 0;
   fCheight = 0;
   fAutoFit = kTRUE;   // check also menu entry

   // Create menus
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&New Canvas",         kFileNewCanvas);
   fFileMenu->AddEntry("&Open...",            kFileOpen);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("Save As...",          kFileSaveAs);
   fFileMenu->AddEntry("Save As canvas.ps",   kFileSaveAsPS);
   fFileMenu->AddEntry("Save As canvas.eps",  kFileSaveAsEPS);
   fFileMenu->AddEntry("Save As canvas.gif",  kFileSaveAsGIF);
   fFileMenu->AddEntry("Save As canvas.C",    kFileSaveAsC);
   fFileMenu->AddEntry("Save As canvas.root", kFileSaveAsRoot);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Print...",           kFilePrint);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Close Canvas",       kFileCloseCanvas);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Quit ROOT",          kFileQuit);

   //fFileMenu->DefaultEntry(kFileNewCanvas);
   //fFileMenu->DisableEntry(kFileOpen);

   fEditMenu = new TGPopupMenu(fClient->GetRoot());
   fEditMenu->AddEntry("&Editor",             kEditEditor);
   fEditMenu->AddEntry("&Undo",               kEditUndo);
   fEditMenu->AddEntry("Clear &Pad",          kEditClearPad);
   fEditMenu->AddEntry("&Clear Canvas",       kEditClearCanvas);

   fEditMenu->DisableEntry(kEditUndo);

   fViewMenu = new TGPopupMenu(fClient->GetRoot());
   fViewMenu->AddEntry("&Colors",             kViewColors);
   fViewMenu->AddEntry("&Fonts",              kViewFonts);
   fViewMenu->AddEntry("&Markers",            kViewMarkers);
   fViewMenu->AddSeparator();
   fViewMenu->AddEntry("&Iconify",            kViewIconify);
   fViewMenu->AddSeparator();
   fViewMenu->AddEntry("&View with X3D",      kViewX3D);
   fViewMenu->AddEntry("View with &OpenGL",   kViewOpenGL);
   fViewMenu->AddSeparator();
   fViewMenu->AddEntry("Interrupt",           kInterrupt);

   fOptionMenu = new TGPopupMenu(fClient->GetRoot());
   fOptionMenu->AddEntry("&Event Status",         kOptionEventStatus);
   fOptionMenu->AddEntry("&Pad Auto Exec",        kOptionAutoExec);
   fOptionMenu->AddSeparator();
   fOptionMenu->AddEntry("&Auto Resize Canvas",   kOptionAutoResize);
   fOptionMenu->AddEntry("&Resize Canvas",        kOptionResizeCanvas);
   fOptionMenu->AddEntry("&Move Opaque",          kOptionMoveOpaque);
   fOptionMenu->AddEntry("Resize &Opaque",        kOptionResizeOpaque);
   fOptionMenu->AddEntry("R&efresh",              kOptionRefresh);
   fOptionMenu->AddSeparator();
   fOptionMenu->AddEntry("Show &Statistics",      kOptionStatistics);
   fOptionMenu->AddEntry("Show &Histogram Title", kOptionHistTitle);
   fOptionMenu->AddEntry("Show &Fit Parameters",  kOptionFitParams);
   fOptionMenu->AddEntry("Can Edit Histograms",   kOptionCanEdit);

   // Opaque options initialized in InitWindow()
   fOptionMenu->CheckEntry(kOptionAutoResize);
   if (gStyle->GetOptStat())
      fOptionMenu->CheckEntry(kOptionStatistics);
   if (gStyle->GetOptTitle())
      fOptionMenu->CheckEntry(kOptionHistTitle);
   if (gStyle->GetOptFit())
      fOptionMenu->CheckEntry(kOptionFitParams);
   if (gROOT->GetEditHistograms())
      fOptionMenu->CheckEntry(kOptionCanEdit);

   fInspectMenu = new TGPopupMenu(fClient->GetRoot());
   fInspectMenu->AddEntry("&ROOT",              kInspectRoot);
   fInspectMenu->AddEntry("&Start Browser",     kInspectBrowser);

   fClassesMenu = new TGPopupMenu(fClient->GetRoot());
   fClassesMenu->AddEntry("&Class Tree",        kClassesTree);

   fHelpMenu = new TGPopupMenu(fClient->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...",        kHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("Help On Canvas...",     kHelpOnCanvas);
   fHelpMenu->AddEntry("Help On Menus...",      kHelpOnMenus);
   fHelpMenu->AddEntry("Help On Graphics Editor...", kHelpOnGraphicsEd);
   fHelpMenu->AddEntry("Help On Browser...",    kHelpOnBrowser);
   fHelpMenu->AddEntry("Help On Objects...",    kHelpOnObjects);
   fHelpMenu->AddEntry("Help On PostScript...", kHelpOnPS);

   // This main frame will process the menu commands
   fFileMenu->Associate(this);
   fEditMenu->Associate(this);
   fViewMenu->Associate(this);
   fOptionMenu->Associate(this);
   fInspectMenu->Associate(this);
   fClassesMenu->Associate(this);
   fHelpMenu->Associate(this);

   // Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   // Create menubar
   fMenuBar = new TGMenuBar(this, 1, 1, kHorizontalFrame);
   fMenuBar->AddPopup("&File",    fFileMenu,    fMenuBarItemLayout);
   fMenuBar->AddPopup("&Edit",    fEditMenu,    fMenuBarItemLayout);
   fMenuBar->AddPopup("&View",    fViewMenu,    fMenuBarItemLayout);
   fMenuBar->AddPopup("&Options", fOptionMenu,  fMenuBarItemLayout);
   fMenuBar->AddPopup("&Inspect", fInspectMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Classes", fClassesMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);

   AddFrame(fMenuBar, fMenuBarLayout);

   // Create canvas and canvas container that will host the ROOT graphics
   fCanvasWindow = new TGCanvas(this, GetWidth()+4, GetHeight()+4,
                                kSunkenFrame | kDoubleBorder);
   fCanvasID = gVirtualX->InitWindow((ULong_t)fCanvasWindow->GetViewPort()->GetId());
   Window_t win = gVirtualX->GetWindowID(fCanvasID);
   fCanvasContainer = new TRootContainer(this, win, fCanvasWindow->GetViewPort());
   fCanvasWindow->SetContainer(fCanvasContainer);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   AddFrame(fCanvasWindow, fCanvasLayout);

   // Create status bar
   int parts[] = { 33, 10, 10, 47 };
   fStatusBar = new TGStatusBar(this, 10, 10);
   fStatusBar->SetParts(parts, 4);

   fStatusBarLayout = new TGLayoutHints(kLHintsBottom | kLHintsLeft | kLHintsExpandX, 2, 2, 1, 1);

   AddFrame(fStatusBar, fStatusBarLayout);

   // Misc

   SetWindowName(name);
   SetIconName(name);
   SetIconPixmap("macro_s.xpm");
   SetClassHints("Canvas", "Canvas");

   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);

   MapSubwindows();

   // by default status bar is hidden
   HideFrame(fStatusBar);

   // we need to use GetDefaultSize() to initialize the layout algorithm...
   Resize(GetDefaultSize());
}

//______________________________________________________________________________
TRootCanvas::~TRootCanvas()
{
   // Delete ROOT basic canvas. Order is significant. Delete in reverse
   // order of creation.

   delete fStatusBar;
   delete fStatusBarLayout;
   delete fCanvasContainer;
   delete fCanvasWindow;
   delete fFileMenu;
   delete fEditMenu;
   delete fViewMenu;
   delete fOptionMenu;
   delete fInspectMenu;
   delete fClassesMenu;
   delete fHelpMenu;
   delete fMenuBar;
   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;
   delete fCanvasLayout;
}

//______________________________________________________________________________
void TRootCanvas::CloseWindow()
{
   // In case window is closed via WM we get here.
   // Forward message to central message handler as button event.

   SendMessage(this, MK_MSG(kC_COMMAND, kCM_BUTTON), kFileCloseCanvas, 0);
}

//______________________________________________________________________________
UInt_t TRootCanvas::GetCwidth() const
{
   // Return width of canvas container.

   return fCanvasContainer->GetWidth();
}

//______________________________________________________________________________
UInt_t TRootCanvas::GetCheight() const
{
   // Return height of canvas container.

   return fCanvasContainer->GetHeight();
}

//______________________________________________________________________________
void TRootCanvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // Gets the size and position of the window containing the canvas. This
   // size includes the menubar and borders.

   gVirtualX->GetWindowSize(fId, x, y, w, h);

   // Get position of window on the screen. For this we need to get the parent
   // of the ROOT canvas, i.e. the window managed by the window manager and get
   // its position
   UInt_t wdum, hdum;
   Window_t id = fId;
   do {
      gVirtualX->GetWindowSize(id, x, y, wdum, hdum);
      id = gVirtualX->GetParent(id);
   } while (id != gClient->GetRoot()->GetId());
}

//______________________________________________________________________________
void TRootCanvas::SetStatusText(const char *txt, Int_t partidx)
{
   // Set text in status bar.

   fStatusBar->SetText(txt, partidx);
}

//______________________________________________________________________________
Bool_t TRootCanvas::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Handle menu and other command generated by the user.

   TRootHelpDialog *hd;
   TList *lc;
   TVirtualPad *savepad;

   switch (GET_MSG(msg)) {

      case kC_COMMAND:

         switch (GET_SUBMSG(msg)) {

            case kCM_BUTTON:
            case kCM_MENU:

               switch (parm1) {
                  // Handle File menu items...
                  case kFileNewCanvas:
                     gROOT->GetMakeDefCanvas()();
                     break;
                  case kFileOpen:
                     {
                        TGFileInfo fi;
                        fi.fFileTypes = (char **) gOpenTypes;
                        new TGFileDialog(fClient->GetRoot(), this, kFDOpen,&fi);
                        if (!fi.fFilename) return kTRUE;
                        new TFile(fi.fFilename, "update");
                        delete [] fi.fFilename;
                     }
                     break;
                  case kFileSaveAs:
                     {
                        TGFileInfo fi;
                        fi.fFileTypes = (char **) gSaveAsTypes;
                        new TGFileDialog(fClient->GetRoot(), this, kFDSave,&fi);
                        if (!fi.fFilename) return kTRUE;
                        if (strstr(fi.fFilename, ".root") ||
                            strstr(fi.fFilename, ".ps")   ||
                            strstr(fi.fFilename, ".eps")  ||
                            strstr(fi.fFilename, ".gif"))
                           fCanvas->SaveAs(fi.fFilename);
                        else if (strstr(fi.fFilename, ".C"))
                           fCanvas->SaveSource(fi.fFilename);
                        else
                           Warning("ProcessMessage", "file cannot be save with this extension (%s)", fi.fFilename);
                        delete [] fi.fFilename;
                     }
                     break;
                  case kFileSaveAsRoot:
                     fCanvas->SaveAs(".root");
                     break;
                  case kFileSaveAsC:
                     fCanvas->SaveSource();
                     break;
                  case kFileSaveAsPS:
                     fCanvas->SaveAs();
                     break;
                  case kFileSaveAsEPS:
                     fCanvas->SaveAs(".eps");
                     break;
                  case kFileSaveAsGIF:
                     fCanvas->SaveAs(".gif");
                     break;
                  case kFilePrint:
                     fCanvas->Print();
                     break;
                  case kFileCloseCanvas:
                     savepad = gPad;
                     gPad = 0;        // hide gPad from CINT
                     gInterpreter->DeleteGlobal(fCanvas);
                     gPad = savepad;  // restore gPad for ROOT
                     delete fCanvas;  // this in turn will delete this object
                     break;
                  case kFileQuit:
                     gApplication->Terminate(0);
                     break;

                  // Handle Edit menu items...
                  case kEditEditor:
                     fCanvas->EditorBar();
                     break;
                  case kEditUndo:
                     // noop
                     break;
                  case kEditClearPad:
                     gPad->Clear();
                     gPad->Modified();
                     gPad->Update();
                     break;
                  case kEditClearCanvas:
                     fCanvas->Clear();
                     fCanvas->Modified();
                     fCanvas->Update();
                     break;

                  // Handle View menu items...
                  case kViewColors:
                     {
                        TVirtualPad *padsav = gPad->GetCanvas();
                        TCanvas *m = new TCanvas("colors","Color Table");
                        TPad::DrawColorTable();
                        m->Update();
                        padsav->cd();
                     }
                     break;
                  case kViewFonts:
                     // noop
                     break;
                  case kViewMarkers:
                     {
                        TVirtualPad *padsav = gPad->GetCanvas();
                        TCanvas *m = new TCanvas("markers","Marker Types",600,200);
                        TMarker::DisplayMarkerTypes();
                        m->Update();
                        padsav->cd();
                     }
                     break;
                  case kViewIconify:
                     Iconify();
                     break;
                  case kViewX3D:
                     gPad->x3d();
                     break;
                  case kViewOpenGL:
                     gPad->x3d("OPENGL");
                     break;
                  case kInterrupt:
                     gROOT->SetInterrupt();
                     break;

                  // Handle Option menu items...
                  case kOptionEventStatus:
                     fCanvas->ToggleEventStatus();
                     if (fCanvas->GetShowEventStatus()) {
                        ShowFrame(fStatusBar);
                        fOptionMenu->CheckEntry(kOptionEventStatus);
                     } else {
                        HideFrame(fStatusBar);
                        fOptionMenu->UnCheckEntry(kOptionEventStatus);
                     }
                     break;
                  case kOptionAutoExec:
                     fCanvas->ToggleAutoExec();
                     if (fCanvas->GetAutoExec()) {
                        fOptionMenu->CheckEntry(kOptionAutoExec);
                     } else {
                        fOptionMenu->UnCheckEntry(kOptionAutoExec);
                     }
                     break;
                  case kOptionAutoResize:
                     {
                        fAutoFit = fAutoFit ? kFALSE : kTRUE;
                        int opt = fCanvasContainer->GetOptions();
                        if (fAutoFit) {
                           opt &= ~kFixedSize;
                           fOptionMenu->CheckEntry(kOptionAutoResize);
                        } else {
                           opt |= kFixedSize;
                           fOptionMenu->UnCheckEntry(kOptionAutoResize);
                        }
                        fCanvasContainer->ChangeOptions(opt);
                        // in case of autofit this will generate a configure
                        // event for the container and this will force the
                        // update of the TCanvas
                        Layout();
                     }
                     break;
                  case kOptionResizeCanvas:
                     FitCanvas();
                     break;
                  case kOptionMoveOpaque:
                     if (fCanvas->OpaqueMoving()) {
                        fCanvas->MoveOpaque(0);
                        fOptionMenu->UnCheckEntry(kOptionMoveOpaque);
                     } else {
                        fCanvas->MoveOpaque(1);
                        fOptionMenu->CheckEntry(kOptionMoveOpaque);
                     }
                     break;
                  case kOptionResizeOpaque:
                     if (fCanvas->OpaqueResizing()) {
                        fCanvas->ResizeOpaque(0);
                        fOptionMenu->UnCheckEntry(kOptionResizeOpaque);
                     } else {
                        fCanvas->ResizeOpaque(1);
                        fOptionMenu->CheckEntry(kOptionResizeOpaque);
                     }
                     break;
                  case kOptionRefresh:
                     fCanvas->Paint();
                     fCanvas->Update();
                     break;
                  case kOptionStatistics:
                     if (gStyle->GetOptStat()) {
                        gStyle->SetOptStat(0);
                        delete gPad->FindObject("stats");
                        fOptionMenu->UnCheckEntry(kOptionStatistics);
                     } else {
                        gStyle->SetOptStat(1);
                        fOptionMenu->CheckEntry(kOptionStatistics);
                     }
                     gPad->Modified();
                     fCanvas->Update();
                     break;
                  case kOptionHistTitle:
                     if (gStyle->GetOptTitle()) {
                        gStyle->SetOptTitle(0);
                        delete gPad->FindObject("title");
                        fOptionMenu->UnCheckEntry(kOptionHistTitle);
                     } else {
                        gStyle->SetOptTitle(1);
                        fOptionMenu->CheckEntry(kOptionHistTitle);
                     }
                     gPad->Modified();
                     fCanvas->Update();
                     break;
                  case kOptionFitParams:
                     if (gStyle->GetOptFit()) {
                        gStyle->SetOptFit(0);
                        fOptionMenu->UnCheckEntry(kOptionFitParams);
                     } else {
                        gStyle->SetOptFit(1);
                        fOptionMenu->CheckEntry(kOptionFitParams);
                     }
                     gPad->Modified();
                     fCanvas->Update();
                     break;
                  case kOptionCanEdit:
                     if (gROOT->GetEditHistograms()) {
                        gROOT->SetEditHistograms(kFALSE);
                        fOptionMenu->UnCheckEntry(kOptionCanEdit);
                     } else {
                        gROOT->SetEditHistograms(kTRUE);
                        fOptionMenu->CheckEntry(kOptionCanEdit);
                     }
                     break;

                  // Handle Inspect menu items...
                  case kInspectRoot:
                     fCanvas->cd();
                     gROOT->Inspect();
                     fCanvas->Update();
                     break;
                  case kInspectBrowser:
                     new TBrowser("browser");
                     break;

                  // Handle Inspect menu items...
                  case kClassesTree:
                     {
                        char cdef[64];
                        lc = (TList*)gROOT->GetListOfCanvases();
                        if (lc->FindObject("ClassTree")) {
                           sprintf(cdef,"ClassTree_%d",lc->GetSize()+1);
                        } else {
                           sprintf(cdef,"%s","ClassTree");
                        }
                        new TClassTree(cdef,"TObject");
                        fCanvas->Update();
                     }
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
                  case kHelpOnCanvas:
                     hd = new TRootHelpDialog(this, "Help on Canvas...", 600, 400);
                     hd->SetText(gHelpCanvas);
                     hd->Popup();
                     break;
                  case kHelpOnMenus:
                     hd = new TRootHelpDialog(this, "Help on Menus...", 600, 400);
                     hd->SetText(gHelpPullDownMenus);
                     hd->Popup();
                     break;
                  case kHelpOnGraphicsEd:
                     hd = new TRootHelpDialog(this, "Help on Graphics Editor...", 600, 400);
                     hd->SetText(gHelpGraphicsEditor);
                     hd->Popup();
                     break;
                  case kHelpOnBrowser:
                     hd = new TRootHelpDialog(this, "Help on Browser...", 600, 400);
                     hd->SetText(gHelpBrowser);
                     hd->Popup();
                     break;
                  case kHelpOnObjects:
                     hd = new TRootHelpDialog(this, "Help on Objects...", 600, 400);
                     hd->SetText(gHelpObjects);
                     hd->Popup();
                     break;
                  case kHelpOnPS:
                     hd = new TRootHelpDialog(this, "Help on PostScript...", 600, 400);
                     hd->SetText(gHelpPostscript);
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
Int_t TRootCanvas::InitWindow()
{
   // Called by TCanvas ctor to get window indetifier.

   if (fCanvas->OpaqueMoving())
      fOptionMenu->CheckEntry(kOptionMoveOpaque);
   if (fCanvas->OpaqueResizing())
      fOptionMenu->CheckEntry(kOptionResizeOpaque);

   return fCanvasID;
}

//______________________________________________________________________________
void TRootCanvas::SetCanvasSize(UInt_t w, UInt_t h)
{
   // Set size of canvas container. Unix in pixels.

   // turn off autofit, we want to stay at the given size
   fAutoFit = kFALSE;
   fOptionMenu->UnCheckEntry(kOptionAutoResize);
   int opt = fCanvasContainer->GetOptions();
   opt |= kFixedSize;    // turn on fixed size mode
   fCanvasContainer->ChangeOptions(opt);
   fCanvasContainer->SetWidth(w);
   fCanvasContainer->SetHeight(h);
   Layout();  // force layout (will update container to given size)
   fCanvas->Resize();
   fCanvas->Update();
}

//______________________________________________________________________________
void TRootCanvas::SetWindowPosition(Int_t x, Int_t y)
{
   // Set canvas position (units in pixels).

   Move(x, y);
}

//______________________________________________________________________________
void TRootCanvas::SetWindowSize(UInt_t w, UInt_t h)
{
   // Set size of canvas( units in pixels).

   Resize(w, h);
}

//______________________________________________________________________________
void TRootCanvas::SetWindowTitle(const char *title)
{
   // Change title on window.

   SetWindowName(title);
   SetIconName(title);
}

//______________________________________________________________________________
void TRootCanvas::FitCanvas()
{
   // Fit canvas container to current window size.

   if (!fAutoFit) {
      int opt = fCanvasContainer->GetOptions();
      int oopt = opt;
      opt &= ~kFixedSize;   // turn off fixed size mode
      fCanvasContainer->ChangeOptions(opt);
      Layout();  // force layout
      fCanvas->Resize();
      fCanvas->Update();
      fCanvasContainer->ChangeOptions(oopt);
   }
}

//______________________________________________________________________________
void TRootCanvas::ShowMenuBar(Bool_t show)
{
   // Show or hide menubar.

   if (show) {
      ShowFrame(fMenuBar);
   } else {
      HideFrame(fMenuBar);
   }
}

//______________________________________________________________________________
void TRootCanvas::ShowStatusBar(Bool_t show)
{
   // Show or hide statusbar.

   if (show) {
      ShowFrame(fStatusBar);
      fOptionMenu->CheckEntry(kOptionEventStatus);
   } else {
      HideFrame(fStatusBar);
      fOptionMenu->UnCheckEntry(kOptionEventStatus);
   }
}

//______________________________________________________________________________
Bool_t TRootCanvas::HandleContainerButton(Event_t *event)
{
   // Handle mouse button events in the canvas container.

   Int_t button = event->fCode;
   Int_t x = event->fX;
   Int_t y = event->fY;

   if (event->fType == kButtonPress) {
      fButton = button;
      if (button == kButton1)
         fCanvas->HandleInput(kButton1Down, x, y);
      if (button == kButton2)
         fCanvas->HandleInput(kButton2Down, x, y);
      if (button == kButton3) {
         fCanvas->HandleInput(kButton3Down, x, y);
         fButton = 0;  // button up is consumed by TContextMenu
      }

   } else if (event->fType == kButtonRelease) {
      if (button == kButton1)
         fCanvas->HandleInput(kButton1Up, x, y);
      if (button == kButton2)
         fCanvas->HandleInput(kButton2Up, x, y);
      if (button == kButton3)
         fCanvas->HandleInput(kButton3Up, x, y);

      fButton = 0;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootCanvas::HandleContainerDoubleClick(Event_t *event)
{
   // Handle mouse button double click events in the canvas container.

   Int_t button = event->fCode;
   Int_t x = event->fX;
   Int_t y = event->fY;

   if (button == kButton1)
      fCanvas->HandleInput(kButton1Double, x, y);
   if (button == kButton2)
      fCanvas->HandleInput(kButton2Double, x, y);
   if (button == kButton3)
      fCanvas->HandleInput(kButton3Double, x, y);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootCanvas::HandleContainerConfigure(Event_t *)
{
   // Handle configure (i.e. resize) event.

   if (fAutoFit) {
      fCanvas->Resize();
      fCanvas->Update();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootCanvas::HandleContainerKey(Event_t *event)
{
   // Handle keyboard events in the canvas container.

  if (event->fType == kGKeyPress) {
      fButton = event->fCode;
      UInt_t keysym;
      char str[2];
      gVirtualX->LookupString(event, str, sizeof(str), keysym);
      if (str[0] == 3)   // ctrl-c sets the interrupt flag
         gROOT->SetInterrupt();
      fCanvas->HandleInput(kKeyPress, str[0], 0);
   } else if (event->fType == kKeyRelease)
      fButton = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootCanvas::HandleContainerMotion(Event_t *event)
{
   // Handle mouse motion event in the canvas container.

   Int_t x = event->fX;
   Int_t y = event->fY;

   if (fButton == 0)
      fCanvas->HandleInput(kMouseMotion, x, y);
   if (fButton == kButton1)
      fCanvas->HandleInput(kButton1Motion, x, y);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootCanvas::HandleContainerExpose(Event_t *event)
{
   // Handle expose events.

   if (event->fCount == 0)
      fCanvas->Flush();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootCanvas::HandleContainerCrossing(Event_t *event)
{
   // Handle enter/leave events. Only leave is activated at the moment.

   Int_t x = event->fX;
   Int_t y = event->fY;

   // pointer grabs create also an enter and leave event but with fCode
   // either kNotifyGrab or kNotifyUngrab, don't propagate these events
   if (event->fType == kLeaveNotify && event->fCode == kNotifyNormal)
      fCanvas->HandleInput(kMouseLeave, x, y);

   return kTRUE;
}
