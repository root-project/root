// @(#)root/gui:$Name:  $:$Id: TRootCanvas.cxx,v 1.97 2006/04/11 06:57:05 antcheva Exp $
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

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "TRootCanvas.h"
#include "TRootApplication.h"
#include "TRootHelpDialog.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "TGMenu.h"
#include "TGWidget.h"
#include "TGFileDialog.h"
#include "TGStatusBar.h"
#include "TGTextEditDialogs.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TBrowser.h"
#include "TClassTree.h"
#include "TMarker.h"
#include "TStyle.h"
#include "TStyleManager.h"
#include "TVirtualX.h"
#include "TApplication.h"
#include "TFile.h"
#include "TInterpreter.h"
#include "TEnv.h"
#include "Riostream.h"
#include "TGDockableFrame.h"

#include "TG3DLine.h"
#include "TGToolBar.h"
#include "TVirtualPadEditor.h"
#include "TRootControlBar.h"
#include "TGLabel.h"
#include "TGuiBuilder.h"
#include "TImage.h"
#include "TError.h"

#include "TPluginManager.h"
#include "TVirtualGL.h"

#ifdef WIN32
#include "TWin32SplashThread.h"
#endif

#include "HelpText.h"


// Canvas menu command ids
enum ERootCanvasCommands {
   kFileNewCanvas,
   kFileOpen,
   kFileSaveAs,
   kFileSaveAsRoot,
   kFileSaveAsC,
   kFileSaveAsPS,
   kFileSaveAsEPS,
   kFileSaveAsPDF,
   kFileSaveAsGIF,
   kFileSaveAsJPG,
   kFilePrint,
   kFileCloseCanvas,
   kFileQuit,

   kEditStyle,
   kEditCut,
   kEditCopy,
   kEditPaste,
   kEditClearPad,
   kEditClearCanvas,
   kEditUndo,
   kEditRedo,

   kViewEditor,
   kViewToolbar,
   kViewEventStatus,
   kViewColors,
   kViewFonts,
   kViewMarkers,
   kViewIconify,
   kViewX3D,
   kViewOpenGL,

   kOptionAutoResize,
   kOptionResizeCanvas,
   kOptionMoveOpaque,
   kOptionResizeOpaque,
   kOptionInterrupt,
   kOptionRefresh,
   kOptionAutoExec,
   kOptionStatistics,
   kOptionHistTitle,
   kOptionFitParams,
   kOptionCanEdit,

   kInspectRoot,
   kInspectBrowser,
   kInspectBuilder,

   kClassesTree,

   kHelpAbout,
   kHelpOnCanvas,
   kHelpOnMenus,
   kHelpOnGraphicsEd,
   kHelpOnBrowser,
   kHelpOnObjects,
   kHelpOnPS,

   kToolModify,
   kToolArc,
   kToolLine,
   kToolArrow,
   kToolDiamond,
   kToolEllipse,
   kToolPad,
   kToolPave,
   kToolPLabel,
   kToolPText,
   kToolPsText,
   kToolGraph,
   kToolCurlyLine,
   kToolCurlyArc,
   kToolLatex,
   kToolMarker,
   kToolCutG

};

static const char *gOpenTypes[] = { "ROOT files",   "*.root",
                                    "All files",    "*",
                                    0,              0 };

static const char *gSaveAsTypes[] = { "PostScript",   "*.ps",
                                      "Encapsulated PostScript", "*.eps",
                                      "PDF",          "*.pdf",
                                      "SVG",          "*.svg",
                                      "GIF",          "*.gif",
                                      "ROOT macros",  "*.C",
                                      "ROOT files",   "*.root",
                                      "XML",          "*.xml",
                                      "PNG",          "*.png",
                                      "XPM",          "*.xpm",
                                      "JPEG",         "*.jpg",
                                      "TIFF",         "*.tiff",
                                      "XCF",          "*.xcf",
                                      "All files",    "*",
                                      0,              0 };

static ToolBarData_t gToolBarData[] = {
   // { filename,      tooltip,            staydown,  id,              button}
   { "newcanvas.xpm",  "New",              kFALSE,    kFileNewCanvas,  NULL },
   { "open.xpm",       "Open",             kFALSE,    kFileOpen,       NULL },
   { "save.xpm",       "Save As",          kFALSE,    kFileSaveAs,     NULL },
   { "printer.xpm",    "Print",            kFALSE,    kFilePrint,      NULL },
   { "",               "",                 kFALSE,    -1,              NULL },
   { "interrupt.xpm",  "Interrupt",        kFALSE,    kOptionInterrupt,NULL },
   { "refresh2.xpm",   "Refresh",          kFALSE,    kOptionRefresh,  NULL },
   { "",               "",                 kFALSE,    -1,              NULL },
   { "inspect.xpm",    "Inspect",          kFALSE,    kInspectRoot,    NULL },
   { "browser.xpm",    "Browser",          kFALSE,    kInspectBrowser, NULL },
   { 0,                0,                  kFALSE,    0,               NULL }
};

static ToolBarData_t gToolBarData1[] = {
   { "pointer.xpm",    "Modify",           kFALSE,    kToolModify,     NULL },
   { "arc.xpm",        "Arc",              kFALSE,    kToolArc,        NULL },
   { "line.xpm",       "Line",             kFALSE,    kToolLine,       NULL },
   { "arrow.xpm",      "Arrow",            kFALSE,    kToolArrow,      NULL },
   { "diamond.xpm",    "Diamond",          kFALSE,    kToolDiamond,    NULL },
   { "ellipse.xpm",    "Ellipse",          kFALSE,    kToolEllipse,    NULL },
   { "pad.xpm",        "Pad",              kFALSE,    kToolPad,        NULL },
   { "pave.xpm",       "Pave",             kFALSE,    kToolPave,       NULL },
   { "pavelabel.xpm",  "Pave Label",       kFALSE,    kToolPLabel,     NULL },
   { "pavetext.xpm",   "Pave Text",        kFALSE,    kToolPText,      NULL },
   { "pavestext.xpm",  "Paves Text",       kFALSE,    kToolPsText,     NULL },
   { "graph.xpm",      "Graph",            kFALSE,    kToolGraph,      NULL },
   { "curlyline.xpm",  "Curly Line",       kFALSE,    kToolCurlyLine,  NULL },
   { "curlyarc.xpm",   "Curly Arc",        kFALSE,    kToolCurlyArc,   NULL },
   { "latex.xpm",      "Text/Latex",       kFALSE,    kToolLatex,      NULL },
   { "marker.xpm",     "Marker",           kFALSE,    kToolMarker,     NULL },
   { "cut.xpm",        "Graphical Cut",    kFALSE,    kToolCutG,       NULL },
   { 0,                0,                  kFALSE,    0,               NULL }
};

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

   Bool_t  HandleButton(Event_t *ev);
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
   void    SavePrimitive(ofstream &out, Option_t *);
   void    SetEditable(Bool_t) { }
};

//______________________________________________________________________________
TRootContainer::TRootContainer(TRootCanvas *c, Window_t id, const TGWindow *p)
   : TGCompositeFrame(gClient, id, p)
{
   // Create a canvas container.

   fCanvas = c;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   AddInput(kKeyPressMask | kKeyReleaseMask | kPointerMotionMask |
            kExposureMask | kStructureNotifyMask | kLeaveWindowMask);
   fEditDisabled = kEditDisable;
}

//______________________________________________________________________________
Bool_t TRootContainer::HandleButton(Event_t *event)
{
   // Directly handle scroll mouse buttons (4 and 5), only pass buttons
   // 1, 2 and 3 on to the TCanvas.

   TGViewPort *vp = (TGViewPort*)fParent;
   Int_t y = vp->GetVPos();
   UInt_t page = vp->GetHeight()/4;
   Int_t newpos;

   gVirtualX->SetInputFocus(GetMainFrame()->GetId());

   if (event->fCode == kButton4) {
      //scroll up
      newpos = y - page;
      if (newpos < 0) newpos = 0;
      fCanvas->fCanvasWindow->SetVsbPosition(newpos);
//      return kTRUE;
   }
   if (event->fCode == kButton5) {
      // scroll down
      newpos = fCanvas->fCanvasWindow->GetVsbPosition() + page;
      fCanvas->fCanvasWindow->SetVsbPosition(newpos);
//      return kTRUE;
   }
   return fCanvas->HandleContainerButton(event);
}

ClassImp(TRootCanvas)

//______________________________________________________________________________
TRootCanvas::TRootCanvas(TCanvas *c, const char *name, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetDefaultRoot(), width, height), TCanvasImp(c)
{
   // Create a basic ROOT canvas.

   CreateCanvas(name);

   ShowToolBar(kFALSE);
   ShowEditor(kFALSE);

   Resize(width, height);
}

//______________________________________________________________________________
TRootCanvas::TRootCanvas(TCanvas *c, const char *name, Int_t x, Int_t y, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetDefaultRoot(), width, height), TCanvasImp(c)
{
   // Create a basic ROOT canvas.

   CreateCanvas(name);

   ShowToolBar(kFALSE);
   ShowEditor(kFALSE);

   MoveResize(x, y, width, height);
   SetWMPosition(x, y);
}

//______________________________________________________________________________
void TRootCanvas::CreateCanvas(const char *name)
{
   // Create the actual canvas.

   fButton    = 0;
   fAutoFit   = kTRUE;   // check also menu entry
   fEditor    = 0;

   // Create menus
   fFileSaveMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fFileSaveMenu->AddEntry(Form("%s.&ps",  name), kFileSaveAsPS);
   fFileSaveMenu->AddEntry(Form("%s.&eps", name), kFileSaveAsEPS);
   fFileSaveMenu->AddEntry(Form("%s.p&df", name), kFileSaveAsPDF);
   fFileSaveMenu->AddEntry(Form("%s.&gif", name), kFileSaveAsGIF);

   static Int_t img = 0;

   if (!img) {
      Int_t sav = gErrorIgnoreLevel;
      gErrorIgnoreLevel = kFatal;
      img = TImage::Create() ? 1 : -1;
      gErrorIgnoreLevel = sav;
   }
   if (img > 0) {
      fFileSaveMenu->AddEntry(Form("%s.&jpg",name),  kFileSaveAsJPG);
   }

   fFileSaveMenu->AddEntry(Form("%s.&C",   name), kFileSaveAsC);
   fFileSaveMenu->AddEntry(Form("%s.&root",name), kFileSaveAsRoot);

   fFileMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fFileMenu->AddEntry("&New Canvas",   kFileNewCanvas);
   fFileMenu->AddEntry("&Open...",      kFileOpen);
   fFileMenu->AddEntry("&Close Canvas", kFileCloseCanvas);
   fFileMenu->AddSeparator();
   fFileMenu->AddPopup("&Save",         fFileSaveMenu);
   fFileMenu->AddEntry("Save &As...",   kFileSaveAs);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Print...",     kFilePrint);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Quit ROOT",    kFileQuit);

   fEditClearMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fEditClearMenu->AddEntry("&Pad",     kEditClearPad);
   fEditClearMenu->AddEntry("&Canvas",  kEditClearCanvas);

   fEditMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fEditMenu->AddEntry("&Style...",     kEditStyle);
   fEditMenu->AddSeparator();
   fEditMenu->AddEntry("Cu&t",          kEditCut);
   fEditMenu->AddEntry("&Copy",         kEditCopy);
   fEditMenu->AddEntry("&Paste",        kEditPaste);
   fEditMenu->AddSeparator();
   fEditMenu->AddPopup("C&lear",        fEditClearMenu);
   fEditMenu->AddSeparator();
   fEditMenu->AddEntry("&Undo",         kEditUndo);
   fEditMenu->AddEntry("&Redo",         kEditRedo);

   fEditMenu->DisableEntry(kEditCut);
   fEditMenu->DisableEntry(kEditCopy);
   fEditMenu->DisableEntry(kEditPaste);
   fEditMenu->DisableEntry(kEditUndo);
   fEditMenu->DisableEntry(kEditRedo);

   fViewWithMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fViewWithMenu->AddEntry("&X3D",      kViewX3D);
   fViewWithMenu->AddEntry("&OpenGL",   kViewOpenGL);

   fViewMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fViewMenu->AddEntry("&Editor",       kViewEditor);
   fViewMenu->AddEntry("&Toolbar",      kViewToolbar);
   fViewMenu->AddEntry("Event &Statusbar", kViewEventStatus);
   fViewMenu->AddSeparator();
   fViewMenu->AddEntry("&Colors",       kViewColors);
   fViewMenu->AddEntry("&Fonts",        kViewFonts);
   fViewMenu->AddEntry("&Markers",      kViewMarkers);
   fViewMenu->AddSeparator();
   fViewMenu->AddEntry("&Iconify",      kViewIconify);
   fViewMenu->AddSeparator();
   fViewMenu->AddPopup("&View With",    fViewWithMenu);

   fViewMenu->DisableEntry(kViewFonts);

   fOptionMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fOptionMenu->AddEntry("&Auto Resize Canvas",  kOptionAutoResize);
   fOptionMenu->AddEntry("&Resize Canvas",       kOptionResizeCanvas);
   fOptionMenu->AddEntry("&Move Opaque",         kOptionMoveOpaque);
   fOptionMenu->AddEntry("Resize &Opaque",       kOptionResizeOpaque);
   fOptionMenu->AddSeparator();
   fOptionMenu->AddEntry("&Interrupt",           kOptionInterrupt);
   fOptionMenu->AddEntry("R&efresh",             kOptionRefresh);
   fOptionMenu->AddSeparator();
   fOptionMenu->AddEntry("&Pad Auto Exec",       kOptionAutoExec);
   fOptionMenu->AddSeparator();
   fOptionMenu->AddEntry("&Statistics",          kOptionStatistics);
   fOptionMenu->AddEntry("Histogram &Title",     kOptionHistTitle);
   fOptionMenu->AddEntry("&Fit Parameters",      kOptionFitParams);
   fOptionMenu->AddEntry("Can Edit &Histograms", kOptionCanEdit);

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

   fInspectMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fInspectMenu->AddEntry("&ROOT",              kInspectRoot);
   fInspectMenu->AddEntry("&Start Browser",     kInspectBrowser);
   fInspectMenu->AddEntry("&Gui Builder",       kInspectBuilder);

   fClassesMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fClassesMenu->AddEntry("&Class Tree",        kClassesTree);

   fHelpMenu = new TGPopupMenu(fClient->GetDefaultRoot());
   fHelpMenu->AddLabel("Basic Help On...");
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("&Canvas",          kHelpOnCanvas);
   fHelpMenu->AddEntry("&Menus",           kHelpOnMenus);
   fHelpMenu->AddEntry("&Graphics Editor", kHelpOnGraphicsEd);
   fHelpMenu->AddEntry("&Browser",         kHelpOnBrowser);
   fHelpMenu->AddEntry("&Objects",         kHelpOnObjects);
   fHelpMenu->AddEntry("&PostScript",      kHelpOnPS);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("&About ROOT...",   kHelpAbout);

   // This main frame will process the menu commands
   fFileMenu->Associate(this);
   fFileSaveMenu->Associate(this);
   fEditMenu->Associate(this);
   fEditClearMenu->Associate(this);
   fViewMenu->Associate(this);
   fViewWithMenu->Associate(this);
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

   fHorizontal1 = new TGHorizontal3DLine(this);
   fHorizontal1Layout = new TGLayoutHints(kLHintsTop | kLHintsExpandX);
   AddFrame(fHorizontal1, fHorizontal1Layout);

   // Create toolbar dock
   fToolDock = new TGDockableFrame(this);
   fToolDock->EnableHide(kFALSE);
   AddFrame(fToolDock, new TGLayoutHints(kLHintsExpandX));

   // will alocate it later
   fToolBar = 0;
   fVertical1 = 0;
   fVertical2 = 0;
   fVertical1Layout = 0;
   fVertical2Layout = 0;

   fToolBarSep = new TGHorizontal3DLine(this);
   fToolBarLayout = new TGLayoutHints(kLHintsTop |  kLHintsExpandX);
   AddFrame(fToolBarSep, fToolBarLayout);

   fMainFrame = new TGCompositeFrame(this, GetWidth() + 4, GetHeight() + 4,
                                      kHorizontalFrame);
   fMainFrameLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);

   // Create editor frame that will host the pad editor
   fEditorFrame = new TGCompositeFrame(fMainFrame, 175, fMainFrame->GetHeight()+4, kFixedWidth);
   fEditorLayout = new TGLayoutHints(kLHintsExpandY | kLHintsLeft);
   fMainFrame->AddFrame(fEditorFrame, fEditorLayout);

   // Create canvas and canvas container that will host the ROOT graphics
   fCanvasWindow = new TGCanvas(fMainFrame, GetWidth()+4, GetHeight()+4,
                                kSunkenFrame | kDoubleBorder);

   fCanvasID = -1;

   if (fCanvas->UseGL()) {
      //first, initialize GL (if not yet)
      if (!gGLManager) {
         TPluginHandler *ph = gROOT->GetPluginManager()->FindHandler("TGLManager");

         if (ph && ph->LoadPlugin() != -1) {
            if (!ph->ExecPlugin(0))
               Warning("CreateCanvas",
                       "Can not load GL, will use default canvas imp instead\n");
         }
      }

      if (gGLManager) {
         fCanvasID = gGLManager->InitGLWindow((ULong_t)fCanvasWindow->GetViewPort()->GetId());
         if (fCanvasID != -1)
            fCanvas->SetSupportGL(kTRUE);
         else {
            fCanvas->SetSupportGL(kFALSE);
            Warning("CreateCanvas", "Cannot init gl window, will use default instead\n");
         }
      }
   }

   if (fCanvasID == -1)
      fCanvasID = gVirtualX->InitWindow((ULong_t)fCanvasWindow->GetViewPort()->GetId());

   Window_t win = gVirtualX->GetWindowID(fCanvasID);
   fCanvasContainer = new TRootContainer(this, win, fCanvasWindow->GetViewPort());
   fCanvasWindow->SetContainer(fCanvasContainer);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY | kLHintsRight);

   fMainFrame->AddFrame(fCanvasWindow, fCanvasLayout);
   AddFrame(fMainFrame, fMainFrameLayout);

   // Create status bar
   int parts[] = { 33, 10, 10, 47 };
   fStatusBar = new TGStatusBar(this, 10, 10);
   fStatusBar->SetParts(parts, 4);

   fStatusBarLayout = new TGLayoutHints(kLHintsBottom | kLHintsLeft | kLHintsExpandX, 2, 2, 1, 1);

   AddFrame(fStatusBar, fStatusBarLayout);

   // Misc
   SetWindowName(name);
   SetIconName(name);
   fIconPic = SetIconPixmap("macro_s.xpm");
   SetClassHints("Canvas", "Canvas");

   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);

   SetEditDisabled(kEditDisable);
   MapSubwindows();

   // by default status bar, tool bar and pad editor are hidden
   HideFrame(fStatusBar);
   HideFrame(fToolDock);
   HideFrame(fToolBarSep);
   HideFrame(fHorizontal1);

   ShowToolBar(kFALSE);
   ShowEditor(kFALSE);

   // we need to use GetDefaultSize() to initialize the layout algorithm...
   Resize(GetDefaultSize());
}

//______________________________________________________________________________
TRootCanvas::~TRootCanvas()
{
   // Delete ROOT basic canvas. Order is significant. Delete in reverse
   // order of creation.

   if (fIconPic) gClient->FreePicture(fIconPic);
   if (fEditor) delete fEditor;
   if (fToolBar) {
      Disconnect(fToolDock, "Docked()",   this, "AdjustSize()");
      Disconnect(fToolDock, "Undocked()", this, "AdjustSize()");
      fToolBar->Cleanup();
      delete fToolBar;
   }

   if (!MustCleanup()) {
      delete fStatusBar;
      delete fStatusBarLayout;
      delete fCanvasContainer;
      delete fCanvasWindow;

      delete fEditorFrame;
      delete fEditorLayout;
      delete fMainFrame;
      delete fMainFrameLayout;
      delete fToolBarSep;
      delete fToolDock;
      delete fToolBarLayout;
      delete fHorizontal1;
      delete fHorizontal1Layout;

      delete fMenuBar;
      delete fMenuBarLayout;
      delete fMenuBarItemLayout;
      delete fMenuBarHelpLayout;
      delete fCanvasLayout;
   }

   delete fFileMenu;
   delete fFileSaveMenu;
   delete fEditMenu;
   delete fEditClearMenu;
   delete fViewMenu;
   delete fViewWithMenu;
   delete fOptionMenu;
   delete fInspectMenu;
   delete fClassesMenu;
   delete fHelpMenu;
}

//______________________________________________________________________________
void TRootCanvas::Close()
{
   // Called via TCanvasImp interface by TCanvas.

   if (fEditor) fEditor->DeleteEditors();
   if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0)
      TVirtualPadEditor::Terminate();

   gVirtualX->CloseWindow();
}

//______________________________________________________________________________
void TRootCanvas::ReallyDelete()
{
   // Really delete the canvas and this GUI.

   if (fEditor) fEditor->DeleteEditors();
   if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0)
      TVirtualPadEditor::Terminate();

   TVirtualPad *savepad = gPad;
   gPad = 0;        // hide gPad from CINT
   gInterpreter->DeleteGlobal(fCanvas);
   gPad = savepad;  // restore gPad for ROOT
   delete fCanvas;  // will in turn delete this object
}

//______________________________________________________________________________
void TRootCanvas::CloseWindow()
{
   // In case window is closed via WM we get here.

   DeleteWindow();
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
UInt_t TRootCanvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // Gets the size and position of the window containing the canvas. This
   // size includes the menubar and borders.

   gVirtualX->GetWindowSize(fId, x, y, w, h);

   Window_t childdum;
   gVirtualX->TranslateCoordinates(fId, gClient->GetDefaultRoot()->GetId(),
                                   0, 0, x, y, childdum);
   if (!fCanvas->GetShowEditor()) return 0;
   return fEditorFrame->GetWidth();
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

   switch (GET_MSG(msg)) {

      case kC_COMMAND:

         switch (GET_SUBMSG(msg)) {

            case kCM_BUTTON:
            case kCM_MENU:

               switch (parm1) {
                  // Handle toolbar items...
                  case kToolModify:
                     gROOT->SetEditorMode();
                     break;
                  case kToolArc:
                     gROOT->SetEditorMode("Arc");
                     break;
                  case kToolLine:
                     gROOT->SetEditorMode("Line");
                     break;
                  case kToolArrow:
                     gROOT->SetEditorMode("Arrow");
                     break;
                  case kToolDiamond:
                     gROOT->SetEditorMode("Diamond");
                     break;
                  case kToolEllipse:
                     gROOT->SetEditorMode("Ellipse");
                     break;
                  case kToolPad:
                     gROOT->SetEditorMode("Pad");
                     break;
                  case kToolPave:
                     gROOT->SetEditorMode("Pave");
                     break;
                  case kToolPLabel:
                     gROOT->SetEditorMode("PaveLabel");
                     break;
                  case kToolPText:
                     gROOT->SetEditorMode("PaveText");
                     break;
                  case kToolPsText:
                     gROOT->SetEditorMode("PavesText");
                     break;
                  case kToolGraph:
                     gROOT->SetEditorMode("PolyLine");
                     break;
                  case kToolCurlyLine:
                     gROOT->SetEditorMode("CurlyLine");
                     break;
                  case kToolCurlyArc:
                     gROOT->SetEditorMode("CurlyArc");
                     break;
                  case kToolLatex:
                     gROOT->SetEditorMode("Text");
                     break;
                  case kToolMarker:
                     gROOT->SetEditorMode("Marker");
                     break;
                  case kToolCutG:
                     gROOT->SetEditorMode("CutG");
                     break;

                  // Handle File menu items...
                  case kFileNewCanvas:
                     gROOT->GetMakeDefCanvas()();
                     break;
                  case kFileOpen:
                     {
                        static TString dir(".");
                        TGFileInfo fi;
                        fi.fFileTypes = gOpenTypes;
                        fi.fIniDir    = StrDup(dir);
                        new TGFileDialog(fClient->GetDefaultRoot(), this, kFDOpen,&fi);
                        if (!fi.fFilename) return kTRUE;
                        dir = fi.fIniDir;
                        new TFile(fi.fFilename, "update");
                     }
                     break;
                  case kFileSaveAs:
                     {
                        static TString dir(".");
                        static Int_t typeidx = 0;
                        static Bool_t overwr = kFALSE;
                        TGFileInfo fi;
                        fi.fFileTypes   = gSaveAsTypes;
                        fi.fIniDir      = StrDup(dir);
                        fi.fFileTypeIdx = typeidx;
                        fi.fOverwrite = overwr;
                        new TGFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);
                        if (!fi.fFilename) return kTRUE;
                        Bool_t  appendedType = kFALSE;
                        TString fn = fi.fFilename;
                        TString ft = fi.fFileTypes[fi.fFileTypeIdx+1];
                        dir     = fi.fIniDir;
                        typeidx = fi.fFileTypeIdx;
                        overwr  = fi.fOverwrite;
again:
                        if (fn.EndsWith(".root") ||
                            fn.EndsWith(".ps")   ||
                            fn.EndsWith(".eps")  ||
                            fn.EndsWith(".pdf")  ||
                            fn.EndsWith(".svg")  ||
                            fn.EndsWith(".gif")  ||
                            fn.EndsWith(".xml")  ||
                            fn.EndsWith(".xpm")  ||
                            fn.EndsWith(".jpg")  ||
                            fn.EndsWith(".png")  ||
                            fn.EndsWith(".xcf")  ||
                            fn.EndsWith(".tiff")) {
                           fCanvas->SaveAs(fn);
                        } else if (fn.EndsWith(".C"))
                           fCanvas->SaveSource(fn);
                        else {
                           if (!appendedType) {
                              if (ft.Index(".") != kNPOS) {
                                 fn += ft(ft.Index("."), ft.Length());
                                 appendedType = kTRUE;
                                 goto again;
                              }
                           }
                           Warning("ProcessMessage", "file %s cannot be saved with this extension", fi.fFilename);
                        }
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
                  case kFileSaveAsPDF:
                     fCanvas->SaveAs(".pdf");
                     break;
                  case kFileSaveAsGIF:
                     fCanvas->SaveAs(".gif");
                     break;
                  case kFileSaveAsJPG:
                     fCanvas->SaveAs(".jpg");
                     break;
                  case kFilePrint:
                     PrintCanvas();
                     break;
                  case kFileCloseCanvas:
                     if (!fEditor && (TVirtualPadEditor::GetPadEditor(kFALSE) != 0))
                        TVirtualPadEditor::Terminate();
                     SendCloseMessage();
                     break;
                  case kFileQuit:
                     if (!gApplication->ReturnFromRun()) {
                        if (fEditor) fEditor->DeleteEditors();
                        if (!fEditor && (TVirtualPadEditor::GetPadEditor(kFALSE) != 0))
                           TVirtualPadEditor::Terminate();
                        delete this;
                     }
                     if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0)
                        TVirtualPadEditor::Terminate();
                     if (gROOT->GetClass("TStyleManager"))
                        gROOT->ProcessLine("TStyleManager::Terminate()");
                     gApplication->Terminate(0);
                     break;

                  // Handle Edit menu items...
                  case kEditStyle:
                     if (!gROOT->GetClass("TStyleManager"))
                        gSystem->Load("libGed");
                     gROOT->ProcessLine("TStyleManager::Show()");
                     break;
                  case kEditCut:
                     // still noop
                     break;
                  case kEditCopy:
                     // still noop
                     break;
                  case kEditPaste:
                     // still noop
                     break;
                  case kEditUndo:
                     // noop
                     break;
                  case kEditRedo:
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
                  case kViewEditor:
                     fCanvas->ToggleEditor();
                     if (!fEditor) CreateEditor();
                     break;
                  case kViewToolbar:
                     fCanvas->ToggleToolBar();
                     break;
                  case kViewEventStatus:
                     fCanvas->ToggleEventStatus();
                     break;
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
                     gPad->GetViewer3D("x3d");
                     break;
                  case kViewOpenGL:
                     gPad->GetViewer3D("ogl");
                     break;

                  // Handle Option menu items...
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
                        //Layout();
                     }
                     Layout();
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
                  case kOptionInterrupt:
                     gROOT->SetInterrupt();
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
                  case kInspectBuilder:
                     TGuiBuilder::Instance();
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
   // Set size of canvas container. Units in pixels.

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
   // Set size of canvas (units in pixels).

   Resize(w, h);
}

//______________________________________________________________________________
void TRootCanvas::RaiseWindow()
{
   // Put canvas window on top of the window stack.

   gVirtualX->RaiseWindow(GetId());
}

//______________________________________________________________________________
void TRootCanvas::SetWindowTitle(const char *title)
{
   // Change title on window.

   SetWindowName(title);
   SetIconName(title);
   fToolDock->SetWindowName(Form("ToolBar: %s", title));
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
void TRootCanvas::PrintCanvas()
{
   // Print the canvas.

   Int_t ret = 0;
   Bool_t pname = kTRUE;
   char *printer, *printCmd;
   static TString sprinter, sprintCmd;

   if (sprinter == "")
      printer = StrDup(gEnv->GetValue("Print.Printer", ""));
   else
      printer = StrDup(sprinter);
   if (sprintCmd == "")
#ifndef WIN32
      printCmd = StrDup(gEnv->GetValue("Print.Command", ""));
#else
      printCmd = StrDup(gEnv->GetValue("Print.Command", "start AcroRd32.exe /p"));
#endif
   else
      printCmd = StrDup(sprintCmd);

   new TGPrintDialog(fClient->GetDefaultRoot(), this, 400, 150,
                     &printer, &printCmd, &ret);
   if (ret) {
      sprinter  = printer;
      sprintCmd = printCmd;

      if (sprinter == "")
         pname = kFALSE;

      TString fn = "rootprint";
      FILE *f = gSystem->TempFileName(fn, gEnv->GetValue("Print.Directory", gSystem->TempDirectory()));
      fclose(f);
      fn += Form(".%s",gEnv->GetValue("Print.FileType", "pdf"));
      fCanvas->Print(fn);

      TString cmd = sprintCmd;
      if (cmd.Contains("%p"))
         cmd.ReplaceAll("%p", sprinter);
      else if (pname) {
         cmd += " "; cmd += sprinter; cmd += " ";
      }

      if (cmd.Contains("%f"))
         cmd.ReplaceAll("%f", fn);
      else {
         cmd += " "; cmd += fn; cmd += " ";
      }

      gSystem->Exec(cmd);
      gSystem->Unlink(fn);
   }
   delete [] printer;
   delete [] printCmd;
}

//______________________________________________________________________________
void TRootCanvas::ShowMenuBar(Bool_t show)
{
   // Show or hide menubar.

   if (show)  ShowFrame(fMenuBar);
   else       HideFrame(fMenuBar);
}

//______________________________________________________________________________
void TRootCanvas::ShowStatusBar(Bool_t show)
{
   // Show or hide statusbar.

   UInt_t dh = fClient->GetDisplayHeight();
   UInt_t ch = fCanvas->GetWindowHeight();

   UInt_t h = GetHeight();
   UInt_t sh = fStatusBar->GetHeight()+2;

   if (show) {
      ShowFrame(fStatusBar);
      fViewMenu->CheckEntry(kViewEventStatus);
      if (dh - ch >= sh) h = h + sh;
      else h = ch;
   } else {
      HideFrame(fStatusBar);
      fViewMenu->UnCheckEntry(kViewEventStatus);
      if (dh - ch < sh) h = ch;
      else h = h - sh;
   }
   Resize(GetWidth(), h);
}

//______________________________________________________________________________
void TRootCanvas::ShowEditor(Bool_t show)
{
   // Show or hide side frame.

   TVirtualPad *savedPad = 0;
   savedPad = (TVirtualPad *) gPad;
   gPad = Canvas();

   UInt_t w = GetWidth();
   UInt_t e = fEditorFrame->GetWidth();
   UInt_t h = GetHeight();
   UInt_t s = fHorizontal1->GetHeight();

   if (show) {
      if (!fEditor) CreateEditor();
      if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0) {
            TVirtualPadEditor::HideEditor();
      }
      if (!fViewMenu->IsEntryChecked(kViewToolbar) || fToolDock->IsUndocked()) {
         ShowFrame(fHorizontal1);
         h = h + s;
      }
      fMainFrame->ShowFrame(fEditorFrame);
      fViewMenu->CheckEntry(kViewEditor);
      w = w + e;
   } else {
      if (!fViewMenu->IsEntryChecked(kViewToolbar) || fToolDock->IsUndocked()) {
         HideFrame(fHorizontal1);
         h = h - s;
      }
      fMainFrame->HideFrame(fEditorFrame);
      fViewMenu->UnCheckEntry(kViewEditor);
      w = w - e;
   }
   Resize(w, h);

   if (savedPad) gPad = savedPad;
}

//______________________________________________________________________________
void TRootCanvas::CreateEditor()
{
   // Create embedded editor.

   if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0) {
         TVirtualPadEditor::HideEditor();
   }
   if (fClient->IsEditable()) {
      ((TGWindow*)fClient->GetRoot())->SetEditable(kFALSE);
   }
   SetEditDisabled(kEditEnable);
   fEditorFrame->SetEditable();
   gPad = Canvas();
   // next two lines are related to the old editor
   TString show = gEnv->GetValue("Canvas.ShowEditor","false");
   gEnv->SetValue("Canvas.ShowEditor","true");
   fEditor = TVirtualPadEditor::LoadEditor();
   fEditor->SetGlobal(kFALSE);
   fEditorFrame->SetEditable(0);
   SetEditDisabled(kEditDisable);

   // next line is related to the old editor
   if (show == "false") gEnv->SetValue("Canvas.ShowEditor","false");
}

//______________________________________________________________________________
void TRootCanvas::ShowToolBar(Bool_t show)
{
   // Show or hide toolbar.

   if (show && !fToolBar) {

      fToolBar = new TGToolBar(fToolDock, 60, 20, kHorizontalFrame);
      fToolDock->AddFrame(fToolBar, fHorizontal1Layout);

      Int_t spacing = 6, i;
      for (i = 0; gToolBarData[i].fPixmap; i++) {
         if (strlen(gToolBarData[i].fPixmap) == 0) {
            spacing = 6;
            continue;
         }
         fToolBar->AddButton(this, &gToolBarData[i], spacing);
         spacing = 0;
      }
      fVertical1 = new TGVertical3DLine(fToolBar);
      fVertical2 = new TGVertical3DLine(fToolBar);
      fVertical1Layout = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 4,2,0,0);
      fVertical2Layout = new TGLayoutHints(kLHintsLeft | kLHintsExpandY);
      fToolBar->AddFrame(fVertical1, fVertical1Layout);
      fToolBar->AddFrame(fVertical2, fVertical2Layout);

      spacing = 6;
      for (i = 0; gToolBarData1[i].fPixmap; i++) {
         if (strlen(gToolBarData1[i].fPixmap) == 0) {
            spacing = 6;
            continue;
         }
         fToolBar->AddButton(this, &gToolBarData1[i], spacing);
         spacing = 0;
      }
      fToolDock->MapSubwindows();
      fToolDock->Layout();
      fToolDock->SetWindowName(Form("ToolBar: %s", GetWindowName()));
      fToolDock->Connect("Docked()", "TRootCanvas", this, "AdjustSize()");
      fToolDock->Connect("Undocked()", "TRootCanvas", this, "AdjustSize()");
   }

   if (!fToolBar) return;

   UInt_t h = GetHeight();
   UInt_t sh = fToolBarSep->GetHeight();
   UInt_t dh = fToolBar->GetHeight();

   if (show) {
      ShowFrame(fToolDock);
      if (!fViewMenu->IsEntryChecked(kViewEditor)) {
         ShowFrame(fHorizontal1);
         h = h + sh;
      }
      ShowFrame(fToolBarSep);
      fViewMenu->CheckEntry(kViewToolbar);
      h = h + dh + sh;
   } else {
      if (fToolDock->IsUndocked()) {
         fToolDock->DockContainer();
         h = h + 2*sh;
      } else h = h - dh;

      HideFrame(fToolDock);
      if (!fViewMenu->IsEntryChecked(kViewEditor)) {
         HideFrame(fHorizontal1);
         h = h - sh;
      }
      HideFrame(fToolBarSep);
      h = h - sh;
      fViewMenu->UnCheckEntry(kViewToolbar);
   }
   Resize(GetWidth(), h);
}

//______________________________________________________________________________
void TRootCanvas::AdjustSize()
{
   // Keep the same canvas size while docking/undocking toolbar.

   UInt_t h = GetHeight();
   UInt_t dh = fToolBar->GetHeight();
   UInt_t sh = fHorizontal1->GetHeight();

   if (fToolDock->IsUndocked()) {
      if (!fViewMenu->IsEntryChecked(kViewEditor)) {
         HideFrame(fHorizontal1);
         h = h - sh;
      }
      HideFrame(fToolBarSep);
      h = h - dh - sh;
   } else {
      if (!fViewMenu->IsEntryChecked(kViewEditor)) {
         ShowFrame(fHorizontal1);
         h = h + sh;
      }
      ShowFrame(fToolBarSep);
      h = h + dh + sh;
   }
   Resize(GetWidth(), h);
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
      if (button == kButton1) {
         if (event->fState & kKeyShiftMask)
            fCanvas->HandleInput(EEventType(7), x, y);
         else
            fCanvas->HandleInput(kButton1Down, x, y);
      }
      if (button == kButton2)
         fCanvas->HandleInput(kButton2Down, x, y);
      if (button == kButton3) {
         fCanvas->HandleInput(kButton3Down, x, y);
         fButton = 0;  // button up is consumed by TContextMenu
      }

   } else if (event->fType == kButtonRelease) {
      if (button == kButton4)
         fCanvas->HandleInput(EEventType(5), x, y);//hack
      if (button == kButton5)
         fCanvas->HandleInput(EEventType(6), x, y);//hack
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

   if (fCanvas->HasFixedAspectRatio()) {
      // get menu height
      static Int_t dh = 0;
      if (!dh)
         dh = GetHeight() - fCanvasContainer->GetHeight();
      UInt_t h = TMath::Nint(fCanvasContainer->GetWidth()/
                             fCanvas->GetAspectRatio()) + dh;
      SetWindowSize(GetWidth(), h);
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
      fCanvas->HandleInput(kKeyPress, str[0], keysym);
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
   if (fButton == kButton2)
      fCanvas->HandleInput(kButton2Motion, x, y);

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

//______________________________________________________________________________
void TRootContainer::SavePrimitive(ofstream &out, Option_t *)
{
   // Save a canvas container as a C++ statement(s) on output stream out.

   out << endl << "   // canvas container" << endl;
   out << "   Int_t canvasID = gVirtualX->InitWindow((ULong_t)"
       << GetParent()->GetParent()->GetName() << "->GetId());" << endl;
   out << "   Window_t winC = gVirtualX->GetWindowID(canvasID);" << endl;
   out << "   TGCompositeFrame *";
   out << GetName() << " = new TGCompositeFrame(gClient,winC"
       << "," << GetParent()->GetName() << ");" << endl;
}

