// @(#)root/gl:$Name:  $:$Id: TRootGLViewer.cxx,v 1.5 2000/10/30 11:00:40 rdm Exp $
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
// TRootGLViewer                                                        //
//                                                                      //
// This class creates a toplevel window with menubar, and an OpenGL     //
// drawing area and context using the ROOT native GUI.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRootGLViewer.h"
#include "TRootHelpDialog.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "TGMenu.h"
#include "TGWidget.h"
#include "TVirtualX.h"

#include "TROOT.h"
#include "TError.h"
#include "Buttons.h"
#include "TVirtualPad.h"
#include "TPadOpenGLView.h"
#include "TVirtualGL.h"

#include "HelpText.h"

#include "gdk/win32/gdkwin32.h"

// Canvas menu command ids
enum ERootGLViewerCommands {
   kFileNewViewer,
   kFileSave,
   kFileSaveAs,
   kFilePrint,
   kFileCloseViewer,

   kHelpAbout,
   kHelpOnViewer
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLContainer                                                         //
//                                                                      //
// Utility class used by TRootGLViewer. The TGLContainer is the frame   //
// embedded in the TGCanvas widget. The GL graphics goes into this      //
// frame. This class is used to enable input events on this graphics    //
// frame and forward the events to the TRootGLViewer handlers.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLContainer : public TGCompositeFrame {
private:
   TRootGLViewer  *fViewer;    // pointer back to viewer imp
public:
   TGLContainer(TRootGLViewer *c, Window_t id, const TGWindow *parent);

   Bool_t  HandleButton(Event_t *ev)
                { return fViewer->HandleContainerButton(ev); }
   Bool_t  HandleConfigureNotify(Event_t *ev)
                { TGFrame::HandleConfigureNotify(ev);
                  return fViewer->HandleContainerConfigure(ev); }
   Bool_t  HandleKey(Event_t *ev)
                { return fViewer->HandleContainerKey(ev); }
   Bool_t  HandleMotion(Event_t *ev)
                { return fViewer->HandleContainerMotion(ev); }
   Bool_t  HandleExpose(Event_t *ev)
                { return fViewer->HandleContainerExpose(ev); }
};

//______________________________________________________________________________
TGLContainer::TGLContainer(TRootGLViewer *c, Window_t id, const TGWindow *p)
   : TGCompositeFrame(gClient, id, p)
{
   // Create a canvas container.

   fViewer = c;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask,
                    kNone, kNone);

   gVirtualX->SelectInput(fId, kKeyPressMask | kExposureMask | kPointerMotionMask |
#ifndef GDK_WIN32
                     kStructureNotifyMask);
#else
                     kKeyReleaseMask | kStructureNotifyMask);
   gVirtualX->SetInputFocus(fId);
#endif
}

//ClassImp(TRootGLViewer)

//______________________________________________________________________________
TRootGLViewer::TRootGLViewer(TPadOpenGLView *pad, const char *title, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height)
{
   // Create a ROOT GL viewer.

   fGLView = pad;
   fPaint  = kTRUE;

   CreateViewer(title);

   Resize(width, height);

   SetDrawList(0);
}

//______________________________________________________________________________
TRootGLViewer::TRootGLViewer(TPadOpenGLView *pad, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height)
{
   // Create a ROOT GL viewer.

   fGLView = pad;
   fPaint  = kTRUE;

   CreateViewer(title);

   MoveResize(x, y, width, height);
   SetWMPosition(x, y);

   SetDrawList(0);
}

//______________________________________________________________________________
void TRootGLViewer::CreateViewer(const char *name)
{
   // Create the actual canvas.

   fButton  = 0;

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
   fFileMenu->DisableEntry(kFileSave);
   fFileMenu->DisableEntry(kFileSaveAs);
   fFileMenu->DisableEntry(kFilePrint);

   fHelpMenu = new TGPopupMenu(fClient->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...",           kHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("Help On OpenGL Viewer...", kHelpOnViewer);

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
   fCanvasWindow = new TGCanvas(this, GetWidth()+4, GetHeight()+4,
                                kSunkenFrame | kDoubleBorder);
   InitGLWindow();
   fCanvasContainer = new TGLContainer(this, (Window_t)fGLWin, fCanvasWindow->GetViewPort());
   fCanvasWindow->SetContainer(fCanvasContainer);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   AddFrame(fCanvasWindow, fCanvasLayout);

   // Misc

   SetWindowName(name);
   SetIconName(name);
   SetClassHints("GLViewer", "GLViewer");

   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);

   MapSubwindows();

   // we need to use GetDefaultSize() to initialize the layout algorithm...
   Resize(GetDefaultSize());

   Show();
}

//______________________________________________________________________________
TRootGLViewer::~TRootGLViewer()
{
   // Delete ROOT GL viewer.

   DeleteContext();
   DeleteGLWindow();

   delete fCanvasContainer;
   delete fCanvasWindow;
   delete fFileMenu;
   delete fHelpMenu;
   delete fMenuBar;
   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;
   delete fCanvasLayout;
}

//______________________________________________________________________________
void TRootGLViewer::InitGLWindow()
{
   // X11 specific code to initialize GL window.
   int xval, yval;
   int wval, hval, depth;

   gVirtualGL->SetTrueColorMode();

   GdkWindow *root, *wind = (GdkWindow *) fCanvasWindow->GetViewPort()->GetId();

   gdk_window_get_geometry((GdkWindow *)wind, &xval, &yval, &wval, &hval, &depth);

   // window attributes
   ULong_t mask;
   GdkWindowAttr attr;

   attr.width = wval;
   attr.height = hval;
   attr.x = xval;
   attr.y = yval;
   attr.wclass = GDK_INPUT_OUTPUT;
   attr.event_mask = 0L; //GDK_ALL_EVENTS_MASK;
   attr.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK | GDK_KEY_PRESS_MASK | GDK_KEY_RELEASE_MASK;
   attr.colormap = gdk_colormap_get_system();
//   attr.event_mask = 0;
   mask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP | GDK_WA_WMCLASS | 
       GDK_WA_NOREDIR;

   attr.window_type = GDK_WINDOW_CHILD;
   fGLWin = gdk_window_new((GdkWindow *) wind, &attr, mask);
   gdk_window_set_events(fGLWin,(GdkEventMask)0L);
   gdk_window_show((GdkWindow *) fGLWin);

   static PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),  // size of this pfd
		1,                              // version number
		PFD_DRAW_TO_WINDOW |            // support window
		  PFD_SUPPORT_OPENGL |          // support OpenGL
		  PFD_DOUBLEBUFFER,             // double buffered
		PFD_TYPE_RGBA,                  // RGBA type
		24,                             // 24-bit color depth
		0, 0, 0, 0, 0, 0,               // color bits ignored
		0,                              // no alpha buffer
		0,                              // shift bit ignored
		0,                              // no accumulation buffer
		0, 0, 0, 0,                     // accum bits ignored
		32,                             // 32-bit z-buffer
		0,                              // no stencil buffer
		0,                              // no auxiliary buffer
		PFD_MAIN_PLANE,                 // main layer
		0,                              // reserved
		0, 0, 0                         // layer masks ignored
	};

	int pixelformat;

	if ( (pixelformat = ChoosePixelFormat(GetWindowDC((HWND)GDK_DRAWABLE_XID(fGLWin)), 
        &pfd)) == 0 )
	{
      Error("InitGLWindow", "Barf! ChoosePixelFormat Failed");
	}
	if ( (SetPixelFormat(GetWindowDC((HWND)GDK_DRAWABLE_XID(fGLWin)), pixelformat,
        &pfd)) == FALSE )
	{
      Error("InitGLWindow", "Barf! SetPixelFormat Failed");
	}
   CreateContext();

   MakeCurrent();
}

//______________________________________________________________________________
void TRootGLViewer::DeleteGLWindow()
{
   // X11 specific code to delete GL window.

   // fGLWin is destroyed when parent is destroyed.
}

//______________________________________________________________________________
void TRootGLViewer::CloseWindow()
{
   // In case window is closed via WM we get here.

   delete fGLView;  // this in turn will delete this object
   fGLView = 0;
}

//______________________________________________________________________________
void TRootGLViewer::CreateContext()
{
   // Create OpenGL context.

    fCtx = wglCreateContext(GetWindowDC((HWND)GDK_DRAWABLE_XID(fGLWin)));

}

//______________________________________________________________________________
void TRootGLViewer::DeleteContext()
{
   // Delete OpenGL context.

   if (fCtx) {
      MakeCurrent();
      wglDeleteContext(fCtx);
      fCtx = 0;
   }
}

//______________________________________________________________________________
void TRootGLViewer::MakeCurrent()
{
   // Set this GL context the current one.

   wglMakeCurrent(GetWindowDC((HWND)GDK_DRAWABLE_XID(fGLWin)), fCtx);
}

//______________________________________________________________________________
void TRootGLViewer::SwapBuffers()
{
   // Swap two GL buffers.

   wglSwapLayerBuffers(GetWindowDC((HWND)GDK_DRAWABLE_XID(fGLWin)), WGL_SWAP_MAIN_PLANE);
   // for help debugging, report any OpenGL errors that occur per frame
}

//______________________________________________________________________________
Bool_t TRootGLViewer::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
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
                     if (fGLView && fGLView->GetPad())
                        fGLView->GetPad()->x3d("OPENGL");
                     else
                        fFileMenu->DisableEntry(kFileNewViewer);
                     break;
                  case kFileSave:
                  case kFileSaveAs:
                  case kFilePrint:
                     break;
                  case kFileCloseViewer:
                     SendCloseMessage();
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
                     hd = new TRootHelpDialog(this, "Help on OpenGL Viewer...", 600, 400);
                     hd->SetText(gHelpGLViewer);
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
Bool_t TRootGLViewer::HandleContainerButton(Event_t *event)
{
   // Handle mouse button events in the canvas container.

   Int_t button = event->fCode;
   Int_t x = event->fX;
   Int_t y = event->fY;

   if (event->fType == kButtonPress) {
      fButton = button;
      if (button == kButton1)
         HandleInput(kButton1Down, x, y);
      if (button == kButton2)
         HandleInput(kButton2Down, x, y);
      if (button == kButton3) {
         HandleInput(kButton3Down, x, y);
         fButton = 0;  // button up is consumed by TContextMenu
      }

   } else if (event->fType == kButtonRelease) {
      if (button == kButton1)
         HandleInput(kButton1Up, x, y);
      if (button == kButton2)
         HandleInput(kButton2Up, x, y);
      if (button == kButton3)
         HandleInput(kButton3Up, x, y);
#ifdef GDK_WIN32
      gVirtualX->SetInputFocus((Window_t)fGLWin);
#endif

      fButton = 0;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGLViewer::HandleContainerConfigure(Event_t *event)
{
   // Handle configure (i.e. resize) event.

   gdk_window_resize((GdkWindow *)fGLWin, event->fWidth, event->fHeight);

   MakeCurrent();
   glViewport(0, 0, (GLint) event->fWidth, (GLint) event->fHeight);
   if (fGLView) fGLView->Size((Int_t) event->fWidth, (Int_t) event->fHeight);

   Update();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGLViewer::HandleContainerKey(Event_t *event)
{
   // Handle keyboard events in the canvas container.

  if (event->fType == kGKeyPress) {
      fButton = event->fCode;
      UInt_t keysym;
      char str[2];
      gVirtualX->LookupString(event, str, sizeof(str), keysym);
      HandleInput(kKeyPress, str[0], 0);
   } else if (event->fType == kKeyRelease)
      fButton = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGLViewer::HandleContainerMotion(Event_t *event)
{
   // Handle mouse motion event in the canvas container.

   Int_t x = event->fX;
   Int_t y = event->fY;

   if (fButton == 0)
      HandleInput(kMouseMotion, x, y);
   if (fButton == kButton1)
      HandleInput(kButton1Motion, x, y);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGLViewer::HandleContainerExpose(Event_t *event)
{
   // Handle expose events.

   if (event->fCount == 0)
      Update();

   return kTRUE;
}

