// @(#)root/gl:$Name:  $:$Id: TRootGLViewer.cxx,v 1.2 2000/06/05 07:28:06 brun Exp $
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
                     kStructureNotifyMask);
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
   fCanvasContainer = new TGLContainer(this, fGLWin, fCanvasWindow->GetViewPort());
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

   fDpy = (Display *) gVirtualX->GetDisplay();

   gVirtualGL->SetTrueColorMode();
   static int dblBuf[] = {
       GLX_DOUBLEBUFFER,
#ifdef STEREO_GL
       GLX_STEREO,
#endif
       GLX_RGBA, GLX_DEPTH_SIZE, 16,
       GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1,
       None
   };
   static int *snglBuf = &dblBuf[1];

   fVisInfo = glXChooseVisual(fDpy, DefaultScreen(fDpy), dblBuf);
   if (fVisInfo == 0)
      fVisInfo = glXChooseVisual(fDpy, DefaultScreen(fDpy), snglBuf);

   if (fVisInfo == 0)
      Error("InitGLWindow", "Barf! No good visual");

   CreateContext();

   int xval, yval;
   unsigned int wval, hval, border, depth;
   Window root, wind = (Window) fCanvasWindow->GetViewPort()->GetId();
   XGetGeometry(fDpy, wind, &root, &xval, &yval, &wval, &hval, &border, &depth);

   // window attributes
   ULong_t mask;
   XSetWindowAttributes attr;

   attr.background_pixel = 0;
   attr.border_pixel = 0;
   attr.colormap = XCreateColormap(fDpy, root, fVisInfo->visual, AllocNone);
   attr.event_mask = NoEventMask;
   attr.backing_store = Always;
   attr.bit_gravity = NorthWestGravity;
   mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask |
          CWBackingStore | CWBitGravity;

   fGLWin = XCreateWindow(fDpy, wind, xval, yval, wval, hval,
                          0, fVisInfo->depth, InputOutput,
                          fVisInfo->visual, mask, &attr);

   XMapWindow(fDpy, fGLWin);

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
   // Forward message to central message handler as button event.

   SendMessage(this, MK_MSG(kC_COMMAND, kCM_BUTTON), kFileCloseViewer, 0);
}

//______________________________________________________________________________
void TRootGLViewer::CreateContext()
{
   // Create OpenGL context.

   fCtx = glXCreateContext(fDpy, fVisInfo, None, GL_TRUE);
}

//______________________________________________________________________________
void TRootGLViewer::DeleteContext()
{
   // Delete OpenGL context.

   if (fCtx) {
      MakeCurrent();
      glXDestroyContext(fDpy, fCtx);
      fCtx = 0;
   }
}

//______________________________________________________________________________
void TRootGLViewer::MakeCurrent()
{
   // Set this GL context the current one.

   glXMakeCurrent(fDpy, fGLWin, fCtx);
}

//______________________________________________________________________________
void TRootGLViewer::SwapBuffers()
{
   // Swap two GL buffers.

   glXSwapBuffers(fDpy, fGLWin);
   if (!glXIsDirect(fDpy, fCtx)) {
      glFinish();
   }

   // for help debugging, report any OpenGL errors that occur per frame
   GLenum error;
   while ((error = glGetError()) != GL_NO_ERROR)
      Error("SwapBuffers", "GL error: %s", gluErrorString(error));
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
                     if (fGLView) fGLView->GetPad()->x3d("OPENGL");
                     break;
                  case kFileSave:
                  case kFileSaveAs:
                  case kFilePrint:
                     break;
                  case kFileCloseViewer:
                     delete fGLView;  // this in turn will delete this object
                     fGLView = 0;
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

      fButton = 0;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGLViewer::HandleContainerConfigure(Event_t *event)
{
   // Handle configure (i.e. resize) event.

   XResizeWindow(fDpy, fGLWin, event->fWidth, event->fHeight);

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

