// @(#)root/gl:$Name:$:$Id:$
// Author: Valery Fine & Fons Rademakers   5/10/2000 and 28/4/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootOIViewer                                                        //
//                                                                      //
// This class creates a toplevel window and an OpenInventor             //
// drawing area and context using the ROOT native GUI.                  //
//                                                                      //
// Open Inventor can be downloaded from                                 //
//  ftp://oss.sgi.com/projects/inventor/download/                       //
//                                                                      //
// Free version of OpenGL API:                                          //
//   http://sourceforge.net/project/showfiles.php?group_id=3            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoCallback.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/elements/SoCacheElement.h>
#include <X11/IntrinsicP.h>

#include "TRootOIViewer.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "TGMsgBox.h"
#include "TVirtualX.h"

#include "TROOT.h"
#include "TError.h"
#include "Buttons.h"
#include "TVirtualPad.h"
#include "TPadOpenGLView.h"
#include "TVirtualGL.h"
#include "TColor.h"
#include "TSystem.h"


XtAppContext TRootOIViewer::fgAppContext = 0;


//______________________________________________________________________________
void InventorCallback(void *d, SoAction *action)
{
   // OpenInventor call back function.

   if (!d) return;
   TRootOIViewer *currentViewer = (TRootOIViewer *)d;
   if (currentViewer) {
      if (action->isOfType(SoGLRenderAction::getClassTypeId())) {
         SoCacheElement::invalidate(action->getState());

         // gVirtualGL->SetRootLight(kFALSE);

         glEnable(GL_COLOR_MATERIAL);
         currentViewer->Paint("");
         glDisable(GL_COLOR_MATERIAL);

      } else if (action->isOfType(SoGetBoundingBoxAction::getClassTypeId())) {
         Double_t minBound[3] = { -1000, -1000, -1000 };
         Double_t maxBound[3] = {  1000,  1000,  1000 };

         // currentViewer->GetGLView()->GetRange(minBound,maxBound);
         if (minBound[0] == maxBound[0])
            SoCacheElement::invalidate(action->getState());

         ((SoGetBoundingBoxAction *)action)->
           extendBy(SbBox3f(minBound[0],minBound[1],minBound[2],
                            maxBound[0],maxBound[2],maxBound[2]));
      }
   }
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSoXtEventHandler                                                    //
//                                                                      //
// All X events that are not for any of the windows under the control   //
// of the ROOT GUI are forwarded by this class to Xt.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSoXtEventHandler : public TGUnknownWindowHandler {
private:
   TRootOIViewer  *fViewer;   // pointer back to viewer imp
public:
   TSoXtEventHandler(TRootOIViewer *c) { fViewer = c; }
   Bool_t HandleEvent(Event_t *ev)
      { return SoXt::dispatchEvent((XEvent *)gVirtualX->GetNativeEvent()); }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXtTimerHandler                                                      //
//                                                                      //
// Check every 100 ms if there is an Xt timer event to be processed.    //
// This is the best we can do since there is no way to get access to    //
// the Xt timer queue.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TXtTimerHandler : public TTimer {
public:
   TXtTimerHandler() : TTimer(100) {  }
   Bool_t Notify();
};

Bool_t TXtTimerHandler::Notify()
{
   XtInputMask m = XtAppPending(TRootOIViewer::fgAppContext);
   if ((m & XtIMTimer))
      XtAppProcessEvent(TRootOIViewer::fgAppContext, XtIMTimer);
   Reset();
   return kTRUE;
}



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOIContainer                                                         //
//                                                                      //
// Utility class used by TRootOIViewer. The TOIContainer is the frame   //
// embedded in the TGCanvas widget. The OI graphics goes into this      //
// frame. This class is used to enable input events on this graphics    //
// frame and forward the events to the TRootOIViewer handlers.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TOIContainer : public TGCompositeFrame {
private:
   TRootOIViewer  *fViewer;    // pointer back to viewer imp
public:
   TOIContainer(TRootOIViewer *c, Window_t id, const TGWindow *parent);

   Bool_t  HandleButton(Event_t *ev)
          { return SoXt::dispatchEvent((XEvent *)gVirtualX->GetNativeEvent()); }
   Bool_t  HandleConfigureNotify(Event_t *ev)
          { TGFrame::HandleConfigureNotify(ev);
            return SoXt::dispatchEvent((XEvent *)gVirtualX->GetNativeEvent()); }
   Bool_t  HandleKey(Event_t *ev)
          { return SoXt::dispatchEvent((XEvent *)gVirtualX->GetNativeEvent()); }
   Bool_t  HandleMotion(Event_t *ev)
          { return SoXt::dispatchEvent((XEvent *)gVirtualX->GetNativeEvent()); }
   Bool_t  HandleExpose(Event_t *ev)
          { return SoXt::dispatchEvent((XEvent *)gVirtualX->GetNativeEvent()); }
};

//______________________________________________________________________________
TOIContainer::TOIContainer(TRootOIViewer *c, Window_t id, const TGWindow *p)
   : TGCompositeFrame(gClient, id, p)
{
   // Create a canvas container.

   fViewer = c;
}


//ClassImp(TRootOIViewer)

//______________________________________________________________________________
TRootOIViewer::TRootOIViewer(TPadOpenGLView *pad, const char *title, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height)
{
   // Create a ROOT OpenInventor viewer.

   fGLView = pad;

   CreateViewer(title);

   Resize(width, height);

   SetDrawList(0);
}

//______________________________________________________________________________
TRootOIViewer::TRootOIViewer(TPadOpenGLView *pad, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height)
   : TGMainFrame(gClient->GetRoot(), width, height)
{
   // Create a ROOT OpenInventor viewer.

   fGLView = pad;

   CreateViewer(title);

   MoveResize(x, y, width, height);
   SetWMPosition(x, y);

   SetDrawList(0);
}

//______________________________________________________________________________
TRootOIViewer::~TRootOIViewer()
{
   // Delete ROOT OpenInventor viewer.

   SafeDelete(fInventorViewer);
   DeleteContext();

   fClient->RemoveUnknownWindowHandler(fSoXtHandler);
   delete fSoXtHandler;
   delete fXtTimerHandler;

   delete fCanvasContainer;
   delete fCanvasWindow;
   delete fCanvasLayout;
}

//______________________________________________________________________________
void TRootOIViewer::CreateViewer(const char *title)
{
   // Create the actual canvas.

   fInventorViewer = 0;
   fPaint          = kTRUE;
   fTopLevel       = 0;

   // Create canvas and canvas container to host the OpenInventor graphics
   fCanvasWindow = new TGCanvas(this, GetWidth()+4, GetHeight()+4,
                                kSunkenFrame | kDoubleBorder);

   InitXt();
   if (!fTopLevel) {
      fCanvasContainer = 0;
      fCanvasLayout    = 0;
      return;
   }
   fCanvasContainer = new TOIContainer(this, XtWindow(fTopLevel),
                                       fCanvasWindow->GetViewPort());
   fCanvasWindow->SetContainer(fCanvasContainer);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   AddFrame(fCanvasWindow, fCanvasLayout);

   SoXt::init(fTopLevel);
   fRootNode = new SoSeparator;
   fRootNode->ref();

   fGLNode = new SoSeparator;

   fMaterial = new SoMaterial;
//   fMaterial->shininess = 100.;
//   fMaterial->ambientColor.setValue(0.9, 0.1, 0.0);
//   fMaterial->diffuseColor.setValue(0.5,0.2,0.5);
//   fMaterial->transparency = 0.8;

   fRootCallback = new SoCallback;
   fRootCallback->setCallback(InventorCallback, this);

   fSoXtHandler = new TSoXtEventHandler(this);
   fClient->AddUnknownWindowHandler(fSoXtHandler);
   fXtTimerHandler = new TXtTimerHandler;
   fXtTimerHandler->TurnOn();

   fGLNode->addChild(fMaterial);
   fGLNode->addChild(fRootCallback);

   fRootNode->addChild(fGLNode);

   SoInput viewDecor;
   const char *fileDecor = "root.iv";
   if (!gSystem->AccessPathName(fileDecor) && viewDecor.openFile(fileDecor)) {
     SoSeparator *extraObjects = SoDB::readAll(&viewDecor);
     if (extraObjects) {
        fRootNode->addChild(extraObjects);
     }
   }

   fInventorViewer = new SoXtExaminerViewer(fTopLevel);
   fInventorViewer->setSceneGraph(fRootNode);
   fInventorViewer->setTitle(title);

   // Pick the background color
   TVirtualPad *thisPad = fGLView->GetPad();
   if (thisPad) {
      Color_t color = thisPad->GetFillColor();
      TColor *background = gROOT->GetColor(color);
      if (background) {
          float rgb[3];
          background->GetRGB(rgb[0],rgb[1],rgb[2]);
          fInventorViewer->setBackgroundColor(SbColor(rgb));
      }
   }
   fInventorViewer->show();

   //SoXt::show(fTopLevel);

   InitGLWindow();

   // Misc

   SetWindowName(title);
   SetIconName(title);
   SetClassHints("OIViewer", "OIViewer");

   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);

   MapSubwindows();

   // we need to use GetDefaultSize() to initialize the layout algorithm...
   Resize(GetDefaultSize());

   Show();
}

//______________________________________________________________________________
void TRootOIViewer::InitXt()
{
   // Initialize Xt.

   fDpy = (Display *) gVirtualX->GetDisplay();

   if (!fgAppContext) {
      XtToolkitInitialize();
      fgAppContext = XtCreateApplicationContext();
      int argc = 0;
      XtDisplayInitialize(fgAppContext, fDpy, 0, "Inventor", 0, 0, &argc, 0);
   }

   int xval, yval;
   unsigned int wval, hval, border, depth;
   Window root, wind = (Window) fCanvasWindow->GetViewPort()->GetId();
   XGetGeometry(fDpy, wind, &root, &xval, &yval, &wval, &hval, &border, &depth);

   //fTopLevel = XtAppCreateShell(0, "Inventor", applicationShellWidgetClass,
   fTopLevel = XtAppCreateShell(0, "Inventor", topLevelShellWidgetClass,
                                fDpy, 0, 0);

   // reparent fTopLevel into the ROOT GUI hierarchy
   XtResizeWidget(fTopLevel, 100, 100, 0);
   XtSetMappedWhenManaged(fTopLevel, False);
   XtRealizeWidget(fTopLevel);
   XSync(fDpy, False);    // I want all windows to be created now
   XReparentWindow(fDpy, XtWindow(fTopLevel), wind, xval, yval);
   XtSetMappedWhenManaged(fTopLevel, True);

   Arg reqargs[20];
   Cardinal nargs = 0;
   XtSetArg(reqargs[nargs], XtNx, xval);      nargs++;
   XtSetArg(reqargs[nargs], XtNy, yval);      nargs++;
   XtSetArg(reqargs[nargs], XtNwidth, wval);  nargs++;
   XtSetArg(reqargs[nargs], XtNheight, hval); nargs++;
   //XtSetArg(reqargs[nargs], "mappedWhenManaged", False); nargs++;
   XtSetValues(fTopLevel, reqargs, nargs);

   XtRealizeWidget(fTopLevel);
}

//______________________________________________________________________________
void TRootOIViewer::InitGLWindow()
{
   // Initialize GL window.

   if (fInventorViewer) {
      gVirtualGL->SetTrueColorMode();
   }
}

//______________________________________________________________________________
void TRootOIViewer::DeleteGLWindow()
{
   // X11 specific code to delete GL window.

   // fGLWin is destroyed when parent is destroyed.
}

//______________________________________________________________________________
void TRootOIViewer::CloseWindow()
{
   // In case window is closed via WM we get here.

   delete fGLView;  // this in turn will delete this object
   fGLView = 0;
}

//______________________________________________________________________________
void TRootOIViewer::CreateContext()
{
   // Create OpenGL context.
}

//______________________________________________________________________________
void TRootOIViewer::DeleteContext()
{
   // Delete OpenGL context.
}
//______________________________________________________________________________
void TRootOIViewer::SwapBuffers()
{
   // Swap two GL buffers
}
//______________________________________________________________________________
void TRootOIViewer::Paint(Option_t *)
{
   // Paint scene.

   MakeCurrent();
   gVirtualGL->RunGLList(1+4);
}

//______________________________________________________________________________
void TRootOIViewer::MakeCurrent()
{
   // Set this GL context the current one.

   glXMakeCurrent(fInventorViewer->getDisplay(),
                  fInventorViewer->getNormalWindow(),
                  fInventorViewer->getNormalContext());
}

