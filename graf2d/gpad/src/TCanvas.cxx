// @(#)root/gpad:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string.h>
#include <stdlib.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TStyle.h"
#include "TText.h"
#include "TBox.h"
#include "TCanvasImp.h"
#include "TDialogCanvas.h"
#include "TGuiFactory.h"
#include "TEnv.h"
#include "TError.h"
#include "TContextMenu.h"
#include "TControlBar.h"
#include "TInterpreter.h"
#include "TApplication.h"
#include "TColor.h"
#include "TVirtualPadEditor.h"
#include "TVirtualViewer3D.h"
#include "TPadPainter.h"
#include "TVirtualGL.h"
#include "TVirtualPS.h"
#include "TObjectSpy.h"
#include "TAxis.h"
#include "TView.h"

#include "TVirtualMutex.h"

class TCanvasInit {
public:
   TCanvasInit() { TApplication::NeedGraphicsLibs(); }
};
static TCanvasInit gCanvasInit;


//*-*x16 macros/layout_canvas

Bool_t TCanvas::fgIsFolder = kFALSE;

const Size_t kDefaultCanvasSize   = 20;

ClassImpQ(TCanvas)


//______________________________________________________________________________
/* Begin_Html
<center><h2>The Canvas class</h2></center>

A Canvas is an area mapped to a window directly under the control of the display
manager. A ROOT session may have several canvases open at any given time.
<p>
A Canvas may be subdivided into independent graphical areas: the <b>Pads</b>.
A canvas has a default pad which has the name of the canvas itself.
An example of a Canvas layout is sketched in the picture below.

<pre>
     ***********************************************************************
     *                       Menus bar for Canvas                          *
     ***********************************************************************
     *                                                                     *
     *  ************************************    *************************  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *              Pad 1               *    *        Pad 2          *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  *                                  *    *                       *  *
     *  ************************************    *************************  *
     *                                                                     *
     ***********************************************************************
</pre>

This canvas contains two pads named P1 and P2. Both Canvas, P1 and P2 can be
moved, grown, shrinked using the normal rules of the Display manager.
<p>
The image below shows a canvas with 4 pads:

<center>
<img src="gif/canvas_layout.gif">
</center>

Once objects have been drawn in a canvas, they can be edited/moved by pointing
directly to them. The cursor shape is changed to suggest the type of action that
one can do on this object. Clicking with the right mouse button on an object
pops-up a contextmenu with a complete list of actions possible on this object.
<p>
A graphical editor may be started from the canvas "View" menu under the menu
entry "Toolbar".
<p>
An interactive HELP is available by clicking on the HELP button at the top right
of the canvas. It gives a short explanation about the canvas' menus.
<p>
A canvas may be automatically divided into pads via <tt>TPad::Divide</tt>.
<p>
At creation time, in interactive mode, the canvas size defines the size of the
canvas window (including the window manager's decoration). To define precisely
the graphics area size of a canvas, the following four lines of code should be
used:
<pre>
   {
      Double_t w = 600;
      Double_t h = 600;
      TCanvas * c1 = new TCanvas("c", "c", w, h);
      c->SetWindowSize(w + (w - c->GetWw()), h + (h - c->GetWh()));
   }
</pre>
in batch mode simply do:
<pre>
      c->SetCanvasSize(w,h);
</pre>
End_Html */


//______________________________________________________________________________
TCanvas::TCanvas(Bool_t build) : TPad(), fDoubleBuffer(0)
{
   // Canvas default constructor.

   fPainter = 0;
   fUseGL = gStyle->GetCanvasPreferGL();

   if (!build || TClass::IsCallingNew() != TClass::kRealNew) {
      Constructor();
   } else {
      const char *defcanvas = gROOT->GetDefCanvasName();
      char *cdef;

      TList *lc = (TList*)gROOT->GetListOfCanvases();
      if (lc->FindObject(defcanvas)) {
         Int_t n = lc->GetSize()+1;
         while (lc->FindObject(Form("%s_n%d",defcanvas,n))) n++;
         cdef = StrDup(Form("%s_n%d",defcanvas,n));
      } else {
         cdef = StrDup(Form("%s",defcanvas));
      }
      Constructor(cdef, cdef, 1);
      delete [] cdef;
   }
}

//______________________________________________________________________________
void TCanvas::Constructor()
{
   // Canvas default constructor

   if (gThreadXAR) {
      void *arr[2];
      arr[1] = this;
      if ((*gThreadXAR)("CANV", 2, arr, 0)) return;
   }

   fCanvas    = 0;
   fCanvasID  = -1;
   fCanvasImp = 0;
   fBatch     = kTRUE;
   fUpdating  = kFALSE;

   fContextMenu   = 0;
   fSelected      = 0;
   fClickSelected = 0;
   fSelectedPad   = 0;
   fClickSelectedPad = 0;
   fPadSave       = 0;
   SetBit(kAutoExec);
   SetBit(kShowEditor);
   SetBit(kShowToolBar);
}

//______________________________________________________________________________
TCanvas::TCanvas(const char *name, Int_t ww, Int_t wh, Int_t winid) : TPad(), fDoubleBuffer(0)
{
   // Create an embedded canvas, i.e. a canvas that is in a TGCanvas widget
   // which is placed in a TGFrame. This ctor is only called via the
   // TRootEmbeddedCanvas class.
   //
   //  If "name" starts with "gl" the canvas is ready to receive GL output.

   fPainter = 0;
   Init();

   fCanvasID     = winid;
   fWindowTopX   = 0;
   fWindowTopY   = 0;
   fWindowWidth  = ww;
   fWindowHeight = wh;
   fCw           = ww + 4;
   fCh           = wh +28;
   fBatch        = kFALSE;
   fUpdating     = kFALSE;

   //This is a very special ctor. A window exists already!
   //Can create painter now.
   fUseGL = gStyle->GetCanvasPreferGL();

   if (fUseGL) {
      fGLDevice = gGLManager->CreateGLContext(winid);
      if (fGLDevice == -1)
         fUseGL = kFALSE;
   }

   CreatePainter();

   fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
   if (!fCanvasImp) return;
   SetName(name);
   Build();
}

//_____________________________________________________________________________
TCanvas::TCanvas(const char *name, const char *title, Int_t form) : TPad(), fDoubleBuffer(0)
{
   //  Create a new canvas with a predefined size form.
   //  If form < 0  the menubar is not shown.
   //
   //  form = 1    700x500 at 10,10 (set by TStyle::SetCanvasDefH,W,X,Y)
   //  form = 2    500x500 at 20,20
   //  form = 3    500x500 at 30,30
   //  form = 4    500x500 at 40,40
   //  form = 5    500x500 at 50,50
   //
   //  If "name" starts with "gl" the canvas is ready to receive GL output.

   fPainter = 0;
   fUseGL = gStyle->GetCanvasPreferGL();

   Constructor(name, title, form);
}

//_____________________________________________________________________________
void TCanvas::Constructor(const char *name, const char *title, Int_t form)
{
   //  Create a new canvas with a predefined size form.
   //  If form < 0  the menubar is not shown.
   //
   //  form = 1    700x500 at 10,10 (set by TStyle::SetCanvasDefH,W,X,Y)
   //  form = 2    500x500 at 20,20
   //  form = 3    500x500 at 30,30
   //  form = 4    500x500 at 40,40
   //  form = 5    500x500 at 50,50

   if (gThreadXAR) {
      void *arr[6];
      static Int_t ww = 500;
      static Int_t wh = 500;
      arr[1] = this; arr[2] = (void*)name; arr[3] = (void*)title; arr[4] =&ww; arr[5] = &wh;
      if ((*gThreadXAR)("CANV", 6, arr, 0)) return;
   }

   Init();
   SetBit(kMenuBar,1);
   if (form < 0) {
      form     = -form;
      SetBit(kMenuBar,0);
   }

   fCanvas = this;

   fCanvasID = -1;
   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(name);
   if (old && old->IsOnHeap()) {
      Warning("Constructor","Deleting canvas with same name: %s",name);
      delete old;
   }
   if (!name[0] || gROOT->IsBatch()) {   //We are in Batch mode
      fWindowTopX = fWindowTopY = 0;
      if (form == 1) {
         fWindowWidth  = gStyle->GetCanvasDefW();
         fWindowHeight = gStyle->GetCanvasDefH();
      } else {
         fWindowWidth  = 500;
         fWindowHeight = 500;
      }
      fCw           = fWindowWidth;
      fCh           = fWindowHeight;
      fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
      if (!fCanvasImp) return;
      fBatch        = kTRUE;
   } else {                  //normal mode with a screen window
      Float_t cx = gStyle->GetScreenFactor();
      if (form < 1 || form > 5) form = 1;
      if (form == 1) {
         UInt_t uh = UInt_t(cx*gStyle->GetCanvasDefH());
         UInt_t uw = UInt_t(cx*gStyle->GetCanvasDefW());
         Int_t  ux = Int_t(cx*gStyle->GetCanvasDefX());
         Int_t  uy = Int_t(cx*gStyle->GetCanvasDefY());
         fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, ux, uy, uw, uh);
      }
      fCw = 500;
      fCh = 500;
      if (form == 2) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 20, 20, UInt_t(cx*500), UInt_t(cx*500));
      if (form == 3) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 30, 30, UInt_t(cx*500), UInt_t(cx*500));
      if (form == 4) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 40, 40, UInt_t(cx*500), UInt_t(cx*500));
      if (form == 5) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 50, 50, UInt_t(cx*500), UInt_t(cx*500));
      if (!fCanvasImp) return;

      if (!gROOT->IsBatch() && fCanvasID == -1)
         fCanvasID = fCanvasImp->InitWindow();

      fCanvasImp->ShowMenuBar(TestBit(kMenuBar));
      fBatch = kFALSE;
   }

   CreatePainter();

   SetName(name);
   SetTitle(title); // requires fCanvasImp set
   Build();

   // Popup canvas
   fCanvasImp->Show();
}

//_____________________________________________________________________________
TCanvas::TCanvas(const char *name, const char *title, Int_t ww, Int_t wh) : TPad(), fDoubleBuffer(0)
{
   //  Create a new canvas at a random position.
   //
   //  ww is the canvas size in pixels along X
   //      (if ww < 0  the menubar is not shown)
   //  wh is the canvas size in pixels along Y
   //
   //  If "name" starts with "gl" the canvas is ready to receive GL output.
   fPainter = 0;
   fUseGL = gStyle->GetCanvasPreferGL();

   Constructor(name, title, ww, wh);
}

//_____________________________________________________________________________
void TCanvas::Constructor(const char *name, const char *title, Int_t ww, Int_t wh)
{
   //  Create a new canvas at a random position.
   //
   //  ww is the canvas size in pixels along X
   //      (if ww < 0  the menubar is not shown)
   //  wh is the canvas size in pixels along Y

   if (gThreadXAR) {
      void *arr[6];
      arr[1] = this; arr[2] = (void*)name; arr[3] = (void*)title; arr[4] =&ww; arr[5] = &wh;
      if ((*gThreadXAR)("CANV", 6, arr, 0)) return;
   }

   Init();
   SetBit(kMenuBar,1);
   if (ww < 0) {
      ww       = -ww;
      SetBit(kMenuBar,0);
   }
   fCw       = ww;
   fCh       = wh;
   fCanvasID = -1;
   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(name);
   if (old && old->IsOnHeap()) {
      Warning("Constructor","Deleting canvas with same name: %s",name);
      delete old;
   }
   if (!name[0] || gROOT->IsBatch()) {   //We are in Batch mode
      fWindowTopX   = fWindowTopY = 0;
      fWindowWidth  = ww;
      fWindowHeight = wh;
      fCw           = ww;
      fCh           = wh;
      fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
      if (!fCanvasImp) return;
      fBatch        = kTRUE;
   } else {
      Float_t cx = gStyle->GetScreenFactor();
      fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, UInt_t(cx*ww), UInt_t(cx*wh));
      if (!fCanvasImp) return;

      if (!gROOT->IsBatch() && fCanvasID == -1)
         fCanvasID = fCanvasImp->InitWindow();

      fCanvasImp->ShowMenuBar(TestBit(kMenuBar));
      fBatch = kFALSE;
   }

   CreatePainter();

   SetName(name);
   SetTitle(title); // requires fCanvasImp set
   Build();

   // Popup canvas
   fCanvasImp->Show();
}

//_____________________________________________________________________________
TCanvas::TCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy, Int_t ww, Int_t wh)
        : TPad(), fDoubleBuffer(0)
{
   //  Create a new canvas.
   //
   //  wtopx,wtopy are the pixel coordinates of the top left corner of
   //  the canvas (if wtopx < 0) the menubar is not shown)
   //  ww is the canvas size in pixels along X
   //  wh is the canvas size in pixels along Y
   //
   //  If "name" starts with "gl" the canvas is ready to receive GL output.

   fPainter = 0;
   fUseGL = gStyle->GetCanvasPreferGL();

   Constructor(name, title, wtopx, wtopy, ww, wh);
}

//_____________________________________________________________________________
void TCanvas::Constructor(const char *name, const char *title, Int_t wtopx,
                          Int_t wtopy, Int_t ww, Int_t wh)
{
   //  Create a new canvas.
   //
   //  wtopx,wtopy are the pixel coordinates of the top left corner of
   //  the canvas (if wtopx < 0) the menubar is not shown)
   //  ww is the canvas size in pixels along X
   //  wh is the canvas size in pixels along Y

   if (gThreadXAR) {
      void *arr[8];
      arr[1] = this;   arr[2] = (void*)name;   arr[3] = (void*)title;
      arr[4] = &wtopx; arr[5] = &wtopy; arr[6] = &ww; arr[7] = &wh;
      if ((*gThreadXAR)("CANV", 8, arr, 0)) return;
   }

   Init();
   SetBit(kMenuBar,1);
   if (wtopx < 0) {
      wtopx    = -wtopx;
      SetBit(kMenuBar,0);
   }
   fCw       = ww;
   fCh       = wh;
   fCanvasID = -1;
   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(name);
   if (old && old->IsOnHeap()) {
      Warning("Constructor","Deleting canvas with same name: %s",name);
      delete old;
   }
   if (!name[0] || gROOT->IsBatch()) {   //We are in Batch mode
      fWindowTopX   = fWindowTopY = 0;
      fWindowWidth  = ww;
      fWindowHeight = wh;
      fCw           = ww;
      fCh           = wh;
      fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
      if (!fCanvasImp) return;
      fBatch        = kTRUE;
   } else {                   //normal mode with a screen window
      Float_t cx = gStyle->GetScreenFactor();
      fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, Int_t(cx*wtopx), Int_t(cx*wtopy), UInt_t(cx*ww), UInt_t(cx*wh));
      if (!fCanvasImp) return;

      if (!gROOT->IsBatch() && fCanvasID == -1)
         fCanvasID = fCanvasImp->InitWindow();

      fCanvasImp->ShowMenuBar(TestBit(kMenuBar));
      fBatch = kFALSE;
   }

   CreatePainter();

   SetName(name);
   SetTitle(title); // requires fCanvasImp set
   Build();

   // Popup canvas
   fCanvasImp->Show();
}

//_____________________________________________________________________________
void TCanvas::Init()
{
   // Initialize the TCanvas members. Called by all constructors.

   // Make sure the application environment exists. It is need for graphics
   // (colors are initialized in the TApplication ctor).
   if (!gApplication)
      TApplication::CreateApplication();

   // Load and initialize graphics libraries if
   // TApplication::NeedGraphicsLibs() has been called by a
   // library static initializer.
   if (gApplication)
      gApplication->InitializeGraphics();

   // Get some default from .rootrc. Used in fCanvasImp->InitWindow().
   fHighLightColor     = gEnv->GetValue("Canvas.HighLightColor", kRed);
   SetBit(kMoveOpaque,   gEnv->GetValue("Canvas.MoveOpaque", 0));
   SetBit(kResizeOpaque, gEnv->GetValue("Canvas.ResizeOpaque", 0));
   if (gEnv->GetValue("Canvas.ShowEventStatus", kFALSE)) SetBit(kShowEventStatus);
   if (gEnv->GetValue("Canvas.ShowToolTips", kFALSE)) SetBit(kShowToolTips);
   if (gEnv->GetValue("Canvas.ShowToolBar", kFALSE)) SetBit(kShowToolBar);
   if (gEnv->GetValue("Canvas.ShowEditor", kFALSE)) SetBit(kShowEditor);
   if (gEnv->GetValue("Canvas.AutoExec", kTRUE)) SetBit(kAutoExec);

   // Fill canvas ROOT data structure
   fXsizeUser = 0;
   fYsizeUser = 0;
   fXsizeReal = kDefaultCanvasSize;
   fYsizeReal = kDefaultCanvasSize;

   fDISPLAY         = "$DISPLAY";
   fUpdating        = kFALSE;
   fRetained        = kTRUE;
   fSelected        = 0;
   fClickSelected   = 0;
   fSelectedX       = 0;
   fSelectedY       = 0;
   fSelectedPad     = 0;
   fClickSelectedPad= 0;
   fPadSave         = 0;
   fEvent           = -1;
   fEventX          = -1;
   fEventY          = -1;
   fContextMenu     = 0;
}

//_____________________________________________________________________________
void TCanvas::Build()
{
   // Build a canvas. Called by all constructors.

   // Get window identifier
   if (fCanvasID == -1 && fCanvasImp)
      fCanvasID = fCanvasImp->InitWindow();
   if (fCanvasID == -1) return;

   if (fCw !=0 && fCh !=0) {
      if (fCw < fCh) fXsizeReal = fYsizeReal*Float_t(fCw)/Float_t(fCh);
      else           fYsizeReal = fXsizeReal*Float_t(fCh)/Float_t(fCw);
   }

   // Set Pad parameters
   gPad            = this;
   fCanvas         = this;
   fMother         = (TPad*)gPad;

   if (!IsBatch()) {    //normal mode with a screen window
      // Set default physical canvas attributes
      //Should be done via gVirtualX, not via fPainter (at least now). No changes here.
      gVirtualX->SelectWindow(fCanvasID);
      gVirtualX->SetFillColor(1);         //Set color index for fill area
      gVirtualX->SetLineColor(1);         //Set color index for lines
      gVirtualX->SetMarkerColor(1);       //Set color index for markers
      gVirtualX->SetTextColor(1);         //Set color index for text
      // Clear workstation
      gVirtualX->ClearWindow();

      // Set Double Buffer on by default
      SetDoubleBuffer(1);

      // Get effective window parameters (with borders and menubar)
      fCanvasImp->GetWindowGeometry(fWindowTopX, fWindowTopY,
                                    fWindowWidth, fWindowHeight);

      // Get effective canvas parameters without borders
      Int_t dum1, dum2;
      gVirtualX->GetGeometry(fCanvasID, dum1, dum2, fCw, fCh);

      fContextMenu = new TContextMenu("ContextMenu");
   } else {
      // Make sure that batch interactive canvas sizes are the same
      fCw -= 4;
      fCh -= 28;
   }
   gROOT->GetListOfCanvases()->Add(this);

   if (!fPrimitives) {
      fPrimitives     = new TList;
      SetFillColor(gStyle->GetCanvasColor());
      SetFillStyle(1001);
      SetGrid(gStyle->GetPadGridX(),gStyle->GetPadGridY());
      SetTicks(gStyle->GetPadTickX(),gStyle->GetPadTickY());
      SetLogx(gStyle->GetOptLogx());
      SetLogy(gStyle->GetOptLogy());
      SetLogz(gStyle->GetOptLogz());
      SetBottomMargin(gStyle->GetPadBottomMargin());
      SetTopMargin(gStyle->GetPadTopMargin());
      SetLeftMargin(gStyle->GetPadLeftMargin());
      SetRightMargin(gStyle->GetPadRightMargin());
      SetBorderSize(gStyle->GetCanvasBorderSize());
      SetBorderMode(gStyle->GetCanvasBorderMode());
      fBorderMode=gStyle->GetCanvasBorderMode(); // do not call SetBorderMode (function redefined in TCanvas)
      SetPad(0, 0, 1, 1);
      Range(0, 0, 1, 1);   //pad range is set by default to [0,1] in x and y

      TVirtualPadPainter *vpp = GetCanvasPainter();
      if (vpp) vpp->SelectDrawable(fPixmapID);//gVirtualX->SelectPixmap(fPixmapID);    //pixmap must be selected
      PaintBorder(GetFillColor(), kTRUE);    //paint background
   }

   // transient canvases have typically no menubar and should not get
   // by default the event status bar (if set by default)
   if (TestBit(kMenuBar) && fCanvasImp) {
      if (TestBit(kShowEventStatus)) fCanvasImp->ShowStatusBar(kTRUE);
      // ... and toolbar + editor
      if (TestBit(kShowToolBar))     fCanvasImp->ShowToolBar(kTRUE);
      if (TestBit(kShowEditor))      fCanvasImp->ShowEditor(kTRUE);
      if (TestBit(kShowToolTips))    fCanvasImp->ShowToolTips(kTRUE);
   }
}

//______________________________________________________________________________
TCanvas::~TCanvas()
{
   // Canvas destructor

   Destructor();
}

//______________________________________________________________________________
void TCanvas::Browse(TBrowser *b)
{
   // Browse.

   Draw();
   cd();
   if (fgIsFolder) fPrimitives->Browse(b);
}

//______________________________________________________________________________
void TCanvas::Destructor()
{
   // Actual canvas destructor.

   if (gThreadXAR) {
      void *arr[2];
      arr[1] = this;
      if ((*gThreadXAR)("CDEL", 2, arr, 0)) return;
   }

   if (!TestBit(kNotDeleted)) return;

   if (fContextMenu) { delete fContextMenu; fContextMenu = 0; }
   if (!gPad) return;

   Close();

   //If not yet (batch mode?).
   delete fPainter;
}

//______________________________________________________________________________
TVirtualPad *TCanvas::cd(Int_t subpadnumber)
{
   // Set current canvas & pad. Returns the new current pad,
   // or 0 in case of failure.
   // See TPad::cd() for an explanation of the parameter.

   if (fCanvasID == -1) return 0;

   TPad::cd(subpadnumber);

   // in case doublebuffer is off, draw directly onto display window
   if (!IsBatch()) {
      if (!fDoubleBuffer)
         gVirtualX->SelectWindow(fCanvasID);//Ok, does not matter for glpad.
   }
   return gPad;
}

//______________________________________________________________________________
void TCanvas::Clear(Option_t *option)
{
   // Remove all primitives from the canvas.
   // If option "D" is specified, direct subpads are cleared but not deleted.
   // This option is not recursive, i.e. pads in direct subpads are deleted.

   if (fCanvasID == -1) return;

   R__LOCKGUARD2(gROOTMutex);

   TString opt = option;
   opt.ToLower();
   if (opt.Contains("d")) {
      // clear subpads, but do not delete pads in case the canvas
      // has been divided (note: option "D" is propagated so could cause
      // conflicts for primitives using option "D" for something else)
      if (fPrimitives) {
         TIter next(fPrimitives);
         TObject *obj;
         while ((obj=next())) {
            obj->Clear(option);
         }
      }
   } else {
      //default, clear everything in the canvas. Subpads are deleted
      TPad::Clear(option);   //Remove primitives from pad
   }

   fSelected      = 0;
   fClickSelected = 0;
   fSelectedPad   = 0;
   fClickSelectedPad = 0;
}

//______________________________________________________________________________
void TCanvas::Cleared(TVirtualPad *pad)
{
   // Emit pad Cleared signal.

   Emit("Cleared(TVirtualPad*)", (Long_t)pad);
}

//______________________________________________________________________________
void TCanvas::Closed()
{
   // Emit Closed signal.

   Emit("Closed()");
}

//______________________________________________________________________________
void TCanvas::Close(Option_t *option)
{
   // Close canvas.
   //
   //  Delete window/pads data structure

   TPad    *padsave = (TPad*)gPad;
   TCanvas *cansave = 0;
   if (padsave) cansave = (TCanvas*)gPad->GetCanvas();

   if (fCanvasID != -1) {

      if ((!gROOT->IsLineProcessing()) && (!gVirtualX->IsCmdThread())) {
         gInterpreter->Execute(this, IsA(), "Close", option);
         return;
      }

      R__LOCKGUARD2(gROOTMutex);

      FeedbackMode(kFALSE);

      cd();
      TPad::Close(option);

      if (!IsBatch()) {
         gVirtualX->SelectWindow(fCanvasID);    //select current canvas

         DeleteCanvasPainter();

         if (fCanvasImp) fCanvasImp->Close();
      }
      fCanvasID = -1;
      fBatch    = kTRUE;

      gROOT->GetListOfCanvases()->Remove(this);

      // Close actual window on screen
      SafeDelete(fCanvasImp);
   }

   if (cansave == this) {
      gPad = (TCanvas *) gROOT->GetListOfCanvases()->First();
   } else {
      gPad = padsave;
   }

   Closed();
}

//______________________________________________________________________________
void TCanvas::CopyPixmaps()
{
   // Copy the canvas pixmap of the pad to the canvas.

   if (!IsBatch()) {
      CopyPixmap();
      TPad::CopyPixmaps();
   }
}

//______________________________________________________________________________
void TCanvas::Draw(Option_t *)
{
   //  Draw a canvas.
   //  If a canvas with the name is already on the screen, the canvas is repainted.
   //  This function is useful when a canvas object has been saved in a Root file.
   //  One can then do:
   //     Root > Tfile f("file.root");
   //     Root > canvas.Draw();

   // Load and initialize graphics libraries if
   // TApplication::NeedGraphicsLibs() has been called by a
   // library static initializer.
   if (gApplication)
      gApplication->InitializeGraphics();

   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(GetName());
   if (old == this) {
      Paint();
      return;
   }
   if (old) { gROOT->GetListOfCanvases()->Remove(old); delete old;}

   if (fWindowWidth  == 0) {
      if (fCw !=0) fWindowWidth = fCw+4;
      else         fWindowWidth = 800;
   }
   if (fWindowHeight == 0) {
      if (fCh !=0) fWindowHeight = fCh+28;
      else         fWindowHeight = 600;
   }
   if (gROOT->IsBatch()) {   //We are in Batch mode
      fCanvasImp  = gBatchGuiFactory->CreateCanvasImp(this, GetName(), fWindowWidth, fWindowHeight);
      if (!fCanvasImp) return;
      fBatch = kTRUE;

   } else {                   //normal mode with a screen window
      fCanvasImp = gGuiFactory->CreateCanvasImp(this, GetName(), fWindowTopX, fWindowTopY,
                                                fWindowWidth, fWindowHeight);
      if (!fCanvasImp) return;
      fCanvasImp->ShowMenuBar(TestBit(kMenuBar));
   }
   Build();
   ResizePad();
   fCanvasImp->Show();
   Modified();
}

//______________________________________________________________________________
TObject *TCanvas::DrawClone(Option_t *option) const
{
   // Draw a clone of this canvas
   // A new canvas is created that is a clone of this canvas

   const char *defcanvas = gROOT->GetDefCanvasName();
   char *cdef;

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (lc->FindObject(defcanvas))
      cdef = Form("%s_n%d",defcanvas,lc->GetSize()+1);
   else
      cdef = Form("%s",defcanvas);

   TCanvas *newCanvas = (TCanvas*)Clone();
   newCanvas->SetName(cdef);

   newCanvas->Draw(option);
   newCanvas->Update();
   return newCanvas;
}

//______________________________________________________________________________
TObject *TCanvas::DrawClonePad()
{
   // Draw a clone of this canvas into the current pad
   // In an interactive session, select the destination/current pad
   // with the middle mouse button, then point to the canvas area to select
   // the canvas context menu item DrawClonePad.
   // Note that the original canvas may have subpads.

   TPad *padsav = (TPad*)gPad;
   TPad *selpad = (TPad*)gROOT->GetSelectedPad();
   TPad *pad = padsav;
   if (pad == this) pad = selpad;
   if (padsav == 0 || pad == 0 || pad == this) {
      TCanvas *newCanvas = (TCanvas*)DrawClone();
      newCanvas->SetWindowSize(GetWindowWidth(),GetWindowHeight());
      return newCanvas;
   }
   if (fCanvasID == -1) {
      fCanvasImp = gGuiFactory->CreateCanvasImp(this, GetName(), fWindowTopX, fWindowTopY,
                                             fWindowWidth, fWindowHeight);
      if (!fCanvasImp) return 0;
      fCanvasImp->ShowMenuBar(TestBit(kMenuBar));
      fCanvasID = fCanvasImp->InitWindow();
   }
   this->cd();
   TObject *obj, *clone;
   //copy pad attributes
   pad->Range(fX1,fY1,fX2,fY2);
   pad->SetTickx(GetTickx());
   pad->SetTicky(GetTicky());
   pad->SetGridx(GetGridx());
   pad->SetGridy(GetGridy());
   pad->SetLogx(GetLogx());
   pad->SetLogy(GetLogy());
   pad->SetLogz(GetLogz());
   pad->SetBorderSize(GetBorderSize());
   pad->SetBorderMode(GetBorderMode());
   TAttLine::Copy((TAttLine&)*pad);
   TAttFill::Copy((TAttFill&)*pad);
   TAttPad::Copy((TAttPad&)*pad);

   //copy primitives
   TIter next(GetListOfPrimitives());
   while ((obj=next())) {
      pad->cd();
      clone = obj->Clone();
      pad->GetListOfPrimitives()->Add(clone,next.GetOption());
   }
   pad->ResizePad();
   pad->Modified();
   pad->Update();
   if (padsav) padsav->cd();
   return 0;
}

//______________________________________________________________________________
void TCanvas::DrawEventStatus(Int_t event, Int_t px, Int_t py, TObject *selected)
{
   // Report name and title of primitive below the cursor.
   //
   //    This function is called when the option "Event Status"
   //    in the canvas menu "Options" is selected.

   const Int_t kTMAX=256;
   static char atext[kTMAX];

   if (!TestBit(kShowEventStatus) || !selected) return;

   if (!fCanvasImp) return; //this may happen when closing a TAttCanvas

   TVirtualPad* savepad;
   savepad = gPad;
   gPad = GetSelectedPad();

   fCanvasImp->SetStatusText(selected->GetTitle(),0);
   fCanvasImp->SetStatusText(selected->GetName(),1);
   if (event == kKeyPress)
      snprintf(atext, kTMAX, "%c", (char) px);
   else
      snprintf(atext, kTMAX, "%d,%d", px, py);
   fCanvasImp->SetStatusText(atext,2);
   fCanvasImp->SetStatusText(selected->GetObjectInfo(px,py),3);
   gPad = savepad;
}

//______________________________________________________________________________
void TCanvas::EditorBar()
{
   // Get editor bar.

   TVirtualPadEditor::GetPadEditor();
}

//______________________________________________________________________________
void TCanvas::EmbedInto(Int_t winid, Int_t ww, Int_t wh)
{
   // Embedded a canvas into a TRootEmbeddedCanvas. This method is only called
   // via TRootEmbeddedCanvas::AdoptCanvas.

   // If fCanvasImp already exists, no need to go further.
   if(fCanvasImp) return;

   fCanvasID     = winid;
   fWindowTopX   = 0;
   fWindowTopY   = 0;
   fWindowWidth  = ww;
   fWindowHeight = wh;
   fCw           = ww;
   fCh           = wh;
   fBatch        = kFALSE;
   fUpdating     = kFALSE;

   fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, GetName(), fCw, fCh);
   if (!fCanvasImp) return;
   Build();
   Resize();
}

//______________________________________________________________________________
void TCanvas::EnterLeave(TPad *prevSelPad, TObject *prevSelObj)
{
   // Generate kMouseEnter and kMouseLeave events depending on the previously
   // selected object and the currently selected object. Does nothing if the
   // selected object does not change.

   if (prevSelObj == fSelected) return;

   TPad *padsav = (TPad *)gPad;
   Int_t sevent = fEvent;

   if (prevSelObj) {
      gPad = prevSelPad;
      prevSelObj->ExecuteEvent(kMouseLeave, fEventX, fEventY);
      fEvent = kMouseLeave;
      RunAutoExec();
      ProcessedEvent(kMouseLeave, fEventX, fEventY, prevSelObj);  // emit signal
   }

   gPad = fSelectedPad;

   if (fSelected) {
      fSelected->ExecuteEvent(kMouseEnter, fEventX, fEventY);
      fEvent = kMouseEnter;
      RunAutoExec();
      ProcessedEvent(kMouseEnter, fEventX, fEventY, fSelected);  // emit signal
   }

   fEvent = sevent;
   gPad   = padsav;
}

//______________________________________________________________________________
void TCanvas::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //  This member function must be implemented to realize the action
   //  corresponding to the mouse click on the object in the canvas
   //
   //  Only handle mouse motion events in TCanvas, all other events are
   //  ignored for the time being

   if (gROOT->GetEditorMode()) {
      TPad::ExecuteEvent(event,px,py);
      return;
   }

   switch (event) {

   case kMouseMotion:
      SetCursor(kCross);
      break;
   }
}

//______________________________________________________________________________
void TCanvas::FeedbackMode(Bool_t set)
{
   // Turn rubberband feedback mode on or off.

   if (set) {
      SetDoubleBuffer(0);             // turn off double buffer mode
      gVirtualX->SetDrawMode(TVirtualX::kInvert);  // set the drawing mode to XOR mode
   } else {
      SetDoubleBuffer(1);             // turn on double buffer mode
      gVirtualX->SetDrawMode(TVirtualX::kCopy); // set drawing mode back to normal (copy) mode
   }
}

//______________________________________________________________________________
void TCanvas::Flush()
{
   // Flush canvas buffers.

   if (fCanvasID == -1) return;

   TPad *padsav = (TPad*)gPad;
   cd();
   if (!IsBatch()) {
      if (!UseGL()) {
         gVirtualX->SelectWindow(fCanvasID);
         gPad = padsav; //don't do cd() because than also the pixmap is changed
         CopyPixmaps();
         gVirtualX->UpdateWindow(1);
      } else {
         TVirtualPS *tvps = gVirtualPS;
         gVirtualPS = 0;
         gGLManager->MakeCurrent(fGLDevice);
         fPainter->InitPainter();
         Paint();
         if (padsav && padsav->GetCanvas() == this) {
            padsav->cd();
            padsav->HighLight(padsav->GetHighLightColor());
            //cd();
         }
         fPainter->LockPainter();
         gGLManager->Flush(fGLDevice);
         gVirtualPS = tvps;
      }
   }
   if (padsav) padsav->cd();
}

//______________________________________________________________________________
void TCanvas::UseCurrentStyle()
{
   // Force a copy of current style for all objects in canvas.

   if ((!gROOT->IsLineProcessing()) && (!gVirtualX->IsCmdThread())) {
      gInterpreter->Execute(this, IsA(), "UseCurrentStyle", "");
      return;
   }

   R__LOCKGUARD2(gROOTMutex);

   TPad::UseCurrentStyle();

   if (gStyle->IsReading()) {
      SetFillColor(gStyle->GetCanvasColor());
      fBorderSize = gStyle->GetCanvasBorderSize();
      fBorderMode = gStyle->GetCanvasBorderMode();
   } else {
      gStyle->SetCanvasColor(GetFillColor());
      gStyle->SetCanvasBorderSize(fBorderSize);
      gStyle->SetCanvasBorderMode(fBorderMode);
   }
}

//______________________________________________________________________________
Int_t TCanvas::GetWindowTopX()
{
   // Returns current top x position of window on screen.

   if (fCanvasImp) fCanvasImp->GetWindowGeometry(fWindowTopX, fWindowTopY,
                                                 fWindowWidth,fWindowHeight);

   return fWindowTopX;
}

//______________________________________________________________________________
Int_t TCanvas::GetWindowTopY()
{
   // Returns current top y position of window on screen.

   if (fCanvasImp) fCanvasImp->GetWindowGeometry(fWindowTopX, fWindowTopY,
                                                 fWindowWidth,fWindowHeight);

   return fWindowTopY;
}

//______________________________________________________________________________
void TCanvas::HandleInput(EEventType event, Int_t px, Int_t py)
{
   // Handle Input Events.
   //
   //  Handle input events, like button up/down in current canvas.

   TPad    *pad;
   TPad    *prevSelPad = (TPad*) fSelectedPad;
   TObject *prevSelObj = fSelected;

   fPadSave = (TPad*)gPad;
   cd();        // make sure this canvas is the current canvas

   fEvent  = event;
   fEventX = px;
   fEventY = py;

   switch (event) {

   case kMouseMotion:
      // highlight object tracked over
      pad = Pick(px, py, prevSelObj);
      if (!pad) return;

      EnterLeave(prevSelPad, prevSelObj);

      gPad = pad;   // don't use cd() we will use the current
                    // canvas via the GetCanvas member and not via
                    // gPad->GetCanvas

      if (fSelected) {
         fSelected->ExecuteEvent(event, px, py);
         RunAutoExec();
      }

      break;

   case kMouseEnter:
      // mouse enters canvas
      if (!fDoubleBuffer) FeedbackMode(kTRUE);
      break;

   case kMouseLeave:
      // mouse leaves canvas
      {
         // force popdown of tooltips
         TObject *sobj = fSelected;
         TPad    *spad = fSelectedPad;
         fSelected     = 0;
         fSelectedPad  = 0;
         EnterLeave(prevSelPad, prevSelObj);
         fSelected     = sobj;
         fSelectedPad  = spad;
         if (!fDoubleBuffer) FeedbackMode(kFALSE);
      }
      break;

   case kButton1Double:
      // triggered on the second button down within 350ms and within
      // 3x3 pixels of the first button down, button up finishes action

   case kButton1Down:
      // find pad in which input occured
      pad = Pick(px, py, prevSelObj);
      if (!pad) return;

      gPad = pad;   // don't use cd() because we won't draw in pad
                    // we will only use its coordinate system

      if (fSelected) {
         FeedbackMode(kTRUE);   // to draw in rubberband mode
         fSelected->ExecuteEvent(event, px, py);

         RunAutoExec();
      }

      break;

   case kArrowKeyPress:
   case kArrowKeyRelease:
   case kButton1Motion:
   case kButton1ShiftMotion: //8 == kButton1Motion + shift modifier
      if (fSelected) {
         gPad = fSelectedPad;

         fSelected->ExecuteEvent(event, px, py);
         gVirtualX->Update();

         if (!fSelected->InheritsFrom(TAxis::Class())) {
            Bool_t resize = kFALSE;
            if (fSelected->InheritsFrom(TBox::Class()))
               resize = ((TBox*)fSelected)->IsBeingResized();
            if (fSelected->InheritsFrom(TVirtualPad::Class()))
               resize = ((TVirtualPad*)fSelected)->IsBeingResized();

            if ((!resize && TestBit(kMoveOpaque)) || (resize && TestBit(kResizeOpaque))) {
               gPad = fPadSave;
               Update();
               FeedbackMode(kTRUE);
            }
         }

         RunAutoExec();
      }

      break;

   case kButton1Up:

      if (fSelected) {
         gPad = fSelectedPad;

         fSelected->ExecuteEvent(event, px, py);

         RunAutoExec();

         if (fPadSave)
            gPad = fPadSave;
         else {
            gPad     = this;
            fPadSave = this;
         }

         Update();    // before calling update make sure gPad is reset
      }
      break;

//*-*----------------------------------------------------------------------

   case kButton2Down:
      // find pad in which input occured
      pad = Pick(px, py, prevSelObj);
      if (!pad) return;

      gPad = pad;   // don't use cd() because we won't draw in pad
                    // we will only use its coordinate system

      FeedbackMode(kTRUE);

      if (fSelected) fSelected->Pop();  // pop object to foreground
      pad->cd();                        // and make its pad the current pad
      if (gDebug)
         printf("Current Pad: %s / %s\n", pad->GetName(), pad->GetTitle());

      // loop over all canvases to make sure that only one pad is highlighted
      {
         TIter next(gROOT->GetListOfCanvases());
         TCanvas *tc;
         while ((tc = (TCanvas *)next()))
            tc->Update();
      }

      //if (pad->GetGLDevice() != -1 && fSelected)
      //   fSelected->ExecuteEvent(event, px, py);

      break;   // don't want fPadSave->cd() to be executed at the end

   case kButton2Motion:
      //was empty!
   case kButton2Up:
      if (fSelected) {
         gPad = fSelectedPad;

         fSelected->ExecuteEvent(event, px, py);
         RunAutoExec();
      }
      break;

   case kButton2Double:
      break;

//*-*----------------------------------------------------------------------

   case kButton3Down:
      // popup context menu
      pad = Pick(px, py, prevSelObj);
      if (!pad) return;

      if (!fDoubleBuffer) FeedbackMode(kFALSE);

      if (fContextMenu && fSelected && !fSelected->TestBit(kNoContextMenu) &&
         !pad->TestBit(kNoContextMenu) && !TestBit(kNoContextMenu))
         fContextMenu->Popup(px, py, fSelected, this, pad);

      break;

   case kButton3Motion:
      break;

   case kButton3Up:
      if (!fDoubleBuffer) FeedbackMode(kTRUE);
      break;

   case kButton3Double:
      break;

   case kKeyPress:
      if (!fSelectedPad || !fSelected) return;
      gPad = fSelectedPad;   // don't use cd() because we won't draw in pad
                    // we will only use its coordinate system
      fSelected->ExecuteEvent(event, px, py);

      RunAutoExec();

      break;

   case kButton1Shift:
      // Try to select
      pad = Pick(px, py, prevSelObj);

      if (!pad) return;

      EnterLeave(prevSelPad, prevSelObj);

      gPad = pad;   // don't use cd() we will use the current
                    // canvas via the GetCanvas member and not via
                    // gPad->GetCanvas
      if (fSelected) {
         fSelected->ExecuteEvent(event, px, py);
         RunAutoExec();
      }
      break;

   case kWheelUp:
   case kWheelDown:
      pad = Pick(px, py, prevSelObj);
      if (!pad) return;

      gPad = pad;
      if (fSelected)
         fSelected->ExecuteEvent(event, px, py);
      break;

   default:
      break;
   }

   if (fPadSave && event != kButton2Down)
      fPadSave->cd();

   if (event != kMouseLeave) { // signal was already emitted for this event
      ProcessedEvent(event, px, py, fSelected);  // emit signal
      DrawEventStatus(event, px, py, fSelected);
   }
}

//______________________________________________________________________________
Bool_t TCanvas::IsFolder() const
{
   // Is folder ?

   return fgIsFolder;
}

//______________________________________________________________________________
void TCanvas::ls(Option_t *option) const
{
   // List all pads.

   TROOT::IndentLevel();
   std::cout <<"Canvas Name=" <<GetName()<<" Title="<<GetTitle()<<" Option="<<option<<std::endl;
   TROOT::IncreaseDirLevel();
   TPad::ls(option);
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
TCanvas *TCanvas::MakeDefCanvas()
{
   // Static function to build a default canvas.

   const char *defcanvas = gROOT->GetDefCanvasName();
   char *cdef;

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (lc->FindObject(defcanvas)) {
      Int_t n = lc->GetSize() + 1;
      cdef = new char[strlen(defcanvas)+15];
      do {
         strlcpy(cdef,Form("%s_n%d", defcanvas, n++),strlen(defcanvas)+15);
      } while (lc->FindObject(cdef));
   } else
      cdef = StrDup(Form("%s",defcanvas));

   TCanvas *c = new TCanvas(cdef, cdef, 1);

   ::Info("TCanvas::MakeDefCanvas"," created default TCanvas with name %s",cdef);
   delete [] cdef;
   return c;
}

//______________________________________________________________________________
void TCanvas::MoveOpaque(Int_t set)
{
   // Set option to move objects/pads in a canvas.
   //
   //  if set = 1 (default) graphics objects are moved in opaque mode
   //         = 0 only the outline of objects is drawn when moving them
   //  The option opaque produces the best effect. It requires however a
   //  a reasonably fast workstation or response time.

   SetBit(kMoveOpaque,set);
}

//______________________________________________________________________________
void TCanvas::Paint(Option_t *option)
{
   // Paint canvas.

   if (fCanvas) TPad::Paint(option);
}

//______________________________________________________________________________
TPad *TCanvas::Pick(Int_t px, Int_t py, TObject *prevSelObj)
{
   // Prepare for pick, call TPad::Pick() and when selected object
   // is different from previous then emit Picked() signal.

   TObjLink *pickobj = 0;

   fSelected    = 0;
   fSelectedOpt = "";
   fSelectedPad = 0;

   TPad *pad = Pick(px, py, pickobj);
   if (!pad) return 0;

   if (!pickobj) {
      fSelected    = pad;
      fSelectedOpt = "";
   } else {
      if (!fSelected) {   // can be set via TCanvas::SetSelected()
         fSelected    = pickobj->GetObject();
         fSelectedOpt = pickobj->GetOption();
      }
   }
   fSelectedPad = pad;

   if (fSelected != prevSelObj)
      Picked(fSelectedPad, fSelected, fEvent);  // emit signal

   if ((fEvent == kButton1Down) || (fEvent == kButton2Down) || (fEvent == kButton3Down)) {
      if (fSelected && !fSelected->InheritsFrom(TView::Class())) {
         fClickSelected = fSelected;
         fClickSelectedPad = fSelectedPad;
         Selected(fSelectedPad, fSelected, fEvent);  // emit signal
         fSelectedX = px;
         fSelectedY = py;
      }
   }
   return pad;
}

//______________________________________________________________________________
void TCanvas::Picked(TPad *pad, TObject *obj, Int_t event)
{
   // Emit Picked() signal.

   Long_t args[3];

   args[0] = (Long_t) pad;
   args[1] = (Long_t) obj;
   args[2] = event;

   Emit("Picked(TPad*,TObject*,Int_t)", args);
}

//______________________________________________________________________________
void TCanvas::Selected(TVirtualPad *pad, TObject *obj, Int_t event)
{
   // Emit Selected() signal.

   Long_t args[3];

   args[0] = (Long_t) pad;
   args[1] = (Long_t) obj;
   args[2] = event;

   Emit("Selected(TVirtualPad*,TObject*,Int_t)", args);
}

//______________________________________________________________________________
void TCanvas::ProcessedEvent(Int_t event, Int_t x, Int_t y, TObject *obj)
{
   // Emit ProcessedEvent() signal.

   Long_t args[4];

   args[0] = event;
   args[1] = x;
   args[2] = y;
   args[3] = (Long_t) obj;

   Emit("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", args);
}

//______________________________________________________________________________
void TCanvas::Resize(Option_t *)
{
   // Recompute canvas parameters following a X11 Resize.

   if (fCanvasID == -1) return;

   if ((!gROOT->IsLineProcessing()) && (!gVirtualX->IsCmdThread())) {
      gInterpreter->Execute(this, IsA(), "Resize", "");
      return;
   }

   R__LOCKGUARD2(gROOTMutex);

   TPad *padsav  = (TPad*)gPad;
   cd();

   if (!IsBatch()) {
      gVirtualX->SelectWindow(fCanvasID);      //select current canvas
      gVirtualX->ResizeWindow(fCanvasID);      //resize canvas and off-screen buffer

      // Get effective window parameters including menubar and borders
      fCanvasImp->GetWindowGeometry(fWindowTopX, fWindowTopY,
                                    fWindowWidth, fWindowHeight);

      // Get effective canvas parameters without borders
      Int_t dum1, dum2;
      gVirtualX->GetGeometry(fCanvasID, dum1, dum2, fCw, fCh);
   }

   if (fXsizeUser && fYsizeUser) {
      UInt_t nwh = fCh;
      UInt_t nww = fCw;
      Double_t rxy = fXsizeUser/fYsizeUser;
      if (rxy < 1) {
         UInt_t twh = UInt_t(Double_t(fCw)/rxy);
         if (twh > fCh)
            nww = UInt_t(Double_t(fCh)*rxy);
         else
            nwh = twh;
         if (nww > fCw) {
            nww = fCw; nwh = twh;
         }
         if (nwh > fCh) {
            nwh = fCh; nww = UInt_t(Double_t(fCh)/rxy);
         }
      } else {
         UInt_t twh = UInt_t(Double_t(fCw)*rxy);
         if (twh > fCh)
            nwh = UInt_t(Double_t(fCw)/rxy);
         else
            nww = twh;
         if (nww > fCw) {
            nww = fCw; nwh = twh;
         }
         if (nwh > fCh) {
            nwh = fCh; nww = UInt_t(Double_t(fCh)*rxy);
         }
      }
      fCw = nww;
      fCh = nwh;
   }

   if (fCw < fCh) {
      fYsizeReal = kDefaultCanvasSize;
      fXsizeReal = fYsizeReal*Double_t(fCw)/Double_t(fCh);
   }
   else {
      fXsizeReal = kDefaultCanvasSize;
      fYsizeReal = fXsizeReal*Double_t(fCh)/Double_t(fCw);
   }

//*-*- Loop on all pads to recompute conversion coefficients
   TPad::ResizePad();

   if (padsav) padsav->cd();
}

//______________________________________________________________________________
void TCanvas::ResizeOpaque(Int_t set)
{
   // Set option to resize objects/pads in a canvas.
   //
   //  if set = 1 (default) graphics objects are resized in opaque mode
   //         = 0 only the outline of objects is drawn when resizing them
   //  The option opaque produces the best effect. It requires however a
   //  a reasonably fast workstation or response time.

   SetBit(kResizeOpaque,set);
}

//______________________________________________________________________________
void TCanvas::RunAutoExec()
{
   // Execute the list of TExecs in the current pad.

   if (!TestBit(kAutoExec)) return;
   if (!gPad) return;
   ((TPad*)gPad)->AutoExec();
}


//______________________________________________________________________________
void TCanvas::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   // Save primitives in this canvas in C++ macro file with GUI.

   // Write canvas options (in $TROOT or $TStyle)
   if (gStyle->GetOptFit()) {
      out<<"   gStyle->SetOptFit(1);"<<std::endl;
   }
   if (!gStyle->GetOptStat()) {
      out<<"   gStyle->SetOptStat(0);"<<std::endl;
   }
   if (!gStyle->GetOptTitle()) {
      out<<"   gStyle->SetOptTitle(0);"<<std::endl;
   }
   if (gROOT->GetEditHistograms()) {
      out<<"   gROOT->SetEditHistograms();"<<std::endl;
   }
   if (GetShowEventStatus()) {
      out<<"   "<<GetName()<<"->ToggleEventStatus();"<<std::endl;
   }
   if (GetShowToolTips()) {
      out<<"   "<<GetName()<<"->ToggleToolTips();"<<std::endl;
   }
   if (GetShowToolBar()) {
      out<<"   "<<GetName()<<"->ToggleToolBar();"<<std::endl;
   }
   if (GetHighLightColor() != 5) {
      if (GetHighLightColor() > 228) {
         TColor::SaveColor(out, GetHighLightColor());
         out<<"   "<<GetName()<<"->SetHighLightColor(ci);" << std::endl;
      } else
         out<<"   "<<GetName()<<"->SetHighLightColor("<<GetHighLightColor()<<");"<<std::endl;
   }

   // Now recursively scan all pads of this canvas
   cd();
   TPad::SavePrimitive(out,option);
}

//______________________________________________________________________________
void TCanvas::SaveSource(const char *filename, Option_t *option)
{
   // Save primitives in this canvas as a C++ macro file.
   // This function loops on all the canvas primitives and for each primitive
   // calls the object SavePrimitive function.
   // When outputing floating point numbers, the default precision is 7 digits.
   // The precision can be changed (via system.rootrc) by changing the value
   // of the environment variable "Canvas.SavePrecision"

   //    reset bit TClass::kClassSaved for all classes
   TIter next(gROOT->GetListOfClasses());
   TClass *cl;
   while((cl = (TClass*)next())) {
      cl->ResetBit(TClass::kClassSaved);
   }

   char quote = '"';
   std::ofstream out;
   Int_t lenfile = strlen(filename);
   char * fname;
   char lcname[10];
   const char *cname = GetName();
   Bool_t invalid = kFALSE;
   //    if filename is given, open this file, otherwise create a file
   //    with a name equal to the canvasname.C
   if (lenfile) {
      fname = (char*)filename;
      out.open(fname, std::ios::out);
   } else {
      Int_t nch = strlen(cname);
      if (nch < 10) {
         strlcpy(lcname,cname,10);
         for (Int_t k=1;k<=nch;k++) {if (lcname[nch-k] == ' ') lcname[nch-k] = 0;}
         if (lcname[0] == 0) {invalid = kTRUE; strlcpy(lcname,"c1",10); nch = 2;}
         cname = lcname;
      }
      fname = new char[nch+3];
      strlcpy(fname,cname,nch+3);
      strncat(fname,".C",2);
      out.open(fname, std::ios::out);
   }
   if (!out.good ()) {
      Error("SaveSource", "Cannot open file: %s",fname);
      if (!lenfile) delete [] fname;
      return;
   }

   //set precision
   Int_t precision = gEnv->GetValue("Canvas.SavePrecision",7);
   out.precision(precision);

   //   Write macro header and date/time stamp
   TDatime t;
   Float_t cx = gStyle->GetScreenFactor();
   Int_t topx,topy;
   UInt_t w, h;
   if (!fCanvasImp) {
      Error("SaveSource", "Cannot open TCanvas");
      return;
   }
   UInt_t editorWidth = fCanvasImp->GetWindowGeometry(topx,topy,w,h);
   w = UInt_t((fWindowWidth - editorWidth)/cx);
   h = UInt_t((fWindowHeight)/cx);
   topx = GetWindowTopX();
   topy = GetWindowTopY();

   if (w == 0) {
      w = GetWw()+4; h = GetWh()+4;
      topx = 1;    topy = 1;
   }

   TString mname(fname);
   Int_t p = mname.Last('.');
   Int_t s = mname.Last('/')+1;
   out <<"void " << mname(s,p-s) << "()" <<std::endl;
   out <<"{"<<std::endl;
   out <<"//=========Macro generated from canvas: "<<GetName()<<"/"<<GetTitle()<<std::endl;
   out <<"//=========  ("<<t.AsString()<<") by ROOT version"<<gROOT->GetVersion()<<std::endl;

   if (gStyle->GetCanvasPreferGL())
      out <<std::endl<<"   gStyle->SetCanvasPreferGL(kTRUE);"<<std::endl<<std::endl;

   //   Write canvas parameters (TDialogCanvas case)
   if (InheritsFrom(TDialogCanvas::Class())) {
      out<<"   "<<ClassName()<<" *"<<cname<<" = new "<<ClassName()<<"("<<quote<<GetName()
         <<quote<<", "<<quote<<GetTitle()<<quote<<","<<w<<","<<h<<");"<<std::endl;
   } else {
   //   Write canvas parameters (TCanvas case)
      out<<"   TCanvas *"<<cname<<" = new TCanvas("<<quote<<GetName()<<quote<<", "<<quote<<GetTitle()
         <<quote;
      if (!HasMenuBar())
         out<<",-"<<topx<<","<<topy<<","<<w<<","<<h<<");"<<std::endl;
      else
         out<<","<<topx<<","<<topy<<","<<w<<","<<h<<");"<<std::endl;
   }
   //   Write canvas options (in $TROOT or $TStyle)
   if (gStyle->GetOptFit()) {
      out<<"   gStyle->SetOptFit(1);"<<std::endl;
   }
   if (!gStyle->GetOptStat()) {
      out<<"   gStyle->SetOptStat(0);"<<std::endl;
   }
   if (!gStyle->GetOptTitle()) {
      out<<"   gStyle->SetOptTitle(0);"<<std::endl;
   }
   if (gROOT->GetEditHistograms()) {
      out<<"   gROOT->SetEditHistograms();"<<std::endl;
   }
   if (GetShowEventStatus()) {
      out<<"   "<<GetName()<<"->ToggleEventStatus();"<<std::endl;
   }
   if (GetShowToolTips()) {
      out<<"   "<<GetName()<<"->ToggleToolTips();"<<std::endl;
   }
   if (GetHighLightColor() != 5) {
      if (GetHighLightColor() > 228) {
         TColor::SaveColor(out, GetHighLightColor());
         out<<"   "<<GetName()<<"->SetHighLightColor(ci);" << std::endl;
      } else
         out<<"   "<<GetName()<<"->SetHighLightColor("<<GetHighLightColor()<<");"<<std::endl;
   }

   //   Now recursively scan all pads of this canvas
   cd();
   if (invalid) SetName("c1");
   TPad::SavePrimitive(out,option);
   //   Write canvas options related to pad editor
   out<<"   "<<GetName()<<"->SetSelected("<<GetName()<<");"<<std::endl;
   if (GetShowToolBar()) {
      out<<"   "<<GetName()<<"->ToggleToolBar();"<<std::endl;
   }
   if (invalid) SetName(" ");

   out <<"}"<<std::endl;
   out.close();
   Info("SaveSource","C++ Macro file: %s has been generated", fname);

   //    reset bit TClass::kClassSaved for all classes
   next.Reset();
   while((cl = (TClass*)next())) {
      cl->ResetBit(TClass::kClassSaved);
   }
   if (!lenfile) delete [] fname;
}

//______________________________________________________________________________
void TCanvas::SetBatch(Bool_t batch)
{
   // Toggle batch mode. However, if the canvas is created without a window
   // then batch mode always stays set.

   if (gROOT->IsBatch())
      fBatch = kTRUE;
   else
      fBatch = batch;
}

//______________________________________________________________________________
void TCanvas::SetCanvasSize(UInt_t ww, UInt_t wh)
{
   // Set Width and Height of canvas to ww and wh respectively
   // If ww and/or wh are greater than the current canvas window
   // a scroll bar is automatically generated.
   // Use this function to zoom in a canvas and naviguate via
   // the scroll bars.

   if (fCanvasImp) {
      fCanvasImp->SetCanvasSize(ww, wh);
      fCw = ww;
      fCh = wh;
      ResizePad();
   }
}

//______________________________________________________________________________
void TCanvas::SetCursor(ECursor cursor)
{
   // Set cursor.

   if (IsBatch()) return;
   gVirtualX->SetCursor(fCanvasID, cursor);
}

//______________________________________________________________________________
void TCanvas::SetDoubleBuffer(Int_t mode)
{
   // Set Double Buffer On/Off.

   if (IsBatch()) return;
   fDoubleBuffer = mode;
   gVirtualX->SetDoubleBuffer(fCanvasID, mode);

   // depending of the buffer mode set the drawing window to either
   // the canvas pixmap or to the canvas on-screen window
   if (fDoubleBuffer) {
      if (fPixmapID != -1) fPainter->SelectDrawable(fPixmapID);
   } else
      if (fCanvasID != -1) fPainter->SelectDrawable(fCanvasID);
}

//______________________________________________________________________________
void TCanvas::SetFixedAspectRatio(Bool_t fixed)
{
   // Fix canvas aspect ratio to current value if fixed is true.

   if (fixed) {
      if (!fFixedAspectRatio) {
         if (fCh != 0)
            fAspectRatio = Double_t(fCw) / fCh;
         else {
            Error("SetAspectRatio", "cannot fix aspect ratio, height of canvas is 0");
            return;
         }
         fFixedAspectRatio = kTRUE;
      }
   } else {
      fFixedAspectRatio = kFALSE;
      fAspectRatio = 0;
   }
}

//______________________________________________________________________________
void TCanvas::SetFolder(Bool_t isfolder)
{
   // If isfolder=kTRUE, the canvas can be browsed like a folder
   // by default a canvas is not browsable.

   fgIsFolder = isfolder;
}

//______________________________________________________________________________
void TCanvas::SetSelected(TObject *obj)
{
   // Set selected canvas.

   fSelected = obj;
   if (obj) obj->SetBit(kMustCleanup);
}

//______________________________________________________________________________
void TCanvas::SetTitle(const char *title)
{
   // Set canvas title.

   fTitle = title;
   if (fCanvasImp) fCanvasImp->SetWindowTitle(title);
}

//______________________________________________________________________________
void TCanvas::Size(Float_t xsize, Float_t ysize)
{
   // Set the canvas scale in centimeters.
   //
   //  This information is used by PostScript to set the page size.
   //  xsize  = size of the canvas in centimeters along X
   //  ysize  = size of the canvas in centimeters along Y
   //   if xsize and ysize are not equal to 0, then the scale factors will
   //   be computed to keep the ratio ysize/xsize independently of the canvas
   //   size (parts of the physical canvas will be unused).
   //
   //   if xsize = 0 and ysize is not zero, then xsize will be computed
   //      to fit to the current canvas scale. If the canvas is resized,
   //      a new value for xsize will be recomputed. In this case the aspect
   //      ratio is not preserved.
   //
   //   if both xsize = 0 and ysize = 0, then the scaling is automatic.
   //   the largest dimension will be allocated a size of 20 centimeters.

   fXsizeUser = xsize;
   fYsizeUser = ysize;

   Resize();
}

//_______________________________________________________________________
void TCanvas::Streamer(TBuffer &b)
{
   // Stream a class object.

   UInt_t R__s, R__c;
   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      gPad    = this;
      fCanvas = this;
      if (v>7) b.ClassBegin(TCanvas::IsA());
      if (v>7) b.ClassMember("TPad");
      TPad::Streamer(b);
      gPad    = this;
      //restore the colors
      TObjArray *colors = (TObjArray*)fPrimitives->FindObject("ListOfColors");
      if (colors) {
         TIter next(colors);
         TColor *colold;
         while ((colold = (TColor*)next())) {
            if (colold) {
               Int_t cn = 0;
               if (colold) cn = colold->GetNumber();
               TColor *colcur = gROOT->GetColor(cn);
               if (colcur) {
                  colcur->SetRGB(colold->GetRed(),colold->GetGreen(),colold->GetBlue());
               } else {
                  colcur = new TColor(cn,colold->GetRed(),
                                        colold->GetGreen(),
                                        colold->GetBlue(),
                                        colold->GetName());
                  if (!colcur) return;
               }
            }
         }
         fPrimitives->Remove(colors);
         colors->Delete();
         delete colors;
      }
      if (v>7) b.ClassMember("fDISPLAY","TString");
      fDISPLAY.Streamer(b);
      if (v>7) b.ClassMember("fDoubleBuffer", "Int_t");
      b >> fDoubleBuffer;
      if (v>7) b.ClassMember("fRetained", "Bool_t");
      b >> fRetained;
      if (v>7) b.ClassMember("fXsizeUser", "Size_t");
      b >> fXsizeUser;
      if (v>7) b.ClassMember("fYsizeUser", "Size_t");
      b >> fYsizeUser;
      if (v>7) b.ClassMember("fXsizeReal", "Size_t");
      b >> fXsizeReal;
      if (v>7) b.ClassMember("fYsizeReal", "Size_t");
      b >> fYsizeReal;
      fCanvasID = -1;
      if (v>7) b.ClassMember("fWindowTopX", "Int_t");
      b >> fWindowTopX;
      if (v>7) b.ClassMember("fWindowTopY", "Int_t");
      b >> fWindowTopY;
      if (v > 2) {
         if (v>7) b.ClassMember("fWindowWidth", "UInt_t");
         b >> fWindowWidth;
         if (v>7) b.ClassMember("fWindowHeight", "UInt_t");
         b >> fWindowHeight;
      }
      if (v>7) b.ClassMember("fCw", "UInt_t");
      b >> fCw;
      if (v>7) b.ClassMember("fCh", "UInt_t");
      b >> fCh;
      if (v <= 2) {
         fWindowWidth  = fCw;
         fWindowHeight = fCh;
      }
      if (v>7) b.ClassMember("fCatt", "TAttCanvas");
      fCatt.Streamer(b);
      Bool_t dummy;
      if (v>7) b.ClassMember("kMoveOpaque", "Bool_t");
      b >> dummy; if (dummy) MoveOpaque(1);
      if (v>7) b.ClassMember("kResizeOpaque", "Bool_t");
      b >> dummy; if (dummy) ResizeOpaque(1);
      if (v>7) b.ClassMember("fHighLightColor", "Color_t");
      b >> fHighLightColor;
      if (v>7) b.ClassMember("fBatch", "Bool_t");
      b >> dummy; //was fBatch
      if (v < 2) return;
      if (v>7) b.ClassMember("kShowEventStatus", "Bool_t");
      b >> dummy; if (dummy) SetBit(kShowEventStatus);

      if (v > 3) {
         if (v>7) b.ClassMember("kAutoExec", "Bool_t");
         b >> dummy; if (dummy) SetBit(kAutoExec);
      }
      if (v>7) b.ClassMember("kMenuBar", "Bool_t");
      b >> dummy; if (dummy) SetBit(kMenuBar);
      fBatch = gROOT->IsBatch();
      if (v>7) b.ClassEnd(TCanvas::IsA());
      b.CheckByteCount(R__s, R__c, TCanvas::IsA());
   } else {
      //save list of colors
      //we must protect the case when two or more canvases are saved
      //in the same buffer. If the list of colors has already been saved
      //in the buffer, do not add the list of colors to the list of primitives.
      TObjArray *colors = 0;
      if (!b.CheckObject(gROOT->GetListOfColors(),TObjArray::Class())) {
         colors = (TObjArray*)gROOT->GetListOfColors();
         fPrimitives->Add(colors);
      }
      R__c = b.WriteVersion(TCanvas::IsA(), kTRUE);
      b.ClassBegin(TCanvas::IsA());
      b.ClassMember("TPad");
      TPad::Streamer(b);
      if(colors) fPrimitives->Remove(colors);
      b.ClassMember("fDISPLAY","TString");
      fDISPLAY.Streamer(b);
      b.ClassMember("fDoubleBuffer", "Int_t");
      b << fDoubleBuffer;
      b.ClassMember("fRetained", "Bool_t");
      b << fRetained;
      b.ClassMember("fXsizeUser", "Size_t");
      b << fXsizeUser;
      b.ClassMember("fYsizeUser", "Size_t");
      b << fYsizeUser;
      b.ClassMember("fXsizeReal", "Size_t");
      b << fXsizeReal;
      b.ClassMember("fYsizeReal", "Size_t");
      b << fYsizeReal;
      UInt_t w   = fWindowWidth,  h    = fWindowHeight;
      Int_t topx = fWindowTopX,   topy = fWindowTopY;
      UInt_t editorWidth = 0;
      if(fCanvasImp) editorWidth = fCanvasImp->GetWindowGeometry(topx,topy,w,h);
      b.ClassMember("fWindowTopX", "Int_t");
      b << topx;
      b.ClassMember("fWindowTopY", "Int_t");
      b << topy;
      b.ClassMember("fWindowWidth", "UInt_t");
      b << (UInt_t)(w-editorWidth);
      b.ClassMember("fWindowHeight", "UInt_t");
      b << h;
      b.ClassMember("fCw", "UInt_t");
      b << fCw;
      b.ClassMember("fCh", "UInt_t");
      b << fCh;
      b.ClassMember("fCatt", "TAttCanvas");
      fCatt.Streamer(b);
      b.ClassMember("kMoveOpaque", "Bool_t");
      b << TestBit(kMoveOpaque);      //please remove in ROOT version 6
      b.ClassMember("kResizeOpaque", "Bool_t");
      b << TestBit(kResizeOpaque);    //please remove in ROOT version 6
      b.ClassMember("fHighLightColor", "Color_t");
      b << fHighLightColor;
      b.ClassMember("fBatch", "Bool_t");
      b << fBatch;                    //please remove in ROOT version 6
      b.ClassMember("kShowEventStatus", "Bool_t");
      b << TestBit(kShowEventStatus); //please remove in ROOT version 6
      b.ClassMember("kAutoExec", "Bool_t");
      b << TestBit(kAutoExec);        //please remove in ROOT version 6
      b.ClassMember("kMenuBar", "Bool_t");
      b << TestBit(kMenuBar);         //please remove in ROOT version 6
      b.ClassEnd(TCanvas::IsA());
      b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
void TCanvas::ToggleAutoExec()
{
   // Toggle pad auto execution of list of TExecs.

   Bool_t autoExec = TestBit(kAutoExec);
   SetBit(kAutoExec,!autoExec);
}

//______________________________________________________________________________
void TCanvas::ToggleEventStatus()
{
   // Toggle event statusbar.

   Bool_t showEventStatus = !TestBit(kShowEventStatus);
   SetBit(kShowEventStatus,showEventStatus);

   if (fCanvasImp) fCanvasImp->ShowStatusBar(showEventStatus);
}

//______________________________________________________________________________
void TCanvas::ToggleToolBar()
{
   // Toggle toolbar.

   Bool_t showToolBar = !TestBit(kShowToolBar);
   SetBit(kShowToolBar,showToolBar);

   if (fCanvasImp) fCanvasImp->ShowToolBar(showToolBar);
}

//______________________________________________________________________________
void TCanvas::ToggleEditor()
{
   // Toggle editor.

   Bool_t showEditor = !TestBit(kShowEditor);
   SetBit(kShowEditor,showEditor);

   if (fCanvasImp) fCanvasImp->ShowEditor(showEditor);
}

//______________________________________________________________________________
void TCanvas::ToggleToolTips()
{
   // Toggle tooltip display.

   Bool_t showToolTips = !TestBit(kShowToolTips);
   SetBit(kShowToolTips, showToolTips);

   if (fCanvasImp) fCanvasImp->ShowToolTips(showToolTips);
}


//______________________________________________________________________________
Bool_t TCanvas::SupportAlpha()
{
   // Static function returning "true" if transparency is supported.
   return gPad && (gVirtualX->InheritsFrom("TGQuartz") ||
                   gPad->GetGLDevice() != -1);
}


//______________________________________________________________________________
void TCanvas::Update()
{
   // Update canvas pad buffers.

   if (fUpdating) return;

   if (fPixmapID == -1) return;

   if (gThreadXAR) {
      void *arr[2];
      arr[1] = this;
      if ((*gThreadXAR)("CUPD", 2, arr, 0)) return;
   }

   if (!fCanvasImp) return;

   if (!gVirtualX->IsCmdThread()) {
      gInterpreter->Execute(this, IsA(), "Update", "");
      return;
   }

   R__LOCKGUARD2(gROOTMutex);

   fUpdating = kTRUE;

   if (!IsBatch()) FeedbackMode(kFALSE);      // Goto double buffer mode

   if (!UseGL())
      PaintModified();           // Repaint all modified pad's

   Flush();                   // Copy all pad pixmaps to the screen

   SetCursor(kCross);
   fUpdating = kFALSE;
}

//______________________________________________________________________________
void TCanvas::DisconnectWidget()
{
   // Used by friend class TCanvasImp.

   fCanvasID    = 0;
   fContextMenu = 0;
}

//______________________________________________________________________________
Bool_t TCanvas::IsGrayscale()
{
   // Check whether this canvas is to be drawn in grayscale mode.

   return TestBit(kIsGrayscale);
}

//______________________________________________________________________________
void TCanvas::SetGrayscale(Bool_t set /*= kTRUE*/)
{
   // Set whether this canvas should be painted in grayscale, and re-paint
   // it if necessary.

   if (IsGrayscale() == set) return;
   SetBit(kIsGrayscale, set);
   Paint(); // update canvas and all sub-pads, unconditionally!
}

//______________________________________________________________________________
void TCanvas::CreatePainter()
{
   // Probably, TPadPainter must be placed in a separate ROOT module -
   // "padpainter" (the same as "histpainter"). But now, it's directly in a
   // gpad dir, so, in case of default painter, no *.so should be loaded,
   // no need in plugin managers.
   // May change in future.

   //Even for batch mode painter is still required, just to delegate
   //some calls to batch "virtual X".
   if (!UseGL() || fBatch)
      fPainter = new TPadPainter;//Do not need plugin manager for this!
   else {
      fPainter = TVirtualPadPainter::PadPainter("gl");
      if (!fPainter) {
         Error("CreatePainter", "GL Painter creation failed! Will use default!");
         fPainter = new TPadPainter;
         fUseGL = kFALSE;
      }
   }
}

//______________________________________________________________________________
TVirtualPadPainter *TCanvas::GetCanvasPainter()
{
   // Access and (probably) creation of pad painter.

   if (!fPainter) CreatePainter();
   return fPainter;
}


//______________________________________________________________________________
void TCanvas::DeleteCanvasPainter()
{
   //assert on IsBatch() == false?

   if (fGLDevice != -1) {
      //fPainter has a font manager.
      //Font manager will delete textures.
      //If context is wrong (we can have several canvases) -
      //wrong texture will be deleted, damaging some of our fonts.
      gGLManager->MakeCurrent(fGLDevice);
   }

   delete fPainter;
   fPainter = 0;

   if (fGLDevice != -1) {
      gGLManager->DeleteGLContext(fGLDevice);//?
      fGLDevice = -1;
   }
}
