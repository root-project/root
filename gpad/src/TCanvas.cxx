// @(#)root/gpad:$Name:  $:$Id: TCanvas.cxx,v 1.6 2000/08/23 08:11:59 brun Exp $
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
#include <fstream.h>
#include <iostream.h>

#include "TROOT.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TDirectory.h"
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


// This small class and the static object makedefcanvas_init make sure that
// the TCanvas::MakeDefCanvas method is registered with TROOT as soon as
// the shared library containing TCanvas is loaded.

class TInitMakeDefCanvas {
public:
   TInitMakeDefCanvas() { TROOT::SetMakeDefCanvas(&TCanvas::MakeDefCanvas); }
};

static TInitMakeDefCanvas makedefcanvas_init;




//*-*x16 macros/layout_canvas

const Size_t kDefaultCanvasSize   = 20;

ClassImp(TCanvas)

//______________________________________________________________________________
//
//   A Canvas is an area mapped to a window directly under
//   the control of the display manager.
//   A ROOT session may have several canvases open at any given time.
//
//   A Canvas may be subdivided into independent graphical areas: the PADs
//   A canvas has a default pad which has the name of the canvas itself.
//   An example of a Canvas layout is sketched in the picture below.
//
//   ***********************************************************************
//   *          Tool Bar menus    for Canvas                               *
//   ***********************************************************************
//   *                                                                     *
//   *  ************************************    *************************  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *              P1                  *    *        P2             *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  *                                  *    *                       *  *
//   *  ************************************    *************************  *
//   *                                                                     *
//   ***********************************************************************
//
//   This canvas contains two pads named P1 and P2.
//   Both Canvas, P1 and P2 can be moved, grown, shrinked using the
//   normal rules of the Display manager.
//   A copy of a real canvas with 4 pads is shown in the picture below.
//Begin_Html
/*
<img src="gif/canvas_layout.gif">
*/
//End_Html
//
//  Once objects have been drawn in a canvas, they can be edited/moved
//  by pointing directly to them. The cursor shape is changed
//  to suggest the type of action that one can do on this object.
//  Clicking with the right mouse button on an object pops-up
//  a contextmenu with a complete list of actions possible on this object.
//
//  A graphical editor may be started from the canvas "Edit" menu.
//  Select the "Editor" item. This will show the following editor menu.
//Begin_Html
/*
<img src="gif/editor_menu.gif">
*/
//End_Html
//  For example, to draw a new TText primitive, select the option Text,
//  then click at the position where you want to draw the text. Type <CR>
//  when you have finished typing the string.
//
//  A canvas may be automatically divided into pads via TPad::Divide.
//

//______________________________________________________________________________
TCanvas::TCanvas() : TPad()
{
   // Canvas default constructor.

   Constructor();
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

   fContextMenu = 0;
   fEditorBar   = 0;
   fSelected    = 0;
   fSelectedPad = 0;
   fPadSave     = 0;
   fAutoExec    = kTRUE;
}

//______________________________________________________________________________
TCanvas::TCanvas(const char *name, Int_t ww, Int_t wh, Int_t winid)
{
   // Create an embedded canvas, i.e. a canvas that is in a TGCanvas widget
   // which is placed in a TGFrame. This ctor is only called via the
   // TRootEmbeddedCanvas class.

   // Allow embedded canvas with same name.
   //TCanvas *old = (TCanvas*)gROOT->FindObject(name);
   //if (old && old->IsOnHeap()) delete old;

   fCanvas       = 0;
   fCanvasID     = winid;
   fWindowTopX   = 0;
   fWindowTopY   = 0;
   fWindowWidth  = ww;
   fWindowHeight = wh;
   fCw           = ww;
   fCh           = wh;
   fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
   fBatch        = kFALSE;

   fMenuBar      = kFALSE;
   fContextMenu  = 0;
   fEditorBar    = 0;

   SetName((char *)name);
   Build();
}

//_____________________________________________________________________________
TCanvas::TCanvas(const char *name, const char *title, Int_t form) : TPad()
{
   //  Create a new canvas with a predefined size form.
   //  If form < 0  the menubar is not shown.
   //
   //  form = 1    700x500 at 10,10 (set by TStyle::SetCanvasDefH,W,X,Y)
   //  form = 2    500x500 at 20,20
   //  form = 3    500x500 at 30,30
   //  form = 4    500x500 at 40,40
   //  form = 5    500x500 at 50,50

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
      void *arr[5];
      arr[1] = this; arr[2] = (void*)name; arr[3] = (void*)title; arr[4] =&form;
      if ((*gThreadXAR)("CANV", 5, arr, NULL)) return;
   }

   fMenuBar = kTRUE;
   if (form < 0) {
      form     = -form;
      fMenuBar = kFALSE;
   }
   fCanvasID = -1;
   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(name);
   if (old && old->IsOnHeap()) delete old;
   if (strlen(name) == 0 || gROOT->IsBatch()) {   //We are in Batch mode
      fWindowTopX   = fWindowTopY = 0;
      fWindowWidth  = 712;                // default size corresponds to A4
      fWindowHeight = 950;
      fCw           = fWindowWidth;
      fCh           = fWindowHeight;
      fCanvasImp  = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
      fBatch      = kTRUE;
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
      if (form == 2) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 20, 20, UInt_t(cx*500), UInt_t(cx*500));
      if (form == 3) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 30, 30, UInt_t(cx*500), UInt_t(cx*500));
      if (form == 4) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 40, 40, UInt_t(cx*500), UInt_t(cx*500));
      if (form == 5) fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, 50, 50, UInt_t(cx*500), UInt_t(cx*500));
      fCanvasImp->ShowMenuBar(fMenuBar);
      fCanvasImp->Show();
      fBatch = kFALSE;
   }
   SetName(name);
   SetTitle(title); // requires fCanvasImp set
   Build();
}

//_____________________________________________________________________________
TCanvas::TCanvas(const char *name, const char *title, Int_t ww, Int_t wh) : TPad()
{
   //  Create a new canvas at a random position.
   //
   //  ww is the canvas size in pixels along X
   //      (if ww < 0  the menubar is not shown)
   //  wh is the canvas size in pixels along Y

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
       if ((*gThreadXAR)("CANV", 6, arr, NULL)) return;
   }

   fMenuBar = kTRUE;
   if (ww < 0) {
      ww       = -ww;
      fMenuBar = kFALSE;
   }
   fCanvasID = -1;
   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(name);
   if (old && old->IsOnHeap()) delete old;
   if (strlen(name) == 0 || gROOT->IsBatch()) {   //We are in Batch mode
      fWindowTopX   = fWindowTopY = 0;
      fWindowWidth  = ww;
      fWindowHeight = wh;
      fCw           = ww;
      fCh           = wh;
      fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
      fBatch        = kTRUE;
   } else {
      Float_t cx = gStyle->GetScreenFactor();
      fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, UInt_t(cx*ww), UInt_t(cx*wh));
      fCanvasImp->ShowMenuBar(fMenuBar);
      fCanvasImp->Show();
      fBatch = kFALSE;
   }
   SetName(name);
   SetTitle(title); // requires fCanvasImp set
   Build();
}

//_____________________________________________________________________________
TCanvas::TCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy, Int_t ww, Int_t wh)
        : TPad()
{
   //  Create a new canvas.
   //
   //  wtopx,wtopy are the pixel coordinates of the top left corner of
   //  the canvas (if wtopx < 0) the menubar is not shown)
   //  ww is the canvas size in pixels along X
   //  wh is the canvas size in pixels along Y

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
      if ((*gThreadXAR)("CANV", 8, arr, NULL)) return;
   }

   fMenuBar = kTRUE;
   if (wtopx < 0) {
      wtopx    = -wtopx;
      fMenuBar = kFALSE;
   }
   fCanvasID = -1;
   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(name);
   if (old && old->IsOnHeap()) delete old;
   if (strlen(name) == 0 || gROOT->IsBatch()) {   //We are in Batch mode
      fWindowTopX   = fWindowTopY = 0;
      fWindowWidth  = ww;
      fWindowHeight = wh;
      fCw           = ww;
      fCh           = wh;
      fCanvasImp    = gBatchGuiFactory->CreateCanvasImp(this, name, fCw, fCh);
      fBatch        = kTRUE;
   } else {                   //normal mode with a screen window
      Float_t cx = gStyle->GetScreenFactor();
      fCanvasImp = gGuiFactory->CreateCanvasImp(this, name, Int_t(cx*wtopx), Int_t(cx*wtopy), UInt_t(cx*ww), UInt_t(cx*wh));
      fCanvasImp->ShowMenuBar(fMenuBar);
      fCanvasImp->Show();
      fBatch = kFALSE;
   }
   SetName(name);
   SetTitle(title); // requires fCanvasImp set
   Build();
}

//_____________________________________________________________________________
void TCanvas::Build()
{
   // Build a canvas. Called by all constructors.

   // Make sure the application environment exists. It is need for graphics
   // (colors are initialized in the TApplication ctor).
   if (!gApplication)
      TApplication::CreateApplication();

   // Get some default from .rootrc. Used in fCanvasImp->InitWindow().
   fMoveOpaque      = gEnv->GetValue("Canvas.MoveOpaque", 0);
   fResizeOpaque    = gEnv->GetValue("Canvas.ResizeOpaque", 0);
   fHighLightColor  = gEnv->GetValue("Canvas.HighLightColor", kRed);
   fShowEventStatus = gEnv->GetValue("Canvas.ShowEventStatus", kFALSE);
   fAutoExec        = gEnv->GetValue("Canvas.AutoExec", kTRUE);

   // Get window identifier
   if (fCanvasID == -1)
      fCanvasID = fCanvasImp->InitWindow();
#ifndef WIN32
   if (fCanvasID < 0) return;
#else
   // fCanvasID is in fact a pointer to the TGWin32 class
   if (fCanvasID  == -1) return;
#endif

   fContextMenu = 0;
   if (!IsBatch()) {    //normal mode with a screen window
      // Set default physical canvas attributes
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

      fContextMenu = new TContextMenu( "ContextMenu" );
   }
   // Fill canvas ROOT data structure
   fXsizeUser = 0;
   fYsizeUser = 0;
   if ( fCw < fCh ) {
      fYsizeReal = kDefaultCanvasSize;
      fXsizeReal = fYsizeReal*Float_t(fCw)/Float_t(fCh);
   }
   else {
      fXsizeReal = kDefaultCanvasSize;
      fYsizeReal = fXsizeReal*Float_t(fCh)/Float_t(fCw);
   }

   fDISPLAY         = "$DISPLAY";
   fRetained        = 1;

   // transient canvases have typically no menubar and should not get
   // by default the event status bar (if set by default)
   if (fShowEventStatus && fMenuBar && fCanvasImp)
      fCanvasImp->ShowStatusBar(fShowEventStatus);

   fSelected        = 0;
   fSelectedPad     = 0;
   fPadSave         = 0;
   fEditorBar       = 0;
   fEvent           = -1;
   fEventX          = -1;
   fEventY          = -1;
   gROOT->GetListOfCanvases()->Add(this);

   // Set Pad parameters
   gPad            = this;
   fCanvas         = this;
   fMother         = (TPad*)gPad;
   if (!fPrimitives) {
      fPrimitives     = new TList(this);
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
      Range(0, 0, 1, 1);   //Pad range is set by default to [0,1] in x and y
      PaintBorder(GetFillColor(), kTRUE);    //Paint background
   }
   SetBit(kObjInCanvas);
#ifdef WIN32
   gVirtualX->UpdateWindow(1);
#endif
}

//______________________________________________________________________________
TCanvas::~TCanvas()
{
   // Canvas destructor

   Destructor();
}

//______________________________________________________________________________
void TCanvas::Destructor()
{
   // Actual canvas destructor.

   if (gThreadXAR) {
      void *arr[2];
      arr[1] = this;
      if ((*gThreadXAR)("CDEL", 2, arr, NULL)) return;
   }

   if (!TestBit(kNotDeleted)) return;

   if (fContextMenu) { delete fContextMenu; fContextMenu = 0;}
   if (!gPad) return;

   Close();
}

//______________________________________________________________________________
void TCanvas::cd(Int_t subpadnumber)
{
//*-*-*-*-*-*-*-*-*-*-*-*Set current canvas & pad*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================
//
//    see TPad::cd for explanation of parameter

   if (fCanvasID == -1) return;

   TPad::cd(subpadnumber);

   // in case doublebuffer is off, draw directly onto display window
   if (IsBatch()) return;
   if (!fDoubleBuffer)
      gVirtualX->SelectWindow(fCanvasID);
}

//______________________________________________________________________________
void TCanvas::Clear(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Clear canvas*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============
//  Remove all primitives in default pad
//  Remove all pads (except default pad)
//

   if (fCanvasID == -1) return;
   TPad::Clear(option);   //Remove primitives from pad

   fSelected    = 0;
   fSelectedPad = 0;
}

//______________________________________________________________________________
void TCanvas::Close(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Close canvas*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============
//  Delete window/pads data structure
//

   if (fCanvasID == -1) return;
   TCanvas *cansave = (TCanvas*)gPad->GetCanvas();
   TPad    *padsave = (TPad*)gPad;

   FeedbackMode(kFALSE);

   TPad::Close(option);

   if (!IsBatch()) {
      gVirtualX->SelectWindow(fCanvasID);    //select current canvas
#ifndef WIN32
      gVirtualX->CloseWindow();
#endif
   }
   fCanvasID = -1;
   fBatch    = kTRUE;

   // Close actual window on screen
   SafeDelete(fCanvasImp);

   gROOT->GetListOfCanvases()->Remove(this);

   if (cansave == this) {
      gPad = (TCanvas *) gROOT->GetListOfCanvases()->First();
   } else {
      gPad = padsave;
   }
}

//______________________________________________________________________________
void TCanvas::CopyPixmaps()
{
//*-*-*-*-*-*-*-*-*Copy the canvas pixmap of the pad to the canvas*-*-*-*-*-*-*
//*-*              ===============================================

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


   TCanvas *old = (TCanvas*)gROOT->GetListOfCanvases()->FindObject(GetName());
   if (old == this) {
      Paint();
      return;
   }
   if (old) gROOT->GetListOfCanvases()->Remove(old);
   if (fWindowWidth  == 0) fWindowWidth  = 800;
   if (fWindowHeight == 0) fWindowHeight = 600;
   fCanvasImp = gGuiFactory->CreateCanvasImp(this, GetName(), fWindowTopX, fWindowTopY,
                                             fWindowWidth, fWindowHeight);
   fCanvasImp->ShowMenuBar(fMenuBar);
   fCanvasImp->Show();

   Build();
   ResizePad();
   Modified();
}

//______________________________________________________________________________
void TCanvas::DrawClone(Option_t *option)
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
}


//______________________________________________________________________________
void TCanvas::DrawClonePad()
{
   // Draw a clone of this canvas into the current pad
   // In an interactive session, select the destination/current pad
   // with the middle mouse button, then point to the canvas area to select
   // the canvas context menu item DrawClonePad.
   // Note that the original canvas may have subpads.

  TPad *padsav = (TPad*)gPad;
  TPad *pad = (TPad*)gROOT->GetSelectedPad();
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
     gROOT->SetSelectedPad(pad);
     clone = obj->Clone();
     pad->GetListOfPrimitives()->Add(clone,obj->GetDrawOption());
  }
  pad->Modified();
  pad->Update();
  padsav->cd();
}


//______________________________________________________________________________
void TCanvas::DrawEventStatus(Int_t event, Int_t px, Int_t py, TObject *selected)
{
//*-*-*-*-*-*-*Report name and title of primitive below the cursor*-*-*-*-*-*
//*-*          ===================================================
//
//    This function is called when the option "Event Status"
//    in the canvas menu "Options" is selected.
//

   const Int_t kTMAX=256;
   static char atext[kTMAX];

   if (!selected) return;

//#ifndef WIN32
#if 0
   static Int_t pxt, pyt;
   gPad->SetDoubleBuffer(0);           // Turn off double buffer mode
   gVirtualX->SetTextColor(1);
   gVirtualX->SetTextAlign(11);

   pxt = gPad->GetCanvas()->XtoAbsPixel(gPad->GetCanvas()->GetX1()) + 5;
   pyt = gPad->GetCanvas()->YtoAbsPixel(gPad->GetCanvas()->GetY1()) - 5;

   sprintf(atext,"%s / %s ", selected->GetName()
                           , selected->GetObjectInfo(px,py));
   for (Int_t i=strlen(atext);i<kTMAX-1;i++) atext[i] = ' ';
   atext[kTMAX-1] = 0;
   gVirtualX->DrawText(pxt, pyt, 0, 1, atext, TVirtualX::kOpaque);
#else
   if (!fCanvasImp) return; //this may happen when closing a TAttCanvas
   fCanvasImp->SetStatusText(selected->GetTitle(),0);
   fCanvasImp->SetStatusText(selected->GetName(),1);
   if (event == kKeyPress)
      sprintf(atext, "%c", (char) px);
   else
      sprintf(atext, "%d,%d", px, py);
   fCanvasImp->SetStatusText(atext,2);
   fCanvasImp->SetStatusText(selected->GetObjectInfo(px,py),3);
#endif
}

//______________________________________________________________________________
void TCanvas::EditorBar()
{
//*-*-*-*-*-*-*-*-*-*-*Create the Editor Controlbar*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

   TControlBar *ed = new TControlBar("vertical", "Editor");
   ed->AddButton("Arc",       "gROOT->SetEditorMode(\"Arc\")",       "Create an arc of circle");
   ed->AddButton("Line",      "gROOT->SetEditorMode(\"Line\")",      "Create a line segment");
   ed->AddButton("Arrow",     "gROOT->SetEditorMode(\"Arrow\")",     "Create an Arrow");
   ed->AddButton("Button",    "gROOT->SetEditorMode(\"Button\")",    "Create a user interface Button");
   ed->AddButton("Diamond",   "gROOT->SetEditorMode(\"Diamond\")",   "Create a diamond");
   ed->AddButton("Ellipse",   "gROOT->SetEditorMode(\"Ellipse\")",   "Create an Ellipse");
   ed->AddButton("Pad",       "gROOT->SetEditorMode(\"Pad\")",       "Create a pad");
   ed->AddButton("Pave",      "gROOT->SetEditorMode(\"Pave\")",      "Create a Pave");
   ed->AddButton("PaveLabel", "gROOT->SetEditorMode(\"PaveLabel\")", "Create a PaveLabel (prompt for label)");
   ed->AddButton("PaveText",  "gROOT->SetEditorMode(\"PaveText\")",  "Create a PaveText");
   ed->AddButton("PavesText", "gROOT->SetEditorMode(\"PavesText\")", "Create a PavesText");
   ed->AddButton("PolyLine",  "gROOT->SetEditorMode(\"PolyLine\")",  "Create a PolyLine (TGraph)");
   ed->AddButton("CurlyLine", "gROOT->SetEditorMode(\"CurlyLine\")", "Create a Curly/WavyLine");
   ed->AddButton("CurlyArc",  "gROOT->SetEditorMode(\"CurlyArc\")",  "Create a Curly/WavyArc");
   ed->AddButton("Text/Latex","gROOT->SetEditorMode(\"Text\")",      "Create a Text/Latex string");
   ed->AddButton("Marker",    "gROOT->SetEditorMode(\"Marker\")",    "Create a marker");
   ed->AddButton("<...Graphical Cut...>",  "gROOT->SetEditorMode(\"CutG\")","Create a Graphical Cut");
   ed->Show();
   fEditorBar = ed;
}

//______________________________________________________________________________
void TCanvas::EnterLeave(TPad *prevSelPad, TObject *prevSelObj)
{
   // Generate kMouseEnter and kMouseLeave events depending on the previously
   // selected object and the currently selected object. Does nothing if the
   // selected object does not change.

   if (prevSelObj == fSelected) return;

   TPad *padsav = (TPad *)gPad;

   if (prevSelObj) {
      gPad = prevSelPad;
      prevSelObj->ExecuteEvent(kMouseLeave, 0, 0);
   }

   gPad = fSelectedPad;

   if (fSelected)
      fSelected->ExecuteEvent(kMouseEnter, 0, 0);

   gPad = padsav;
}

//______________________________________________________________________________
void TCanvas::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function must be implemented to realize the action
//  corresponding to the mouse click on the object in the canvas
//
//  Only handle mouse motion events in TCanvas, all other events are
//  ignored for the time being
//

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
//*-*-*-*-*-*-*-*-*Turn rubberband feedback mode on or off*-*-*-*-*-*-*-*-*-*-*
//*-*              =======================================
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
//*-*-*-*-*-*-*-*-*Flush canvas buffers*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ====================
   if (fCanvasID == -1) return;

   TPad *padsav = (TPad*)gPad;
   cd();
   if (!IsBatch()) {
      gVirtualX->SelectWindow(fCanvasID);
      gPad = padsav; //don't do cd() because than also the pixmap is changed
      CopyPixmaps();
      gVirtualX->UpdateWindow(1);
   }
   padsav->cd();
}

//______________________________________________________________________________
void TCanvas::UseCurrentStyle()
{
//*-*-*-*-*-*Force a copy of current style for all objects in canvas*-*-*-*-*
//*-*        =======================================================

   TPad::UseCurrentStyle();

   SetFillColor(gStyle->GetCanvasColor());
   fBorderSize = gStyle->GetCanvasBorderSize();
   fBorderMode = gStyle->GetCanvasBorderMode();
}

//______________________________________________________________________________
void *TCanvas::GetPadDivision(Int_t, Int_t)
{
//*-*-*-*-*-*-*-*-*Return pad corresponding to one canvas division*-*-*-*-*
//*-*              ===============================================
   return 0;
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
//*-*-*-*-*-*-*-*-*-*Handle Input Events*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ===================
//  Handle input events, like button up/down in current canvas.
//

   TPad         *pad;
   TObjLink     *pickobj;
   TObject      *prevSelObj = 0;
   TPad         *prevSelPad = 0;

   if (fSelected && fSelected->TestBit(kNotDeleted))
      prevSelObj = fSelected;
   if (fSelectedPad && fSelectedPad->TestBit(kNotDeleted))
      prevSelPad = (TPad*) fSelectedPad;

   fPadSave = (TPad*)gPad;
   cd();        // make sure this canvas is the current canvas

   fEvent  = event;
   fEventX = px;
   fEventY = py;

   switch (event) {

   case kMouseMotion:
      // highlight object tracked over
      fSelected    = 0;
      fSelectedOpt = "";
      fSelectedPad = 0;
      pickobj      = 0;
      pad = Pick(px, py, pickobj);
      if (!pad) return;

      if (!pickobj) {
         fSelected    = pad;
         fSelectedOpt = "";
      } else {
         if (!fSelected) {
            fSelected    = pickobj->GetObject();
            fSelectedOpt = pickobj->GetOption();
         }
      }
      fSelectedPad = pad;

      EnterLeave(prevSelPad, prevSelObj);

      gPad = pad;   // don't use cd() we will use the current
                    // canvas via the GetCanvas member and not via
                    // gPad->GetCanvas

      fSelected->ExecuteEvent(event, px, py);

      if (fShowEventStatus) DrawEventStatus(event, px, py, fSelected);
      if (fAutoExec)        RunAutoExec();

      break;

   case kMouseLeave:
      // mouse leaves canvas
      {
         // force popdown of tooltips
         TObject     *sobj = fSelected;
         TVirtualPad *spad = fSelectedPad;
         fSelected    = 0;
         fSelectedPad = 0;
         EnterLeave(prevSelPad, prevSelObj);
         fSelected    = sobj;
         fSelectedPad = spad;
      }
      break;

   case kButton1Double:
      // triggered on the second button down within 350ms and within
      // 3x3 pixels of the first button down, button up finishes action

   case kButton1Down:

      // find pad in which input occured
      fSelected    = 0;
      fSelectedOpt = "";
      fSelectedPad = 0;
      pickobj      = 0;
      pad = Pick(px, py, pickobj);
      if (!pad) return;

      if (!pickobj) {
         fSelected    = pad;
         fSelectedOpt = "";
      } else {
         if (!fSelected) {
            fSelected    = pickobj->GetObject();
            fSelectedOpt = pickobj->GetOption();
         }
      }
      fSelectedPad = pad;

      gPad = pad;   // don't use cd() because we won't draw in pad
                    // we will only use its coordinate system

      FeedbackMode(kTRUE);   // to draw in rubberband mode

      fSelected->ExecuteEvent(event, px, py);

      if (fShowEventStatus) DrawEventStatus(event, px, py, fSelected);
      if (fAutoExec)        RunAutoExec();

      break;

   case kButton1Motion:

      if (fSelected) {
         gPad = fSelectedPad;

         fSelected->ExecuteEvent(event, px, py);

         {
            Bool_t resize = kFALSE;
            if (fSelected->InheritsFrom(TBox::Class()))
               resize = ((TBox*)fSelected)->IsBeingResized();
            if (fSelected->InheritsFrom(TVirtualPad::Class()))
               resize = ((TVirtualPad*)fSelected)->IsBeingResized();

            if ((!resize && fMoveOpaque) || (resize && fResizeOpaque)) {
               gPad = fPadSave;
               Update();
               FeedbackMode(kTRUE);
            }
         }
         if (fShowEventStatus) DrawEventStatus(event, px, py, fSelected);
         if (fAutoExec)        RunAutoExec();
      }

      break;

   case kButton1Up:

      if (fSelected) {
         gPad = fSelectedPad;

         fSelected->ExecuteEvent(event, px, py);

         if (fShowEventStatus) DrawEventStatus(event, px, py, fSelected);
         if (fAutoExec)        RunAutoExec();

         if (fPadSave->TestBit(kNotDeleted))
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
      fSelected    = 0;
      fSelectedOpt = "";
      fSelectedPad = 0;
      pickobj      = 0;
      pad = Pick(px, py, pickobj);
      if (!pad) return;

      if (!pickobj) {
         fSelected    = pad;
         fSelectedOpt = "";
      } else {
         if (!fSelected) {
            fSelected    = pickobj->GetObject();
            fSelectedOpt = pickobj->GetOption();
         }
      }
      fSelectedPad = pad;

      gPad = pad;   // don't use cd() because we won't draw in pad
                    // we will only use its coordinate system

      FeedbackMode(kTRUE);

      fSelected->Pop();           // pop object to foreground
      pad->cd();                  // and make its pad the current pad
      if (gDebug)
         printf("Current Pad: %s / %s\n", pad->GetName(), pad->GetTitle());

      // loop over all canvases to make sure that only one pad is highlighted
      {
         TIter next(gROOT->GetListOfCanvases());
         TCanvas *tc;
         while ((tc = (TCanvas *)next()))
            tc->Update();
      }

      return;   // don't want fPadSave->cd() to be executed at the end

   case kButton2Motion:
      break;

   case kButton2Up:
      break;

   case kButton2Double:
      break;

//*-*----------------------------------------------------------------------

   case kButton3Down:
   {
      pad = Pick(px, py, pickobj);
      if (!pad) return;

      if (!pickobj) {
         fSelected    = pad;
         fSelectedOpt = "";
      } else {
         if (!fSelected) {
            fSelected    = pickobj->GetObject();
            fSelectedOpt = pickobj->GetOption();
         }
      }
      fSelectedPad = pad;

      if (fContextMenu)
          fContextMenu->Popup(px, py, fSelected, this, pad);

      break;
   }
   case kButton3Motion:
      break;

   case kButton3Up:
      break;

   case kButton3Double:
      break;

   case kKeyPress:

      // find pad in which input occured
      fSelected    = 0;
      fSelectedOpt = "";
      fSelectedPad = 0;
      pickobj      = 0;
      pad = Pick(px, py, pickobj);
      if (!pad) return;

      if (!pickobj) {
         fSelected    = pad;
         fSelectedOpt = "";
      } else {
         if (!fSelected) {
            fSelected    = pickobj->GetObject();
            fSelectedOpt = pickobj->GetOption();
         }
      }
      fSelectedPad = pad;

      gPad = pad;   // don't use cd() because we won't draw in pad
                    // we will only use its coordinate system

      fSelected->ExecuteEvent(event, px, py);

      if (fShowEventStatus) DrawEventStatus(event, px, py, fSelected);
      if (fAutoExec)        RunAutoExec();

      break;

   default:
      break;
   }

   if (fPadSave) fPadSave->cd();
}

//______________________________________________________________________________
void TCanvas::ls(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*List all pads*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          =============
   TROOT::IndentLevel();
   cout <<"Canvas Name=" <<GetName()<<" Title="<<GetTitle()<<" Option="<<option<<endl;
   TROOT::IncreaseDirLevel();
   TPad::ls(option);
   TROOT::DecreaseDirLevel();
}


//______________________________________________________________________________
void TCanvas::MakeDefCanvas()
{
//*-*-*-*-*-*-*Static function to build a default canvas*-*-*-*-*-*-*-*-*-*-*
//*-*          =========================================

   const char *defcanvas = gROOT->GetDefCanvasName();
   char *cdef;

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (lc->FindObject(defcanvas))
      cdef = StrDup(Form("%s_n%d",defcanvas,lc->GetSize()+1));
   else
      cdef = StrDup(Form("%s",defcanvas));

//   if (gInterpreter)
//      gROOT->ProcessLine(Form("TCanvas *%s = new TCanvas(\"%s\",\"%s\",1);",cdef,cdef,cdef));
//   else
      new TCanvas(cdef, cdef, 1);

   Printf("<TCanvas::MakeDefCanvas>: created default TCanvas with name %s",cdef);
   delete [] cdef;
}

//______________________________________________________________________________
void TCanvas::MoveOpaque(Int_t set)
{
//*-*-*-*-*-*-*-*-*Set option to move objects/pads in a canvas*-*-*-*-*-*-*-*
//*-*              ===========================================
//
//  if set = 1 (default) graphics objects are moved in opaque mode
//         = 0 only the outline of objects is drawn when moving them
//  The option opaque produces the best effect. It requires however a
//  a reasonably fast workstation or response time.
//
   fMoveOpaque = set;
}

//______________________________________________________________________________
void TCanvas::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Paint canvas*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ============

   if (fCanvas)
      TPad::Paint(option);

}

//______________________________________________________________________________
void TCanvas::Resize(Option_t *)
{
//*-*-*-*-*-*-*Recompute canvas parameters following a X11 Resize*-*-*-*-*-*-*
//*-*          ==================================================

   if (fCanvasID == -1) return;

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

   padsav->cd();
}

//______________________________________________________________________________
void TCanvas::RunAutoExec()
{
   // Execute the list of TExecs in the current pad.

   if (!gPad) return;
   ((TPad*)gPad)->AutoExec();
}

//______________________________________________________________________________
void TCanvas::SaveSource(const char *filename, Option_t *option)
{
//*-*-*-*-*-*-*Save primitives in this canvas as a C++ macro file*-*-*-*-*-*
//*-*          ==================================================

//    reset bit TClass::kClassSaved for all classes
   TIter next(gROOT->GetListOfClasses());
   TClass *cl;
   while((cl = (TClass*)next())) {
      cl->ResetBit(TClass::kClassSaved);
   }

   char quote = '"';
   ofstream out;
   Int_t lenfile = strlen(filename);
   char * fname;
//    if filename is given, open this file, otherwise create a file
//    with a name equal to the canvasname.C
   if (lenfile) {
       fname = (char*)filename;
       out.open(fname, ios::out);
   } else {
       Int_t nch = strlen(GetName());
       fname = new char[nch+3];
       strcpy(fname,GetName());
       strcat(fname,".C");
       out.open(fname, ios::out);
   }
   if (!out.good ()) {
      Printf("SaveSource cannot open file: %s",fname);
      if (!lenfile) delete [] fname;
      return;
   }

//   Write macro header and date/time stamp
   TDatime t;
   Float_t cx = gStyle->GetScreenFactor();
   Int_t w = Int_t((fWindowWidth)/cx);
   Int_t h = Int_t((fWindowHeight)/cx);

   out <<"{"<<endl;
   out <<"//=========Macro generated from canvas: "<<GetName()<<"/"<<GetTitle()<<endl;
   out <<"//=========  ("<<t.AsString()<<") by ROOT version"<<gROOT->GetVersion()<<endl;
//   out <<"   gROOT->Reset();"<<endl;

//   Write canvas parameters (TDialogCanvas case)
   if (InheritsFrom(TDialogCanvas::Class())) {
      out<<"   "<<ClassName()<<" *"<<GetName()<<" = new "<<ClassName()<<"("<<quote<<GetName()<<quote<<", "<<quote<<GetTitle()
         <<quote<<","<<w<<","<<h<<");"<<endl;
   } else {
//   Write canvas parameters (TCanvas case)
      out<<"   TCanvas *"<<GetName()<<" = new TCanvas("<<quote<<GetName()<<quote<<", "<<quote<<GetTitle()
         <<quote<<","<<GetWindowTopX()<<","<<GetWindowTopY()<<","<<w<<","<<h<<");"<<endl;
   }
//   Write canvas options (in $TROOT or $TStyle)
   if (gStyle->GetOptFit()) {
      out<<"   gStyle->SetOptFit(1);"<<endl;
   }
   if (!gStyle->GetOptStat()) {
      out<<"   gStyle->SetOptStat(0);"<<endl;
   }
   if (gROOT->GetEditHistograms()) {
      out<<"   gROOT->SetEditHistograms();"<<endl;
   }
   if (GetShowEventStatus()) {
      out<<"   "<<GetName()<<"->ToggleEventStatus();"<<endl;
   }
   if (GetHighLightColor() != 5) {
      out<<"   "<<GetName()<<"->SetHighLightColor("<<GetHighLightColor()<<");"<<endl;
   }


//   Now recursively scan all pads of this canvas
   cd();
   TPad::SavePrimitive(out,option);

   out <<"}"<<endl;
   out.close();
   Printf("C++ Macro file: %s has been generated", fname);

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

   if (fCanvasImp) fCanvasImp->SetCanvasSize(ww, wh);
}

//______________________________________________________________________________
void TCanvas::SetCursor(ECursor cursor)
{
//*-*-*-*-*-*-*-*-*-*-*Set cursor*-*-**-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==========

   if (IsBatch()) return;
   gVirtualX->SetCursor(fCanvasID, cursor);
}

//______________________________________________________________________________
void TCanvas::SetDoubleBuffer(Int_t mode)
{
//*-*-*-*-*-*-*-*-*-*-*Set Double Buffer On/Off*-*-**-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
   if (IsBatch()) return;
   fDoubleBuffer = mode;
   gVirtualX->SetDoubleBuffer(fCanvasID, mode);

   // depending of the buffer mode set the drawing window to either
   // the canvas pixmap or to the canvas on-screen window
#ifndef WIN32
   if (fDoubleBuffer)
      gVirtualX->SelectWindow(fPixmapID);
   else
#endif
      gVirtualX->SelectWindow(fCanvasID);
}

//______________________________________________________________________________
void TCanvas::SetTitle(const char *title)
{
//*-*-*-*-*-*-*-*-*-*-*Set Canvas title*-*-**-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============

   fTitle = title;
   if (fCanvasImp) fCanvasImp->SetWindowTitle(title);
}

//______________________________________________________________________________
void TCanvas::Size(Float_t xsize, Float_t ysize)
{
//*-*-*-*-*-*-*Set the canvas scale in centimeters*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*          ===================================
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
//

   fXsizeUser = xsize;
   fYsizeUser = ysize;

   Resize();
}

//_______________________________________________________________________
void TCanvas::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   UInt_t R__s, R__c;
   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      gPad    = this;
      fCanvas = this;
      TPad::Streamer(b);
      gPad    = this;
      fDISPLAY.Streamer(b);
      b >> fDoubleBuffer;
      b >> fRetained;
      b >> fXsizeUser;
      b >> fYsizeUser;
      b >> fXsizeReal;
      b >> fYsizeReal;
      fCanvasID = -1;
      b >> fWindowTopX;
      b >> fWindowTopY;
      if (v > 2) {
         b >> fWindowWidth;
         b >> fWindowHeight;
      }
      b >> fCw;
      b >> fCh;
      if (v <= 2) {
         fWindowWidth  = fCw;
         fWindowHeight = fCh;
      }
      fCatt.Streamer(b);
      b >> fMoveOpaque;
      b >> fResizeOpaque;
      b >> fHighLightColor;
      b >> fBatch;
      fBatch = gROOT->IsBatch();
      if (v < 2) return;
      b >> fShowEventStatus;
      if (v > 3)
         b >> fAutoExec;
      b >> fMenuBar;
      b.CheckByteCount(R__s, R__c, TCanvas::IsA());
   } else {
      R__c = b.WriteVersion(TCanvas::IsA(), kTRUE);
      TPad::Streamer(b);
      fDISPLAY.Streamer(b);
      b << fDoubleBuffer;
      b << fRetained;
      b << fXsizeUser;
      b << fYsizeUser;
      b << fXsizeReal;
      b << fYsizeReal;
      UInt_t w = fWindowWidth; // must be saved, modified by GetWindowTopX in batch
      UInt_t h = fWindowHeight;
      b << GetWindowTopX();
      b << GetWindowTopY();
      fWindowWidth  = w;
      fWindowHeight = h;
      b << w;
      b << h;
      b << fCw;
      b << fCh;
      fCatt.Streamer(b);
      b << fMoveOpaque;
      b << fResizeOpaque;
      b << fHighLightColor;
      b << fBatch;
      b << fShowEventStatus;
      b << fAutoExec;
      b << fMenuBar;
      b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
void TCanvas::ToggleAutoExec()
{
   // Toggle pad auto execution of list of TExecs.

   fAutoExec = fAutoExec ? kFALSE : kTRUE;
}

//______________________________________________________________________________
void TCanvas::ToggleEventStatus()
{
   // Toggle event statusbar.

   fShowEventStatus = fShowEventStatus ? kFALSE : kTRUE;

   if (fCanvasImp) fCanvasImp->ShowStatusBar(fShowEventStatus);
}

//______________________________________________________________________________
void TCanvas::Update()
{
   // Update canvas pad buffers

   if (gThreadXAR) {
      void *arr[2];
      arr[1] = this;
      if ((*gThreadXAR)("CUPD", 2, arr, NULL)) return;
   }

   if (!IsBatch()) FeedbackMode(kFALSE);      // Goto double buffer mode

   PaintModified();           // Repaint all modified pad's

   Flush();                   // Copy all pad pixmaps to the screen

   SetCursor(kCross);
}
