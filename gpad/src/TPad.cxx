// @(#)root/gpad:$Name:  $:$Id: TPad.cxx,v 1.61 2001/12/14 13:31:30 brun Exp $
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
#include "TError.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TH1.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TClassTable.h"
#include "TVirtualPS.h"
#include "TVirtualX.h"
#include "TView.h"
#include "TPoint.h"
#include "TGraph.h"
#include "TArrow.h"
#include "TPaveText.h"
#include "TPavesText.h"
#include "TPaveLabel.h"
#include "TCurlyArc.h"
#include "TArc.h"
#include "TLatex.h"
#include "TMarker.h"
#include "TDiamond.h"
#include "TGroupButton.h"
#include "TBrowser.h"
#include "TVirtualGL.h"
#include "TString.h"
#include "TLine.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TFrame.h"
#include "TGaxis.h"
#include "TExec.h"
#include "TPadView3D.h"
#include "TDatime.h"
#include "TColor.h"

// Local scratch buffer for screen points, faster than allocating buffer on heap
const Int_t kPXY       = 1002;
const Int_t kButton    = 101;
const Int_t kCutG      = 100;
const Int_t kCurlyLine = 200;
const Int_t kCurlyArc  = 201;

static TPoint gPXY[kPXY];
static Int_t readLevel = 0;

Int_t TPad::fgMaxPickDistance = 5;

ClassImpQ(TPad)

//______________________________________________________________________________
//  The Pad class is the most important graphics class in the ROOT system.
//Begin_Html
/*
<img src="gif/tpad_classtree.gif">
*/
//End_Html
//  A Pad is contained in a Canvas.
//  A Pad may contain other pads (unlimited pad hierarchy).
//  A pad is a linked list of primitives of any type (graphics objects,
//  histograms, detectors, tracks, etc.).
//  Adding a new element into a pad is in general performed by the Draw
//  member function of the object classes.
//  It is important to realize that the pad is a linked list of references
//  to the original object.
//  For example, in case of an histogram, the histogram.Draw() operation
//  only stores a reference to the histogram object and not a graphical
//  representation of this histogram.
//  When the mouse is used to change (say the bin content), the bin content
//  of the original histogram is changed !!
//
//  The convention used in ROOT is that a Draw operation only adds
//  a reference to the object. The effective drawing is performed when
//  when the canvas receives a signal to be painted.
//  This signal is generally sent when typing carriage return in the
//  command input or when a graphical operation has been performed on one
//  of the pads of this canvas.
//  When a Canvas/Pad is repainted, the member function Paint for all
//  objects in the Pad linked list is invoked.
//
//  When the mouse is moved on the Pad, The member function DistancetoPrimitive
//  is called for all the elements in the pad. DistancetoPrimitive returns
//  the distance in pixels to this object.
//  when the object is within the distance window, the member function
//  ExecuteEvent is called for this object.
//  in ExecuteEvent, move, changes can be performed on the object.
//  For examples of DistancetoPrimitive and ExecuteEvent functions,
//  see classes TLine::DistancetoPrimitive, TLine::ExecuteEvent
//              TBox::DistancetoPrimitive,  TBox::ExecuteEvent
//              TH1::DistancetoPrimitive,   TH1::ExecuteEvent
//
//  A Pad supports linear and log scales coordinate systems.
//  The transformation coefficients are explained in TPad::ResizePad.
//  An example of pads hierarchy is shown below:
//Begin_Html
/*
<img src="examples/gif/canvas.gif">
*/
//End_Html
//

//______________________________________________________________________________
TPad::TPad(): TVirtualPad()
{
//*-*-*-*-*-*-*-*-*-*-*Pad default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================

   fTip        = 0;
   fPadPointer = 0;
   fPrimitives = 0;
   fExecs      = 0;
   fCanvas     = 0;
   fMother     = 0;
   fPadPaint   = 0;
   fPixmapID   = -1;
   fTheta      = 30;
   fPhi        = 30;
   fNumber     = 0;
   fAbsCoord   = kFALSE;
   fEditable   = kTRUE;
   fCrosshair  = 0;
   fCrosshairPos = 0;
   fPadView3D  = 0;

   fFixedAspectRatio = kFALSE;
   fAspectRatio      = 0.;

   fLogx  = 0;
   fLogy  = 0;
   fLogz  = 0;
   fGridx = 0;
   fGridy = 0;
   fTickx = 0;
   fTicky = 0;
   fFrame = 0;
   fView  = 0;

   fUxmin = fUymin = fUxmax = fUymax = 0;

//*-*- Set default world coordinates to NDC [0,1]
   fX1 = 0;
   fX2 = 1;
   fY1 = 0;
   fY2 = 1;

//*-*- Set default pad range
   fXlowNDC = 0;
   fYlowNDC = 0;
   fWNDC    = 1;
   fHNDC    = 1;
}

//______________________________________________________________________________
TPad::TPad(const char *name, const char *title, Double_t xlow,
           Double_t ylow, Double_t xup, Double_t yup,
           Color_t color, Short_t bordersize, Short_t bordermode)
          : TVirtualPad(name,title,xlow,ylow,xup,yup,color,bordersize,bordermode)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Pad constructor-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          ===============
//  A pad is a linked list of primitives.
//  A pad is contained in a canvas. It may contain other pads.
//  A pad has attributes. When a pad is created, the attributes
//  defined in the current style are copied to the pad attributes.
//
//  xlow [0,1] is the position of the bottom left point of the pad
//             expressed in the mother pad reference system
//  ylow [0,1] is the Y position of this point.
//  xup  [0,1] is the x position of the top right point of the pad
//             expressed in the mother pad reference system
//  yup  [0,1] is the Y position of this point.
//
//  the bordersize is in pixels
//  bordermode = -1 box looks as it is behind the screen
//  bordermode = 0  no special effects
//  bordermode = 1  box looks as it is in front of the screen
//

   fTip        = 0;
   fBorderSize = bordersize;
   fBorderMode = bordermode;
   if (gPad)   fCanvas = gPad->GetCanvas();
   else        fCanvas = (TCanvas*)this;
   fMother     = (TPad*)gPad;
   fPrimitives = new TList;
   fExecs      = new TList;
   fPadPointer = 0;
   fTheta      = 30;
   fPhi        = 30;
   fGridx      = gStyle->GetPadGridX();
   fGridy      = gStyle->GetPadGridY();
   fTickx      = gStyle->GetPadTickX();
   fTicky      = gStyle->GetPadTickY();
   fFrame      = 0;
   fView       = 0;
   fPadPaint   = 0;
   fPixmapID   = -1;      // -1 means pixmap will be created by ResizePad()
   fNumber     = 0;
   fAbsCoord   = kFALSE;
   fEditable   = kTRUE;
   fPadView3D  = 0;
   fCrosshair  = 0;
   fCrosshairPos = 0;

   fFixedAspectRatio = kFALSE;
   fAspectRatio      = 0.;

//*-*- Set default world coordinates to NDC [0,1]
   fX1 = 0;
   fX2 = 1;
   fY1 = 0;
   fY2 = 1;

   if (!gPad) {
      Error("TPad", "You must create a TCanvas before creating a TPad");
      MakeZombie();
      return;
   }

   TPad *padsav = (TPad*)gPad;

   if ((xlow < 0) || (xlow > 1) || (ylow < 0) || (ylow > 1)) {
      Error("TPad", "illegal bottom left position: x=%f, y=%f", xlow, ylow);
      goto zombie;
   }
   if ((xup < 0) || (xup > 1) || (yup < 0) || (yup > 1)) {
      Error("TPad", "illegal top right position: x=%f, y=%f", xup, yup);
      goto zombie;
   }

   fLogx = gStyle->GetOptLogx();
   fLogy = gStyle->GetOptLogy();
   fLogz = gStyle->GetOptLogz();

   fUxmin = fUymin = fUxmax = fUymax = 0;

//*-*- Set pad parameters and Compute conversion coeeficients
   SetPad(name, title, xlow, ylow, xup, yup, color, bordersize, bordermode);
   Range(0, 0, 1, 1);
   SetBit(kCanDelete);

   padsav->cd();
   return;

zombie:
   // error in creating pad occured, make this pad a zombie
   MakeZombie();
   padsav->cd();
}

//______________________________________________________________________________
TPad::~TPad()
{
//*-*-*-*-*-*-*-*-*-*-*Pad destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============

   if (!TestBit(kNotDeleted)) return;
   Close();
   CloseToolTip(fTip);
   DeleteToolTip(fTip);
   SafeDelete(fPrimitives);
   SafeDelete(fExecs);
   if (fPadView3D) {
      fPadView3D->SetPad();
      fPadView3D = 0;
   }
}

//______________________________________________________________________________
void TPad::AddExec(const char *name, const char*command)
{
// Add a new TExec object to the list of Execs.
// When an event occurs in the pad (mouse click, etc) the list of CINT commands
// in the list of Execs are executed via TPad::AutoExec.
//  When a pad event occurs (mouse move, click, etc) all the commands
//  contained in the fExecs list are executed in the order found in the list.
//  This facility is activated by default. It can be deactivated by using
//  the canvas "Option" menu.
//  The following examples of TExec commands are provided in the tutorials:
//  macros exec1.C and exec2.C.
//  Example1 of use of exec1.C
//  ==========================
//  Root > TFile f("hsimple.root")
//  Root > hpx.Draw()
//  Root > c1.AddExec("ex1",".x exec1.C")
//   At this point you can use the mouse to click on the contour of
//   the histogram hpx. When the mouse is clicked, the bin number and its
//   contents are printed.
//  Example2 of use of exec1.C
//  ==========================
//  Root > TFile f("hsimple.root")
//  Root > hpxpy.Draw()
//  Root > c1.AddExec("ex2",".x exec2.C")
//    When moving the mouse in the canvas, a second canvas shows the
//    projection along X of the bin corresponding to the Y position
//    of the mouse. The resulting histogram is fitted with a gaussian.
//    A "dynamic" line shows the current bin position in Y.
//    This more elaborated example can be used as a starting point
//    to develop more powerful interactive applications exploiting CINT
//    as a development engine.

   if (!fExecs) fExecs = new TList;
   TExec *ex = new TExec(name,command);
   fExecs->Add(ex);
}

//______________________________________________________________________________
void TPad::AutoExec()
{
// Execute the list of Execs when a pad event occurs.

   if (fCrosshair) DrawCrosshair();

   if (!fExecs) fExecs = new TList;
   TIter next(fExecs);
   TExec *exec;
   while ((exec = (TExec*)next())) {
      exec->Exec();
   }
}

//______________________________________________________________________________
void TPad::Browse(TBrowser *b)
{
    cd();
    fPrimitives->Browse(b);
}


//______________________________________________________________________________
void TPad::cd(Int_t subpadnumber)
{
//*-*-*-*-*-*-*-*-*Set Current Pad*-*-*-*-*-**-*-*-*-*-*-*-*
//*-*              ====================
//  When a canvas/pad is divided via TPad::Divide, one can directly
//  set the current path to one of the subdivisions.
//  See TPad::Divide for the convention to number subpads.
//  For example:
//    c1.Divide(2,3); // create 6 pads (2 divisions along x, 3 along y).
//    To set the current pad to the bottom right pad, do
//    c1.cd(6);
//  Note1:  c1.cd() is equivalent to c1.cd(0) and sets the current pad
//          to c1 itself.
//  Note2:  after a statement like c1.cd(6), the global variable gPad
//          points to the current pad. One can use gPad to set attributes
//          of the current pad.

   if (!subpadnumber) {
      gPad = this;
      if (!gPad->IsBatch()) gVirtualX->SelectWindow(fPixmapID);
      return;
   }

   TObject *obj;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         Int_t n = ((TPad*)obj)->GetNumber();
         if (n == subpadnumber) {
            ((TPad*)obj)->cd();
            return;
         }
      }
   }
}

//______________________________________________________________________________
void TPad::Clear(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Delete all pad primitives*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================
//
//   If the bit kClearAfterCR has been set for this pad, the Clear function
//   will execute only after having pressed a CarriageReturn
//   Set the bit with mypad->SetBit(TPad::kClearAfterCR)

   if (!IsEditable()) return;

   if (!fPadPaint) {
      SafeDelete(fView);
      if (fPrimitives) fPrimitives->Clear(option);
      delete fFrame; fFrame = 0;
   }

   cd();

   if (TestBit(kClearAfterCR)) getchar();

   if (!gPad->IsBatch()) gVirtualX->ClearWindow();
   if (gVirtualPS && gPad == gPad->GetCanvas()) gVirtualPS->NewPage();

   PaintBorder(GetFillColor(), kTRUE);
   fCrosshairPos = 0;
   ResetBit(TGraph::kClipFrame);
}

//___________________________________________________________
Int_t TPad::Clip(Float_t *x, Float_t *y, Float_t xclipl, Float_t yclipb, Float_t xclipr, Float_t yclipt)
{
//   Clipping routine: Cohen Sutherland algorithm.
// If Clip ==2 the segment is outside the boundary.
// If Clip ==1 the segment has one point outside the boundary.
// If Clip ==0 the segment is inside the boundary.
//
// _Input parameters:
//
//  x[2], y[2] : Segment coordinates
//  xclipl, yclipb, xclipr, yclipt : Clipping boundary
//
// _Output parameters:
//
//  x[2], y[2] : New segment coordinates
//
   const Float_t P=10000;
   Int_t clip = 0;

   for (Int_t i=0;i<2;i++) {
      if (TMath::Abs(xclipl-x[i]) <= TMath::Abs(xclipr-xclipl)/P) x[i] = xclipl;
      if (TMath::Abs(xclipr-x[i]) <= TMath::Abs(xclipr-xclipl)/P) x[i] = xclipr;
      if (TMath::Abs(yclipb-y[i]) <= TMath::Abs(yclipt-yclipb)/P) y[i] = yclipb;
      if (TMath::Abs(yclipt-y[i]) <= TMath::Abs(yclipt-yclipb)/P) y[i] = yclipt;
   }

//Compute the first endpoint codes.
   Int_t code1 = ClippingCode(x[0],y[0],xclipl,yclipb,xclipr,yclipt);
   Int_t code2 = ClippingCode(x[1],y[1],xclipl,yclipb,xclipr,yclipt);

   Double_t xt=0, yt=0;
   Int_t clipped = 0; //this variable could be used in a future version
   while(code1 + code2) {
      clipped = 1;

//The line lies entirely outside the clipping boundary

      if (code1&code2) {
         clip = 2;
         return clip;
      }
//The line is subdivided into several parts

      Int_t ic = code1;
      if (ic == 0) ic = code2;
      if (ic & 0x1) {
         yt = y[0] + (y[1]-y[0])*(xclipl-x[0])/(x[1]-x[0]);
         xt = xclipl;
      }
      if (ic & 0x2) {
         yt = y[0] + (y[1]-y[0])*(xclipr-x[0])/(x[1]-x[0]);
         xt = xclipr;
      }
      if (ic & 0x4) {
         xt = x[0] + (x[1]-x[0])*(yclipb-y[0])/(y[1]-y[0]);
         yt = yclipb;
      }
      if (ic & 0x8) {
         xt = x[0] + (x[1]-x[0])*(yclipt-y[0])/(y[1]-y[0]);
         yt = yclipt;
      }
      if (ic == code1) {
         x[0]  = xt;
         y[0]  = yt;
         code1 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      } else {
         x[1]  = xt;
         y[1]  = yt;
         code2 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      }
   }
   clip = clipped;
   return clip;
}


//___________________________________________________________
Int_t TPad::Clip(Double_t *x, Double_t *y, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt)
{
//   Clipping routine: Cohen Sutherland algorithm.
// If Clip ==2 the segment is outside the boundary.
// If Clip ==1 the segment has one point outside the boundary.
// If Clip ==0 the segment is inside the boundary.
//
// _Input parameters:
//
//  x[2], y[2] : Segment coordinates
//  xclipl, yclipb, xclipr, yclipt : Clipping boundary
//
// _Output parameters:
//
//  x[2], y[2] : New segment coordinates
//
   const Double_t P=10000;
   Int_t clip = 0;

   for (Int_t i=0;i<2;i++) {
      if (TMath::Abs(xclipl-x[i]) <= TMath::Abs(xclipr-xclipl)/P) x[i] = xclipl;
      if (TMath::Abs(xclipr-x[i]) <= TMath::Abs(xclipr-xclipl)/P) x[i] = xclipr;
      if (TMath::Abs(yclipb-y[i]) <= TMath::Abs(yclipt-yclipb)/P) y[i] = yclipb;
      if (TMath::Abs(yclipt-y[i]) <= TMath::Abs(yclipt-yclipb)/P) y[i] = yclipt;
   }

//Compute the first endpoint codes.
   //Int_t code1 = ClippingCode(x[0],y[0],xclipl,yclipb,xclipr,yclipt);
   //Int_t code2 = ClippingCode(x[1],y[1],xclipl,yclipb,xclipr,yclipt);
   Int_t code1 = 0;
   if (x[0] < xclipl) code1 = code1 | 0x1;
   if (x[0] > xclipr) code1 = code1 | 0x2;
   if (y[0] < yclipb) code1 = code1 | 0x4;
   if (y[0] > yclipt) code1 = code1 | 0x8;
   Int_t code2 = 0;
   if (x[1] < xclipl) code2 = code2 | 0x1;
   if (x[1] > xclipr) code2 = code2 | 0x2;
   if (y[1] < yclipb) code2 = code2 | 0x4;
   if (y[1] > yclipt) code2 = code2 | 0x8;

   Double_t xt=0, yt=0;
   Int_t clipped = 0; //this variable could be used in a future version
   while(code1 + code2) {
      clipped = 1;

//The line lies entirely outside the clipping boundary

      if (code1&code2) {
         clip = 2;
         return clip;
      }
//The line is subdivided into several parts

      Int_t ic = code1;
      if (ic == 0) ic = code2;
      if (ic & 0x1) {
         yt = y[0] + (y[1]-y[0])*(xclipl-x[0])/(x[1]-x[0]);
         xt = xclipl;
      }
      if (ic & 0x2) {
         yt = y[0] + (y[1]-y[0])*(xclipr-x[0])/(x[1]-x[0]);
         xt = xclipr;
      }
      if (ic & 0x4) {
         xt = x[0] + (x[1]-x[0])*(yclipb-y[0])/(y[1]-y[0]);
         yt = yclipb;
      }
      if (ic & 0x8) {
         xt = x[0] + (x[1]-x[0])*(yclipt-y[0])/(y[1]-y[0]);
         yt = yclipt;
      }
      if (ic == code1) {
         x[0]  = xt;
         y[0]  = yt;
         code1 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      } else {
         x[1]  = xt;
         y[1]  = yt;
         code2 = ClippingCode(xt,yt,xclipl,yclipb,xclipr,yclipt);
      }
   }
   clip = clipped;
   return clip;
}


//___________________________________________________________
Int_t TPad::ClippingCode(Double_t x, Double_t y, Double_t xcl1, Double_t ycl1, Double_t xcl2, Double_t ycl2)
{
//   Compute the endpoint codes for TPad::Clip.

   Int_t code = 0;
   if (x < xcl1) code = code | 0x1;
   if (x > xcl2) code = code | 0x2;
   if (y < ycl1) code = code | 0x4;
   if (y > ycl2) code = code | 0x8;
   return code;
}


//______________________________________________________________________________
void TPad::Close(Option_t *)
{
//*-*-*-*-*-*-*-*-*Delete all primitives in pad and pad itself*-*-*-*-*-*-*
//*-*              ============================
//   NB. Pad cannot be used anymore after this call
//
   if (!TestBit(kNotDeleted)) return;
   if (!fMother) return;

   if (fPrimitives)
      fPrimitives->Clear();
   if (fView) {
      if (fView->TestBit(kNotDeleted)) delete fView;
      fView = 0;
   }
   if (fFrame) {
      if (fFrame->TestBit(kNotDeleted)) delete fFrame;
      fFrame = 0;
   }

   if (fPixmapID != -1) {
      if (gPad) {
         if (!gPad->IsBatch()) {
            gVirtualX->SelectWindow(fPixmapID);
            gVirtualX->ClosePixmap();
         }
      }
      fPixmapID = -1;

      if (!gROOT->GetListOfCanvases()) return;
      if (fMother == this) {
         gROOT->GetListOfCanvases()->Remove(this);
         return;   // in case of TCanvas
      }

      // remove from the mother's list of primitives
      if (fMother->GetListOfPrimitives()) fMother->GetListOfPrimitives()->Remove(this);

      if (gPad == this) fMother->cd();
      if (gROOT->GetSelectedPad() == this) gROOT->SetSelectedPad(0);
   }
   fMother = 0;
   if (gROOT->GetSelectedPad()== this) gROOT->SetSelectedPad(0);
}

//______________________________________________________________________________
void TPad::CopyPixmap()
{
//*-*-*-*-*-*-*-*-*Copy the pixmap of the pad to the canvas*-*-*-*-*-*-*
//*-*              ========================================

   int px, py;
   XYtoAbsPixel(fX1, fY2, px, py);
   gVirtualX->CopyPixmap(fPixmapID, px, py);
   if (this == gPad) HighLight(gPad->GetHighLightColor());
}

//______________________________________________________________________________
void TPad::CopyPixmaps()
{
//*-*-*-*-*-*-*-*-*Copy the sub-pixmaps of the pad to the canvas*-*-*-*-*-*-*
//*-*              =============================================

   TObject *obj;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         ((TPad*)obj)->CopyPixmap();
         ((TPad*)obj)->CopyPixmaps();
      }
   }
}

//______________________________________________________________________________
void TPad::CreateNewEllipse(Int_t event, Int_t px, Int_t py, Int_t mode)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new arc/ellipse in this pad*-*-*-*-*-*-*-*-*-*
//*-*                  ====================================
//
//  Function called by TPad::ExecuteEvent when the action "Create NewEllipse"
//  has been selected in the Canvas pulldown menu.
//
//  Click left button to indicate arrow starting position.
//  Release left button to terminate the arrow.
//


   static Double_t x0, y0, x1, y1;

   static Int_t pxold, pyold;
   static Int_t px0, py0;
   static Int_t linedrawn;
   Double_t xc,yc,r1,r2;
   TEllipse *el = 0;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      x0 = gPad->AbsPixeltoX(px);
      y0 = gPad->AbsPixeltoY(py);
      px0   = px; py0   = py;
      pxold = px; pyold = py;
      linedrawn = 0;
      break;

   case kButton1Motion:
      if (linedrawn) gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      break;

   case kButton1Up:
      x1 = gPad->AbsPixeltoX(px);
      y1 = gPad->AbsPixeltoY(py);
      xc = 0.5*(x0+x1);
      yc = 0.5*(y0+y1);

      if (mode == kArc) {
         r1 = 0.5*TMath::Abs(x1-x0);
         el = new TArc(xc, yc, r1);
      }
      if (mode == kEllipse) {
         r1 = 0.5*TMath::Abs(x1-x0);
         r2 = 0.5*TMath::Abs(y1-y0);
         el = new TEllipse(xc, yc, r1, r2);
      }
      gPad->GetCanvas()->FeedbackMode(kFALSE);
      gPad->Modified(kTRUE);
      el->Draw();
      gROOT->SetEditorMode();
      break;
   }
}

//______________________________________________________________________________
void TPad::CreateNewLine(Int_t event, Int_t px, Int_t py, Int_t mode)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new arrow in this pad*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================
//
//  Function called by TPad::ExecuteEvent when the action "Create NewArrow"
//  has been selected in the Canvas pulldown menu.
//
//  Click left button to indicate arrow starting position.
//  Release left button to terminate the arrow.
//
//

   static Double_t x0, y0, x1, y1;

   static Int_t pxold, pyold;
   static Int_t px0, py0;
   static Int_t linedrawn;
   TLine *line;
   TArrow *arrow;
   TCurlyLine *cline;
   Double_t radius, phimin,phimax;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      x0 = gPad->AbsPixeltoX(px);
      y0 = gPad->AbsPixeltoY(py);
      px0   = px; py0   = py;
      pxold = px; pyold = py;
      linedrawn = 0;
      break;

   case kButton1Motion:
      if (linedrawn) gVirtualX->DrawLine(px0, py0, pxold, pyold);
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawLine(px0, py0, pxold, pyold);
      break;

   case kButton1Up:
      if (px == px0 && py == py0) break;
      x1 = gPad->AbsPixeltoX(px);
      y1 = gPad->AbsPixeltoY(py);
      gPad->Modified(kTRUE);

      if (mode == kLine) {
         line = new TLine(x0,y0,x1,y1);
         line->Draw();
      }
      if (mode == kArrow) {
         arrow = new TArrow(x0,y0,x1,y1,0.03,"|>");
         arrow->SetFillColor(1);
         arrow->Draw();
      }
      if (mode == kCurlyLine) {
         cline = new TCurlyLine(x0,y0,x1,y1);
         cline->Draw();
      }
      if (mode == kCurlyArc) {
         radius = TMath::Sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
         phimin = 0;
         phimax = 360;
         cline = new TCurlyArc(x0,y0,radius,phimin,phimax);
         cline->Draw();
      }
      gROOT->SetEditorMode();
      break;
   }
}

//______________________________________________________________________________
void TPad::CreateNewPad(Int_t event, Int_t px, Int_t py, Int_t)
{
//*-*-*-*-*-*-*-*-*-*-*Create a new pad in this pad*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================
//
//  Function called by TPad::ExecuteEvent when the action "Create NewPad"
//  has been selected in the Canvas pulldown menu.
//
//  Click left button to indicate one corner of the pad
//  Click left button to indicate the opposite corner
//
//  The new pad is inserted in the pad where the first point is selected.
//
   static Int_t px1old, py1old, px2old, py2old;
   static Int_t px1, py1, px2, py2, pxl, pyl, pxt, pyt;
   static Bool_t boxdrawn;
   static TPad *padsav;
   Double_t xlow, ylow, xup, yup;
   TPad * newpad;

   switch (event) {

   case kButton1Down:
      padsav = (TPad*)gPad;
      cd();
      gVirtualX->SetLineColor(-1);
      px1 = gPad->XtoAbsPixel(gPad->GetX1());
      py1 = gPad->YtoAbsPixel(gPad->GetY1());
      px2 = gPad->XtoAbsPixel(gPad->GetX2());
      py2 = gPad->YtoAbsPixel(gPad->GetY2());
      px1old = px; py1old = py;
      boxdrawn = 0;
      break;

   case kButton1Motion:
      if (boxdrawn) gVirtualX->DrawBox(pxl, pyl, pxt, pyt, TVirtualX::kHollow);
      px2old = px;
      px2old = TMath::Max(px2old, px1);
      px2old = TMath::Min(px2old, px2);
      py2old = py;
      py2old = TMath::Max(py2old, py2);
      py2old = TMath::Min(py2old, py1);
      pxl = TMath::Min(px1old, px2old);
      pxt = TMath::Max(px1old, px2old);
      pyl = TMath::Max(py1old, py2old);
      pyt = TMath::Min(py1old, py2old);
      boxdrawn = 1;
      gVirtualX->DrawBox(pxl, pyl, pxt, pyt, TVirtualX::kHollow);
      break;

   case kButton1Up:
      gPad->Modified(kTRUE);
      gPad->SetDoubleBuffer(1);   // Turn on double buffer mode
      gVirtualX->SetDrawMode(TVirtualX::kCopy);       // set drawing mode back to normal (copy) mode
      xlow = (Double_t(pxl) - Double_t(px1))/(Double_t(px2) - Double_t(px1));
      ylow = (Double_t(py1) - Double_t(pyl))/(Double_t(py1) - Double_t(py2));
      xup  = (Double_t(pxt) - Double_t(px1))/(Double_t(px2) - Double_t(px1));
      yup  = (Double_t(py1) - Double_t(pyt))/(Double_t(py1) - Double_t(py2));
      gROOT->SetEditorMode();
      boxdrawn = 0;
      if (xup <= xlow || yup <= ylow) return;
      newpad = new TPad("newpad","newpad",xlow, ylow, xup, yup);
      if (newpad->IsZombie()) break;
      newpad->SetFillColor(gStyle->GetPadColor());
      newpad->Draw();
      padsav->cd();
      break;
   }
}

//______________________________________________________________________________
void TPad::CreateNewPave(Int_t event, Int_t px, Int_t py, Int_t mode)
{
//*-*-*-*-*-*-*-*-*-*Create a new pavetext in this pad*-*-*-*-*-*-*-*-*-*-*
//*-*                =================================
//
//  Function called by TPad::ExecuteEvent when the action "Create NewPaveText"
//  has been selected in the Canvas pulldown menu.
//
//  Click left button to indicate one corner of the pavelabel.
//  Release left button at the opposite corner.
//


   static Double_t x0, y0, x1, y1;

   static Int_t pxold, pyold;
   static Int_t px0, py0;
   static Int_t linedrawn;
   const Int_t kTMAX=100;
   Int_t i,pxl,pyl;
   Double_t temp;
   Double_t xp0,xp1,yp0,yp1;
   static char atext[kTMAX];
   TObject *pave = 0;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      x0 = gPad->AbsPixeltoX(px);
      y0 = gPad->AbsPixeltoY(py);
      px0   = px; py0   = py;
      pxold = px; pyold = py;
      linedrawn = 0;
      break;

   case kButton1Motion:
      if (linedrawn) gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      break;

   case kButton1Up:
      if (px == px0) px = px0+10;
      if (py == py0) py = py0-10;
      x1 = gPad->AbsPixeltoX(px);
      y1 = gPad->AbsPixeltoY(py);

      if (x1 < x0) {temp = x0; x0 = x1; x1 = temp;}
      if (y1 < y0) {temp = y0; y0 = y1; y1 = temp;}
      xp0 = PadtoX(x0);
      xp1 = PadtoX(x1);
      yp0 = PadtoY(y0);
      yp1 = PadtoY(y1);
      if (mode == kPave)       pave = new TPave(xp0,yp0,xp1,yp1);
      if (mode == kPaveText )  pave = new TPaveText(xp0,yp0,xp1,yp1);
      if (mode == kPavesText)  pave = new TPavesText(xp0,yp0,xp1,yp1);
      if (mode == kDiamond)    pave = new TDiamond(x0,y0,x1,y1);
      if (mode == kPaveLabel || mode == kButton) {
         pxl = (px0 + px)/2;
         pyl = (py0 + py)/2;
         for (i=0;i<kTMAX;i++) atext[i] = ' ';
         atext[kTMAX-1] = 0;
         gVirtualX->RequestString(pxl, pyl, atext);
         for (i=kTMAX-2;i>=0;i--) {
            if (atext[i] != ' ') {
               atext[i+1] = 0;
               break;
            }
         }
         if (mode == kPaveLabel) pave = new TPaveLabel(xp0,yp0,xp1,yp1,atext);
         if (mode == kButton)    pave = new TButton(atext,"",
                              (x0-gPad->GetX1())/(gPad->GetX2() - gPad->GetX1()),
                              (y0-gPad->GetY1())/(gPad->GetY2() - gPad->GetY1()),
                              (x1-gPad->GetX1())/(gPad->GetX2() - gPad->GetX1()),
                              (y1-gPad->GetY1())/(gPad->GetY2() -
                              gPad->GetY1()));
      }
      gPad->GetCanvas()->FeedbackMode(kFALSE);
      gPad->Modified(kTRUE);
      pave->Draw();
      gROOT->SetEditorMode();
      break;
   }
}

//______________________________________________________________________________
void TPad::CreateNewPolyLine(Int_t event, Int_t px, Int_t py, Int_t mode)
{
//*-*-*-*-*-*-*-*-*-*Create a new PolyLine*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ======================
//  Click left button to indicate a new point
//  Click right button to close the polyline
//  mode = 0 normal polyline
//  mode = 0 smooth polyline
// This function is called when the item NewPolyLine is selected in the
// canvas pulldown menu.
//
   const Int_t kMAX=100;
   static Double_t xline[kMAX], yline[kMAX];

   static Int_t pxold, pyold, px1old, py1old;
   static Int_t npoints = 0;
   static Int_t linedrawn;
   Int_t dp;
   TGraph *gr;

   switch (event) {

   case kButton1Double:
   case kButton1Down:
      if (npoints == 0) {
         gVirtualX->SetLineColor(-1);
      } else {
         gPad->GetCanvas()->FeedbackMode(kFALSE);
         gVirtualX->DrawLine(px1old, py1old, pxold, pyold);
      }
      // stop collecting new points if new point is close ( < 5 pixels) of previous point
      if (event == kButton1Double) {
         px = px1old;
         py = py1old;
      }
      dp = TMath::Abs(px-px1old) +TMath::Abs(py-py1old);
      if (npoints && dp < 5) {
         gPad->Modified(kTRUE);
         if      (mode == kPolyLine) {
            gr = new TGraph(npoints,xline,yline);
            gr->ResetBit(TGraph::kClipFrame);
            gr->Draw("L");
         } else {
            xline[npoints] = xline[0];
            yline[npoints] = yline[0];
            npoints++;
            gr = (TGraph*)gROOT->ProcessLineFast(
                 Form("new %s(\"CUTG\",%d,(Double_t*)0x%lx,(Double_t*)0x%lx)",
                      gROOT->GetCutClassName(),npoints,(Long_t)xline,(Long_t)yline));
            if (gr) gr->Draw("L");
         }
         npoints = 0;
         linedrawn = 0;
         gROOT->SetEditorMode();
         break;
      }
      xline[npoints] = gPad->PadtoX(gPad->AbsPixeltoX(px));
      yline[npoints] = gPad->PadtoY(gPad->AbsPixeltoY(py));
      px1old = px; py1old = py;
      pxold  = px; pyold  = py;
      linedrawn = 0;
      npoints++;

      if (npoints >= kMAX) {
         Error("CreateNewPolyLine", "too many points (more than %d)", kMAX);
         npoints = kMAX;
      }
      break;

   case kMouseMotion:
   case kButton1Motion:
   case kButton1Up:
      if (npoints == 0) return;
      gPad->GetCanvas()->FeedbackMode(kTRUE);
      if (linedrawn) gVirtualX->DrawLine(px1old, py1old, pxold, pyold);
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawLine(px1old, py1old, pxold, pyold);
      break;
   }
}

//______________________________________________________________________________
void TPad::CreateNewText(Int_t event, Int_t px, Int_t py, Int_t mode)
{
//*-*-*-*-*-*-*-*-*-*Create a new TLatex at the cursor position*-*-*-*-*-*-*-*
//*-*                ==========================================
//  Click left button to indicate the text position
// This function is called when the item NewText is selected in the
// canvas pulldown menu.
//

   const Int_t kTMAX=100;
   static char atext[kTMAX];
   Int_t i, lentext;
   TLatex *newtext;
   TMarker *marker;
   Double_t x, y;

   switch (event) {

   case kButton1Down:
      x = gPad->AbsPixeltoX(px);
      y = gPad->AbsPixeltoY(py);
      if (mode == kMarker) {
         marker = new TMarker(x,y,gStyle->GetMarkerStyle());
         marker->Draw();
         gROOT->SetEditorMode();
         break;
      }
      for (i=0;i<kTMAX;i++) atext[i] = ' ';
      atext[kTMAX-1] = 0;
      lentext = kTMAX;
      newtext = new TLatex();
      gVirtualX->SetLineColor(-1);
      newtext->TAttText::Modify();
      gVirtualX->RequestString(px, py, atext);
      lentext = strlen(atext);
      for (i=lentext-1;i>=0;i--) {
         if (atext[i] != ' ') {
            atext[lentext] = 0;
            break;
         }
         lentext--;
      }
      if (!lentext) break;
      newtext->DrawLatex(x, y, atext);
      gPad->Modified(kTRUE);
      gROOT->SetEditorMode();
      break;
   }
}

//______________________________________________________________________________
void TPad::DeleteExec(const char *name)
{
// Remove TExec name from the list of Execs.

   if (!fExecs) fExecs = new TList;
   TExec *ex = (TExec*)fExecs->FindObject(name);
   if (!ex) return;
   fExecs->Remove(ex);
   delete ex;
}

//______________________________________________________________________________
Int_t TPad::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a box*-*-*-*-*-*
//*-*                  ==========================================
//  Compute the closest distance of approach from point px,py to the
//  edges of this pad.
//  The distance is computed in pixels units.
//

   Int_t pxl, pyl, pxt, pyt;
   Int_t px1 = gPad->XtoAbsPixel(fX1);
   Int_t py1 = gPad->YtoAbsPixel(fY1);
   Int_t px2 = gPad->XtoAbsPixel(fX2);
   Int_t py2 = gPad->YtoAbsPixel(fY2);
   if (px1 < px2) {pxl = px1; pxt = px2;}
   else           {pxl = px2; pxt = px1;}
   if (py1 < py2) {pyl = py1; pyt = py2;}
   else           {pyl = py2; pyt = py1;}

//*-*- Are we inside the box?
//*-*  ======================
   if ( (px > pxl && px < pxt) && (py > pyl && py < pyt) ) {
      if (GetFillStyle()) return 0;  //*-* if pad is filled
   }

//*-*- Are we on the edges?
//*-*  ====================
   Int_t dxl = TMath::Abs(px - pxl);
   if (py < pyl) dxl += pyl - py; if (py > pyt) dxl += py - pyt;
   Int_t dxt = TMath::Abs(px - pxt);
   if (py < pyl) dxt += pyl - py; if (py > pyt) dxt += py - pyt;
   Int_t dyl = TMath::Abs(py - pyl);
   if (px < pxl) dyl += pxl - px; if (px > pxt) dyl += px - pxt;
   Int_t dyt = TMath::Abs(py - pyt);
   if (px < pxl) dyt += pxl - px; if (px > pxt) dyt += px - pxt;

   Int_t distance = dxl;
   if (dxt < distance) distance = dxt;
   if (dyl < distance) distance = dyl;
   if (dyt < distance) distance = dyt;

   return distance - Int_t(0.5*fLineWidth);
}

//______________________________________________________________________________
void TPad::Divide(Int_t nx, Int_t ny, Float_t xmargin, Float_t ymargin, Int_t color)
{
//*-*-*-*-*-*-*-*-*Automatic pad generation by division*-*-*-*-*-*-*-*-*-*-*
//*-*              ====================================
//  The current canvas is divided in nx by ny equal divisions(pads).
//  xmargin is the space along x between pads in percent of canvas.
//  ymargin is the space along y between pads in percent of canvas.
//  color is the color of the new pads. If 0, color is the canvas color.
//  Pads are automatically named canvasname_n where n is the division number
//  starting from top left pad.
//       Example if canvasname=c1 , nx=2, ny=3
//
//    ...............................................................
//    .                               .                             .
//    .                               .                             .
//    .                               .                             .
//    .           c1_1                .           c1_2              .
//    .                               .                             .
//    .                               .                             .
//    .                               .                             .
//    ...............................................................
//    .                               .                             .
//    .                               .                             .
//    .                               .                             .
//    .           c1_3                .           c1_4              .
//    .                               .                             .
//    .                               .                             .
//    .                               .                             .
//    ...............................................................
//    .                               .                             .
//    .                               .                             .
//    .                               .                             .
//    .           c1_5                .           c1_6              .
//    .                               .                             .
//    .                               .                             .
//    ...............................................................
//
//
//    Once a pad is divided into subpads, one can set the current pad
//    to a subpad with a given division number as illustrated above
//    with TPad::cd(subpad_number).
//    For example, to set the current pad to c1_4, one can do:
//    c1->cd(4)
//
//  Note1:  c1.cd() is equivalent to c1.cd(0) and sets the current pad
//          to c1 itself.
//  Note2:  after a statement like c1.cd(6), the global variable gPad
//          points to the current pad. One can use gPad to set attributes
//          of the current pad.
//

   if (!IsEditable()) return;


   if (gThreadXAR) {
      void *arr[7];
      arr[1] = this; arr[2] = (void*)&nx;arr[3] = (void*)& ny;
      arr[4] = (void*)&xmargin; arr[5] = (void *)& ymargin; arr[6] = (void *)&color;
      if ((*gThreadXAR)("PDCD", 7, arr, NULL)) return;
   }

   TPad *padsav = (TPad*)gPad;
   cd();
   if (nx <= 0) nx = 1;
   if (ny <= 0) ny = 1;
   Int_t ix,iy;
   Double_t x1,y1,x2,y2;
   Double_t dy = 1/Double_t(ny);
   Double_t dx = 1/Double_t(nx);
   TPad *pad;
   char *name = new char [strlen(GetName())+6];
   Int_t n = 0;
   if (color == 0) color = GetFillColor();
   for (iy=0;iy<ny;iy++) {
      y2 = 1 - iy*dy - ymargin;
      y1 = y2 - dy + 2*ymargin;
      if (y1 < 0) y1 = 0;
      if (y1 > y2) continue;
      for (ix=0;ix<nx;ix++) {
         x1 = ix*dx + xmargin;
         x2 = x1 +dx -2*xmargin;
         if (x1 > x2) continue;
         n++;
         sprintf(name,"%s_%d",GetName(),n);
         pad = new TPad(name,name,x1,y1,x2,y2,color);
         pad->SetNumber(n);
         pad->Draw();
      }
   }
   delete [] name;
   Modified();
   if (padsav) padsav->cd();
}

//______________________________________________________________________________
void TPad::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Draw Pad in Current pad (re-parent pad if necessary)*-*-*-*
//*-*              ====================================================

   // if no canvas opened yet create a default canvas
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }

   // pad cannot be in itself and it can only be in one other pad at a time
   if (gPad != this) {
      if (fMother) fMother->GetListOfPrimitives()->Remove(this);
      TPad *oldMother = fMother;
      fCanvas = gPad->GetCanvas();
      fMother = (TPad*)gPad;
      if (oldMother != fMother || fPixmapID == -1) ResizePad();
   }

   Paint();

   if (gPad->IsRetained() && gPad != this && fMother)
      fMother->GetListOfPrimitives()->Add(this, option);
}

//______________________________________________________________________________
void TPad::DrawClassObject(const TObject *classobj, Option_t *option)
{
   // Draw class inheritance tree of the class to which obj belongs.
   // If a class B inherits from a class A, description of B is drawn
   // on the right side of description of A.
   // Member functions overridden by B are shown in class A with a blue line
   // crossing-out the corresponding member function.
   // The following picture is the class inheritance tree of class TPaveLabel:
   //Begin_Html
   /*
   <img src="gif/drawclass.gif">
   */
   //End_Html

   char dname[256];
   const Int_t MAXLEVELS = 10;
   TClass *clevel[MAXLEVELS], *cl, *cll;
   TBaseClass *base, *cinherit;
   TText *ptext = 0;
   TString opt=option;
   Double_t x,y,dy,y1,v1,v2,dv;
   Int_t nd,nf,nc,nkd,nkf,i,j;
   TPaveText *pt;
   Int_t maxlev = 4;
   if (opt.Contains("2")) maxlev = 2;
   if (opt.Contains("3")) maxlev = 3;
   if (opt.Contains("5")) maxlev = 5;
   if (opt.Contains("6")) maxlev = 6;
   if (opt.Contains("7")) maxlev = 7;

      // Clear and Set Pad range
   Double_t xpad = 20.5;
   Double_t ypad = 27.5;
   Clear();
   Range(0,0,xpad,ypad);

   // Find number of levels
   Int_t nlevel = 0;
   TClass *obj = (TClass*)classobj;
   clevel[nlevel] = obj;
   TList *lbase = obj->GetListOfBases();
   while(lbase) {
      base = (TBaseClass*)lbase->First();
      if (!base) break;
      if ( base->GetClassPointer() == 0) break;
      nlevel++;
      clevel[nlevel] = base->GetClassPointer();
      lbase = clevel[nlevel]->GetListOfBases();
      if (nlevel >= maxlev-1) break;
   }
   Int_t maxelem = 0;
   Int_t ncdraw  = 0;
   Int_t ilevel, nelem;
   for (ilevel=nlevel;ilevel>=0;ilevel--) {
      cl = clevel[ilevel];
      nelem = cl->GetNdata() + cl->GetNmethods();
      if (nelem > maxelem) maxelem = nelem;
      nc = (nelem/50) + 1;
      ncdraw += nc;
   }

   Double_t tsizcm = 0.40;
   Double_t x1 = 0.25;
   Double_t x2 = 0;
   Double_t dx = 3.5;
   if (ncdraw > 4) {
      dx = dx - 0.42*Double_t(ncdraw-5);
      tsizcm = tsizcm - 0.03*Double_t(ncdraw-5);
   }
   Double_t tsiz = 1.2*tsizcm/ypad;

   // Now loop on levels
   for (ilevel=nlevel;ilevel>=0;ilevel--) {
      cl    = clevel[ilevel];
      nelem = cl->GetNdata() + cl->GetNmethods();
      if (nelem > maxelem) maxelem = nelem;
      nc    = (nelem/50) + 1;
      dy    = 0.45;
      if (ilevel < nlevel) x1 = x2 + 0.5;
      x2    = x1 + nc*dx;
      v2    = ypad - 0.5;
      lbase = cl->GetListOfBases();
      cinherit = 0;
      if (lbase) cinherit = (TBaseClass*)lbase->First();

      do {
         nd = cl->GetNdata();
         nf = cl->GetNmethods() - 2; //do not show default constructor and destructor
         if (cl->GetListOfMethods()->FindObject("Dictionary")) {
            nf -= 6;  // do not count the Dictionary/ClassDef functions
         }
         nkf= nf/nc +1;
         nkd= nd/nc +1;
         if (nd == 0) nkd=0;
         if (nf == 0) nkf=0;
         y1 = v2 - 0.7;
         v1 = y1 - Double_t(nkf+nkd+nc-1)*dy;
         dv = v2 - v1;

         // Create a new PaveText
         pt = new TPaveText(x1,v1,x2,v2);
         pt->SetBit(kCanDelete);
         pt->SetFillColor(19);
         pt->Draw();
         pt->SetTextColor(4);
         pt->SetTextFont(61);
         pt->SetTextAlign(12);
         pt->SetTextSize(tsiz);
         TBox *box = pt->AddBox(0,(y1+0.01-v1)/dv,0,(v2-0.01-v1)/dv);
         box->SetFillColor(17);
         pt->AddLine(0,(y1-v1)/dv,0,(y1-v1)/dv);
         TText *title = pt->AddText(0.5,(0.5*(y1+v2)-v1)/dv,(char*)cl->GetName());
         title->SetTextAlign(22);
         title->SetTextSize(0.6*(v2-y1)/ypad);

         // Draw data Members
         i = 0;
         x = 0.03;
         y = y1 + 0.5*dy;
         TDataMember *d;
         TIter        nextd(cl->GetListOfDataMembers());
         while ((d = (TDataMember *) nextd())) {
            if (i >= nkd) { i = 1; y = y1 - 0.5*dy; x += 1/Double_t(nc); }
            else { i++; y -= dy; }

            // Take in account the room the array index will occupy

            Int_t dim = d->GetArrayDim();
            Int_t indx = 0;
            sprintf(dname,"%s",obj->EscapeChars((char*)d->GetName()));
            Int_t ldname = 0;
            while (indx < dim ){
               ldname = strlen(dname);
               sprintf(&dname[ldname],"[%d]",d->GetMaxIndex(indx));
               indx++;
            }
            ptext = pt->AddText(x,(y-v1)/dv,dname);
         }

         // Draw a separator line
         Double_t ysep;
         if (nd) {
            ysep = y1 - Double_t(nkd)*dy;
            pt->AddLine(0,(ysep-v1)/dv,0,(ysep-v1)/dv);
            ysep -= 0.5*dy;
         } else  ysep = y1;

         // Draw Member Functions
         Int_t fcount = 0;
         i = 0;
         x = 0.03;
         y = ysep + 0.5*dy;
         TMethod *m;
         TIter        nextm(cl->GetListOfMethods());
         while ((m = (TMethod *) nextm())) {
            if (
               !strcmp( m->GetName(), "Dictionary"    ) ||
               !strcmp( m->GetName(), "Class_Version" ) ||
               !strcmp( m->GetName(), "DeclFileName"  ) ||
               !strcmp( m->GetName(), "DeclFileLine"  ) ||
               !strcmp( m->GetName(), "ImplFileName"  ) ||
               !strcmp( m->GetName(), "ImplFileLine"  )
            ) continue;
            fcount++;
            if (fcount > nf) break;
            if (i >= nkf) { i = 1; y = ysep - 0.5*dy; x += 1/Double_t(nc); }
            else { i++; y -= dy; }
            ptext = pt->AddText(x,(y-v1)/dv,obj->EscapeChars((char*)m->GetName()));

            // Check if method is overloaded in a derived class
            // If yes, Change the color of the text to blue
            for (j=ilevel-1;j>=0;j--) {
               if (cl == clevel[ilevel]) {
                  if (clevel[j]->GetMethodAny((char*)m->GetName())) {
                     ptext->SetTextColor(15);
                     break;
                  }
               }
            }
         }

         // Draw second inheritance classes for this class
         cll = 0;
         if (cinherit) {
            cinherit = (TBaseClass*)lbase->After(cinherit);
            if (cinherit) {
               cl  = cinherit->GetClassPointer();
               cll = cl;
               v2  = v1 -0.4;
               dy  = 0.35;
            }
         }
      } while (cll);
   }
   Update();
}

//______________________________________________________________________________
void TPad::DrawCrosshair()
{
   //Function called to draw a crosshair in the canvas
   // Example:
   // Root > TFile f("hsimple.root");
   // Root > hpxpy.Draw();
   // Root > c1.SetCrosshair();
   // When moving the mouse in the canvas, a crosshair is drawn

   if (gPad->GetEvent() == kMouseEnter) return;

   TPad *cpad = (TPad*)gPad;
   cpad->GetCanvas()->FeedbackMode(kTRUE);

   //erase old position and draw a line at current position
   int pxold = fCrosshairPos%10000;
   int pyold = fCrosshairPos/10000;
   int px    = cpad->GetEventX();
   int py    = cpad->GetEventY()+1;
   int pxmin = cpad->XtoAbsPixel(fX1);
   int pxmax = cpad->XtoAbsPixel(fX2);
   int pymin = cpad->YtoAbsPixel(fY1);
   int pymax = cpad->YtoAbsPixel(fY2);
//printf("event=%d, pxold=%d, pyold=%d, px=%d, py=%d, pxmin=%d, pymin=%d, pxmax=%d, pymax=%d\n",gPad->GetEvent(),pxold,pyold,px,py,pxmin,pymin,pxmax,pymax);
   if(pxold) gVirtualX->DrawLine(pxold,pymin,pxold,pymax);
   if(pyold) gVirtualX->DrawLine(pxmin,pyold,pxmax,pyold);
   if (cpad->GetEvent() == kButton1Down ||
       cpad->GetEvent() == kButton1Up   ||
       cpad->GetEvent() == kMouseLeave) {
      fCrosshairPos = 0;
      return;
   }
   gVirtualX->DrawLine(px,pymin,px,pymax);
   gVirtualX->DrawLine(pxmin,py,pxmax,py);
   fCrosshairPos = px + 10000*py;
}

//______________________________________________________________________________
TH1F *TPad::DrawFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, const char *title)
{
//  Draw a pad frame
//  Compute real pad range taking into account all margins
//  Use services of TH1F class

   if (!IsEditable()) return 0;

   TH1F *hframe = (TH1F*)FindObject("hframe");
   if (hframe) delete hframe;
   hframe = new TH1F("hframe",title,1000,xmin,xmax);
   hframe->SetBit(TH1::kNoStats);
   hframe->SetBit(kCanDelete);
   hframe->SetMinimum(ymin);
   hframe->SetMaximum(ymax);
   hframe->GetYaxis()->SetLimits(ymin,ymax);
   hframe->SetDirectory(0);
   hframe->Draw("a");
   Update();
   return hframe;
}

//______________________________________________________________________________
void TPad::DrawColorTable()
{
//*-*-*-*-static function to Display Color Table in a pad*-*-*-*-*-*-*
//*-*     ===============================================

   Int_t i, j;
   Int_t color;
   Double_t xlow, ylow, xup, yup, hs, ws;
   Double_t x1, y1, x2, y2;
   x1 = y1 = 0;
   x2 = y2 = 20;

   gPad->SetFillColor(0);
   gPad->Clear();
   gPad->Range(x1,y1,x2,y2);

   TText *text = new TText(0,0,"");
   text->SetTextFont(61);
   text->SetTextSize(0.05);
   text->SetTextAlign(22);

   TBox *box;
   char label[8];
//*-* draw colortable boxes
   hs = (y2-y1)/Double_t(5);
   ws = (x2-x1)/Double_t(10);
   for (i=0;i<10;i++) {
      xlow = x1 + ws*(Double_t(i)+0.1);
      xup  = x1 + ws*(Double_t(i)+0.9);
      for (j=0;j<5;j++) {
         ylow = y1 + hs*(Double_t(j)+0.1);
         yup  = y1 + hs*(Double_t(j)+0.9);
         color = 10*j + i + 1;
         sprintf(label,"%d",color);
         box = new TBox(xlow, ylow, xup, yup);
         box->SetFillColor(color);
         box->Draw();
         text->DrawText(0.5*(xlow+xup), 0.5*(ylow+yup),label);
      }
   }

}

//______________________________________________________________________________
void TPad::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
//*-*-*-*-*-*-*-*-*Draw line in CurrentPad World coordinates*-*-*-*-*-*-*-*
//*-*              =========================================

   Int_t px1 = gPad->XtoPixel(x1);
   Int_t px2 = gPad->XtoPixel(x2);
   Int_t py1 = gPad->YtoPixel(y1);
   Int_t py2 = gPad->YtoPixel(y2);

   gVirtualX->DrawLine(px1, py1, px2, py2);
}

//______________________________________________________________________________
void TPad::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
//*-*-*-*-*-*-*-*-*Draw line in CurrentPad NDC coordinates*-*-*-*-*-*-*-*
//*-*              =======================================

   Int_t px1 = gPad->UtoPixel(u1);
   Int_t px2 = gPad->UtoPixel(u2);
   Int_t py1 = gPad->VtoPixel(v1);
   Int_t py2 = gPad->VtoPixel(v2);

   gVirtualX->DrawLine(px1, py1, px2, py2);
}

//______________________________________________________________________________
void TPad::DrawText(Double_t x, Double_t y, const char *text)
{
//*-*-*-*-*-*-*-*-*Draw text in CurrentPad World coordinates*-*-*-*-*-*-*-*
//*-*              =========================================

   Int_t px = gPad->XtoPixel(x);
   Int_t py = gPad->YtoPixel(y);

   Float_t angle = gVirtualX->GetTextAngle();
   Float_t mgn   = gVirtualX->GetTextSize();
   gVirtualX->DrawText(px, py, angle, mgn, text, TVirtualX::kClear);
}

//______________________________________________________________________________
void TPad::DrawTextNDC(Double_t u, Double_t v, const char *text)
{
//*-*-*-*-*-*-*-*-*Draw text in CurrentPad NDC coordinates*-*-*-*-*-*-*-*
//*-*              =======================================

   Int_t px = gPad->UtoPixel(u);
   Int_t py = gPad->VtoPixel(v);

   Float_t angle = gVirtualX->GetTextAngle();
   Float_t mgn   = gVirtualX->GetTextSize();
   gVirtualX->DrawText(px, py, angle, mgn, text, TVirtualX::kClear);
}

//______________________________________________________________________________
void TPad::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a TPad object is clicked.
//
//  If the mouse is clicked in one of the 4 corners of the pad (PA,PB,PC,PD)
//  the pad is resized with the rubber rectangle.
//
//  If the mouse is clicked inside the pad, the pad is moved.
//
//  If the mouse is clicked on the 4 edges (L,R,T,B), the pad is scaled
//  parallel to this edge.
//
//    PA                    T                       PB
//     +--------------------------------------------+
//     |                                            |
//     |                                            |
//     |                                            |
//    L|                  INSIDE                    |R
//     |                                            |
//     |                                            |
//     |                                            |
//     |                                            |
//     +--------------------------------------------+
//    PD                    B                      PC
//
//
//  Note that this function duplicates on purpose the functionality
//  already implemented in TBox::ExecuteEvent.
//  If somebody modifies this function, may be similar changes should also
//  be applied to TBox::ExecuteEvent.

   static Double_t xmin;
   static Double_t xmax;
   static Double_t ymin;
   static Double_t ymax;

   const Int_t kMaxDiff = 5;
   const Int_t kMinSize = 20;
   static Int_t pxorg, pyorg;
   static Int_t px1, px2, py1, py2, pxl, pyl, pxt, pyt, pxold, pyold;
   static Int_t px1p, px2p, py1p, py2p, pxlp, pylp, pxtp, pytp;
   static Bool_t PA, PB, PC, PD, T, L, R, B, INSIDE;
   Int_t  wx, wy;
   TVirtualPad  *parent;
   Bool_t opaque  = OpaqueMoving();
   Bool_t ropaque = OpaqueResizing();
   Bool_t fixedr  = HasFixedAspectRatio();

   if (!IsEditable() && event != kMouseEnter) return;

   HideToolTip(event);

   parent = GetMother();
   if (fXlowNDC < 0 && event != kButton1Down) return;
   if (fYlowNDC < 0 && event != kButton1Down) return;

   // keep old range and mouse position
   if (event == kButton1Down) {
      xmin = fX1;
      xmax = fX2;
      ymin = fY1;
      ymax = fY2;

      pxorg = px;
      pyorg = py;
   }

   Int_t newcode = gROOT->GetEditorMode();
   switch (newcode) {
      case kPad:
         CreateNewPad(event,px,py,0);
         break;
      case kMarker:
      case kText:
#ifndef WIN32
         CreateNewText(event,px,py,newcode);
#else
#ifndef GDK_WIN32
        {
            if (event == kButton1Down) gROOT->SetEditorMode();
            gROOT->ProcessLine(Form("((TPad *)0x%lx)->CreateNewText(%d,%d,%d,%d);",
                                    (Long_t)this, event, px, py, newcode));
        }
#else
         CreateNewText(event,px,py,newcode);
#endif
#endif
         break;
      case kLine:
         CreateNewLine(event,px,py,kLine);
         break;
      case kArrow:
         CreateNewLine(event,px,py,kArrow);
         break;
      case kCurlyLine:
         CreateNewLine(event,px,py,kCurlyLine);
         break;
      case kCurlyArc:
         CreateNewLine(event,px,py,kCurlyArc);
         break;
      case kPolyLine:
         CreateNewPolyLine(event,px,py,kPolyLine);
         break;
      case kCutG:
         CreateNewPolyLine(event,px,py,kCutG);
         break;
      case kArc:
         CreateNewEllipse(event,px,py,kArc);
         break;
      case kEllipse:
         CreateNewEllipse(event,px,py,kEllipse);
         break;
      case kButton:
      case kPave:
      case kPaveLabel:
      case kPaveText:
      case kPavesText:
      case kDiamond:
#ifdef WIN32
#ifndef GDK_WIN32
         if (newcode == kPaveLabel || newcode == kButton) {
            if (event == kButton1Up) gROOT->SetEditorMode();
            gROOT->ProcessLine(Form("((TPad *)0x%lx)->CreateNewPave(%d,%d,%d,%d);",
                                    (Long_t)this, event, px, py,newcode));
         }
         else
#endif
#endif
           CreateNewPave(event,px,py,newcode);
         return;
      default:
         break;
      }
      if (newcode) return;

   Bool_t doing_again = kFALSE;
again:

   switch (event) {

   case kMouseEnter:
      if (fTip)
         ResetToolTip(fTip);
      break;

   case kButton1Down:

#ifdef WIN32
      Pop(); //this should be for cases where mouse has only two buttons
#endif
      gVirtualX->SetLineColor(-1);
      TAttLine::Modify();  //Change line attributes only if necessary
      if (GetFillColor())
         gVirtualX->SetLineColor(GetFillColor());
      else
         gVirtualX->SetLineColor(1);
      gVirtualX->SetLineWidth(2);

      // No break !!!

   case kMouseMotion:

      px1 = XtoAbsPixel(fX1);
      py1 = YtoAbsPixel(fY1);
      px2 = XtoAbsPixel(fX2);
      py2 = YtoAbsPixel(fY2);

      if (px1 < px2) {
         pxl = px1;
         pxt = px2;
      } else {
         pxl = px2;
         pxt = px1;
      }
      if (py1 < py2) {
         pyl = py1;
         pyt = py2;
      } else {
         pyl = py2;
         pyt = py1;
      }

      px1p = parent->XtoAbsPixel(parent->GetX1()) + parent->GetBorderSize();
      py1p = parent->YtoAbsPixel(parent->GetY1()) - parent->GetBorderSize();
      px2p = parent->XtoAbsPixel(parent->GetX2()) - parent->GetBorderSize();
      py2p = parent->YtoAbsPixel(parent->GetY2()) + parent->GetBorderSize();

      if (px1p < px2p) {
         pxlp = px1p;
         pxtp = px2p;
      } else {
         pxlp = px2p;
         pxtp = px1p;
      }
      if (py1p < py2p) {
         pylp = py1p;
         pytp = py2p;
      } else {
         pylp = py2p;
         pytp = py1p;
      }

      PA = PB = PC = PD = T = L = R = B = INSIDE = kFALSE;

                                                         // case PA
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxl; pyold = pyl; PA = kTRUE;
         SetCursor(kTopLeft);
      }
                                                         // case PB
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxt; pyold = pyl; PB = kTRUE;
         SetCursor(kTopRight);
      }
                                                         // case PC
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxt; pyold = pyt; PC = kTRUE;
         SetCursor(kBottomRight);
      }
                                                         // case PD
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxl; pyold = pyt; PD = kTRUE;
         SetCursor(kBottomLeft);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyl) < kMaxDiff) {             // top edge
         pxold = pxl; pyold = pyl; T = kTRUE;
         SetCursor(kTopSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyt) < kMaxDiff) {             // bottom edge
         pxold = pxt; pyold = pyt; B = kTRUE;
         SetCursor(kBottomSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxl) < kMaxDiff) {             // left edge
         pxold = pxl; pyold = pyl; L = kTRUE;
         SetCursor(kLeftSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxt) < kMaxDiff) {             // right edge
          pxold = pxt; pyold = pyt; R = kTRUE;
          SetCursor(kRightSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          (py > pyl+kMaxDiff && py < pyt-kMaxDiff)) {    // inside box
         pxold = px; pyold = py; INSIDE = kTRUE;
         if (event == kButton1Down)
            SetCursor(kMove);
         else
            SetCursor(kCross);
      }

      fResizing = kFALSE;
      if (PA || PB || PC || PD || T || L || R || B)
         fResizing = kTRUE;

      if (!PA && !PB && !PC && !PD && !T && !L && !R && !B && !INSIDE)
         SetCursor(kCross);

      break;

   case kButton1Motion:

      wx = wy = 0;

      if (PA) {
         if (!ropaque) gVirtualX->DrawBox(pxold, pyt, pxt, pyold, TVirtualX::kHollow);  // draw the old box
         if (px > pxt-kMinSize) { px = pxt-kMinSize; wx = px; }
         if (py > pyt-kMinSize) { py = pyt-kMinSize; wy = py; }
         if (px < pxlp) { px = pxlp; wx = px; }
         if (py < pylp) { py = pylp; wy = py; }
         if (fixedr) {
            Double_t dy = Double_t(TMath::Abs(pxt-px))/parent->UtoPixel(1.) /
                          fAspectRatio;
            Int_t npy2 = pyt - TMath::Abs(parent->VtoAbsPixel(dy) -
                                          parent->VtoAbsPixel(0));
            if (npy2 < pylp) {
               px = pxold;
               py = pyold;
            } else
               py = npy2;

            wx = wy = 0;
         }
         if (!ropaque) gVirtualX->DrawBox(px   , pyt, pxt, py,    TVirtualX::kHollow);  // draw the new box
      }
      if (PB) {
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyt, pxold, pyold, TVirtualX::kHollow);
         if (px < pxl+kMinSize) { px = pxl+kMinSize; wx = px; }
         if (py > pyt-kMinSize) { py = pyt-kMinSize; wy = py; }
         if (px > pxtp) { px = pxtp; wx = px; }
         if (py < pylp) { py = pylp; wy = py; }
         if (fixedr) {
            Double_t dy = Double_t(TMath::Abs(pxl-px))/parent->UtoPixel(1.) /
                          fAspectRatio;
            Int_t npy2 = pyt - TMath::Abs(parent->VtoAbsPixel(dy) -
                                          parent->VtoAbsPixel(0));
            if (npy2 < pylp) {
               px = pxold;
               py = pyold;
            } else
               py = npy2;

            wx = wy = 0;
         }
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyt, px ,  py,    TVirtualX::kHollow);
      }
      if (PC) {
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyl, pxold, pyold, TVirtualX::kHollow);
         if (px < pxl+kMinSize) { px = pxl+kMinSize; wx = px; }
         if (py < pyl+kMinSize) { py = pyl+kMinSize; wy = py; }
         if (px > pxtp) { px = pxtp; wx = px; }
         if (py > pytp) { py = pytp; wy = py; }
         if (fixedr) {
            Double_t dy = Double_t(TMath::Abs(pxl-px))/parent->UtoPixel(1.) /
                          fAspectRatio;
            Int_t npy2 = pyl + TMath::Abs(parent->VtoAbsPixel(dy) -
                                          parent->VtoAbsPixel(0));
            if (npy2 > pytp) {
               px = pxold;
               py = pyold;
            } else
               py = npy2;

            wx = wy = 0;
         }
         if (!ropaque) gVirtualX->DrawBox(pxl  , pyl, px ,   py,    TVirtualX::kHollow);
      }
      if (PD) {
         if (!ropaque) gVirtualX->DrawBox(pxold, pyold, pxt, pyl, TVirtualX::kHollow);
         if (px > pxt-kMinSize) { px = pxt-kMinSize; wx = px; }
         if (py < pyl+kMinSize) { py = pyl+kMinSize; wy = py; }
         if (px < pxlp) { px = pxlp; wx = px; }
         if (py > pytp) { py = pytp; wy = py; }
         if (fixedr) {
            Double_t dy = Double_t(TMath::Abs(pxt-px))/parent->UtoPixel(1.) /
                          fAspectRatio;
            Int_t npy2 = pyl + TMath::Abs(parent->VtoAbsPixel(dy) -
                                          parent->VtoAbsPixel(0));
            if (npy2 > pytp) {
               px = pxold;
               py = pyold;
            } else
               py = npy2;

            wx = wy = 0;
         }
         if (!ropaque) gVirtualX->DrawBox(px   , py ,   pxt, pyl, TVirtualX::kHollow);
      }
      if (T) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         py2 += py - pyold;
         if (py2 > py1-kMinSize) { py2 = py1-kMinSize; wy = py2; }
         if (py2 < py2p) { py2 = py2p; wy = py2; }
         if (fixedr) {
            Double_t dx = Double_t(TMath::Abs(py2-py1))/parent->VtoPixel(0) *
                          fAspectRatio;
            Int_t npx2 = px1 + parent->UtoPixel(dx);
            if (npx2 > px2p)
               py2 -= py - pyold;
            else
               px2 = npx2;
         }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (B) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         py1 += py - pyold;
         if (py1 < py2+kMinSize) { py1 = py2+kMinSize; wy = py1; }
         if (py1 > py1p) { py1 = py1p; wy = py1; }
         if (fixedr) {
            Double_t dx = Double_t(TMath::Abs(py2-py1))/parent->VtoPixel(0) *
                          fAspectRatio;
            Int_t npx2 = px1 + parent->UtoPixel(dx);
            if (npx2 > px2p)
               py1 -= py - pyold;
            else
               px2 = npx2;
         }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (L) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px1 += px - pxold;
         if (px1 > px2-kMinSize) { px1 = px2-kMinSize; wx = px1; }
         if (px1 < px1p) { px1 = px1p; wx = px1; }
         if (fixedr) {
            Double_t dy = Double_t(TMath::Abs(px2-px1))/parent->UtoPixel(1.) /
                          fAspectRatio;
            Int_t npy2 = py1 - TMath::Abs(parent->VtoAbsPixel(dy) -
                                          parent->VtoAbsPixel(0));
            if (npy2 < py2p)
               px1 -= px - pxold;
            else
               py2 = npy2;
         }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (R) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px2 += px - pxold;
         if (px2 < px1+kMinSize) { px2 = px1+kMinSize; wx = px2; }
         if (px2 > px2p) { px2 = px2p; wx = px2; }
         if (fixedr) {
            Double_t dy = Double_t(TMath::Abs(px2-px1))/parent->UtoPixel(1.) /
                          fAspectRatio;
            Int_t npy2 = py1 - TMath::Abs(parent->VtoAbsPixel(dy) -
                                          parent->VtoAbsPixel(0));
            if (npy2 < py2p)
               px2 -= px - pxold;
            else
               py2 = npy2;
         }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (INSIDE) {
         if (!opaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);  // draw the old box
         Int_t dx = px - pxold;
         Int_t dy = py - pyold;
         px1 += dx; py1 += dy; px2 += dx; py2 += dy;
         if (px1 < px1p) { dx = px1p - px1; px1 += dx; px2 += dx; wx = px+dx; }
         if (px2 > px2p) { dx = px2 - px2p; px1 -= dx; px2 -= dx; wx = px-dx; }
         if (py1 > py1p) { dy = py1 - py1p; py1 -= dy; py2 -= dy; wy = py-dy; }
         if (py2 < py2p) { dy = py2p - py2; py1 += dy; py2 += dy; wy = py+dy; }
         if (!opaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);  // draw the new box
      }

      if (wx || wy) {
         if (wx) px = wx;
         if (wy) py = wy;
         gVirtualX->Warp(px, py);
      }

      pxold = px;
      pyold = py;

      if ((!fResizing && opaque) || (fResizing && ropaque)) {
         event = kButton1Up;
         doing_again = kTRUE;
         goto again;
      }

      break;

   case kButton1Up:

      if (PA) {
         fX1 = AbsPixeltoX(pxold);
         fY1 = AbsPixeltoY(pyt);
         fX2 = AbsPixeltoX(pxt);
         fY2 = AbsPixeltoY(pyold);
      }
      if (PB) {
         fX1 = AbsPixeltoX(pxl);
         fY1 = AbsPixeltoY(pyt);
         fX2 = AbsPixeltoX(pxold);
         fY2 = AbsPixeltoY(pyold);
      }
      if (PC) {
         fX1 = AbsPixeltoX(pxl);
         fY1 = AbsPixeltoY(pyold);
         fX2 = AbsPixeltoX(pxold);
         fY2 = AbsPixeltoY(pyl);
      }
      if (PD) {
         fX1 = AbsPixeltoX(pxold);
         fY1 = AbsPixeltoY(pyold);
         fX2 = AbsPixeltoX(pxt);
         fY2 = AbsPixeltoY(pyl);
      }
      if (T || B || L || R || INSIDE) {
         fX1 = AbsPixeltoX(px1);
         fY1 = AbsPixeltoY(py1);
         fX2 = AbsPixeltoX(px2);
         fY2 = AbsPixeltoY(py2);
      }

      if (INSIDE)
         if (!doing_again) gPad->SetCursor(kCross);

      if (PA || PB || PC || PD || T || L || R || B)
         Modified(kTRUE);

      gVirtualX->SetLineColor(-1);
      gVirtualX->SetLineWidth(-1);

      if (px != pxorg || py != pyorg) {

         // Get parent corners pixels coordinates
         TPad *parent = fMother;
         Int_t parentpx1 = parent->XtoAbsPixel(parent->GetX1());
         Int_t parentpx2 = parent->XtoAbsPixel(parent->GetX2());
         Int_t parentpy1 = parent->YtoAbsPixel(parent->GetY1());
         Int_t parentpy2 = parent->YtoAbsPixel(parent->GetY2());

         // Get pad new corners pixels coordinates
         Int_t px1 = XtoAbsPixel(fX1); if (px1 < parentpx1) {px1 = parentpx1; }
         Int_t px2 = XtoAbsPixel(fX2); if (px2 > parentpx2) {px2 = parentpx2; }
         Int_t py1 = YtoAbsPixel(fY1); if (py1 > parentpy1) {py1 = parentpy1; }
         Int_t py2 = YtoAbsPixel(fY2); if (py2 < parentpy2) {py2 = parentpy2; }

         // Compute new pad positions in the NDC space of parent
         fXlowNDC = Double_t(px1 - parentpx1)/Double_t(parentpx2 - parentpx1);
         fYlowNDC = Double_t(py1 - parentpy1)/Double_t(parentpy2 - parentpy1);
         fWNDC    = Double_t(px2 - px1)/Double_t(parentpx2 - parentpx1);
         fHNDC    = Double_t(py2 - py1)/Double_t(parentpy2 - parentpy1);
      }

      // Restore old range
      fX1 = xmin;
      fX2 = xmax;
      fY1 = ymin;
      fY2 = ymax;

      // Reset pad parameters and recompute conversion coefficients
      ResizePad();

      break;

   case kButton1Locate:

      ExecuteEvent(kButton1Down, px, py);

      while (1) {
         px = py = 0;
         event = gVirtualX->RequestLocator(1, 1, px, py);

         ExecuteEvent(kButton1Motion, px, py);

         if (event != -1) {                     // button is released
            ExecuteEvent(kButton1Up, px, py);
            return;
         }
      }

   case kButton2Down:

      Pop();
      break;

   }
}

//______________________________________________________________________________
TObject *TPad::FindObject(const char *name) const
{
   if (fPrimitives) return fPrimitives->FindObject(name);
   return 0;
}

//______________________________________________________________________________
TObject *TPad::FindObject(const TObject *) const
{
   //not implemented yet
   return 0;
}

//______________________________________________________________________________
Int_t TPad::GetCanvasID() const
{
   return fCanvas->GetCanvasID();
}

//______________________________________________________________________________
Int_t TPad::GetEvent() const
{
   return fCanvas->GetEvent();
}

//______________________________________________________________________________
Int_t TPad::GetEventX() const
{
   return fCanvas->GetEventX();
}

//______________________________________________________________________________
Int_t TPad::GetEventY() const
{
   return fCanvas->GetEventY();
}

//______________________________________________________________________________
TVirtualPad *TPad::GetVirtCanvas() const
{
   return (TVirtualPad*) fCanvas;
}

//______________________________________________________________________________
Color_t TPad::GetHighLightColor() const
{
   return fCanvas->GetHighLightColor();
}

//______________________________________________________________________________
Int_t TPad::GetMaxPickDistance() 
{
   //static function (see also TPad::SetMaxPickDistance)
   return fgMaxPickDistance;
}

//______________________________________________________________________________
TObject *TPad::GetSelected() const
{
   return fCanvas->GetSelected();
}

//______________________________________________________________________________
TVirtualPad *TPad::GetSelectedPad() const
{
   return fCanvas->GetSelectedPad();
}

//______________________________________________________________________________
TVirtualPad *TPad::GetPadSave() const
{
   return fCanvas->GetPadSave();
}

//______________________________________________________________________________
UInt_t TPad::GetWh() const
{
   return fCanvas->GetWh();
}

//______________________________________________________________________________
UInt_t TPad::GetWw() const
{
   return fCanvas->GetWw();
}

//______________________________________________________________________________
void TPad::HideToolTip(Int_t event)
{
   // Hide tool tip depending on the event type. Typically tool tips
   // are hidden when event is not a kMouseEnter and not a kMouseMotion
   // event.

   if (event != kMouseEnter && event != kMouseMotion && fTip)
      gPad->CloseToolTip(fTip);
}

//______________________________________________________________________________
Bool_t TPad::IsBatch() const
{
   return fCanvas->IsBatch();
}

//______________________________________________________________________________
Bool_t TPad::IsRetained() const
{
   return fCanvas->IsRetained();
}

//______________________________________________________________________________
Bool_t TPad::OpaqueMoving() const
{
   return fCanvas->OpaqueMoving();
}

//______________________________________________________________________________
Bool_t TPad::OpaqueResizing() const
{
   return fCanvas->OpaqueResizing();
}

//______________________________________________________________________________
void TPad::SetBatch(Bool_t batch)
{
   fCanvas->SetBatch(batch);
}

//______________________________________________________________________________
void TPad::SetCanvasSize(UInt_t ww, UInt_t wh)
{
   fCanvas->SetCanvasSize(ww,wh);
}

//______________________________________________________________________________
void TPad::SetCursor(ECursor cursor)
{
   fCanvas->SetCursor(cursor);
}

//______________________________________________________________________________
void TPad::SetDoubleBuffer(Int_t mode)
{
   fCanvas->SetDoubleBuffer(mode);
}

//______________________________________________________________________________
void TPad::SetSelected(TObject *obj)
{
   fCanvas->SetSelected(obj);
}

//______________________________________________________________________________
void TPad::Update()
{
   fCanvas->Update();
}

//______________________________________________________________________________
TFrame *TPad::GetFrame()
{
   TFrame     *frame = (TFrame*)GetListOfPrimitives()->FindObject(fFrame);
   if (!frame) frame = (TFrame*)GetListOfPrimitives()->FindObject("TFrame");
   fFrame = frame;
   if (!fFrame) {
      if (!frame) fFrame = new TFrame(0,0,1,1);
      Int_t framecolor = GetFrameFillColor();
      if (!framecolor) framecolor = GetFillColor();
      fFrame->SetFillColor(framecolor);
      fFrame->SetFillStyle(GetFrameFillStyle());
      fFrame->SetLineColor(GetFrameLineColor());
      fFrame->SetLineStyle(GetFrameLineStyle());
      fFrame->SetLineWidth(GetFrameLineWidth());
      fFrame->SetBorderSize(GetFrameBorderSize());
      fFrame->SetBorderMode(GetFrameBorderMode());
   }
   return fFrame;
}

//______________________________________________________________________________
TObject *TPad::GetPrimitive(const char *name) const
{
   if (fPrimitives) return fPrimitives->FindObject(name);
   return 0;
}

//______________________________________________________________________________
void TPad::GetPadPar(Double_t &xlow, Double_t &ylow, Double_t &xup, Double_t &yup)
{
//*-*-*-*-*-*-*-*Return lower and upper bounds of the pad in NDC coordinates
//*-*            ===========================================================
  xlow = fXlowNDC;
  ylow = fYlowNDC;
  xup  = fXlowNDC+fWNDC;
  yup  = fYlowNDC+fHNDC;
}

//______________________________________________________________________________
void TPad::GetRange(Double_t &x1, Double_t &y1, Double_t &x2, Double_t &y2)
{
//*-*-*-*-*-*-*-*Return pad world coordinates range*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ==================================
  x1 = fX1;
  y1 = fY1;
  x2 = fX2;
  y2 = fY2;
}

//______________________________________________________________________________
void TPad::GetRangeAxis(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax)
{
//*-*-*-*-*-*-*-*Return pad axis coordinates range*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ==================================
  xmin = fUxmin;
  ymin = fUymin;
  xmax = fUxmax;
  ymax = fUymax;
}

//______________________________________________________________________________
void TPad::HighLight(Color_t color, Bool_t set)
{
   AbsCoordinates(kTRUE);

   // We do not want to have active(executable) buttons, etc highlighted
   // in this manner, unless we want to edit'em
   if (GetMother()->IsEditable() && !InheritsFrom(TButton::Class())) {
      gROOT->SetSelectedPad(this);
      if (set)
         PaintBorder(-color, kFALSE);
      else
         PaintBorder(-GetFillColor(), kFALSE);
   }

   AbsCoordinates(kFALSE);
}

//______________________________________________________________________________
void TPad::ls(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*List all primitives in pad*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================
   TROOT::IndentLevel();
   cout <<IsA()->GetName()<<" fXlowNDC=" <<fXlowNDC<<" fYlowNDC="<<fYlowNDC<<" fWNDC="<<GetWNDC()<<" fHNDC="<<GetHNDC()
        <<" Name= "<<GetName()<<" Title= "<<GetTitle()<<" Option="<<option<<endl;
   TROOT::IncreaseDirLevel();
   fPrimitives->ls(option);
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
Double_t TPad::PadtoX(Double_t x) const
{
   if (fLogx && x < 50) return Double_t(TMath::Exp(2.302585092994*x));
   return x;
}

//______________________________________________________________________________
Double_t TPad::PadtoY(Double_t y) const
{
   if (fLogy && y < 50) return Double_t(TMath::Exp(2.302585092994*y));
   return y;
}

//______________________________________________________________________________
Double_t TPad::XtoPad(Double_t x) const
{
   if (fLogx) {
      if (x > 0) x = TMath::Log10(x);
      else       x = fUxmin;
   }
   return x;
}

//______________________________________________________________________________
Double_t TPad::YtoPad(Double_t y) const
{
   if (fLogy) {
      if (y > 0) y = TMath::Log10(y);
      else       y = fUymin;
   }
   return y;
}

//______________________________________________________________________________
void TPad::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Paint all primitives in pad*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ===========================

   TPad *padsav = (TPad*)gPad;

   fPadPaint = 1;
   cd();
//*-*   TView must be drawn here (is TView contains OpenGL context)

//*-*   Start OpenGL Model if present
   if (fPadView3D)
       fPadView3D->PaintBeginModel(option);
//   else
   PaintBorder(GetFillColor(), kTRUE);

   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   TObject *obj;

   while (lnk) {
      obj = lnk->GetObject();
      obj->Paint(lnk->GetOption());
      lnk = (TObjOptLink*)lnk->Next();
   }

   if (fPadView3D) {
//*-*   Finish 3D Model
       fPadView3D->PaintEnd(option);
//*-*   Start 3D scene
       fPadView3D->PaintScene(option);
   }

   padsav->cd();
   fPadPaint = 0;
   Modified(kFALSE);
}

//______________________________________________________________________________
void TPad::PaintBorder(Color_t color, Bool_t tops)
{
//*-*-*-*-*-*-*-*-*-*Paint the pad border*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ====================

//*-*- Draw first  a box as a normal filled box
   if(color >= 0) {
      TAttLine::Modify();  //Change line attributes only if necessary
      TAttFill::Modify();  //Change fill area attributes only if necessary
      PaintBox(fX1,fY1,fX2,fY2);
   }
   if (color < 0) color = -color;
//*-*- then paint 3d frame (depending on bordermode)
   if (IsTransparent()) return;
   // Paint a 3D frame around the pad.

   if (fBorderMode == 0) return;
   Int_t bordersize = fBorderSize;
   if (bordersize <= 0) bordersize = 2;

   Short_t pxl,pyl,pxt,pyt,px1,py1,px2,py2;
   Double_t xl, xt, yl, yt;

   // GetDarkColor() and GetLightColor() use GetFillColor()
   Color_t oldcolor = GetFillColor();
   SetFillColor(color);
   TAttFill::Modify();
   Color_t light;
   Color_t dark;
   if (color <= 50) {
      light    = color + 150;
      dark     = color + 100;
   } else {
      Float_t r, g, b, h, l, s;
      TColor *c = gROOT->GetColor(color);
      c->GetRGB(r, g, b);
      TColor::RGBtoHLS(r, g, b, h, l, s);
      TColor::HLStoRGB(h, 0.7*l, s, r, g, b);
      dark = TColor::GetColor(r, g, b);
      TColor::HLStoRGB(h, 1.2*l, s, r, g, b);
      light = TColor::GetColor(r, g, b);
   }

//*-*- Compute real left bottom & top right of the box in pixels
   px1 = XtoPixel(fX1);   py1 = YtoPixel(fY1);
   px2 = XtoPixel(fX2);   py2 = YtoPixel(fY2);
   if (px1 < px2) {pxl = px1; pxt = px2; xl = fX1; xt = fX2; }
   else           {pxl = px2; pxt = px1; xl = fX2; xt = fX1;}
   if (py1 > py2) {pyl = py1; pyt = py2; yl = fY1; yt = fY2;}
   else           {pyl = py2; pyt = py1; yl = fY2; yt = fY1;}

   if (!IsBatch()) {
      TPoint frame[7];

//*-*- Draw top&left part of the box
      frame[0].fX = pxl;                 frame[0].fY = pyl;
      frame[1].fX = pxl + bordersize;    frame[1].fY = pyl - bordersize;
      frame[2].fX = frame[1].fX;         frame[2].fY = pyt + bordersize;
      frame[3].fX = pxt - bordersize;    frame[3].fY = frame[2].fY;
      frame[4].fX = pxt;                 frame[4].fY = pyt;
      frame[5].fX = pxl;                 frame[5].fY = pyt;
      frame[6].fX = pxl;                 frame[6].fY = pyl;

      if (fBorderMode == -1) gVirtualX->SetFillColor(dark);
      else                   gVirtualX->SetFillColor(light);
      gVirtualX->DrawFillArea(7, frame);

//*-*- Draw bottom&right part of the box
      frame[0].fX = pxl;                 frame[0].fY = pyl;
      frame[1].fX = pxl + bordersize;    frame[1].fY = pyl - bordersize;
      frame[2].fX = pxt - bordersize;    frame[2].fY = frame[1].fY;
      frame[3].fX = frame[2].fX;         frame[3].fY = pyt + bordersize;
      frame[4].fX = pxt;                 frame[4].fY = pyt;
      frame[5].fX = pxt;                 frame[5].fY = pyl;
      frame[6].fX = pxl;                 frame[6].fY = pyl;

      if (fBorderMode == -1) gVirtualX->SetFillColor(light);
      else                   gVirtualX->SetFillColor(dark);
      gVirtualX->DrawFillArea(7, frame);

//*-* If this pad is a button, highlight it
      if (InheritsFrom("TButton") && fBorderMode == -1) {
         if (TestBit(kFraming)) {  // bit set in TButton::SetFraming
            if (GetFillColor() != 2) gVirtualX->SetLineColor(2);
            else                     gVirtualX->SetLineColor(4);
            gVirtualX->DrawBox(px1+2,py1-2,px2-2,py2+2, TVirtualX::kHollow);
         }
      }
      gVirtualX->SetFillColor(-1);
      SetFillColor(oldcolor);
   }

   if (!tops) return;

//*-*- same for PostScript
//   Double_t dx   = (xt - xl) *Double_t(bordersize)/Double_t(pxt - pxl);
//   Int_t border = gVirtualPS->XtoPS(xt) - gVirtualPS->XtoPS(xt-dx);

   PaintBorderPS(xl, yl, xt, yt, fBorderMode, bordersize, dark, light);
}

//______________________________________________________________________________
void TPad::PaintBorderPS(Double_t xl,Double_t yl,Double_t xt,Double_t yt,Int_t bmode,Int_t bsize,Int_t dark,Int_t light)
{
//*-*-*-*-*-*-*-*Paint a frame border with Postscript*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================================

   if (!gVirtualPS) return;
   gVirtualPS->DrawFrame(xl, yl, xt, yt, bmode,bsize,dark,light);
}

//______________________________________________________________________________
void TPad::PaintPadFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
//*-*-*-*-*-*-*-*-*-*Paint histogram/graph frame*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ===========================

   TList *glist  = GetListOfPrimitives();
   TFrame *frame = GetFrame();
   frame->SetX1(xmin);
   frame->SetX2(xmax);
   frame->SetY1(ymin);
   frame->SetY2(ymax);
   if (!glist->FindObject(fFrame)) {
      glist->AddFirst(frame);
      fFrame->SetBit(kMustCleanup);
   }
   frame->Paint();
}

//______________________________________________________________________________
void TPad::PaintModified()
{
//*-*-*-*-*-*-*Traverse pad hierarchy and (re)paint only modified pads*-*-*-*
//*-*          =======================================================

   TPad *padsav = (TPad*)gPad;
   TVirtualPS *saveps = gVirtualPS;
   if (gVirtualPS) {
      if (gVirtualPS->TestBit(kPrintingPS)) gVirtualPS = 0;
   }
   fPadPaint = 1;
   cd();
   if (IsModified() || IsTransparent()) {
       if (fPadView3D)
#if 1
         fPadView3D->PaintBeginModel();
#else
         gROOT->ProcessLine(Form("((TPadView3D *)0x%lx)->PaintBeginModel();",(Long_t)fPadView3D));
#endif
       PaintBorder(GetFillColor(), kTRUE);
   }

   if (fCanvas == this && gStyle->GetOptDate()) {
      if (!fPrimitives->FindObject("DATE")) {
          TDatime dt;
          TText *tdate = new TText(gStyle->GetDateX(),gStyle->GetDateY(),dt.AsString());
          tdate->SetName("DATE");
          tdate->SetTextSize( gStyle->GetAttDate()->GetTextSize());
          tdate->SetTextFont( gStyle->GetAttDate()->GetTextFont());
          tdate->SetTextColor(gStyle->GetAttDate()->GetTextColor());
          tdate->SetTextAlign(gStyle->GetAttDate()->GetTextAlign());
          tdate->SetTextAngle(gStyle->GetAttDate()->GetTextAngle());
          tdate->SetNDC();
          tdate->Draw();
       }
   }
   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   TObject *obj;

   while (lnk) {
      obj = lnk->GetObject();
      if (obj->InheritsFrom(TPad::Class())) {
         ((TPad*)obj)->PaintModified();
      } else if (IsModified() || IsTransparent()) {
         obj->Paint(lnk->GetOption());
      }
      lnk = (TObjOptLink*)lnk->Next();
   }

   if (IsModified() && fPadView3D)
   {
#if 1
//*-*   Finish 3D Model
       fPadView3D->PaintEnd();
//*-*   Start 3D scene
       fPadView3D->PaintScene();
#else
//*-*   Finish 3D Model
       gROOT->ProcessLine(Form("((TPadView3D *)0x%lx)->PaintEnd();",(Long_t)fPadView3D));
//*-*   Start 3D scene
       gROOT->ProcessLine(Form("((TPadView3D *)0x%lx)->PaintScene();",(Long_t)fPadView3D));
#endif
   }

   if (padsav) padsav->cd();
   fPadPaint = 0;
   Modified(kFALSE);
   gVirtualPS = saveps;
}

//______________________________________________________________________________
void TPad::PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint box in CurrentPad World coordinates*-*-*-*-*-*-*-*-*-*
//*-*              =========================================

   Color_t fcolor;
   if (!gPad->IsBatch()) {
      Int_t px1 = XtoPixel(x1);
      Int_t px2 = XtoPixel(x2);
      Int_t py1 = YtoPixel(y1);
      Int_t py2 = YtoPixel(y2);
      //box width must be at least one pixel
      if (TMath::Abs(px2-px1) < 1) px2 = px1+1;
      if (TMath::Abs(py1-py2) < 1) py1 = py2+1;

      Int_t style = gVirtualX->GetFillStyle();
      if (style) {
         if (style > 3000 && style < 4000) {
            fcolor = -1;
#ifndef WIN32
            if (style < 3020) {
               // set solid background color
               gVirtualX->SetFillStyle(1001);
               gVirtualX->DrawBox(px1,py1,px2,py2,TVirtualX::kFilled);
            }

            if (style > 3100) {
               char gifname[32];
               sprintf(gifname,"gif%d",style);
               TNamed *gifnamed = (TNamed*)gROOT->GetListOfSpecials()->FindObject(gifname);
               if (gifnamed) gVirtualX->ReadGIF(px1,py2,gifnamed->GetTitle());
            }
            if (style > 3020 && style < 3100) {
               gVirtualX->SetFillStyle(style-10);
            } else {
               fcolor = gVirtualX->GetFillColor(); // save fill color
               gVirtualX->SetFillColor(1);
               gVirtualX->SetFillStyle(style);
            }
#else
#ifdef GDK_WIN32
            if (style < 3020) {
               // set solid background color
               gVirtualX->SetFillStyle(1001);
               gVirtualX->DrawBox(px1,py1,px2,py2,TVirtualX::kFilled);
            }

            if (style > 3100) {
               char gifname[32];
               sprintf(gifname,"gif%d",style);
               TNamed *gifnamed = (TNamed*)gROOT->GetListOfSpecials()->FindObject(gifname);
               if (gifnamed) gVirtualX->ReadGIF(px1,py2,gifnamed->GetTitle());
            }
            if (style > 3020 && style < 3100) {
               gVirtualX->SetFillStyle(style-10);
            } else {
               fcolor = gVirtualX->GetFillColor(); // save fill color
               gVirtualX->SetFillColor(1);
               gVirtualX->SetFillStyle(style);
            }
#endif
#endif
            // draw stipples
            gVirtualX->DrawBox(px1,py1,px2,py2,TVirtualX::kFilled);

#ifndef WIN32
            // restore fill color
            if (fcolor >= 0) {
               gVirtualX->SetFillColor(fcolor);
            }
#else
#ifdef GDK_WIN32
            // restore fill color
            if (fcolor >= 0) {
               gVirtualX->SetFillColor(fcolor);
            }
#endif
#endif
            } else if (style >= 4000 && style <= 4100) {
            // For style >=4000 we make the window transparent.
            // From 4000 to 4100 the window is 100% transparent to 100% opaque

            //ignore this style option when this is the canvas itself
            if (this == fMother)
               gVirtualX->DrawBox(px1,py1,px2,py2,TVirtualX::kFilled);
            else {
               //draw background by blitting all bottom pads
               int px, py;
               XYtoAbsPixel(fX1, fY2, px, py);

               fMother->CopyBackgroundPixmap(px, py);
               CopyBackgroundPixmaps(fMother, this, px, py);

               gVirtualX->SetOpacity(style-4000);
            }
         } else {
            gVirtualX->DrawBox(px1,py1,px2,py2,TVirtualX::kFilled);
         }
      } else {
         gVirtualX->DrawBox(px1,py1,px2,py2,TVirtualX::kHollow);
      }
   }

   if (gVirtualPS) {
      gVirtualPS->DrawBox(x1, y1, x2, y2);
   }

   Modified();
}

//______________________________________________________________________________
void TPad::CopyBackgroundPixmaps(TPad *start, TPad *stop, Int_t x, Int_t y)
{
   // Copy pixmaps of pads laying below pad "stop" into pad "stop". This
   // gives the effect of pad "stop" being transparent.

   TObject *obj;
   TIter next(start->GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         if (obj == stop) break;
         ((TPad*)obj)->CopyBackgroundPixmap(x, y);
         ((TPad*)obj)->CopyBackgroundPixmaps((TPad*)obj, stop, x, y);
      }
   }
}

//______________________________________________________________________________
void TPad::CopyBackgroundPixmap(Int_t x, Int_t y)
{
   // Copy pixmap of this pad as background of the current pad.

   int px, py;
   XYtoAbsPixel(fX1, fY2, px, py);
   gVirtualX->CopyPixmap(GetPixmapID(), px-x, py-y);
}

//______________________________________________________________________________
void TPad::PaintFillArea(Int_t nn, Float_t *xx, Float_t *yy, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint fill area in CurrentPad World coordinates*-*-*-*-*-*-*
//*-*              ===============================================

   if (nn <3) return;
   Int_t i,iclip,n=0;
   Double_t xmin,xmax,ymin,ymax;
   Double_t u1, v1, u[2],v[2];
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Double_t *x = new Double_t[2*nn+1];
   Double_t *y = new Double_t[2*nn+1];

   for (i=0;i<nn;i++) {
      u[0] = xx[i];
      v[0] = yy[i];
      if (i == nn-1) {
         u[1] = xx[0];
         v[1] = yy[0];
      } else {
         u[1] = xx[i+1];
         v[1] = yy[i+1];
      }
      u1 = u[1];
      v1 = v[1];
      iclip = Clip(u,v,xmin,ymin,xmax,ymax);
      if (iclip == 2) continue;
      if (iclip == 1) {
         if (u[0] == u[1] && v[0] == v[1]) continue;
      }
      x[n] = u[0];
      y[n] = v[0];
      n++;
      if (iclip) {
         if (u[1] != u1 || v[1] != v1) {
            x[n] = u[1];
            y[n] = v[1];
            n++;
         }
      }
   }
   x[n] = x[0];
   y[n] = y[0];
   if (n < 3) return;

   TPoint *pxy;

//*-*- Create temporary array to store array in pixel coordinates

   if (!gPad->IsBatch()) {
      if (n <kPXY) pxy = &gPXY[0];
      else         pxy = new TPoint[n+1];
//*-*- convert points from world to pixel coordinates
      for (i=0;i<n;i++) {
         pxy[i].fX = gPad->XtoPixel(x[i]);
         pxy[i].fY = gPad->YtoPixel(y[i]);
      }
//*-*- invoke the graphics subsystem
      if (gVirtualX->GetFillStyle() == 0) {
         pxy[n].fX = pxy[0].fX;
         pxy[n].fY = pxy[0].fY;
         gVirtualX->DrawFillArea(n+1,pxy);
      } else {
         gVirtualX->DrawFillArea(n,pxy);
      }
      if (n > kPXY) delete [] pxy;
   }

   if (gVirtualPS) {
      gVirtualPS->DrawPS(-n, x, y);
   }
   delete [] x;
   delete [] y;
   Modified();
}

//______________________________________________________________________________
void TPad::PaintFillArea(Int_t nn, Double_t *xx, Double_t *yy, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint fill area in CurrentPad World coordinates*-*-*-*-*-*-*
//*-*              ===============================================

   if (nn <3) return;
   Int_t i,iclip,n=0;
   Double_t xmin,xmax,ymin,ymax;
   Double_t u1, v1, u[2],v[2];
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Double_t *x = new Double_t[2*nn+1];
   Double_t *y = new Double_t[2*nn+1];

   for (i=0;i<nn;i++) {
      u[0] = xx[i];
      v[0] = yy[i];
      if (i == nn-1) {
         u[1] = xx[0];
         v[1] = yy[0];
      } else {
         u[1] = xx[i+1];
         v[1] = yy[i+1];
      }
      u1 = u[1];
      v1 = v[1];
      iclip = Clip(u,v,xmin,ymin,xmax,ymax);
      if (iclip == 2) continue;
      if (iclip == 1) {
         if (u[0] == u[1] && v[0] == v[1]) continue;
      }
      x[n] = u[0];
      y[n] = v[0];
      n++;
      if (iclip) {
         if (u[1] != u1 || v[1] != v1) {
            x[n] = u[1];
            y[n] = v[1];
            n++;
         }
      }
   }
   x[n] = x[0];
   y[n] = y[0];
   if (n < 3) return;

   TPoint *pxy;

//*-*- Create temporary array to store array in pixel coordinates

   if (!gPad->IsBatch()) {
      if (n <kPXY) pxy = &gPXY[0];
      else         pxy = new TPoint[n+1];
//*-*- convert points from world to pixel coordinates
      for (i=0;i<n;i++) {
         pxy[i].fX = gPad->XtoPixel(x[i]);
         pxy[i].fY = gPad->YtoPixel(y[i]);
      }
//*-*- invoke the graphics subsystem
      if (gVirtualX->GetFillStyle() == 0) {
         pxy[n].fX = pxy[0].fX;
         pxy[n].fY = pxy[0].fY;
         gVirtualX->DrawFillArea(n+1,pxy);
      } else {
         gVirtualX->DrawFillArea(n,pxy);
      }
      if (n > kPXY) delete [] pxy;
   }

   if (gVirtualPS) {
      gVirtualPS->DrawPS(-n, x, y);
   }
   delete [] x;
   delete [] y;
   Modified();
}

//______________________________________________________________________________
void TPad::PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
//*-*-*-*-*-*-*-*-*Paint line in CurrentPad World coordinates*-*-*-*-*-*-*-*
//*-*              ==========================================

   Double_t x[2], y[2];
   x[0] = x1;   x[1] = x2;   y[0] = y1;   y[1] = y2;

   //If line is totally clipped, return
   if (TestBit(TGraph::kClipFrame)) {
      if (Clip(x,y,fUxmin,fUymin,fUxmax,fUymax) == 2) return;
   } else {
      if (Clip(x,y,fX1,fY1,fX2,fY2) == 2) return;
   }
   if (!gPad->IsBatch()) {
      Int_t px1 = XtoPixel(x[0]);
      Int_t px2 = XtoPixel(x[1]);
      Int_t py1 = YtoPixel(y[0]);
      Int_t py2 = YtoPixel(y[1]);

      gVirtualX->DrawLine(px1, py1, px2, py2);
   }

   if (gVirtualPS) {
      gVirtualPS->DrawPS(2, x, y);
   }

   Modified();
}

//______________________________________________________________________________
void TPad::PaintLineNDC(Double_t u1, Double_t v1,Double_t u2, Double_t v2)
{
//*-*-*-*-*-*-*-*Draw a line with coordinates in NDC*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ===================================
   static Double_t xw[2], yw[2];
   if (!gPad->IsBatch()) {
      Int_t px1 = UtoPixel(u1);
      Int_t py1 = VtoPixel(v1);
      Int_t px2 = UtoPixel(u2);
      Int_t py2 = VtoPixel(v2);

      gVirtualX->DrawLine(px1, py1, px2, py2);
   }

   if (gVirtualPS) {
      xw[0] = fX1 + u1*(fX2 - fX1);
      xw[1] = fX1 + u2*(fX2 - fX1);
      yw[0] = fY1 + v1*(fY2 - fY1);
      yw[1] = fY1 + v2*(fY2 - fY1);
      gVirtualPS->DrawPS(2, xw, yw);
      gVirtualPS->PrintFast(2," s");
   }

   Modified();
}

//______________________________________________________________________________
void TPad::PaintLine3D(Float_t *p1, Float_t *p2)
{
//*-*-*-*-*-*-*-*-*Paint 3-D line in the CurrentPad*-*-*-*-*-*-*-*-*-*-*
//*-*              ================================

   if (!fView) return;

//*-*- convert from 3-D to 2-D pad coordinate system
   Double_t xpad[6];
   Double_t temp[3];
   Int_t i;
   for (i=0;i<3;i++) temp[i] = p1[i];
   fView->WCtoNDC(temp, &xpad[0]);
   for (i=0;i<3;i++) temp[i] = p2[i];
   fView->WCtoNDC(temp, &xpad[3]);
   PaintLine(xpad[0],xpad[1],xpad[3],xpad[4]);
}

//______________________________________________________________________________
void TPad::PaintLine3D(Double_t *p1, Double_t *p2)
{
//*-*-*-*-*-*-*-*-*Paint 3-D line in the CurrentPad*-*-*-*-*-*-*-*-*-*-*
//*-*              ================================

   if (!fView) return;

//*-*- convert from 3-D to 2-D pad coordinate system
   Double_t xpad[6];
   fView->WCtoNDC(p1, &xpad[0]);
   fView->WCtoNDC(p2, &xpad[3]);
   PaintLine(xpad[0],xpad[1],xpad[3],xpad[4]);
}

//______________________________________________________________________________
void TPad::PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint polyline in CurrentPad World coordinates*-*-*-*-*-*-*
//*-*              ==============================================

   if (n < 2) return;
   TPoint *pxy = &gPXY[0];
   if (!gPad->IsBatch()) {
      if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   }
   Double_t xmin,xmax,ymin,ymax;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,j,i1=-1,np=1;
   for (i=0; i<n-1; i++) {
      Double_t x1=x[i];
      Double_t y1=y[i];
      Double_t x2=x[i+1];
      Double_t y2=y[i+1];
      Int_t iclip = Clip(&x[i],&y[i],xmin,ymin,xmax,ymax);
      if (iclip == 2) {
         i1 = -1;
         continue;
      }
      np++;
      if (i1 < 0) i1 = i;
      if (iclip == 0 && i < n-2) continue;
      if (!gPad->IsBatch()) {
         for (j=0;j<np;j++) {
            pxy[j].fX = XtoPixel(x[i1+j]);
            pxy[j].fY = YtoPixel(y[i1+j]);
         }
         gVirtualX->DrawPolyLine(np,pxy);
      }
      if (gVirtualPS) {
         gVirtualPS->DrawPS(np, &x[i1], &y[i1]);
      }
      if (iclip) {
         x[i] = x1;
         y[i] = y1;
         x[i+1] = x2;
         y[i+1] = y2;
      }
      i1 = -1;
      np = 1;
   }
   if (!gPad->IsBatch()) {
      if (n >= kPXY)   delete [] pxy;
   }
   Modified();
}

//______________________________________________________________________________
void TPad::PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint polyline in CurrentPad World coordinates*-*-*-*-*-*-*
//*-*              ==============================================

   if (n < 2) return;
   TPoint *pxy = &gPXY[0];
   if (!gPad->IsBatch()) {
      if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   }
   Double_t xmin,xmax,ymin,ymax;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,j,i1=-1,np=1;
   for (i=0; i<n-1; i++) {
      Double_t x1=x[i];
      Double_t y1=y[i];
      Double_t x2=x[i+1];
      Double_t y2=y[i+1];
      Int_t iclip = Clip(&x[i],&y[i],xmin,ymin,xmax,ymax);
      if (iclip == 2) {
         i1 = -1;
         continue;
      }
      np++;
      if (i1 < 0) i1 = i;
      if (iclip == 0 && i < n-2) continue;
      if (!gPad->IsBatch()) {
         for (j=0;j<np;j++) {
            pxy[j].fX = XtoPixel(x[i1+j]);
            pxy[j].fY = YtoPixel(y[i1+j]);
         }
         gVirtualX->DrawPolyLine(np,pxy);
      }
      if (gVirtualPS) {
         gVirtualPS->DrawPS(np, &x[i1], &y[i1]);
      }
      if (iclip) {
         x[i] = x1;
         y[i] = y1;
         x[i+1] = x2;
         y[i+1] = y2;
      }
      i1 = -1;
      np = 1;
   }
   if (!gPad->IsBatch()) {
      if (n >= kPXY)   delete [] pxy;
   }
   Modified();
}

//______________________________________________________________________________
void TPad::PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint polyline in CurrentPad NDC coordinates*-*-*-*-*-*-*
//*-*              ============================================

   TPoint *pxy;

//*-*- Create temporary array to store array in pixel coordinates
   if (n <=0) return;

   if (!gPad->IsBatch()) {
      if (n <kPXY) pxy = &gPXY[0];
      else         pxy = new TPoint[n+1]; if (!pxy) return;
//*-*- convert points from world to pixel coordinates
      for (Int_t i=0;i<n;i++) {
         pxy[i].fX = UtoPixel(x[i]);
         pxy[i].fY = VtoPixel(y[i]);
      }
//*-*- invoke the graphics subsystem
      gVirtualX->DrawPolyLine(n,pxy);
      if (n > kPXY)  delete [] pxy;
   }

   if (gVirtualPS) {
      gVirtualPS->DrawPS(n, x, y);
   }

   Modified();

}


//______________________________________________________________________________
void TPad::PaintPolyLine3D(Int_t n, Double_t *p)
{
//*-*-*-*-*-*-*-*-*Paint 3-D polyline in the CurrentPad*-*-*-*-*-*-*-*-*-*-*
//*-*              ====================================

   if (!fView) return;

//*-*- Loop on each individual line

   for (Int_t i = 1; i < n; i++)
      PaintLine3D(&p[3*i-3], &p[3*i]);

   Modified();
}

//______________________________________________________________________________
void TPad::PaintPolyMarker(Int_t nn, Float_t *x, Float_t *y, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint polymarker in CurrentPad World coordinates*-*-*-*-*-*
//*-*              ================================================

   Int_t n = TMath::Abs(nn);
   TPoint *pxy = &gPXY[0];
   if (!gPad->IsBatch()) {
      if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   }
   Double_t xmin,xmax,ymin,ymax;
   if (nn > 0 || TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,j,i1=-1,np=0;
   for (i=0; i<n; i++) {
      if (x[i] >= xmin && x[i] <= xmax && y[i] >= ymin && y[i] <= ymax) {
         np++;
         if (i1 < 0) i1 = i;
         if (i < n-1) continue;
      }
      if (np == 0) continue;
      if (!gPad->IsBatch()) {
         for (j=0;j<np;j++) {
            pxy[j].fX = XtoPixel(x[i1+j]);
            pxy[j].fY = YtoPixel(y[i1+j]);
         }
         gVirtualX->DrawPolyMarker(np,pxy);
      }
      if (gVirtualPS) {
         gVirtualPS->DrawPolyMarker(np, &x[i1], &y[i1]);
      }
      i1 = -1;
      np = 0;
   }
   if (!gPad->IsBatch()) {
      if (n >= kPXY)   delete [] pxy;
   }
   Modified();
}

//______________________________________________________________________________
void TPad::PaintPolyMarker(Int_t nn, Double_t *x, Double_t *y, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint polymarker in CurrentPad World coordinates*-*-*-*-*-*
//*-*              ================================================

   Int_t n = TMath::Abs(nn);
   TPoint *pxy = &gPXY[0];
   if (!gPad->IsBatch()) {
      if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   }
   Double_t xmin,xmax,ymin,ymax;
   if (nn > 0 || TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,j,i1=-1,np=0;
   for (i=0; i<n; i++) {
      if (x[i] >= xmin && x[i] <= xmax && y[i] >= ymin && y[i] <= ymax) {
         np++;
         if (i1 < 0) i1 = i;
         if (i < n-1) continue;
      }
      if (np == 0) continue;
      if (!gPad->IsBatch()) {
         for (j=0;j<np;j++) {
            pxy[j].fX = XtoPixel(x[i1+j]);
            pxy[j].fY = YtoPixel(y[i1+j]);
         }
         gVirtualX->DrawPolyMarker(np,pxy);
      }
      if (gVirtualPS) {
         gVirtualPS->DrawPolyMarker(np, &x[i1], &y[i1]);
      }
      i1 = -1;
      np = 0;
   }
   if (!gPad->IsBatch()) {
      if (n >= kPXY)   delete [] pxy;
   }
   Modified();
}

//______________________________________________________________________________
void TPad::PaintText(Double_t x, Double_t y, const char *text)
{
//*-*-*-*-*-*-*-*-*Paint text in CurrentPad World coordinates*-*-*-*-*-*-*-*
//*-*              ==========================================

   Modified();

   if (!gPad->IsBatch()) {
      Int_t px = XtoPixel(x);
      Int_t py = YtoPixel(y);
      Float_t angle = gVirtualX->GetTextAngle();
      gVirtualX->DrawText(px, py, angle, gVirtualX->GetTextMagnitude(), text, TVirtualX::kClear);
   }

   if (gVirtualPS) {
      if (x < fX1) x = fX1;
      if( x > fX2) x = fX2;
      if (y < fY1) y = fY1;
      if( y > fY2) y = fY2;
      gVirtualPS->Text(x, y, text);
   }

}

//______________________________________________________________________________
void TPad::PaintTextNDC(Double_t u, Double_t v, const char *text)
{
//*-*-*-*-*-*-*-*-*Paint text in CurrentPad NDC coordinates*-*-*-*-*-*-*-*
//*-*              ========================================


   Modified();

   if (!gPad->IsBatch()) {
      Int_t px = UtoPixel(u);
      Int_t py = VtoPixel(v);
      Float_t angle = gVirtualX->GetTextAngle();

      gVirtualX->DrawText(px, py, angle, gVirtualX->GetTextMagnitude(), text, TVirtualX::kClear);
   }

   if (gVirtualPS) {
      Double_t x = fX1 + u*(fX2 - fX1);
      Double_t y = fY1 + v*(fY2 - fY1);
      if (x < fX1 || x > fX2) return;
      if (y < fY1 || y > fY2) return;
      gVirtualPS->Text(x, y, text);
   }

}

//______________________________________________________________________________
TPad *TPad::Pick(Int_t px, Int_t py, TObjLink *&pickobj)
{
//*-*-*-*-*-*-*-*-*Search for an object at pixel position px,py*-*-*-*-*-*-*
//*-*              ============================================
//  Check if point is in this pad.
//  If yes, check if it is in one of the subpads
//  If found in the pad, compute closest distance of approach
//  to each primitive.
//  If one distance of approach is found to be within the limit Distancemaximum
//  the corresponding primitive is selected and the routine returns.
//

   //the two following statements are necessary under NT (multithreaded)
   //when a TCanvas object is being created and a thread calling TPad::Pick
   //before the TPad constructor has completed in the other thread
   if (gPad == 0) return 0; //Andy Haas
   if (GetListOfPrimitives() == 0) return 0; //Andy Haas

   Int_t dist;
//*-*- Search if point is in pad itself
   Double_t x = AbsPixeltoX(px);
   Double_t y = AbsPixeltoY(py);
   if (this != gPad->GetCanvas()) {
      if (!((x >= fX1 && x <= fX2) && (y >= fY1 && y <= fY2))) return 0;
   }

//*-*- search for a primitive in this pad or its subpads
   static TObjOptLink dummyLink(0,"");  //place holder for when no link available
   TPad *padsav = (TPad*)gPad;
   gPad  = this;    // since no drawing will be done, don't use cd() for efficiency reasons
   TPad *pick   = 0;
   TPad *picked = this;
   pickobj      = 0;
   if (DistancetoPrimitive(px,py) < fgMaxPickDistance) {
      dummyLink.SetObject(this);
      pickobj = &dummyLink;
   }

   // Loop backwards over the list of primitives. The first non-pad primitive
   // found is the selected one. However, we have to keep going down the
   // list to see if there is maybe a pad overlaying the primitive. In that
   // case look into the pad for a possible primitive. Once a pad has been
   // found we can terminate the loop.
   Bool_t gotPrim = kFALSE;      // true if found a non pad primitive
   TObjLink *lnk = GetListOfPrimitives()->LastLink();
   while (lnk) {
      TObject *obj = lnk->GetObject();
      fPadPointer  = obj;
      if (obj->InheritsFrom(TPad::Class())) {
         pick = ((TPad*)obj)->Pick(px, py, pickobj);
         if (pick) {
            picked = pick;
            break;
         }
      } else if (!gROOT->GetEditorMode()) {
         if (!gotPrim) {
            if (!obj->TestBit(kCannotPick)) {
               dist = obj->DistancetoPrimitive(px, py);
               if (dist < fgMaxPickDistance) {
                  pickobj = lnk;
                  gotPrim = kTRUE;
                  if (dist == 0) break;
               }
            }
         }
      }

      lnk = lnk->Prev();
   }

   if (picked->InheritsFrom(TButton::Class())) {
      TButton *button = (TButton*)picked;
      if (!button->IsEditable()) pickobj = 0;
   }

   gPad = padsav;
   return picked;
}

//___________________________________________________________________________
void TPad::Pop()
{
//*-*-*-*-*-*-*-*-*-*-*Pop pad to the top of the stack*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
   if (this == fMother->GetListOfPrimitives()->Last()) return;

   TListIter next(fMother->GetListOfPrimitives());
   TObject *obj;
   while ((obj = next()))
      if (obj == this) {
         char *opt = StrDup(next.GetOption());
         fMother->GetListOfPrimitives()->Remove(this);
         fMother->GetListOfPrimitives()->AddLast(this, opt);
         delete [] opt;
         return;
      }
}


//______________________________________________________________________________
void TPad::Print(const char *filename) const
{
//  Save Pad contents on a  file in various formats
//  ===============================================
//
//   if filename is "", the file produced is padname.ps
//   if filename starts with a dot, the padname is added in front
//   if filename contains .eps, an Encapsulated Postscript file is produced
//   if filename contains .gif, a GIF file is produced
//   if filename contains .C or .cxx, a C++ macro file is produced
//   if filename contains .root, a Root file is produced
//
//  See comments in TPad::SaveAs or the TPad::Print function below
//
//

   ((TPad*)this)->SaveAs(filename);

}

//______________________________________________________________________________
void TPad::Print(const char *filename, Option_t *option)
{
//*-*-*-*-*Save Pad contents on a file in various formats*-*-*-*-*-*
//*-*      ==============================================
//
//   if option  =  0   - as "ps"
//               "ps"  - Postscript file is produced
//               "eps" - an Encapsulated Postscript file is produced
//               "gif" - a GIF file is produced
//               "cxx" - a C++ macro file is produced
//
//     filename = 0 - filename  is defined by the GetName and its
//                    extension is defined with the option
//
//   When Postscript output is selected (ps, eps), the pad is saved
//   to filename.ps or filename.eps. The aspect ratio of the pad is preserved
//   on the Postscript file. When the "ps" option is selected, the Postscript
//   page will be landscape format if the pad is in landscape format, otherwise
//   portrait format is selected.
//   The physical size of the Postscript page is the one selected in the
//   current style. This size can be modified via TStyle::SetPaperSize.
//   Examples:
//        gStyle->SetPaperSize(kA4);  //default
//        gStyle->SetPaperSize(kUSLetter);
//     where kA4 and kUSLetter are defined in the enum EPaperSize in TStyle.h
//    An alternative is to call:
//        gStyle->SetPaperSize(20,26);  same as kA4
// or     gStyle->SetPaperSize(20,24);  same as kUSLetter
//   The above numbers take into account some margins and are in centimeters.
//
//  To generate a Postscript file containing more than one picture, see
//  class TPostScript.

   char psname[264];
   Int_t lenfil =  filename ? strlen(filename) : 0;
   const char *opt = option;

//*-*   Set the default option as "Postscript" (Should be a data member of TPad)

   const char *opt_default="ps";
   if( !opt ) opt = opt_default;

   if ( !lenfil )  sprintf(psname,"%s.%s",GetName(),opt);
   else            strcpy(psname,filename);

   // line below protected against case like c1->SaveAs( "../ps/cs.ps" );
   if ((psname[0] == '.') && (strchr(psname,'/') == 0)) sprintf(psname,"%s%s",GetName(),filename);

   if (strstr(opt,"gif")) {
      if (GetCanvas()->IsBatch()) {
         Printf("Cannot create gif file in batch mode.");
         return;
      }
      Update();
      Int_t wid = (this == GetCanvas()) ? GetCanvas()->GetCanvasID() : GetPixmapID();
      gVirtualX->SelectWindow(wid);
      gVirtualX->WriteGIF(psname);
      Info("TPad::Print", "GIF file %s has been created", psname);
      return;
   }

   if (strstr(opt,"cxx")) {
      GetCanvas()->SaveSource(psname,"");
      return;
   }
   if (strstr(opt,"root")) {
      TDirectory *dirsav = gDirectory;
      TFile *fsave = new TFile(psname,"RECREATE");
      Write();
      fsave->Close();
      delete fsave;
      if (dirsav) dirsav->cd();
      Info("TPad::Print", "ROOT file %s has been created", psname);
      return;
   }

   // in case we read directly from a Root file and the canvas
   // is not on the screen, set batch mode
   Bool_t noScreen = kFALSE;
   if (!GetCanvas()->IsBatch() && GetCanvas()->GetCanvasID() == -1) {
      noScreen = kTRUE;
      GetCanvas()->SetBatch(kTRUE);
   }
   Int_t pstype = 111;
   Double_t xcanvas = GetCanvas()->XtoPixel(GetCanvas()->GetX2());
   Double_t ycanvas = GetCanvas()->YtoPixel(GetCanvas()->GetY1());
   Double_t ratio   = ycanvas/xcanvas;
   if (ratio < 1)               pstype = 112;
   if (strstr(opt,"Portrait"))  pstype = 111;
   if (strstr(opt,"Landscape")) pstype = 112;
   if (strstr(opt,"eps")) pstype = 113;
   TPad *padsav = (TPad*)gPad;
   cd();
   TVirtualPS *psave = gVirtualPS;
   // check if Postscript class is in memory. If not, dynamic link
   if (gROOT->LoadClass("TPostScript","Postscript")) return;

   gROOT->ProcessLineFast("new TPostScript()");
   gVirtualPS->Open(psname,pstype);
   gVirtualPS->SetBit(kPrintingPS);
   Paint();
   if (noScreen)  GetCanvas()->SetBatch(kFALSE);
   Info("TPad::Print", "PostScript file %s has been created", psname);
   delete gVirtualPS;
   gVirtualPS = psave;
   padsav->cd();
}

//______________________________________________________________________________
void TPad::Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Set world coordinate system for the pad.
   // Emits signal "RangeChanged()", in the slot get the range
   // via GetRange().

   if ((x1 >= x2) || (y1 >= y2)) {
      Error("Range", "illegal world coordinates range: x1=%f, y1=%f, x2=%f, y2=%f",x1,y1,x2,y2);
      return;
   }

   fUxmin = x1;
   fUxmax = x2;
   fUymin = y1;
   fUymax = y2;

   if (fX1 == x1 && fY1 == y1 && fX2 == x2 && fY2 == y2) return;

   fX1  = x1;
   fY1  = y1;
   fX2  = x2;
   fY2  = y2;

   // compute pad conversion coefficients
   ResizePad();

   // emit signal
   RangeChanged();
}

//______________________________________________________________________________
void TPad::RangeAxis(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
   // Set axis coordinate system for the pad.
   // The axis coordinate system is a subset of the world coordinate system
   // xmin,ymin is the origin of the current coordinate system,
   // xmax is the end of the X axis, ymax is the end of the Y axis.
   // By default a margin of 10 per cent is left on all sides of the pad
   // Emits signal "RangeAxisChanged()", in the slot get the axis range
   // via GetRangeAxis().

   if ((xmin >= xmax) || (ymin >= ymax)) {
      Error("RangeAxis", "illegal axis coordinates range: xmin=%f, ymin=%f, xmax=%f, ymax=%f",
            xmin, ymin, xmax, ymax);
      return;
   }

   fUxmin  = xmin;
   fUymin  = ymin;
   fUxmax  = xmax;
   fUymax  = ymax;

   // emit signal
   RangeAxisChanged();
}

//______________________________________________________________________________
void TPad::RecursiveRemove(TObject *obj)
{
//*-*-*-*-*-*-*-*Recursively remove object from a pad and its subpads*-*-*-*-*
//*-*            ====================================================

   if (obj == fCanvas->GetSelected()) fCanvas->SetSelected(0);
   Int_t nold = fPrimitives->GetSize();
   fPrimitives->RecursiveRemove(obj);
   if (nold != fPrimitives->GetSize()) fModified = kTRUE;
}

//______________________________________________________________________________
void TPad::RedrawAxis(Option_t *option)
{
//  Redraw the frame axis
//  Redrawing axis may be necessary in case of superimposed histograms
//  when one or more histograms have a fill color
//  Instead of calling this function, it may be more convenient
//  to call directly h1->Draw("sameaxis") where h1 is the pointer
//  to the first histogram drawn in the pad.
//
   // get first histogram in the list of primitives
   TIter next(fPrimitives);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom("TH1")) {
         TH1 *hobj = (TH1*)obj;
         hobj->Draw("sameaxis");
         return;
      }
   }
}

//______________________________________________________________________________
void TPad::ResizePad(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Compute pad conversion coefficients*-*-*-*-*-*-*-*-*
//*-*                  ===================================
//
//   Conversion from x to px & y to py
//   =================================
//
//       x - xmin     px - pxlow              xrange  = xmax-xmin
//       --------  =  ----------      with
//        xrange        pxrange               pxrange = pxmax-pxmin
//
//               pxrange(x-xmin)
//   ==>  px =   ---------------  + pxlow   = fXtoPixelk + fXtoPixel * x
//                    xrange
//
//   ==>  fXtoPixelk = pxlow - pxrange*xmin/xrange
//        fXtoPixel  = pxrange/xrange
//           where  pxlow   = fAbsXlowNDC*fCw
//                  pxrange = fAbsWNDC*fCw
//
//
//       y - ymin     py - pylow              yrange  = ymax-ymin
//       --------  =  ----------      with
//        yrange        pyrange               pyrange = pymax-pymin
//
//               pyrange(y-ymin)
//   ==>  py =   ---------------  + pylow   = fYtoPixelk + fYtoPixel * y
//                    yrange
//
//   ==>  fYtoPixelk = pylow - pyrange*ymin/yrange
//        fYtoPixel  = pyrange/yrange
//           where  pylow   = (1-fAbsYlowNDC)*fCh
//                  pyrange = -fAbsHNDC*fCh
//
//-  Conversion from px to x & py to y
//   =================================
//
//             xrange(px-pxlow)
//   ==>  x =  ----------------  + xmin  = fPixeltoXk + fPixeltoX * px
//                 pxrange
//-
//   ==>  fPixeltoXk = xmin - pxlow*xrange/pxrange
//        fPixeltoX  = xrange/pxrange
//
//             yrange(py-pylow)
//   ==>  y =  ----------------  + ymin  = fPixeltoYk + fPixeltoY * py
//                 pyrange
//-
//   ==>  fPixeltoYk = ymin - pylow*yrange/pyrange
//        fPixeltoY  = yrange/pyrange
//
//-----------------------------------------------------------------------
//
//  Computation of the coefficients in case of LOG scales
//- =====================================================
//
//   A, Conversion from pixel coordinates to world coordinates
//
//       Log(x) - Log(xmin)      Log(x/xmin)       px - pxlow
//  u = --------------------- =  -------------  =  -----------
//      Log(xmax) - Log(xmin)    Log(xmax/xmin)     pxrange
//
//  ==> Log(x/xmin) = u*Log(xmax/xmin)
//      x = xmin*exp(u*Log(xmax/xmin)
//   Let alfa = Log(xmax/xmin)/fAbsWNDC
//
//      x = xmin*exp(-alfa*pxlow) + exp(alfa*px)
//      x = fPixeltoXk*exp(fPixeltoX*px)
//  ==> fPixeltoXk = xmin*exp(-alfa*pxlow)
//      fPixeltoX  = alfa
//
//       Log(y) - Log(ymin)      Log(y/ymin)       pylow - py
//  v = --------------------- =  -------------  =  -----------
//      Log(ymax) - Log(ymin)    Log(ymax/ymin)     pyrange
//
//   Let beta = Log(ymax/ymin)/pyrange
//      Log(y/ymin) = beta*pylow - beta*py
//      y/ymin = exp(beta*pylow - beta*py)
//      y = ymin*exp(beta*pylow)*exp(-beta*py)
//  ==> y = fPixeltoYk*exp(fPixeltoY*py)
//      fPixeltoYk = ymin*exp(beta*pylow)
//      fPixeltoY  = -beta
//
//-  B, Conversion from World coordinates to pixel coordinates
//
//  px = pxlow + u*pxrange
//     = pxlow + Log(x/xmin)/alfa
//     = pxlow -Log(xmin)/alfa  + Log(x)/alfa
//     = fXtoPixelk + fXtoPixel*Log(x)
//  ==> fXtoPixelk = pxlow -Log(xmin)/alfa
//  ==> fXtoPixel  = 1/alfa
//
//  py = pylow - Log(y/ymin)/beta
//     = fYtoPixelk + fYtoPixel*Log(y)
//  ==> fYtoPixelk = pylow - Log(ymin)/beta
//      fYtoPixel  = 1/beta
//


//*-*- Recompute subpad positions in case pad has been moved/resized
   TPad *parent = fMother;
   if (this == gPad->GetCanvas()) {
      fAbsXlowNDC  = fXlowNDC;
      fAbsYlowNDC  = fYlowNDC;
      fAbsWNDC     = fWNDC;
      fAbsHNDC     = fHNDC;
   }
   else {
      fAbsXlowNDC  = fXlowNDC*parent->GetAbsWNDC() + parent->GetAbsXlowNDC();
      fAbsYlowNDC  = fYlowNDC*parent->GetAbsHNDC() + parent->GetAbsYlowNDC();
      fAbsWNDC     = fWNDC*parent->GetAbsWNDC();
      fAbsHNDC     = fHNDC*parent->GetAbsHNDC();
   }

   Double_t ww = (Double_t)gPad->GetWw();
   Double_t wh = (Double_t)gPad->GetWh();
   Double_t pxlow   = fAbsXlowNDC*ww;
   Double_t pylow   = (1-fAbsYlowNDC)*wh;
   Double_t pxrange = fAbsWNDC*ww;
   Double_t pyrange = -fAbsHNDC*wh;

//*-*- Linear X axis
   Double_t xrange  = fX2 - fX1;
   fXtoAbsPixelk = 0.5 + pxlow - pxrange*fX1/xrange;      //origin at left
   fXtoPixelk = -pxrange*fX1/xrange;
   fXtoPixel  = pxrange/xrange;
   fAbsPixeltoXk = fX1 - pxlow*xrange/pxrange;
   fPixeltoXk = fX1;
   fPixeltoX  = xrange/pxrange;
//*-*- Linear Y axis
   Double_t yrange  = fY2 - fY1;
   fYtoAbsPixelk = 0.5 + pylow - pyrange*fY1/yrange;      //origin at top
   fYtoPixelk = -pyrange - pyrange*fY1/yrange;
   fYtoPixel  = pyrange/yrange;
   fAbsPixeltoYk = fY1 - pylow*yrange/pyrange;
   fPixeltoYk = fY1;
   fPixeltoY  = yrange/pyrange;

//*-*- Coefficients to convert from pad NDC coordinates to pixel coordinates

   fUtoAbsPixelk = 0.5 + pxlow;
   fUtoPixelk = 0.0;
   fUtoPixel  = pxrange;
   fVtoAbsPixelk = 0.5 + pylow;
   fVtoPixelk = -pyrange;
   fVtoPixel  = pyrange;

//*-*- Coefficients to convert from canvas pixels to pad world coordinates

//*-*- Resize all subpads
   TObject *obj;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class()))
         ((TPad*)obj)->ResizePad(option);
   }

//*-*- Reset all current sizes
   if (gPad->IsBatch())
      fPixmapID = 0;
   else {
      gVirtualX->SetLineWidth(-1);
      gVirtualX->SetTextSize(-1);

      // create or re-create off-screen pixmap
      if (fPixmapID) {
         int w = TMath::Abs(XtoPixel(fX2) - XtoPixel(fX1));
         int h = TMath::Abs(YtoPixel(fY2) - YtoPixel(fY1));
         //protection in case of wrong pad parameters.
         //without this protection, the OpenPixmap or ResizePixmap crashes with
         //the message "Error in <RootX11ErrorHandler>: BadValue (integer parameter out of range for operation)"
         //resulting in a frozen xterm
         if (w <= 0 || w > 10000) {
            Warning("ResizePad", "%s width changed from %d to %d\n",GetName(),w,10);
            w = 10;
         }
         if (h <= 0 || h > 10000) {
            Warning("ResizePad", "%s height changed from %d to %d\n",GetName(),h,10);
            h = 10;
         }
         if (fPixmapID == -1)       // this case is handled via the ctor
            fPixmapID = gVirtualX->OpenPixmap(w, h);
         else
            if (gVirtualX->ResizePixmap(fPixmapID, w, h)) Modified(kTRUE);
      }
   }
}


//______________________________________________________________________________
void TPad::SaveAs(const char *filename)
{
//*-*-*-*-*Save Pad contents on a  file in various formats*-*-*-*-*-*
//*-*      ===============================================
//
//   if filename is "", the file produced is padname.ps
//   if filename starts with a dot, the padname is added in front
//   if filename contains .eps, an Encapsulated Postscript file is produced
//   if filename contains .gif, a GIF file is produced
//   if filename contains .C or .cxx, a C++ macro file is produced
//   if filename contains .root, a Root file is produced
//
//   See comments in TPad::Print for the Postscript formats

   char psname[264];
   Int_t lenfil =  filename ? strlen(filename) : 0;

   if (!lenfil)  sprintf(psname,"%s.ps",GetName());
   else          strcpy(psname,filename);

   // line below protected against case like c1->SaveAs( "../ps/cs.ps" );
   if ((psname[0] == '.') && (strchr(psname,'/') == 0)) sprintf(psname,"%s%s",GetName(),filename);

   if (strstr(psname,".gif"))
               Print(psname,"gif");
   else if (strstr(psname,".C") || strstr(psname,".cxx") || strstr(psname,".cpp"))
               Print(psname,"cxx");
   else if (strstr(psname,".root"))
               Print(psname,"root");
   else if (strstr(psname,".eps"))
               Print(psname,"eps");
   else
               Print(psname,"ps");

 }

//______________________________________________________________________________
void TPad::SavePrimitive(ofstream &out, Option_t *)
{
//*-*-*-*-*-*Save primitives in this pad on the C++ source file out*-*-*-*-*-*
//*-*        ======================================================

   TPad *padsav = (TPad*)gPad;
   char quote='"';
   char lcname[10];
   const char *cname = GetName();
   Int_t nch = strlen(cname);
   if (nch < 10) {
      strcpy(lcname,cname);
      for (Int_t k=1;k<=nch;k++) {if (lcname[nch-k] == ' ') lcname[nch-k] = 0;}
      if (lcname[0] == 0) {
         if (this == gPad->GetCanvas()) {strcpy(lcname,"c1");  nch = 2;}
         else                           {strcpy(lcname,"pad"); nch = 3;}
      }
      cname = lcname;
   }
   
//   Write pad parameters
   if (this != gPad->GetCanvas()) {
      out <<"  "<<endl;
      out <<"// ------------>Primitives in pad: "<<GetName()<<endl;

      if (gROOT->ClassSaved(TPad::Class())) {
         out<<"   ";
      } else {
         out<<"   TPad *";
      }
      out<<cname<<" = new TPad("<<quote<<GetName()<<quote<<", "<<quote<<GetTitle()
      <<quote
      <<","<<fXlowNDC
      <<","<<fYlowNDC
      <<","<<fXlowNDC+fWNDC
      <<","<<fYlowNDC+fHNDC
      <<");"<<endl;
      out<<"   "<<cname<<"->Draw();"<<endl;
      out<<"   "<<cname<<"->cd();"<<endl;
   }
   out<<"   "<<cname<<"->Range("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2<<");"<<endl;
   TView *view = GetView();
   Double_t rmin[3], rmax[3];
   if (view) {
      view->GetRange(rmin, rmax);
      out<<"   TView *view = new TView(1);"<<endl;
      out<<"   view->SetRange("<<rmin[0]<<","<<rmin[1]<<","<<rmin[2]<<","
                               <<rmax[0]<<","<<rmax[1]<<","<<rmax[2]<<");"<<endl;
   }
   if (GetFillColor() != 19) {
      out<<"   "<<cname<<"->SetFillColor("<<GetFillColor()<<");"<<endl;
   }
   if (GetFillStyle() != 1001) {
      out<<"   "<<cname<<"->SetFillStyle("<<GetFillStyle()<<");"<<endl;
   }
   if (GetBorderMode() != 1) {
      out<<"   "<<cname<<"->SetBorderMode("<<GetBorderMode()<<");"<<endl;
   }
   if (GetBorderSize() != 4) {
      out<<"   "<<cname<<"->SetBorderSize("<<GetBorderSize()<<");"<<endl;
   }
   if (GetLogx()) {
      out<<"   "<<cname<<"->SetLogx();"<<endl;
   }
   if (GetLogy()) {
      out<<"   "<<cname<<"->SetLogy();"<<endl;
   }
   if (GetLogz()) {
      out<<"   "<<cname<<"->SetLogz();"<<endl;
   }
   if (GetGridx()) {
      out<<"   "<<cname<<"->SetGridx();"<<endl;
   }
   if (GetGridy()) {
      out<<"   "<<cname<<"->SetGridy();"<<endl;
   }
   if (GetTickx()) {
      out<<"   "<<cname<<"->SetTickx();"<<endl;
   }
   if (GetTicky()) {
      out<<"   "<<cname<<"->SetTicky();"<<endl;
   }
   if (GetTheta() != 30) {
      out<<"   "<<cname<<"->SetTheta("<<GetTheta()<<");"<<endl;
   }
   if (GetPhi() != 30) {
      out<<"   "<<cname<<"->SetPhi("<<GetPhi()<<");"<<endl;
   }
   if (TMath::Abs(fLeftMargin-0.1) > 0.01) {
      out<<"   "<<cname<<"->SetLeftMargin("<<GetLeftMargin()<<");"<<endl;
   }
   if (TMath::Abs(fRightMargin-0.1) > 0.01) {
      out<<"   "<<cname<<"->SetRightMargin("<<GetRightMargin()<<");"<<endl;
   }
   if (TMath::Abs(fTopMargin-0.1) > 0.01) {
      out<<"   "<<cname<<"->SetTopMargin("<<GetTopMargin()<<");"<<endl;
   }
   if (TMath::Abs(fBottomMargin-0.1) > 0.01) {
      out<<"   "<<cname<<"->SetBottomMargin("<<GetBottomMargin()<<");"<<endl;
   }

   if (GetFrameFillColor() != GetFillColor()) {
      out<<"   "<<cname<<"->SetFrameFillColor("<<GetFrameFillColor()<<");"<<endl;
   }
   if (GetFrameFillStyle() != 1001) {
      out<<"   "<<cname<<"->SetFrameFillStyle("<<GetFrameFillStyle()<<");"<<endl;
   }
   if (GetFrameLineStyle() != 1) {
      out<<"   "<<cname<<"->SetFrameLineStyle("<<GetFrameLineStyle()<<");"<<endl;
   }
   if (GetFrameLineColor() != 1) {
      out<<"   "<<cname<<"->SetFrameLineColor("<<GetFrameLineColor()<<");"<<endl;
   }
   if (GetFrameLineWidth() != 1) {
      out<<"   "<<cname<<"->SetFrameLineWidth("<<GetFrameLineWidth()<<");"<<endl;
   }
   if (GetFrameBorderMode() != 1) {
      out<<"   "<<cname<<"->SetFrameBorderMode("<<GetFrameBorderMode()<<");"<<endl;
   }
   if (GetFrameBorderSize() != 1) {
         out<<"   "<<cname<<"->SetFrameBorderSize("<<GetFrameBorderSize()<<");"<<endl;
   }

   TFrame *frame = fFrame;
   if (!frame) frame = (TFrame*)GetPrimitive("TFrame");
   if (frame) {
      if (frame->GetFillColor() != GetFillColor()) {
         out<<"   "<<cname<<"->SetFrameFillColor("<<frame->GetFillColor()<<");"<<endl;
      }
      if (frame->GetFillStyle() != 1001) {
         out<<"   "<<cname<<"->SetFrameFillStyle("<<frame->GetFillStyle()<<");"<<endl;
      }
      if (frame->GetLineStyle() != 1) {
         out<<"   "<<cname<<"->SetFrameLineStyle("<<frame->GetLineStyle()<<");"<<endl;
      }
      if (frame->GetLineColor() != 1) {
         out<<"   "<<cname<<"->SetFrameLineColor("<<frame->GetLineColor()<<");"<<endl;
      }
      if (frame->GetLineWidth() != 1) {
         out<<"   "<<cname<<"->SetFrameLineWidth("<<frame->GetLineWidth()<<");"<<endl;
      }
      if (frame->GetBorderMode() != 1) {
         out<<"   "<<cname<<"->SetFrameBorderMode("<<frame->GetBorderMode()<<");"<<endl;
      }
      if (frame->GetBorderSize() != 1) {
         out<<"   "<<cname<<"->SetFrameBorderSize("<<frame->GetBorderSize()<<");"<<endl;
      }
   }

   TIter next(GetListOfPrimitives());
   TObject *obj;

   while ((obj = next()))
         obj->SavePrimitive(out, (Option_t *)next.GetOption());
   out<<"   "<<cname<<"->Modified();"<<endl;
   out<<"   "<<GetMother()->GetName()<<"->cd();"<<endl;
   if (padsav) padsav->cd();
}

//______________________________________________________________________________
void TPad::SetFixedAspectRatio(Bool_t fixed)
{
   // Fix pad aspect ratio to current value if fixed is true.

   if (fixed) {
      if (!fFixedAspectRatio) {
         if (fHNDC != 0.)
            fAspectRatio = fWNDC / fHNDC;
         else {
            Error("SetAspectRatio", "cannot fix aspect ratio, height of pad is 0");
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
void TPad::SetEditable(Bool_t mode)
{
   // Set pad editable yes/no
   // If a pad is not editable, one cannot modify the pad and its objects
   // via the mouse.

   fEditable = mode;

   TObject *obj;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         TPad *pad = (TPad*)obj;
         pad->SetEditable(mode);
      }
   }
}

//______________________________________________________________________________
void TPad::SetFillStyle(Style_t fstyle)
{
   // Overrride TAttFill::FillStyle for TPad because we want to handle style=0
   // as style 4000.

   if (fstyle == 0) fstyle = 4000;
   TAttFill::SetFillStyle(fstyle);
}

//______________________________________________________________________________
void TPad::SetLogx(Int_t value)
{
//*-*-*-*-*-*-*-*-*Set Lin/Log scale for X
//*-*              ========================
   if (value) fLogx = 1;
   else       fLogx = 0;
   delete fView; fView=0;
}

//______________________________________________________________________________
void TPad::SetLogy(Int_t value)
{
//*-*-*-*-*-*-*-*-*Set Lin/Log scale for Y
//*-*              ========================
   if (value) fLogy = 1;
   else       fLogy = 0;
   delete fView; fView=0;
}

//______________________________________________________________________________
void TPad::SetLogz(Int_t value)
{
//*-*-*-*-*-*-*-*-*Set Lin/Log scale for Z
//*-*              ========================
   if (value) fLogz = 1;
   else       fLogz = 0;
   delete fView; fView=0;
}

//______________________________________________________________________________
void TPad::SetPad(Double_t xlow, Double_t ylow, Double_t xup, Double_t yup)
{
   // Set canvas range for pad and resize the pad. If the aspect ratio
   // was fixed before the call it will be un-fixed.

   //if (!IsEditable()) return;

   // Reorder points to make sure xlow,ylow is bottom left point and
   // xup,yup is top right point.
   if (xup < xlow) {
      Double_t x = xlow;
      xlow = xup;
      xup  = x;
   }
   if (yup < ylow) {
      Double_t y = ylow;
      ylow = yup;
      yup  = y;
   }

   fXlowNDC = xlow;
   fYlowNDC = ylow;
   fWNDC    = xup - xlow;
   fHNDC    = yup - ylow;

   SetFixedAspectRatio(kFALSE);

   ResizePad();
}

//______________________________________________________________________________
void TPad::SetPad(const char *name, const char *title,
                  Double_t xlow, Double_t ylow, Double_t xup, Double_t yup,
                  Color_t color, Short_t bordersize, Short_t bordermode)
{
//*-*-*-*-*-*-*-*-*Set all pad parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ======================
   //if (!IsEditable()) return;

   fName  = name;
   fTitle = title;
   SetFillStyle(1001);
   SetBottomMargin(gStyle->GetPadBottomMargin());
   SetTopMargin(gStyle->GetPadTopMargin());
   SetLeftMargin(gStyle->GetPadLeftMargin());
   SetRightMargin(gStyle->GetPadRightMargin());
   if (color >= 0)   SetFillColor(color);
   else              SetFillColor(gStyle->GetPadColor());
   if (bordersize <  0) fBorderSize = gStyle->GetPadBorderSize();
   else                 fBorderSize = bordersize;
   if (bordermode < -1) fBorderMode = gStyle->GetPadBorderMode();
   else                 fBorderMode = bordermode;

   SetPad(xlow, ylow, xup, yup);
}

//______________________________________________________________________________
void TPad::SetAttFillPS(Color_t color, Style_t style)
{
//*-*-*-*-*-*-*-*-*Set postscript fill area attributes*-*-*-*-*-*-*-*-*-*-*
//*-*              ===================================

   if (gVirtualPS) {
      gVirtualPS->SetFillColor(color);
      gVirtualPS->SetFillStyle(style);
   }
}

//______________________________________________________________________________
void TPad::SetAttLinePS(Color_t color, Style_t style, Width_t lwidth)
{
//*-*-*-*-*-*-*-*-*Set postscript line attributes*-*-*-*-*-*-*-*-*-*-*
//*-*              ==============================

   if (gVirtualPS) {
      gVirtualPS->SetLineColor(color);
      gVirtualPS->SetLineStyle(style);
      gVirtualPS->SetLineWidth(lwidth);
   }
}

//______________________________________________________________________________
void TPad::SetAttMarkerPS(Color_t color, Style_t style, Size_t msize)
{
//*-*-*-*-*-*-*-*-*Set postscript marker attributes*-*-*-*-*-*-*-*-*-*-*
//*-*              ================================

   if (gVirtualPS) {
      gVirtualPS->SetMarkerColor(color);
      gVirtualPS->SetMarkerStyle(style);
      gVirtualPS->SetMarkerSize(msize);
   }
}

//______________________________________________________________________________
void TPad::SetAttTextPS(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize)
{
//*-*-*-*-*-*-*-*-*Set postscript text attributes*-*-*-*-*-*-*-*-*-*-*
//*-*              ==============================

   if (gVirtualPS) {
      gVirtualPS->SetTextAlign(align);
      gVirtualPS->SetTextAngle(angle);
      gVirtualPS->SetTextColor(color);
      gVirtualPS->SetTextFont(font);
      if (font%10 > 2) {
         gVirtualPS->SetTextSize(tsize);
         Float_t dy = AbsPixeltoY(0) - AbsPixeltoY(Int_t(tsize));
         tsize = dy/(fY2-fY1);
      }
      gVirtualPS->SetTextSize(tsize);
   }
}

//______________________________________________________________________________
void TPad::SetCrosshair(Int_t crhair)
{
   // set crosshair active/inactive
   // if crhair != 0, a crosshair will be drawn in the pad and its subpads

   fCrosshair = crhair;
   fCrosshairPos = 0;

   TObject *obj;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         TPad *pad = (TPad*)obj;
         pad->SetCrosshair(crhair);
      }
   }
}

//______________________________________________________________________________
void TPad::SetMaxPickDistance(Int_t maxPick)
{
   // static function to set the maximum Pick Distance fgMaxPickDistance
   // This parameter is used in TPad::Pick to select an object if
   // its DistancetoPrimitive returns a value < fgMaxPickDistance
   // The default value is 5 pixels. Setting a smaller value will make
   // picking more precise but also more difficult
   
   fgMaxPickDistance = maxPick;
}

//______________________________________________________________________________
void TPad::SetToolTipText(const char *text, Long_t delayms)
{
   // Set tool tip text associated with this pad. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with
   // text = 0.

   if (fTip) {
      DeleteToolTip(fTip);
      fTip = 0;
   }

   if (text && strlen(text))
      fTip = CreateToolTip((TBox*)0, text, delayms);
}

//_______________________________________________________________________
void TPad::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================

   UInt_t R__s, R__c;
   Int_t nch, nobjects;
   Float_t single;
   TObject *obj;
   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 5) {
         if (!gPad) gPad = new TCanvas(GetName());
         TPad *padsave = (TPad*)gPad;
         fMother = (TPad*)gPad;
         if (fMother)  fCanvas = fMother->GetCanvas();
         gPad      = this;
         fPixmapID = -1;      // -1 means pixmap will be created by ResizePad()
         readLevel++;
         gROOT->SetReadingObject(kTRUE);

         TPad::Class()->ReadBuffer(b, this, v, R__s, R__c);

         fModified = kTRUE;
         fPadPointer = 0;
         readLevel--;
         if (readLevel == 0 && IsA() == TPad::Class()) ResizePad();
         gROOT->SetReadingObject(kFALSE);
         gPad = padsave;
         return;
      }

      //====process old versions before automatic schema evolution
      if (v < 5) {   //old TPad in single precision
         if (v < 3) {   //old TPad derived from TWbox
            b.ReadVersion();   //      TVirtualPad::Streamer(b)
            b.ReadVersion();   //      TWbox::Streamer(b)
            b.ReadVersion();   //      TBox::Streamer(b)
            TObject::Streamer(b);
            TAttLine::Streamer(b);
            TAttFill::Streamer(b);
            b >> single; fX1 = single;
            b >> single; fY1 = single;
            b >> single; fX2 = single;
            b >> single; fY2 = single;
            b >> fBorderSize;
            b >> fBorderMode;
            TAttPad::Streamer(b);
         } else {  //new TPad
            TVirtualPad::Streamer(b);
            TAttPad::Streamer(b);
            b >> single; fX1 = single;
            b >> single; fY1 = single;
            b >> single; fX2 = single;
            b >> single; fY2 = single;
            b >> fBorderSize;
            b >> fBorderMode;
         }
         b >> fLogx;
         b >> fLogy;
         b >> fLogz;
         b >> single; fXtoAbsPixelk = single;
         b >> single; fXtoPixelk    = single;
         b >> single; fXtoPixel     = single;
         b >> single; fYtoAbsPixelk = single;
         b >> single; fYtoPixelk    = single;
         b >> single; fYtoPixel     = single;
         b >> single; fUtoAbsPixelk = single;
         b >> single; fUtoPixelk    = single;
         b >> single; fUtoPixel     = single;
         b >> single; fVtoAbsPixelk = single;
         b >> single; fVtoPixelk    = single;
         b >> single; fVtoPixel     = single;
         b >> single; fAbsPixeltoXk = single;
         b >> single; fPixeltoXk    = single;
         b >> single; fPixeltoX     = single;
         b >> single; fAbsPixeltoYk = single;
         b >> single; fPixeltoYk    = single;
         b >> single; fPixeltoY     = single;
         b >> single; fXlowNDC      = single;
         b >> single; fYlowNDC      = single;
         b >> single; fWNDC         = single;
         b >> single; fHNDC         = single;
         b >> single; fAbsXlowNDC   = single;
         b >> single; fAbsYlowNDC   = single;
         b >> single; fAbsWNDC      = single;
         b >> single; fAbsHNDC      = single;
         b >> single; fUxmin        = single;
         b >> single; fUymin        = single;
         b >> single; fUxmax        = single;
         b >> single; fUymax        = single;
      } else {
         TVirtualPad::Streamer(b);
         TAttPad::Streamer(b);
         b >> fX1;
         b >> fY1;
         b >> fX2;
         b >> fY2;
         b >> fBorderSize;
         b >> fBorderMode;
         b >> fLogx;
         b >> fLogy;
         b >> fLogz;
         b >> fXtoAbsPixelk;
         b >> fXtoPixelk;
         b >> fXtoPixel;
         b >> fYtoAbsPixelk;
         b >> fYtoPixelk;
         b >> fYtoPixel;
         b >> fUtoAbsPixelk;
         b >> fUtoPixelk;
         b >> fUtoPixel;
         b >> fVtoAbsPixelk;
         b >> fVtoPixelk;
         b >> fVtoPixel;
         b >> fAbsPixeltoXk;
         b >> fPixeltoXk;
         b >> fPixeltoX;
         b >> fAbsPixeltoYk;
         b >> fPixeltoYk;
         b >> fPixeltoY;
         b >> fXlowNDC;
         b >> fYlowNDC;
         b >> fWNDC;
         b >> fHNDC;
         b >> fAbsXlowNDC;
         b >> fAbsYlowNDC;
         b >> fAbsWNDC;
         b >> fAbsHNDC;
         b >> fUxmin;
         b >> fUymin;
         b >> fUxmax;
         b >> fUymax;
      }

      if (!gPad) gPad = new TCanvas(GetName());
      if (readLevel == 0) fMother = (TPad*)gROOT->GetSelectedPad();
      else                fMother = (TPad*)gPad;
      if (!fMother) fMother = (TPad*)gPad;
      if (fMother)  fCanvas = fMother->GetCanvas();
      gPad      = fMother;
      fPixmapID = -1;      // -1 means pixmap will be created by ResizePad()
//-------------------------
// read objects and their drawing options
//      b >> fPrimitives;
      readLevel++;
      gROOT->SetReadingObject(kTRUE);
      fPrimitives = new TList;
      b >> nobjects;
      if (nobjects > 0) {
         TPad *padsav = (TPad*)gPad;
         gPad = this;
         char drawoption[64];
         for (Int_t i = 0; i < nobjects; i++) {
            b >> obj;
            b >> nch;
            b.ReadFastArray(drawoption,nch);
            fPrimitives->AddLast(obj, drawoption);
            gPad = this; // gPad may be modified in b >> obj if obj is a pad
         }
         gPad = padsav;
      }
      readLevel--;
      gROOT->SetReadingObject(kFALSE);
//-------------------------
      if (v > 3) {
         b >> fExecs;
      }
      fName.Streamer(b);
      fTitle.Streamer(b);
      b >> fPadPaint;
      fModified = kTRUE;
      b >> fGridx;
      b >> fGridy;
      b >> fFrame;
      b >> fView;
      if (v < 5) {
         b >> single; fTheta = single;
         b >> single; fPhi   = single;
      } else {
         b >> fTheta;
         b >> fPhi;
      }
      fPadPointer = 0;
      b >> fNumber;
      b >> fAbsCoord;
      if (v > 1) {
         b >> fTickx;
         b >> fTicky;
      } else {
         fTickx = fTicky = 0;
      }
      if (readLevel == 0 && IsA() == TPad::Class()) ResizePad();
      b.CheckByteCount(R__s, R__c, TPad::IsA());
      //====end of old versions

   } else {
      TPad::Class()->WriteBuffer(b,this);
   }
}

//______________________________________________________________________________
void TPad::UseCurrentStyle()
{
//*-*-*-*-*-*Force a copy of current style for all objects in pad*-*-*-*-*
//*-*        ====================================================

   SetFillColor(gStyle->GetPadColor());
   SetBottomMargin(gStyle->GetPadBottomMargin());
   SetTopMargin(gStyle->GetPadTopMargin());
   SetLeftMargin(gStyle->GetPadLeftMargin());
   SetRightMargin(gStyle->GetPadRightMargin());
   fBorderSize = gStyle->GetPadBorderSize();
   fBorderMode = gStyle->GetPadBorderMode();
   fGridx = gStyle->GetPadGridX();
   fGridy = gStyle->GetPadGridY();
   fTickx = gStyle->GetPadTickX();
   fTicky = gStyle->GetPadTickY();
   fLogx  = gStyle->GetOptLogx();
   fLogy  = gStyle->GetOptLogy();
   fLogz  = gStyle->GetOptLogz();

   TIter next(GetListOfPrimitives());
   TObject *obj;

   while ((obj = next())) {
      obj->UseCurrentStyle();
   }

   TPaveText *stats  = (TPaveText*)FindObject("stats");
   if (stats) {
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetTextFont(gStyle->GetStatFont());
      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      if (!gStyle->GetOptStat()) delete stats;
   }

   TPaveText *title  = (TPaveText*)FindObject("title");
   if (title) {
      title->SetFillColor(gStyle->GetTitleColor());
      title->SetTextFont(gStyle->GetTitleFont());
      title->SetTextColor(gStyle->GetTitleTextColor());
      title->SetBorderSize(gStyle->GetTitleBorderSize());
      if (!gStyle->GetOptTitle()) delete title;
   }
   if (fFrame) {
      fFrame->SetFillColor(gStyle->GetFrameFillColor());
      fFrame->SetFillStyle(gStyle->GetFrameFillStyle());
      fFrame->SetLineColor(gStyle->GetFrameLineColor());
      fFrame->SetLineStyle(gStyle->GetFrameLineStyle());
      fFrame->SetLineWidth(gStyle->GetFrameLineWidth());
      fFrame->SetBorderSize(gStyle->GetFrameBorderSize());
      fFrame->SetBorderMode(gStyle->GetFrameBorderMode());
   }

   Modified();
}

//______________________________________________________________________________
TObject *TPad::WaitPrimitive(const char *pname, const char *emode)
{
   // Loop and sleep until a primitive with name=pname
   // is found in the pad.
   // If emode is given, the editor is automatically set to emode, ie
   // it is not required to have the editor control bar.
   // The possible values for emode are:
   //  emode = "" (default). User will select the mode via the editor bar
   //        = "Arc", "Line", "Arrow", "Button", "Diamond", "Ellipse",
   //        = "Pad","pave", "PaveLabel","PaveText", "PavesText",
   //        = "PolyLine", "CurlyLine", "CurlyArc", "Text", "Marker", "CutG"
   // if emode is specified and it is not valid, "PolyLine" is assumed.
   // if emode is not specified or ="", an attempt is to use pname[1...]
   // for example if pname="TArc", emode="Arc" will be assumed.
   // When this function is called within a macro, the macro execution
   // is suspended until a primitive corresponding to the arguments
   // is found in the pad.
   // If CRTL/C is typed in the pad, the function returns 0.
   // While this function is executing, one can use the mouse, interact
   // with the graphics pads, use the Inspector, Browser, TreeViewer, etc.
   // Examples:
   //   c1.WaitPrimitive();      // Return the first created primitive
   //                            // whatever it is.
   //   c1.WaitPrimitive("ggg"); // Set the editor in mode "PolyLine/Graph"
   //                            // Create a polyline, then using the context
   //                            // menu item "SetName", change the name
   //                            // of the created TGraph to "ggg"
   //   c1.FindObject("Arc");    // Set the editor in mode "Arc". Returns
   //                            // as soon as a TArc object is created.
   //   c1.FindObject("lat","Text"); // Set the editor in Text/Latex mode.
   //                            // Create a text object, then Set its name to "lat"
   //
   // The following macro waits for 10 primitives of any type to be created.
   //{
   //   TCanvas c1("c1");
   //   TObject *obj;
   //   for (Int_t i=0;i<10;i++) {
   //      obj = gPad->WaitPrimitive();
   //      if (!obj) break;
   //      printf("Loop i=%d, found objIsA=%s, name=%s\n",
   //         i,obj->ClassName(),obj->GetName());
   //   }
   //}

   if (strlen(emode)) gROOT->SetEditorMode(emode);
   if (gROOT->GetEditorMode() == 0 && strlen(pname) > 2) gROOT->SetEditorMode(&pname[1]);

   TObject *oldlast = gPad->GetListOfPrimitives()->Last();
   TObject *obj = 0;
   Bool_t testlast = kFALSE;
   if (strlen(pname) == 0 && strlen(emode) == 0) testlast = kTRUE;
   if (testlast) gROOT->SetEditorMode();
   while (!gSystem->ProcessEvents()) {
      obj = FindObject(pname);
      if (obj) {
         gROOT->SetEditorMode();
         return obj;
      }
      if (testlast) {
         obj = gPad->GetListOfPrimitives()->Last();
         if (obj != oldlast) return obj;
      }
   }
   return 0;
}

//______________________________________________________________________________
TObject *TPad::CreateToolTip(const TBox *box, const char *text, Long_t delayms)
{
   // Create a tool tip and return its pointer.

#ifndef WIN32
   if (gPad->IsBatch()) return 0;
   // return new TGToolTip(box, text, delayms);
   return (TObject*)gROOT->ProcessLineFast(Form("new TGToolTip((TBox*)0x%lx,\"%s\",%d)",
                                           (Long_t)box,text,(Int_t)delayms));
#else
#ifdef GDK_WIN32
   if (gPad->IsBatch()) return 0;
   // return new TGToolTip(box, text, delayms);
   return (TObject*)gROOT->ProcessLineFast(Form("new TGToolTip((TBox*)0x%lx,\"%s\",%d)",
                                           (Long_t)box,text,(Int_t)delayms));
#else
   return 0;
#endif
#endif
}

//______________________________________________________________________________
void TPad::DeleteToolTip(TObject *tip)
{
   // Delete tool tip object.

#ifndef WIN32
   // delete tip;
   if (!tip) return;
   gROOT->ProcessLineFast(Form("delete (TGToolTip*)0x%lx", (Long_t)tip));
#else
#ifdef GDK_WIN32
   if (!tip) return;
   gROOT->ProcessLineFast(Form("delete (TGToolTip*)0x%lx", (Long_t)tip));
#endif
#endif
}

//______________________________________________________________________________
void TPad::ResetToolTip(TObject *tip)
{
   // Reset tool tip, i.e. within time specified in CreateToolTip the
   // tool tip will pop up.

#ifndef WIN32
   if (!tip) return;
   // tip->Reset(this);
   gROOT->ProcessLineFast(Form("((TGToolTip*)0x%lx)->Reset((TPad*)0x%lx)",
                          (Long_t)tip,(Long_t)this));
#else
#ifdef GDK_WIN32
   if (!tip) return;
   // tip->Reset(this);
   gROOT->ProcessLineFast(Form("((TGToolTip*)0x%lx)->Reset((TPad*)0x%lx)",
                          (Long_t)tip,(Long_t)this));
#endif
#endif
}

//______________________________________________________________________________
void TPad::CloseToolTip(TObject *tip)
{
   // Hide tool tip.

#ifndef WIN32
   if (!tip) return;
   // tip->Hide();
   gROOT->ProcessLineFast(Form("((TGToolTip*)0x%lx)->Hide()",(Long_t)tip));
#else
#ifdef GDK_WIN32
   if (!tip) return;
   // tip->Hide();
   gROOT->ProcessLineFast(Form("((TGToolTip*)0x%lx)->Hide()",(Long_t)tip));
#endif
#endif
}

//______________________________________________________________________________
void TPad::x3d(Option_t *option)
{
//*-*-*-*-*-*Invokes the x3d package to view the content of a pad*-*-*-*-*-*-*
//*-*        ====================================================

#ifndef WIN32
   if (!strcasecmp(option, "OPENGL"))
#endif
   {
      // Disconnect the previous 3D context
      if (fPadView3D)
         fPadView3D->SetPad(0);
      fPadView3D = gVirtualGL->CreatePadGLView(this);
      if (!fPadView3D) {
         gROOT->LoadClass("TGLKernel", "RGL");
         fPadView3D = gVirtualGL->CreatePadGLView(this);
         if (!fPadView3D)
            Error("x3d", "OpenGL shared libraries not loaded");
      }
      if (fPadView3D) {
          Modified();
          Update();
      }
      return;
   }

#ifndef WIN32
   if (gROOT->LoadClass("TViewerX3D","X3d")) return;

   gROOT->ProcessLine(Form("TViewerX3D *R__x3d = new TViewerX3D((TVirtualPad*)0x%lx,\"%s\")",
                      (Long_t)this, option));
#endif
}
