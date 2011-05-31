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
#include "TError.h"
#include "TMath.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TH1.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TClassTable.h"
#include "TVirtualPS.h"
#include "TVirtualX.h"
#include "TVirtualViewer3D.h"
#include "TView.h"
#include "TPoint.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "THStack.h"
#include "TPaveText.h"
#include "TGroupButton.h"
#include "TBrowser.h"
#include "TVirtualGL.h"
#include "TString.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TFrame.h"
#include "TExec.h"
#include "TDatime.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TPluginManager.h"
#include "TEnv.h"
#include "TImage.h"
#include "TViewer3DPad.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TCreatePrimitives.h"
#include "TLegend.h"
#include "TAtt3D.h"
#include "TObjString.h"
#include "TApplication.h"
#include "TVirtualPadPainter.h"

static Int_t gReadLevel = 0;

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
//  For example, in case of a histogram, the histogram.Draw() operation
//  only stores a reference to the histogram object and not a graphical
//  representation of this histogram.
//  When the mouse is used to change (say the bin content), the bin content
//  of the original histogram is changed !!
//
//  The convention used in ROOT is that a Draw operation only adds
//  a reference to the object. The effective drawing is performed
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
<img src="gif/canvas.gif">
*/
//End_Html
//


//______________________________________________________________________________
TPad::TPad()
{
   // Pad default constructor.

   fModified   = kTRUE;
   fTip        = 0;
   fPadPointer = 0;
   fPrimitives = 0;
   fExecs      = 0;
   fCanvas     = 0;
   fMother     = 0;
   fPadPaint   = 0;
   fPixmapID   = -1;
   fGLDevice   = -1;
   fCopyGLDevice = kFALSE;
   fEmbeddedGL = kFALSE;
   fTheta      = 30;
   fPhi        = 30;
   fNumber     = 0;
   fAbsCoord   = kFALSE;
   fEditable   = kTRUE;
   fCrosshair  = 0;
   fCrosshairPos = 0;
   fPadView3D  = 0;
   fMother     = (TPad*)gPad;

   fAbsHNDC      = 0.;
   fAbsPixeltoXk = 0.;
   fAbsPixeltoYk = 0.;
   fAbsWNDC      = 0.;
   fAbsXlowNDC   = 0.;
   fAbsYlowNDC   = 0.;
   fBorderMode   = 0;
   fBorderSize   = 0;
   fPixeltoX     = 0;
   fPixeltoXk    = 0.;
   fPixeltoY     = 0.;
   fPixeltoYk    = 0.;
   fUtoAbsPixelk = 0.;
   fUtoPixel     = 0.;
   fUtoPixelk    = 0.;
   fVtoAbsPixelk = 0.;
   fVtoPixel     = 0.;
   fVtoPixelk    = 0.;
   fXtoAbsPixelk = 0.;
   fXtoPixel     = 0.;
   fXtoPixelk    = 0.;
   fYtoAbsPixelk = 0.;
   fYtoPixel     = 0.;
   fYtoPixelk   = 0.;

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

   // Set default world coordinates to NDC [0,1]
   fX1 = 0;
   fX2 = 1;
   fY1 = 0;
   fY2 = 1;

   // Set default pad range
   fXlowNDC = 0;
   fYlowNDC = 0;
   fWNDC    = 1;
   fHNDC    = 1;

   fViewer3D = 0;
   SetBit(kMustCleanup);

   // the following line is temporarily disabled. It has side effects
   // when the pad is a TDrawPanelHist or a TFitPanel.
   // the line was supposed to fix a problem with DrawClonePad
   //   gROOT->SetSelectedPad(this);
}


//______________________________________________________________________________
TPad::TPad(const char *name, const char *title, Double_t xlow,
           Double_t ylow, Double_t xup, Double_t yup,
           Color_t color, Short_t bordersize, Short_t bordermode)
          : TVirtualPad(name,title,xlow,ylow,xup,yup,color,bordersize,bordermode)
{
   // Pad constructor.
   //
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

   fModified   = kTRUE;
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
   fPadView3D  = 0;
   fPixmapID   = -1;      // -1 means pixmap will be created by ResizePad()
   fCopyGLDevice = kFALSE;
   fEmbeddedGL = kFALSE;
   fNumber     = 0;
   fAbsCoord   = kFALSE;
   fEditable   = kTRUE;
   fCrosshair  = 0;
   fCrosshairPos = 0;

   fFixedAspectRatio = kFALSE;
   fAspectRatio      = 0.;

   fViewer3D = 0;

   fGLDevice = fCanvas->GetGLDevice();
   // Set default world coordinates to NDC [0,1]
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

   // Set pad parameters and Compute conversion coeeficients
   SetPad(name, title, xlow, ylow, xup, yup, color, bordersize, bordermode);
   Range(0, 0, 1, 1);
   SetBit(kMustCleanup);
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
   // Pad destructor.

   if (!TestBit(kNotDeleted)) return;
   Close();
   CloseToolTip(fTip);
   DeleteToolTip(fTip);
   SafeDelete(fPrimitives);
   SafeDelete(fExecs);
   delete fViewer3D;
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

   if (GetCrosshair()) DrawCrosshair();

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
   // Browse pad.

   cd();
   if (fPrimitives) fPrimitives->Browse(b);
}


//______________________________________________________________________________
TLegend *TPad::BuildLegend(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
                           const char* title)
{
   // Build a legend from the graphical objects in the pad
   //
   // A simple method to to build automatically a TLegend from the primitives in
   // a TPad. Only those deriving from TAttLine, TAttMarker and TAttFill are
   // added, excluding TPave and TFrame derived classes.
   // x1, y1, x2, y2 are the Tlegend coordinates.
   // title is the legend title. By default it is " ".
   //
   // If the pad contains some TMultiGraph or THStack the individual graphs or
   // histograms in them are added to the TLegend.

   TList *lop=GetListOfPrimitives();
   if (!lop) return 0;
   TLegend *leg=0;
   TIter next(lop);
   TString mes;
   TObject *o=0;
   while( (o=next()) ) {
      if((o->InheritsFrom(TAttLine::Class()) || o->InheritsFrom(TAttMarker::Class()) ||
          o->InheritsFrom(TAttFill::Class())) &&
         ( !(o->InheritsFrom(TFrame::Class())) && !(o->InheritsFrom(TPave::Class())) )) {
            if (!leg) leg = new TLegend(x1, y1, x2, y2, title);
            if (o->InheritsFrom(TNamed::Class()) && strlen(((TNamed *)o)->GetTitle()))
               mes = ((TNamed *)o)->GetTitle();
            else if (strlen(o->GetName()))
               mes = o->GetName();
            else
               mes = o->ClassName();
            TString opt("");
            if (o->InheritsFrom(TAttLine::Class()))   opt += "l";
            if (o->InheritsFrom(TAttMarker::Class())) opt += "p";
            if (o->InheritsFrom(TAttFill::Class()))   opt += "f";
            leg->AddEntry(o,mes.Data(),opt.Data());
      } else if ( o->InheritsFrom(TMultiGraph::Class() ) ) {
         if (!leg) leg = new TLegend(x1, y1, x2, y2, title);
         TList * grlist = ((TMultiGraph *)o)->GetListOfGraphs();
         TIter nextgraph(grlist);
         TGraph * gr;
         TObject * obj;
         while ((obj = nextgraph())) {
            gr = (TGraph*) obj;
            if      (strlen(gr->GetTitle())) mes = gr->GetTitle();
            else if (strlen(gr->GetName()))  mes = gr->GetName();
            else                             mes = gr->ClassName();
            leg->AddEntry( obj, mes.Data(), "lpf" );
         }
      } else if ( o->InheritsFrom(THStack::Class() ) ) {
         if (!leg) leg = new TLegend(x1, y1, x2, y2, title);
         TList * hlist = ((THStack *)o)->GetHists();
         TIter nexthist(hlist);
         TH1 * hist;
         TObject * obj;
         while ((obj = nexthist())) {
            hist = (TH1*) obj;
            if      (strlen(hist->GetTitle())) mes = hist->GetTitle();
            else if (strlen(hist->GetName()))  mes = hist->GetName();
            else                               mes = hist->ClassName();
            leg->AddEntry( obj, mes.Data(), "lpf" );
         }
      }
   }
   if (leg) {
      TVirtualPad *gpadsave;
      gpadsave = gPad;
      this->cd();
      leg->Draw();
      gpadsave->cd();
   } else {
      Info("BuildLegend(void)","No object to build a TLegend.");
   }
   return leg;
}


//______________________________________________________________________________
TVirtualPad *TPad::cd(Int_t subpadnumber)
{
   // Set Current pad.
   // When a canvas/pad is divided via TPad::Divide, one can directly
   //  set the current path to one of the subdivisions.
   //  See TPad::Divide for the convention to number subpads.
   //  Returns the new current pad, or 0 in case of failure.
   //  For example:
   //    c1.Divide(2,3); // create 6 pads (2 divisions along x, 3 along y).
   //    To set the current pad to the bottom right pad, do
   //    c1.cd(6);
   //  Note1:  c1.cd() is equivalent to c1.cd(0) and sets the current pad
   //          to c1 itself.
   //  Note2:  after a statement like c1.cd(6), the global variable gPad
   //          points to the current pad. One can use gPad to set attributes
   //          of the current pad.
   //  Note3:  One can get a pointer to one of the sub-pads of pad with:
   //          TPad *subpad = (TPad*)pad->GetPad(subpadnumber);

   if (!subpadnumber) {
      gPad = this;
      if (!gPad->IsBatch()) GetPainter()->SelectDrawable(fPixmapID);
      return gPad;
   }

   TObject *obj;
   if (!fPrimitives) fPrimitives = new TList;
   TIter    next(fPrimitives);
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         Int_t n = ((TPad*)obj)->GetNumber();
         if (n == subpadnumber) {
            return ((TPad*)obj)->cd();
         }
      }
   }
   return 0;
}


//______________________________________________________________________________
void TPad::Clear(Option_t *option)
{
   // Delete all pad primitives.
   //
   //   If the bit kClearAfterCR has been set for this pad, the Clear function
   //   will execute only after having pressed a CarriageReturn
   //   Set the bit with mypad->SetBit(TPad::kClearAfterCR)

   if (!IsEditable()) return;

   if (!fPadPaint) {
      SafeDelete(fView);
      if (fPrimitives) fPrimitives->Clear(option);
      if (fFrame) {
         if (fFrame->TestBit(kNotDeleted)) delete fFrame;
         fFrame = 0;
      }
   }
   if (fCanvas) fCanvas->Cleared(this);

   cd();

   if (TestBit(kClearAfterCR)) getchar();

   if (!gPad->IsBatch()) GetPainter()->ClearDrawable();
   if (gVirtualPS && gPad == gPad->GetCanvas()) gVirtualPS->NewPage();

   PaintBorder(GetFillColor(), kTRUE);
   fCrosshairPos = 0;
   ResetBit(TGraph::kClipFrame);
}


//___________________________________________________________
Int_t TPad::Clip(Float_t *x, Float_t *y, Float_t xclipl, Float_t yclipb, Float_t xclipr, Float_t yclipt)
{
   // Clipping routine: Cohen Sutherland algorithm.
   //
   //   If Clip ==2 the segment is outside the boundary.
   //   If Clip ==1 the segment has one point outside the boundary.
   //   If Clip ==0 the segment is inside the boundary.
   //
   // _Input parameters:
   //
   //  x[2], y[2] : Segment coordinates
   //  xclipl, yclipb, xclipr, yclipt : Clipping boundary
   //
   // _Output parameters:
   //
   //  x[2], y[2] : New segment coordinates

   const Float_t kP=10000;
   Int_t clip = 0;

   for (Int_t i=0;i<2;i++) {
      if (TMath::Abs(xclipl-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipl;
      if (TMath::Abs(xclipr-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipr;
      if (TMath::Abs(yclipb-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipb;
      if (TMath::Abs(yclipt-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipt;
   }

   // Compute the first endpoint codes.
   Int_t code1 = ClippingCode(x[0],y[0],xclipl,yclipb,xclipr,yclipt);
   Int_t code2 = ClippingCode(x[1],y[1],xclipl,yclipb,xclipr,yclipt);

   Double_t xt=0, yt=0;
   Int_t clipped = 0; //this variable could be used in a future version
   while(code1 + code2) {
      clipped = 1;

      // The line lies entirely outside the clipping boundary
      if (code1&code2) {
         clip = 2;
         return clip;
      }

      // The line is subdivided into several parts
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
   // Clipping routine: Cohen Sutherland algorithm.
   //
   //   If Clip ==2 the segment is outside the boundary.
   //   If Clip ==1 the segment has one point outside the boundary.
   //   If Clip ==0 the segment is inside the boundary.
   //
   // _Input parameters:
   //
   //  x[2], y[2] : Segment coordinates
   //  xclipl, yclipb, xclipr, yclipt : Clipping boundary
   //
   // _Output parameters:
   //
   //  x[2], y[2] : New segment coordinates

   const Double_t kP=10000;
   Int_t clip = 0;

   for (Int_t i=0;i<2;i++) {
      if (TMath::Abs(xclipl-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipl;
      if (TMath::Abs(xclipr-x[i]) <= TMath::Abs(xclipr-xclipl)/kP) x[i] = xclipr;
      if (TMath::Abs(yclipb-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipb;
      if (TMath::Abs(yclipt-y[i]) <= TMath::Abs(yclipt-yclipb)/kP) y[i] = yclipt;
   }

   // Compute the first endpoint codes.
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

      // The line lies entirely outside the clipping boundary
      if (code1&code2) {
         clip = 2;
         return clip;
      }

      // The line is subdivided into several parts
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
   // Compute the endpoint codes for TPad::Clip.

   Int_t code = 0;
   if (x < xcl1) code = code | 0x1;
   if (x > xcl2) code = code | 0x2;
   if (y < ycl1) code = code | 0x4;
   if (y > ycl2) code = code | 0x8;
   return code;
}


//___________________________________________________________
Int_t TPad::ClipPolygon(Int_t n, Double_t *x, Double_t *y, Int_t nn, Double_t *xc, Double_t *yc, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt)
{
   // Clip polygon using the Sutherland-Hodgman algorithm.
   //
   // Input parameters:
   //
   //  n: Number of points in the polygon to be clipped
   //  x[n], y[n] : Polygon do be clipped vertices
   //  xclipl, yclipb, xclipr, yclipt : Clipping boundary
   //
   // Output parameters:
   //
   // nn: number of points in xc and yc
   // xc, yc: clipped polygon vertices. The Int_t returned by this function is
   //         the number of points in the clipped polygon. These vectors must
   //         be allocated by the calling function. A size of 2*n for each is
   //         enough.
   //
   // Sutherland and Hodgman's polygon-clipping algorithm uses a divide-and-conquer
   // strategy: It solves a series of simple and identical problems that, when
   // combined, solve the overall problem. The simple problem is to clip a polygon
   // against a single infinite clip edge. Four clip edges, each defining one boundary
   // of the clip rectangle, successively clip a polygon against a clip rectangle.
   //
   // Steps of Sutherland-Hodgman's polygon-clipping algorithm:
   //
   // * Polygons can be clipped against each edge of the window one at a time.
   //   Windows/edge intersections, if any, are easy to find since the X or Y coordinates
   //   are already known.
   // * Vertices which are kept after clipping against one window edge are saved for
   //   clipping against the remaining edges.
   // * Note that the number of vertices usually changes and will often increases.
   //
   // The clip boundary determines a visible and invisible region. The edges from
   // vertex i to vertex i+1 can be one of four types:
   //
   // * Case 1 : Wholly inside visible region - save endpoint
   // * Case 2 : Exit visible region - save the intersection
   // * Case 3 : Wholly outside visible region - save nothing
   // * Case 4 : Enter visible region - save intersection and endpoint

   Int_t nc, nc2;
   Double_t x1, y1, x2, y2, slope; // Segment to be clipped

   Double_t *xc2 = new Double_t[nn];
   Double_t *yc2 = new Double_t[nn];

   // Clip against the left boundary
   x1 = x[n-1]; y1 = y[n-1];
   nc2 = 0;
   Int_t i;
   for (i=0; i<n; i++) {
      x2 = x[i]; y2 = y[i];
      if (x1 == x2) {
         slope = 0;
      } else {
         slope = (y2-y1)/(x2-x1);
      }
      if (x1 >= xclipl) {
         if (x2 < xclipl) {
            xc2[nc2] = xclipl; yc2[nc2++] = slope*(xclipl-x1)+y1;
         } else {
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      } else {
         if (x2 >= xclipl) {
            xc2[nc2] = xclipl; yc2[nc2++] = slope*(xclipl-x1)+y1;
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   // Clip against the top boundary
   x1 = xc2[nc2-1]; y1 = yc2[nc2-1];
   nc = 0;
   for (i=0; i<nc2; i++) {
      x2 = xc2[i]; y2 = yc2[i];
      if (y1 == y2) {
         slope = 0;
      } else {
         slope = (x2-x1)/(y2-y1);
      }
      if (y1 <= yclipt) {
         if (y2 > yclipt) {
            xc[nc] = x1+(yclipt-y1)*slope; yc[nc++] = yclipt;
         } else {
            xc[nc] = x2; yc[nc++] = y2;
         }
      } else {
         if (y2 <= yclipt) {
            xc[nc] = x1+(yclipt-y1)*slope; yc[nc++] = yclipt;
            xc[nc] = x2; yc[nc++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   // Clip against the right boundary
   x1 = xc[nc-1]; y1 = yc[nc-1];
   nc2 = 0;
   for (i=0; i<nc; i++) {
      x2 = xc[i]; y2 = yc[i];
      if (x1 == x2) {
         slope = 0;
      } else {
         slope = (y2-y1)/(x2-x1);
      }
      if (x1 <= xclipr) {
         if (x2 > xclipr) {
            xc2[nc2] = xclipr; yc2[nc2++] = slope*(xclipr-x1)+y1;
         } else {
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      } else {
         if (x2 <= xclipr) {
            xc2[nc2] = xclipr; yc2[nc2++] = slope*(xclipr-x1)+y1;
            xc2[nc2] = x2; yc2[nc2++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   // Clip against the bottom boundary
   x1 = xc2[nc2-1]; y1 = yc2[nc2-1];
   nc = 0;
   for (i=0; i<nc2; i++) {
      x2 = xc2[i]; y2 = yc2[i];
      if (y1 == y2) {
         slope = 0;
      } else {
         slope = (x2-x1)/(y2-y1);
      }
      if (y1 >= yclipb) {
         if (y2 < yclipb) {
            xc[nc] = x1+(yclipb-y1)*slope; yc[nc++] = yclipb;
         } else {
            xc[nc] = x2; yc[nc++] = y2;
         }
      } else {
         if (y2 >= yclipb) {
            xc[nc] = x1+(yclipb-y1)*slope; yc[nc++] = yclipb;
            xc[nc] = x2; yc[nc++] = y2;
         }
      }
      x1 = x2; y1 = y2;
   }

   delete [] xc2;
   delete [] yc2;

   if (nc < 3) nc =0;
   return nc;
}


//______________________________________________________________________________
void TPad::Close(Option_t *)
{
   // Delete all primitives in pad and pad itself.
   // Pad cannot be used anymore after this call.
   // Emits signal "Closed()".

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

   // emit signal
   if (IsA() != TCanvas::Class())
      Closed();

   if (fPixmapID != -1) {
      if (gPad) {
         if (!gPad->IsBatch()) {
            GetPainter()->SelectDrawable(fPixmapID);
            GetPainter()->DestroyDrawable();
         }
      }
      fPixmapID = -1;

      if (!gROOT->GetListOfCanvases()) return;
      if (fMother == this) {
         gROOT->GetListOfCanvases()->Remove(this);
         return;   // in case of TCanvas
      }

      // remove from the mother's list of primitives
      if (fMother) {
         if (fMother->GetListOfPrimitives())
            fMother->GetListOfPrimitives()->Remove(this);

         if (gPad == this) fMother->cd();
      }

      if (fCanvas->GetPadSave() == this)
         fCanvas->ClearPadSave();
      if (fCanvas->GetSelectedPad() == this)
         fCanvas->SetSelectedPad(0);
      if (fCanvas->GetClickSelectedPad() == this)
         fCanvas->SetClickSelectedPad(0);
   }

   fMother = 0;
   if (gROOT->GetSelectedPad() == this) gROOT->SetSelectedPad(0);
}


//______________________________________________________________________________
void TPad::CopyPixmap()
{
   // Copy the pixmap of the pad to the canvas.

   int px, py;
   XYtoAbsPixel(fX1, fY2, px, py);

   if (fPixmapID != -1)
      GetPainter()->CopyDrawable(fPixmapID, px, py);

   if (this == gPad) HighLight(gPad->GetHighLightColor());
}


//______________________________________________________________________________
void TPad::CopyPixmaps()
{
   // Copy the sub-pixmaps of the pad to the canvas.

   TObject *obj;
   if (!fPrimitives) fPrimitives = new TList;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         ((TPad*)obj)->CopyPixmap();
         ((TPad*)obj)->CopyPixmaps();
      }
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
   // Compute distance from point px,py to a box.
   //
   //  Compute the closest distance of approach from point px,py to the
   //  edges of this pad.
   //  The distance is computed in pixels units.

   Int_t pxl, pyl, pxt, pyt;
   Int_t px1 = gPad->XtoAbsPixel(fX1);
   Int_t py1 = gPad->YtoAbsPixel(fY1);
   Int_t px2 = gPad->XtoAbsPixel(fX2);
   Int_t py2 = gPad->YtoAbsPixel(fY2);
   if (px1 < px2) {pxl = px1; pxt = px2;}
   else           {pxl = px2; pxt = px1;}
   if (py1 < py2) {pyl = py1; pyt = py2;}
   else           {pyl = py2; pyt = py1;}

   // Are we inside the box?
   // ======================
   if ( (px > pxl && px < pxt) && (py > pyl && py < pyt) ) {
      if (GetFillStyle()) return 0;  //*-* if pad is filled
   }

   // Are we on the edges?
   // ====================
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
   // Automatic pad generation by division.
   //
   //  The current canvas is divided in nx by ny equal divisions (pads).
   //  xmargin is the space along x between pads in percent of canvas.
   //  ymargin is the space along y between pads in percent of canvas.
   //    (see Note3 below for the special case xmargin <=0 and ymargin <=0)
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
   //  Note3:  in case xmargin <=0 and ymargin <= 0, there is no space
   //          between pads. The current pad margins are recomputed to
   //          optimize the layout.

   if (!IsEditable()) return;


   if (gThreadXAR) {
      void *arr[7];
      arr[1] = this; arr[2] = (void*)&nx;arr[3] = (void*)& ny;
      arr[4] = (void*)&xmargin; arr[5] = (void *)& ymargin; arr[6] = (void *)&color;
      if ((*gThreadXAR)("PDCD", 7, arr, 0)) return;
   }

   TPad *padsav = (TPad*)gPad;
   cd();
   if (nx <= 0) nx = 1;
   if (ny <= 0) ny = 1;
   Int_t ix,iy;
   Double_t x1,y1,x2,y2;
   Double_t dx,dy;
   TPad *pad;
   Int_t nchname  = strlen(GetName())+6;
   Int_t nchtitle = strlen(GetTitle())+6;
   char *name  = new char [nchname];
   char *title = new char [nchtitle];
   Int_t n = 0;
   if (color == 0) color = GetFillColor();
   if (xmargin > 0 && ymargin > 0) {
      //general case
      dy = 1/Double_t(ny);
      dx = 1/Double_t(nx);
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
            snprintf(name,nchname,"%s_%d",GetName(),n);
            pad = new TPad(name,name,x1,y1,x2,y2,color);
            pad->SetNumber(n);
            pad->Draw();
         }
      }
   } else {
      // special case when xmargin <= 0 && ymargin <= 0
      Double_t xl = GetLeftMargin();
      Double_t xr = GetRightMargin();
      Double_t yb = GetBottomMargin();
      Double_t yt = GetTopMargin();
      xl /= (1-xl+xr)*nx;
      xr /= (1-xl+xr)*nx;
      yb /= (1-yb+yt)*ny;
      yt /= (1-yb+yt)*ny;
      SetLeftMargin(xl);
      SetRightMargin(xr);
      SetBottomMargin(yb);
      SetTopMargin(yt);
      dx = (1-xl-xr)/nx;
      dy = (1-yb-yt)/ny;
      Int_t number = 0;
      for (Int_t i=0;i<nx;i++) {
         x1 = i*dx+xl;
         x2 = x1 + dx;
         if (i == 0) x1 = 0;
         if (i == nx-1) x2 = 1-xr;
         for (Int_t j=0;j<ny;j++) {
            number = j*nx + i +1;
            y2 = 1 -j*dy -yt;
            y1 = y2 - dy;
            if (j == 0)    y2 = 1-yt;
            if (j == ny-1) y1 = 0;
            snprintf(name,nchname,"%s_%d",GetName(),number);
            snprintf(title,nchtitle,"%s_%d",GetTitle(),number);
            pad = new TPad(name,title,x1,y1,x2,y2);
            pad->SetNumber(number);
            pad->SetBorderMode(0);
            if (i == 0)    pad->SetLeftMargin(xl*nx);
            else           pad->SetLeftMargin(0);
                           pad->SetRightMargin(0);
                           pad->SetTopMargin(0);
            if (j == ny-1) pad->SetBottomMargin(yb*ny);
            else           pad->SetBottomMargin(0);
            pad->Draw();
         }
      }
   }
   delete [] name;
   delete [] title;
   Modified();
   if (padsav) padsav->cd();
}


//______________________________________________________________________________
void TPad::Draw(Option_t *option)
{
   // Draw Pad in Current pad (re-parent pad if necessary).

   // if no canvas opened yet create a default canvas
   if (!gPad) {
      gROOT->MakeDefCanvas();
   }

   // pad cannot be in itself and it can only be in one other pad at a time
   if (!fPrimitives) fPrimitives = new TList;
   if (gPad != this) {
      if (fMother) fMother->GetListOfPrimitives()->Remove(this);
      TPad *oldMother = fMother;
      fCanvas = gPad->GetCanvas();
      //
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
   const Int_t kMAXLEVELS = 10;
   TClass *clevel[kMAXLEVELS], *cl, *cll;
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
      if (dx < 1.3) dx = 1.3;
      tsizcm = tsizcm - 0.03*Double_t(ncdraw-5);
      if (tsizcm < 0.27) tsizcm = 0.27;
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
            snprintf(dname,256,"%s",obj->EscapeChars(d->GetName()));
            Int_t ldname = 0;
            while (indx < dim ){
               ldname = strlen(dname);
               snprintf(&dname[ldname],256,"[%d]",d->GetMaxIndex(indx));
               indx++;
            }
            pt->AddText(x,(y-v1)/dv,dname);
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
            ptext = pt->AddText(x,(y-v1)/dv,obj->EscapeChars(m->GetName()));

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
   //
   // Example:
   // Root > TFile f("hsimple.root");
   // Root > hpxpy.Draw();
   // Root > c1.SetCrosshair();
   // When moving the mouse in the canvas, a crosshair is drawn
   //
   // if the canvas fCrosshair = 1 , the crosshair spans the full canvas
   // if the canvas fCrosshair > 1 , the crosshair spans only the pad

   if (gPad->GetEvent() == kMouseEnter) return;

   TPad *cpad = (TPad*)gPad;
   TCanvas *canvas = cpad->GetCanvas();
   canvas->FeedbackMode(kTRUE);

   //erase old position and draw a line at current position
   Int_t pxmin,pxmax,pymin,pymax,pxold,pyold,px,py;
   pxold = fCrosshairPos%10000;
   pyold = fCrosshairPos/10000;
   px    = cpad->GetEventX();
   py    = cpad->GetEventY()+1;
   if (canvas->GetCrosshair() > 1) {  //crosshair only in the current pad
      pxmin = cpad->XtoAbsPixel(fX1);
      pxmax = cpad->XtoAbsPixel(fX2);
      pymin = cpad->YtoAbsPixel(fY1);
      pymax = cpad->YtoAbsPixel(fY2);
   } else { //default; crosshair spans the full canvas
      pxmin = 0;
      pxmax = canvas->GetWw();
      pymin = 0;
      pymax = cpad->GetWh();
   }
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
   //
   //  Compute real pad range taking into account all margins
   //  Use services of TH1F class
   if (!IsEditable()) return 0;
   TPad *padsav = (TPad*)gPad;
   if (this !=  padsav) {
      Warning("DrawFrame","Drawframe must be called for the current pad only");
      return padsav->DrawFrame(xmin,ymin,xmax,ymax,title);
   }

   cd();

   TH1F *hframe = (TH1F*)FindObject("hframe");
   if (hframe) delete hframe;
   Int_t nbins = 1000;
   //if log scale in X, use variable bin size linear with log(x)
   //this gives a better precision when zooming on the axis
   if (fLogx && xmin > 0 && xmax > xmin) {
      Double_t xminl = TMath::Log(xmin);
      Double_t xmaxl = TMath::Log(xmax);
      Double_t dx = (xmaxl-xminl)/nbins;
      Double_t *xbins = new Double_t[nbins+1];
      xbins[0] = xmin;
      for (Int_t i=1;i<=nbins;i++) {
         xbins[i] = TMath::Exp(xminl+i*dx);
      }
      hframe = new TH1F("hframe",title,nbins,xbins);
      delete [] xbins;
   } else {
      hframe = new TH1F("hframe",title,nbins,xmin,xmax);
   }
   hframe->SetBit(TH1::kNoStats);
   hframe->SetBit(kCanDelete);
   hframe->SetMinimum(ymin);
   hframe->SetMaximum(ymax);
   hframe->GetYaxis()->SetLimits(ymin,ymax);
   hframe->SetDirectory(0);
   hframe->Draw(" ");
   Update();
   if (padsav) padsav->cd();
   return hframe;
}

//______________________________________________________________________________
void TPad::DrawColorTable()
{
   // Static function to Display Color Table in a pad.

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
   text->SetTextSize(0.07);
   text->SetTextAlign(22);

   TBox *box = new TBox();

   // Draw color table boxes.
   hs = (y2-y1)/Double_t(5);
   ws = (x2-x1)/Double_t(10);
   for (i=0;i<10;i++) {
      xlow = x1 + ws*(Double_t(i)+0.1);
      xup  = x1 + ws*(Double_t(i)+0.9);
      for (j=0;j<5;j++) {
         ylow = y1 + hs*(Double_t(j)+0.1);
         yup  = y1 + hs*(Double_t(j)+0.9);
         color = 10*j + i;
         box->SetFillStyle(1001);
         box->SetFillColor(color);
         box->DrawBox(xlow, ylow, xup, yup);
         box->SetFillStyle(0);
         box->SetLineColor(1);
         box->DrawBox(xlow, ylow, xup, yup);
         if (color == 1) text->SetTextColor(0);
         else            text->SetTextColor(1);
         text->DrawText(0.5*(xlow+xup), 0.5*(ylow+yup), Form("%d",color));
      }
   }
}


//______________________________________________________________________________
void TPad::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //  This member function is called when a TPad object is clicked.
   //
   //  If the mouse is clicked in one of the 4 corners of the pad (pA,pB,pC,pD)
   //  the pad is resized with the rubber rectangle.
   //
   //  If the mouse is clicked inside the pad, the pad is moved.
   //
   //  If the mouse is clicked on the 4 edges (pL,pR,pTop,pBot), the pad is scaled
   //  parallel to this edge.
   //
   //    pA                   pTop                     pB
   //     +--------------------------------------------+
   //     |                                            |
   //     |                                            |
   //     |                                            |
   //   pL|                 pINSIDE                    |pR
   //     |                                            |
   //     |                                            |
   //     |                                            |
   //     |                                            |
   //     +--------------------------------------------+
   //    pD                   pBot                     pC
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
   static Bool_t pA, pB, pC, pD, pTop, pL, pR, pBot, pINSIDE;
   Int_t  wx, wy;
   Bool_t opaque  = OpaqueMoving();
   Bool_t ropaque = OpaqueResizing();
   Bool_t fixedr  = HasFixedAspectRatio();

   if (!IsEditable() && event != kMouseEnter) return;
   TVirtualPad  *parent = GetMother();
   if (!parent->IsEditable()) return;

   HideToolTip(event);

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
   if (newcode)
      pA = pB = pC = pD = pTop = pL = pR = pBot = pINSIDE = kFALSE;
   switch (newcode) {
      case kPad:
         TCreatePrimitives::Pad(event,px,py,0);
         break;
      case kMarker:
      case kText:
         TCreatePrimitives::Text(event,px,py,newcode);
         break;
      case kLine:
         TCreatePrimitives::Line(event,px,py,kLine);
         break;
      case kArrow:
         TCreatePrimitives::Line(event,px,py,kArrow);
         break;
      case kCurlyLine:
         TCreatePrimitives::Line(event,px,py,kCurlyLine);
         break;
      case kCurlyArc:
         TCreatePrimitives::Line(event,px,py,kCurlyArc);
         break;
      case kPolyLine:
         TCreatePrimitives::PolyLine(event,px,py,kPolyLine);
         break;
      case kCutG:
         TCreatePrimitives::PolyLine(event,px,py,kCutG);
         break;
      case kArc:
         TCreatePrimitives::Ellipse(event,px,py,kArc);
         break;
      case kEllipse:
         TCreatePrimitives::Ellipse(event,px,py,kEllipse);
         break;
      case kButton:
      case kPave:
      case kPaveLabel:
      case kPaveText:
      case kPavesText:
      case kDiamond:
         TCreatePrimitives::Pave(event,px,py,newcode);
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

      GetPainter()->SetLineColor(-1);
      TAttLine::Modify();  //Change line attributes only if necessary
      if (GetFillColor())
         GetPainter()->SetLineColor(GetFillColor());
      else
         GetPainter()->SetLineColor(1);
      GetPainter()->SetLineWidth(2);

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

      pA = pB = pC = pD = pTop = pL = pR = pBot = pINSIDE = kFALSE;

                                                         // case pA
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxl; pyold = pyl; pA = kTRUE;
         SetCursor(kTopLeft);
      }
                                                         // case pB
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyl) <= kMaxDiff) {
         pxold = pxt; pyold = pyl; pB = kTRUE;
         SetCursor(kTopRight);
      }
                                                         // case pC
      if (TMath::Abs(px - pxt) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxt; pyold = pyt; pC = kTRUE;
         SetCursor(kBottomRight);
      }
                                                         // case pD
      if (TMath::Abs(px - pxl) <= kMaxDiff && TMath::Abs(py - pyt) <= kMaxDiff) {
         pxold = pxl; pyold = pyt; pD = kTRUE;
         SetCursor(kBottomLeft);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyl) < kMaxDiff) {             // top edge
         pxold = pxl; pyold = pyl; pTop = kTRUE;
         SetCursor(kTopSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          TMath::Abs(py - pyt) < kMaxDiff) {             // bottom edge
         pxold = pxt; pyold = pyt; pBot = kTRUE;
         SetCursor(kBottomSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxl) < kMaxDiff) {             // left edge
         pxold = pxl; pyold = pyl; pL = kTRUE;
         SetCursor(kLeftSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
         TMath::Abs(px - pxt) < kMaxDiff) {             // right edge
         pxold = pxt; pyold = pyt; pR = kTRUE;
         SetCursor(kRightSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          (py > pyl+kMaxDiff && py < pyt-kMaxDiff)) {    // inside box
         pxold = px; pyold = py; pINSIDE = kTRUE;
         if (event == kButton1Down)
            SetCursor(kMove);
         else
            SetCursor(kCross);
      }

      fResizing = kFALSE;
      if (pA || pB || pC || pD || pTop || pL || pR || pBot)
         fResizing = kTRUE;

      if (!pA && !pB && !pC && !pD && !pTop && !pL && !pR && !pBot && !pINSIDE)
         SetCursor(kCross);

      break;

   case kButton1Motion:

      if (TestBit(kCannotMove)) break;
      wx = wy = 0;

      if (pA) {
         if (!ropaque) gVirtualX->DrawBox(pxold, pyt, pxt, pyold, TVirtualX::kHollow);
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
         if (!ropaque) gVirtualX->DrawBox(px, pyt, pxt, py, TVirtualX::kHollow);
      }
      if (pB) {
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
      if (pC) {
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
         if (!ropaque) gVirtualX->DrawBox(pxl, pyl, px, py, TVirtualX::kHollow);
      }
      if (pD) {
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
         if (!ropaque) gVirtualX->DrawBox(px, py, pxt, pyl, TVirtualX::kHollow);
      }
      if (pTop) {
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
      if (pBot) {
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
      if (pL) {
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
      if (pR) {
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
      if (pINSIDE) {
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
      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         break;
      }

      if (pA) {
         fX1 = AbsPixeltoX(pxold);
         fY1 = AbsPixeltoY(pyt);
         fX2 = AbsPixeltoX(pxt);
         fY2 = AbsPixeltoY(pyold);
      }
      if (pB) {
         fX1 = AbsPixeltoX(pxl);
         fY1 = AbsPixeltoY(pyt);
         fX2 = AbsPixeltoX(pxold);
         fY2 = AbsPixeltoY(pyold);
      }
      if (pC) {
         fX1 = AbsPixeltoX(pxl);
         fY1 = AbsPixeltoY(pyold);
         fX2 = AbsPixeltoX(pxold);
         fY2 = AbsPixeltoY(pyl);
      }
      if (pD) {
         fX1 = AbsPixeltoX(pxold);
         fY1 = AbsPixeltoY(pyold);
         fX2 = AbsPixeltoX(pxt);
         fY2 = AbsPixeltoY(pyl);
      }
      if (pTop || pBot || pL || pR || pINSIDE) {
         fX1 = AbsPixeltoX(px1);
         fY1 = AbsPixeltoY(py1);
         fX2 = AbsPixeltoX(px2);
         fY2 = AbsPixeltoY(py2);
      }

      if (pINSIDE)
         if (!doing_again) gPad->SetCursor(kCross);

      if (pA || pB || pC || pD || pTop || pL || pR || pBot)
         Modified(kTRUE);

      gVirtualX->SetLineColor(-1);
      gVirtualX->SetLineWidth(-1);

      if (px != pxorg || py != pyorg) {

         // Get parent corners pixels coordinates
         Int_t parentpx1 = fMother->XtoAbsPixel(parent->GetX1());
         Int_t parentpx2 = fMother->XtoAbsPixel(parent->GetX2());
         Int_t parentpy1 = fMother->YtoAbsPixel(parent->GetY1());
         Int_t parentpy2 = fMother->YtoAbsPixel(parent->GetY2());

         // Get pad new corners pixels coordinates
         Int_t apx1 = XtoAbsPixel(fX1); if (apx1 < parentpx1) {apx1 = parentpx1; }
         Int_t apx2 = XtoAbsPixel(fX2); if (apx2 > parentpx2) {apx2 = parentpx2; }
         Int_t apy1 = YtoAbsPixel(fY1); if (apy1 > parentpy1) {apy1 = parentpy1; }
         Int_t apy2 = YtoAbsPixel(fY2); if (apy2 < parentpy2) {apy2 = parentpy2; }

         // Compute new pad positions in the NDC space of parent
         fXlowNDC = Double_t(apx1 - parentpx1)/Double_t(parentpx2 - parentpx1);
         fYlowNDC = Double_t(apy1 - parentpy1)/Double_t(parentpy2 - parentpy1);
         fWNDC    = Double_t(apx2 - apx1)/Double_t(parentpx2 - parentpx1);
         fHNDC    = Double_t(apy2 - apy1)/Double_t(parentpy2 - parentpy1);
      }

      // Restore old range
      fX1 = xmin;
      fX2 = xmax;
      fY1 = ymin;
      fY2 = ymax;

      // Reset pad parameters and recompute conversion coefficients
      ResizePad();

      // emit signal
      RangeChanged();

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
void TPad::ExecuteEventAxis(Int_t event, Int_t px, Int_t py, TAxis *axis)
{
   // Execute action corresponding to one event for a TAxis object
   // (called by TAxis::ExecuteEvent.)
   //  This member function is called when an axis is clicked with the locator
   //
   //  The axis range is set between the position where the mouse is pressed
   //  and the position where it is released.
   //  If the mouse position is outside the current axis range when it is released
   //  the axis is unzoomed with the corresponding proportions.
   //  Note that the mouse does not need to be in the pad or even canvas
   //  when it is released.

   if (!IsEditable()) return;

   SetCursor(kHand);

   TView *view = GetView();
   static Int_t axisNumber;
   static Double_t ratio1, ratio2;
   static Int_t px1old, py1old, px2old, py2old;
   Int_t bin1, bin2, first, last;
   Double_t temp, xmin,xmax;

   // The CONT4 option, used to paint TH2, is a special case; it uses a 3D
   // drawing technique to paint a 2D plot.
   TString opt = axis->GetParent()->GetDrawOption();
   opt.ToLower();
   Bool_t kCont4 = kFALSE;
   if (strstr(opt,"cont4")) {
      view = 0;
      kCont4 = kTRUE;
   }

   switch (event) {

   case kButton1Down:
      axisNumber = 1;
      if (!strcmp(axis->GetName(),"xaxis")) {
         axisNumber = 1;
         if (!IsVertical()) axisNumber = 2;
      }
      if (!strcmp(axis->GetName(),"yaxis")) {
         axisNumber = 2;
         if (!IsVertical()) axisNumber = 1;
      }
      if (!strcmp(axis->GetName(),"zaxis")) {
         axisNumber = 3;
      }
      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio1);
      } else {
         if (axisNumber == 1) {
            ratio1 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            px1old = XtoAbsPixel(GetUxmin()+ratio1*(GetUxmax() - GetUxmin()));
            py1old = YtoAbsPixel(GetUymin());
            px2old = px1old;
            py2old = YtoAbsPixel(GetUymax());
         } else if (axisNumber == 2) {
            ratio1 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py1old = YtoAbsPixel(GetUymin()+ratio1*(GetUymax() - GetUymin()));
            px1old = XtoAbsPixel(GetUxmin());
            px2old = XtoAbsPixel(GetUxmax());
            py2old = py1old;
         } else {
            ratio1 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py1old = YtoAbsPixel(GetUymin()+ratio1*(GetUymax() - GetUymin()));
            px1old = XtoAbsPixel(GetUxmax());
            px2old = XtoAbsPixel(GetX2());
            py2old = py1old;
         }
         gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
      }
      gVirtualX->SetLineColor(-1);
      // No break !!!

   case kButton1Motion:
      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio2);
      } else {
         gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
         if (axisNumber == 1) {
            ratio2 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            px2old = XtoAbsPixel(GetUxmin()+ratio2*(GetUxmax() - GetUxmin()));
         } else {
            ratio2 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py2old = YtoAbsPixel(GetUymin()+ratio2*(GetUymax() - GetUymin()));
         }
         gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
      }
   break;

   case kWheelUp:
      bin1 = axis->GetFirst()+1;
      bin2 = axis->GetLast()-1;
      axis->SetRange(bin1,bin2);
      gPad->Modified();
      gPad->Update();
   break;

   case kWheelDown:
      bin1 = axis->GetFirst()-1;
      bin2 = axis->GetLast()+1;
      axis->SetRange(bin1,bin2);
      gPad->Modified();
      gPad->Update();
   break;

   case kButton1Up:
      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         break;
      }

      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio2);
         if (ratio1 > ratio2) {
            temp   = ratio1;
            ratio1 = ratio2;
            ratio2 = temp;
         }
         if (ratio2 - ratio1 > 0.05) {
            TH1 *hobj = (TH1*)axis->GetParent();
            if (axisNumber == 3 && hobj && hobj->GetDimension() != 3) {
               Float_t zmin = hobj->GetMinimum();
               Float_t zmax = hobj->GetMaximum();
               if(GetLogz()){
                  if (zmin <= 0 && zmax > 0) zmin = TMath::Min((Double_t)1,
                                                               (Double_t)0.001*zmax);
                  zmin = TMath::Log10(zmin);
                  zmax = TMath::Log10(zmax);
               }
               Float_t newmin = zmin + (zmax-zmin)*ratio1;
               Float_t newmax = zmin + (zmax-zmin)*ratio2;
               if(newmin < zmin)newmin = hobj->GetBinContent(hobj->GetMinimumBin());
               if(newmax > zmax)newmax = hobj->GetBinContent(hobj->GetMaximumBin());
               if(GetLogz()){
                  newmin = TMath::Exp(2.302585092994*newmin);
                  newmax = TMath::Exp(2.302585092994*newmax);
               }
               hobj->SetMinimum(newmin);
               hobj->SetMaximum(newmax);
               hobj->SetBit(TH1::kIsZoomed);
            } else {
               first = axis->GetFirst();
               last  = axis->GetLast();
               bin1 = first + Int_t((last-first+1)*ratio1);
               bin2 = first + Int_t((last-first+1)*ratio2);
               axis->SetRange(bin1, bin2);
            }
            delete view;
            SetView(0);
            Modified(kTRUE);
         }
      } else {
         if (axisNumber == 1) {
            ratio2 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            xmin = GetUxmin() +ratio1*(GetUxmax() - GetUxmin());
            xmax = GetUxmin() +ratio2*(GetUxmax() - GetUxmin());
            if (GetLogx() && !kCont4) {
               xmin = PadtoX(xmin);
               xmax = PadtoX(xmax);
            }
         } else if (axisNumber == 2) {
            ratio2 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            xmin = GetUymin() +ratio1*(GetUymax() - GetUymin());
            xmax = GetUymin() +ratio2*(GetUymax() - GetUymin());
            if (GetLogy() && !kCont4) {
               xmin = PadtoY(xmin);
               xmax = PadtoY(xmax);
            }
         } else {
            ratio2 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            xmin = ratio1;
            xmax = ratio2;
         }
         if (xmin > xmax) {
            temp   = xmin;
            xmin   = xmax;
            xmax   = temp;
            temp   = ratio1;
            ratio1 = ratio2;
            ratio2 = temp;
         }

         // xmin and xmax need to be adjusted in case of CONT4.
         if (kCont4) {
            Double_t low = axis->GetBinLowEdge(axis->GetFirst());
            Double_t up  = axis->GetBinUpEdge(axis->GetLast());
            Double_t xmi = GetUxmin();
            Double_t xma = GetUxmax();
            xmin = ((xmin-xmi)/(xma-xmi))*(up-low)+low;
            xmax = ((xmax-xmi)/(xma-xmi))*(up-low)+low;
         }

         if (!strcmp(axis->GetName(),"xaxis")) axisNumber = 1;
         if (!strcmp(axis->GetName(),"yaxis")) axisNumber = 2;
         if (ratio2 - ratio1 > 0.05) {
            //update object owning this axis
            TH1 *hobj1 = (TH1*)axis->GetParent();
            bin1 = axis->FindFixBin(xmin);
            bin2 = axis->FindFixBin(xmax);
            if (axisNumber == 1) axis->SetRange(bin1,bin2);
            if (axisNumber == 2 && hobj1) {
               if (hobj1->GetDimension() == 1) {
                  if (hobj1->GetNormFactor() != 0) {
                     Double_t norm = hobj1->GetSumOfWeights()/hobj1->GetNormFactor();
                     xmin *= norm;
                     xmax *= norm;
                  }
                  hobj1->SetMinimum(xmin);
                  hobj1->SetMaximum(xmax);
                  hobj1->SetBit(TH1::kIsZoomed);
               } else {
                  axis->SetRange(bin1,bin2);
               }
            }
            //update all histograms in the pad
            TIter next(GetListOfPrimitives());
            TObject *obj;
            while ((obj= next())) {
               if (!obj->InheritsFrom(TH1::Class())) continue;
               TH1 *hobj = (TH1*)obj;
               if (hobj == hobj1) continue;
               bin1 = hobj->GetXaxis()->FindFixBin(xmin);
               bin2 = hobj->GetXaxis()->FindFixBin(xmax);
               if (axisNumber == 1) {
                  hobj->GetXaxis()->SetRange(bin1,bin2);
               } else if (axisNumber == 2) {
                  if (hobj->GetDimension() == 1) {
                     Double_t xxmin = xmin;
                     Double_t xxmax = xmax;
                     if (hobj->GetNormFactor() != 0) {
                        Double_t norm = hobj->GetSumOfWeights()/hobj->GetNormFactor();
                        xxmin *= norm;
                        xxmax *= norm;
                     }
                     hobj->SetMinimum(xxmin);
                     hobj->SetMaximum(xxmax);
                     hobj->SetBit(TH1::kIsZoomed);
                  } else {
                     hobj->GetXaxis()->SetRange(bin1,bin2);
                  }
               }
            }
            Modified(kTRUE);
         }
      }
      gVirtualX->SetLineColor(-1);
      break;
   }
}

//______________________________________________________________________________
TObject *TPad::FindObject(const char *name) const
{
   // Search if object named name is inside this pad or in pads inside this pad.
   //
   //  In case name is in several subpads the first one is returned.

   if (!fPrimitives) return 0;
   TObject *found = fPrimitives->FindObject(name);
   if (found) return found;
   TObject *cur;
   TIter    next(GetListOfPrimitives());
   while ((cur = next())) {
      if (cur->InheritsFrom(TPad::Class())) {
         found = ((TPad*)cur)->FindObject(name);
         if (found) return found;
      }
   }
    return 0;
}


//______________________________________________________________________________
TObject *TPad::FindObject(const TObject *obj) const
{
   // Search if obj is in pad or in pads inside this pad.
   //
   //  In case obj is in several subpads the first one is returned.

   if (!fPrimitives) return 0;
   TObject *found = fPrimitives->FindObject(obj);
   if (found) return found;
   TObject *cur;
   TIter    next(GetListOfPrimitives());
   while ((cur = next())) {
      if (cur->InheritsFrom(TPad::Class())) {
         found = ((TPad*)cur)->FindObject(obj);
         if (found) return found;
      }
   }
   return 0;
}


//______________________________________________________________________________
Int_t TPad::GetCanvasID() const
{
   // Get canvas identifier.

   return fCanvas ? fCanvas->GetCanvasID() : -1;
}

//______________________________________________________________________________
TCanvasImp *TPad::GetCanvasImp() const
{
   // Get canvas implementation pointer if any

   return fCanvas ? fCanvas->GetCanvasImp() : 0;
}


//______________________________________________________________________________
Int_t TPad::GetEvent() const
{
   // Get Event.

   return  fCanvas ? fCanvas->GetEvent() : 0;
}


//______________________________________________________________________________
Int_t TPad::GetEventX() const
{
   // Get X event.

   return  fCanvas ? fCanvas->GetEventX() : 0;
}


//______________________________________________________________________________
Int_t TPad::GetEventY() const
{
   // Get Y event.

   return  fCanvas ? fCanvas->GetEventY() : 0;
}


//______________________________________________________________________________
TVirtualPad *TPad::GetVirtCanvas() const
{
   // Get virtual canvas.

   return  fCanvas ? (TVirtualPad*) fCanvas : 0;
}


//______________________________________________________________________________
Color_t TPad::GetHighLightColor() const
{
   // Get highlight color.

   return  fCanvas ? fCanvas->GetHighLightColor() : 0;
}


//______________________________________________________________________________
Int_t TPad::GetMaxPickDistance()
{
   // Static function (see also TPad::SetMaxPickDistance)

   return fgMaxPickDistance;
}


//______________________________________________________________________________
TObject *TPad::GetSelected() const
{
   // Get selected.

   if (fCanvas == this) return 0;
   return  fCanvas ? fCanvas->GetSelected() : 0;
}


//______________________________________________________________________________
TVirtualPad *TPad::GetSelectedPad() const
{
   // Get selected pad.

   if (fCanvas == this) return 0;
   return  fCanvas ? fCanvas->GetSelectedPad() : 0;
}


//______________________________________________________________________________
TVirtualPad *TPad::GetPadSave() const
{
   // Get save pad.

   if (fCanvas == this) return 0;
   return  fCanvas ? fCanvas->GetPadSave() : 0;
}


//______________________________________________________________________________
UInt_t TPad::GetWh() const
{
   // Get Wh.

   return  fCanvas ? fCanvas->GetWh() : 0;
}


//______________________________________________________________________________
UInt_t TPad::GetWw() const
{
   // Get Ww.

   return  fCanvas ? fCanvas->GetWw() : 0;
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
   // Is pad in batch mode ?

   return  fCanvas ? fCanvas->IsBatch() : 0;
}


//______________________________________________________________________________
Bool_t TPad::IsRetained() const
{
   // Is pad retained ?

   return  fCanvas ? fCanvas->IsRetained() : 0;
}


//______________________________________________________________________________
Bool_t TPad::OpaqueMoving() const
{
   // Is pad moving in opaque mode ?

   return  fCanvas ? fCanvas->OpaqueMoving() : 0;
}


//______________________________________________________________________________
Bool_t TPad::OpaqueResizing() const
{
   // Is pad resizing in opaque mode ?

   return  fCanvas ? fCanvas->OpaqueResizing() : 0;
}


//______________________________________________________________________________
void TPad::SetBatch(Bool_t batch)
{
   // Set pad in batch mode.

   if (fCanvas) fCanvas->SetBatch(batch);
}


//______________________________________________________________________________
void TPad::SetCanvasSize(UInt_t ww, UInt_t wh)
{
   // Set canvas size.

   if (fCanvas) fCanvas->SetCanvasSize(ww,wh);
}


//______________________________________________________________________________
void TPad::SetCursor(ECursor cursor)
{
   // Set cursor type.

   if (fCanvas) fCanvas->SetCursor(cursor);
}


//______________________________________________________________________________
void TPad::SetDoubleBuffer(Int_t mode)
{
   // Set double buffer mode ON or OFF.

   if (fCanvas) fCanvas->SetDoubleBuffer(mode);
}


//______________________________________________________________________________
void TPad::SetSelected(TObject *obj)
{
   // Set selected.

   if (fCanvas) fCanvas->SetSelected(obj);
}


//______________________________________________________________________________
void TPad::Update()
{
   // Update pad.

   if (fCanvas) fCanvas->Update();
}


//______________________________________________________________________________
TFrame *TPad::GetFrame()
{
   // Get frame.

   if (!fPrimitives) fPrimitives = new TList;
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
   // Get primitive.

   if (!fPrimitives) return 0;
   TIter next(fPrimitives);
   TObject *found, *obj;
   while ((obj=next())) {
      if (!strcmp(name, obj->GetName())) return obj;
      if (obj->InheritsFrom(TPad::Class())) continue;
      found = obj->FindObject(name);
      if (found) return found;
   }
   return 0;
}


//______________________________________________________________________________
TVirtualPad *TPad::GetPad(Int_t subpadnumber) const
{
   // Get a pointer to subpadnumber of this pad.

   if (!subpadnumber) {
      return (TVirtualPad*)this;
   }

   TObject *obj;
   if (!fPrimitives) return 0;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TVirtualPad::Class())) {
         TVirtualPad *pad = (TVirtualPad*)obj;
         if (pad->GetNumber() == subpadnumber) return pad;
      }
   }
   return 0;
}


//______________________________________________________________________________
void TPad::GetPadPar(Double_t &xlow, Double_t &ylow, Double_t &xup, Double_t &yup)
{
   // Return lower and upper bounds of the pad in NDC coordinates.

   xlow = fXlowNDC;
   ylow = fYlowNDC;
   xup  = fXlowNDC+fWNDC;
   yup  = fYlowNDC+fHNDC;
}


//______________________________________________________________________________
void TPad::GetRange(Double_t &x1, Double_t &y1, Double_t &x2, Double_t &y2)
{
   // Return pad world coordinates range.

   x1 = fX1;
   y1 = fY1;
   x2 = fX2;
   y2 = fY2;
}


//______________________________________________________________________________
void TPad::GetRangeAxis(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax)
{
   // Return pad axis coordinates range.

   xmin = fUxmin;
   ymin = fUymin;
   xmax = fUxmax;
   ymax = fUymax;
}


//______________________________________________________________________________
void TPad::HighLight(Color_t color, Bool_t set)
{
   // Highlight pad.
   //do not highlight when printing on Postscript
   if (gVirtualPS && gVirtualPS->TestBit(kPrintingPS)) return;

   if (color <= 0) return;

   AbsCoordinates(kTRUE);

   // We do not want to have active(executable) buttons, etc highlighted
   // in this manner, unless we want to edit'em
   if (GetMother()->IsEditable() && !InheritsFrom(TButton::Class())) {
      //When doing a DrawClone from the GUI you would do
      //  - select an empty pad -
      //  - right click on object -
      //     - select DrawClone on menu -
      //
      // Without the SetSelectedPad(); in the HighLight function, the
      // above instruction lead to the clone to be drawn in the
      // same canvas as the original object.  This is because the
      // 'right clicking' (via TCanvas::HandleInput) changes gPad
      // momentarily such that when DrawClone is called, it is
      // not the right value (for DrawClone). Should be FIXED.
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
   // List all primitives in pad.

   TROOT::IndentLevel();
   cout <<IsA()->GetName()<<" fXlowNDC=" <<fXlowNDC<<" fYlowNDC="<<fYlowNDC<<" fWNDC="<<GetWNDC()<<" fHNDC="<<GetHNDC()
        <<" Name= "<<GetName()<<" Title= "<<GetTitle()<<" Option="<<option<<endl;
   TROOT::IncreaseDirLevel();
   if (!fPrimitives) return;
   fPrimitives->ls(option);
   TROOT::DecreaseDirLevel();
}


//______________________________________________________________________________
Double_t TPad::PadtoX(Double_t x) const
{
   // Convert x from pad to X.

   if (fLogx && x < 50) return Double_t(TMath::Exp(2.302585092994*x));
   return x;
}


//______________________________________________________________________________
Double_t TPad::PadtoY(Double_t y) const
{
   // Convert y from pad to Y.

   if (fLogy && y < 50) return Double_t(TMath::Exp(2.302585092994*y));
   return y;
}


//______________________________________________________________________________
Double_t TPad::XtoPad(Double_t x) const
{
   // Convert x from X to pad.

   if (fLogx) {
      if (x > 0) x = TMath::Log10(x);
      else       x = fUxmin;
   }
   return x;
}


//______________________________________________________________________________
Double_t TPad::YtoPad(Double_t y) const
{
   // Convert y from Y to pad.

   if (fLogy) {
      if (y > 0) y = TMath::Log10(y);
      else       y = fUymin;
   }
   return y;
}


//______________________________________________________________________________
void TPad::Paint(Option_t * /*option*/)
{
   // Paint all primitives in pad.

   if (!fPrimitives) fPrimitives = new TList;
   if (fViewer3D && fViewer3D->CanLoopOnPrimitives()) {
      fViewer3D->PadPaint(this);
      Modified(kFALSE);
      if (GetGLDevice()!=-1 && gVirtualPS) {
         TPad *padsav = (TPad*)gPad;
         gPad = this;
         gGLManager->PrintViewer(GetViewer3D());
         gPad = padsav;
      }
      return;
   }

   if (fCanvas) TColor::SetGrayscale(fCanvas->IsGrayscale());

   TPad *padsav = (TPad*)gPad;

   fPadPaint = 1;
   cd();

   PaintBorder(GetFillColor(), kTRUE);
   PaintDate();

   TObjOptLink *lnk = (TObjOptLink*)GetListOfPrimitives()->FirstLink();
   TObject *obj;

   Bool_t began3DScene = kFALSE;
   while (lnk) {
      obj = lnk->GetObject();

      // Create a pad 3D viewer if none exists and we encounter a 3D shape
      if (!fViewer3D && obj->InheritsFrom(TAtt3D::Class())) {
         GetViewer3D("pad");
      }

      // Open a 3D scene if required
      if (fViewer3D && !fViewer3D->BuildingScene()) {
         fViewer3D->BeginScene();
         began3DScene = kTRUE;
      }

      obj->Paint(lnk->GetOption());
      lnk = (TObjOptLink*)lnk->Next();
   }

   if (padsav) padsav->cd();
   fPadPaint = 0;
   Modified(kFALSE);

   // Close the 3D scene if we opened it. This must be done after modified
   // flag is cleared, as some viewers will invoke another paint by marking pad modified again
   if (began3DScene) {
      fViewer3D->EndScene();
   }

///// Generate the PS output using gl2ps
///if (GetGLDevice()!=-1 && gVirtualPS) {
///   gPad = this;
///   gGLManager->PrintViewer(GetViewer3D());
///   gPad = padsav;
///}
}


//______________________________________________________________________________
void TPad::PaintBorder(Color_t color, Bool_t tops)
{
   // Paint the pad border.
   // Draw first  a box as a normal filled box
   if(color >= 0) {
      TAttLine::Modify();  //Change line attributes only if necessary
      TAttFill::Modify();  //Change fill area attributes only if necessary
      PaintBox(fX1,fY1,fX2,fY2);
   }
   if (color < 0) color = -color;
   // then paint 3d frame (depending on bordermode)
   if (IsTransparent()) return;
   // Paint a 3D frame around the pad.

   if (fBorderMode == 0) return;
   Int_t bordersize = fBorderSize;
   if (bordersize <= 0) bordersize = 2;

   const Double_t realBsX = bordersize / (GetAbsWNDC() * GetWw()) * (fX2 - fX1);
   const Double_t realBsY = bordersize / (GetAbsHNDC() * GetWh()) * (fY2 - fY1);

   Short_t px1,py1,px2,py2;
   Double_t xl, xt, yl, yt;

   // GetDarkColor() and GetLightColor() use GetFillColor()
   Color_t oldcolor = GetFillColor();
   SetFillColor(color);
   TAttFill::Modify();
   Color_t light = 0, dark = 0;
   if (color != 0) {
      light = TColor::GetColorBright(color);
      dark  = TColor::GetColorDark(color);
   }

   // Compute real left bottom & top right of the box in pixels
   px1 = XtoPixel(fX1);   py1 = YtoPixel(fY1);
   px2 = XtoPixel(fX2);   py2 = YtoPixel(fY2);
   if (px1 < px2) {xl = fX1; xt = fX2; }
   else           {xl = fX2; xt = fX1;}
   if (py1 > py2) {yl = fY1; yt = fY2;}
   else           {yl = fY2; yt = fY1;}

   Double_t frameXs[7] = {}, frameYs[7] = {};

   if (!IsBatch()) {
      // Draw top&left part of the box
      frameXs[0] = xl;           frameYs[0] = yl;
      frameXs[1] = xl + realBsX; frameYs[1] = yl + realBsY;
      frameXs[2] = frameXs[1];   frameYs[2] = yt - realBsY;
      frameXs[3] = xt - realBsX; frameYs[3] = frameYs[2];
      frameXs[4] = xt;           frameYs[4] = yt;
      frameXs[5] = xl;           frameYs[5] = yt;
      frameXs[6] = xl;           frameYs[6] = yl;

      if (fBorderMode == -1) GetPainter()->SetFillColor(dark);
      else                   GetPainter()->SetFillColor(light);
      GetPainter()->DrawFillArea(7, frameXs, frameYs);

      // Draw bottom&right part of the box
      frameXs[0] = xl;              frameYs[0] = yl;
      frameXs[1] = xl + realBsX;    frameYs[1] = yl + realBsY;
      frameXs[2] = xt - realBsX;    frameYs[2] = frameYs[1];
      frameXs[3] = frameXs[2];      frameYs[3] = yt - realBsY;
      frameXs[4] = xt;              frameYs[4] = yt;
      frameXs[5] = xt;              frameYs[5] = yl;
      frameXs[6] = xl;              frameYs[6] = yl;

      if (fBorderMode == -1) GetPainter()->SetFillColor(light);
      else                   GetPainter()->SetFillColor(dark);
      GetPainter()->DrawFillArea(7, frameXs, frameYs);

      // If this pad is a button, highlight it
      if (InheritsFrom(TButton::Class()) && fBorderMode == -1) {
         if (TestBit(kFraming)) {  // bit set in TButton::SetFraming
            if (GetFillColor() != 2) GetPainter()->SetLineColor(2);
            else                     GetPainter()->SetLineColor(4);
            GetPainter()->DrawBox(xl + realBsX, yl + realBsY, xt - realBsX, yt - realBsY, TVirtualPadPainter::kHollow);
         }
      }
      GetPainter()->SetFillColor(-1);
      SetFillColor(oldcolor);
   }

   if (!tops) return;

   PaintBorderPS(xl, yl, xt, yt, fBorderMode, bordersize, dark, light);
}


//______________________________________________________________________________
void TPad::PaintBorderPS(Double_t xl,Double_t yl,Double_t xt,Double_t yt,Int_t bmode,Int_t bsize,Int_t dark,Int_t light)
{
   // Paint a frame border with Postscript.

   if (!gVirtualPS) return;
   gVirtualPS->DrawFrame(xl, yl, xt, yt, bmode,bsize,dark,light);
}


//______________________________________________________________________________
void TPad::PaintDate()
{
   // Paint the current date and time if the option date is on.

   if (fCanvas == this && gStyle->GetOptDate()) {
      TDatime dt;
      const char *dates;
      char iso[16];
      if (gStyle->GetOptDate() < 10) {
         //by default use format like "Wed Sep 25 17:10:35 2002"
         dates = dt.AsString();
      } else if (gStyle->GetOptDate() < 20) {
         //use ISO format like 2002-09-25
         strlcpy(iso,dt.AsSQLString(),16);
         dates = iso;
      } else {
         //use ISO format like 2002-09-25 17:10:35
         dates = dt.AsSQLString();
      }
      TText tdate(gStyle->GetDateX(),gStyle->GetDateY(),dates);
      tdate.SetTextSize( gStyle->GetAttDate()->GetTextSize());
      tdate.SetTextFont( gStyle->GetAttDate()->GetTextFont());
      tdate.SetTextColor(gStyle->GetAttDate()->GetTextColor());
      tdate.SetTextAlign(gStyle->GetAttDate()->GetTextAlign());
      tdate.SetTextAngle(gStyle->GetAttDate()->GetTextAngle());
      tdate.SetNDC();
      tdate.Paint();
   }
}


//______________________________________________________________________________
void TPad::PaintPadFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
   // Paint histogram/graph frame.

   if (!fPrimitives) fPrimitives = new TList;
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
   if (gROOT->GetForceStyle()) frame->UseCurrentStyle();
   frame->Paint();
}


//______________________________________________________________________________
void TPad::PaintModified()
{
   // Traverse pad hierarchy and (re)paint only modified pads.

   if (fViewer3D && fViewer3D->CanLoopOnPrimitives()) {
      if (IsModified()) {
         fViewer3D->PadPaint(this);
         Modified(kFALSE);
      }
      TList *pList = GetListOfPrimitives();
      TObjOptLink *lnk = 0;
      if (pList) lnk = (TObjOptLink*)pList->FirstLink();
      TObject *obj;
      while (lnk) {
         obj = lnk->GetObject();
         if (obj->InheritsFrom(TPad::Class()))
            ((TPad*)obj)->PaintModified();
         lnk = (TObjOptLink*)lnk->Next();
      }
      return;
   }

   if (fCanvas) TColor::SetGrayscale(fCanvas->IsGrayscale());

   TPad *padsav = (TPad*)gPad;
   TVirtualPS *saveps = gVirtualPS;
   if (gVirtualPS) {
      if (gVirtualPS->TestBit(kPrintingPS)) gVirtualPS = 0;
   }
   fPadPaint = 1;
   cd();
   if (IsModified() || IsTransparent()) {
      if ((fFillStyle < 3026) && (fFillStyle > 3000)) {
         if (!gPad->IsBatch()) GetPainter()->ClearDrawable();
      }
      PaintBorder(GetFillColor(), kTRUE);
   }

   PaintDate();

   TList *pList = GetListOfPrimitives();
   TObjOptLink *lnk = 0;
   if (pList) lnk = (TObjOptLink*)pList->FirstLink();
   TObject *obj;

   Bool_t began3DScene = kFALSE;

   while (lnk) {
      obj = lnk->GetObject();
      if (obj->InheritsFrom(TPad::Class())) {
         ((TPad*)obj)->PaintModified();
      } else if (IsModified() || IsTransparent()) {

         // Create a pad 3D viewer if none exists and we encounter a
         // 3D shape
         if (!fViewer3D && obj->InheritsFrom(TAtt3D::Class())) {
            GetViewer3D("pad");
         }

         // Open a 3D scene if required
         if (fViewer3D && !fViewer3D->BuildingScene()) {
            fViewer3D->BeginScene();
            began3DScene = kTRUE;
         }

         obj->Paint(lnk->GetOption());
      }
      lnk = (TObjOptLink*)lnk->Next();
   }

   if (padsav) padsav->cd();
   fPadPaint = 0;
   Modified(kFALSE);

   // This must be done after modified flag is cleared, as some
   // viewers will invoke another paint by marking pad modified again
   if (began3DScene) {
      fViewer3D->EndScene();
   }

   gVirtualPS = saveps;
}


//______________________________________________________________________________
void TPad::PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *option)
{
   // Paint box in CurrentPad World coordinates.
   //
   // if option[0] = 's' the box is forced to be paint with style=0
   // if option[0] = 'l' the box contour is drawn

   if (!gPad->IsBatch()) {
      Int_t style0 = GetPainter()->GetFillStyle();
      Int_t style  = style0;
      if (option[0] == 's') {
         GetPainter()->SetFillStyle(0);
         style = 0;
      }
      if (style) {
         if (style > 3000 && style < 4000) {
            if (style < 3026) {
               // draw stipples with fFillColor foreground
               GetPainter()->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
            }

            if (style >= 3100 && style < 4000) {
               Double_t xb[4], yb[4];
               xb[0] = x1; xb[1] = x1; xb[2] = x2; xb[3] = x2;
               yb[0] = y1; yb[1] = y2; yb[2] = y2; yb[3] = y1;
               PaintFillAreaHatches(4, xb, yb, style);
               return;
            }
            //special case for TAttFillCanvas
            if (GetPainter()->GetFillColor() == 10) {
               GetPainter()->SetFillColor(1);
               GetPainter()->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
               GetPainter()->SetFillColor(10);
            }
         } else if (style >= 4000 && style <= 4100) {
            // For style >=4000 we make the window transparent.
            // From 4000 to 4100 the window is 100% transparent to 100% opaque

            //ignore this style option when this is the canvas itself
            if (this == fMother)
               GetPainter()->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
            else {
               //draw background by blitting all bottom pads
               int px, py;
               XYtoAbsPixel(fX1, fY2, px, py);

               if (fMother) {
                  fMother->CopyBackgroundPixmap(px, py);
                  CopyBackgroundPixmaps(fMother, this, px, py);
               }

               GetPainter()->SetOpacity(style - 4000);
            }
         } else {
            GetPainter()->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);
         }
         if (option[0] == 'l') GetPainter()->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kHollow);
      } else {
         GetPainter()->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kHollow);
         if (option[0] == 's') GetPainter()->SetFillStyle(style0);
      }
   }

   if (gVirtualPS) {
      Int_t style0 = gVirtualPS->GetFillStyle();
      if (option[0] == 's') {
         gVirtualPS->SetFillStyle(0);
      } else {
         if (style0 >= 3100 && style0 < 4000) {
            Double_t xb[4], yb[4];
            xb[0] = x1; xb[1] = x1; xb[2] = x2; xb[3] = x2;
            yb[0] = y1; yb[1] = y2; yb[2] = y2; yb[3] = y1;
            PaintFillAreaHatches(4, xb, yb, style0);
            return;
         }
      }
      gVirtualPS->DrawBox(x1, y1, x2, y2);
      if (option[0] == 'l') {
         gVirtualPS->SetFillStyle(0);
         gVirtualPS->DrawBox(x1, y1, x2, y2);
      }
      if (option[0] == 's' || option[0] == 'l') gVirtualPS->SetFillStyle(style0);
   }

   Modified();
}


//______________________________________________________________________________
void TPad::CopyBackgroundPixmaps(TPad *start, TPad *stop, Int_t x, Int_t y)
{
   // Copy pixmaps of pads laying below pad "stop" into pad "stop". This
   // gives the effect of pad "stop" being transparent.

   TObject *obj;
   if (!fPrimitives) fPrimitives = new TList;
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
   GetPainter()->CopyDrawable(GetPixmapID(), px-x, py-y);
}


//______________________________________________________________________________
void TPad::PaintFillArea(Int_t nn, Float_t *xx, Float_t *yy, Option_t *)
{
   // Paint fill area in CurrentPad World coordinates.

   Warning("TPad::PaintFillArea", "Float_t signature is obsolete");

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

   if (n < 3) {
      delete [] x;
      delete [] y;
      return;
   }

   // Paint the fill area with hatches
   Int_t fillstyle = GetPainter()->GetFillStyle();
   if (gPad->IsBatch() && gVirtualPS) fillstyle = gVirtualPS->GetFillStyle();
   if (fillstyle >= 3100 && fillstyle < 4000) {
      PaintFillAreaHatches(nn, x, y, fillstyle);
      delete [] x;
      delete [] y;
      return;
   }

   if (!gPad->IsBatch())
      // invoke the graphics subsystem
      GetPainter()->DrawFillArea(n, x, y);

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
   // Paint fill area in CurrentPad World coordinates.

   if (nn <3) return;
   Int_t n=0;
   Double_t xmin,xmax,ymin,ymax;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }

   Int_t nc = 2*nn+1;
   Double_t *x = new Double_t[nc];
   Double_t *y = new Double_t[nc];
   memset(x,0,8*nc);
   memset(y,0,8*nc);

   n = ClipPolygon(nn, xx, yy, nc, x, y,xmin,ymin,xmax,ymax);
   if (!n) {
      delete [] x;
      delete [] y;
      return;
   }

   // Paint the fill area with hatches
   Int_t fillstyle = GetPainter()->GetFillStyle();
   if (gPad->IsBatch() && gVirtualPS) fillstyle = gVirtualPS->GetFillStyle();
   if (fillstyle >= 3100 && fillstyle < 4000) {
      PaintFillAreaHatches(nn, x, y, fillstyle);
      delete [] x;
      delete [] y;
      return;
   }

   if (!gPad->IsBatch())
      // invoke the graphics subsystem
      GetPainter()->DrawFillArea(n, x, y);

   if (gVirtualPS) {
      gVirtualPS->DrawPS(-n, x, y);
   }
   delete [] x;
   delete [] y;
   Modified();
}


//______________________________________________________________________________
void TPad::PaintFillAreaHatches(Int_t nn, Double_t *xx, Double_t *yy, Int_t FillStyle)
{
   //   This function paints hatched fill area arcording to the FillStyle value
   // The convention for the Hatch is the following:
   //
   //            FillStyle = 3ijk
   //
   //    i (1-9) : specify the space between each hatch
   //              1 = minimum  9 = maximum
   //              the final spacing is i*GetHatchesSpacing(). The hatches spacing
   //              is set by SetHatchesSpacing()
   //
   //    j (0-9) : specify angle between 0 and 90 degrees
   //
   //              0 = 0
   //              1 = 10
   //              2 = 20
   //              3 = 30
   //              4 = 45
   //              5 = Not drawn
   //              6 = 60
   //              7 = 70
   //              8 = 80
   //              9 = 90
   //
   //    k (0-9) : specify angle between 90 and 180 degrees
   //              0 = 180
   //              1 = 170
   //              2 = 160
   //              3 = 150
   //              4 = 135
   //              5 = Not drawn
   //              6 = 120
   //              7 = 110
   //              8 = 100
   //              9 = 90

   static Double_t ang1[10] = {0., 10., 20., 30., 45.,5., 60., 70., 80., 90.};
   static Double_t ang2[10] = {180.,170.,160.,150.,135.,5.,120.,110.,100., 90.};

   Int_t fasi  = FillStyle%1000;
   Int_t idSPA = (Int_t)(fasi/100);
   Int_t iAng2 = (Int_t)((fasi-100*idSPA)/10);
   Int_t iAng1 = fasi%10;
   Double_t dy = 0.003*(Double_t)(idSPA)*gStyle->GetHatchesSpacing();
   Int_t lw = gStyle->GetHatchesLineWidth();
   Short_t lws = 0;
   Int_t   lss = 0;
   Int_t   lcs = 0;

   // Save the current line attributes
   if (!gPad->IsBatch()) {
      lws = GetPainter()->GetLineWidth();
      lss = GetPainter()->GetLineStyle();
      lcs = GetPainter()->GetLineColor();
   } else {
      if (gVirtualPS) {
         lws = gVirtualPS->GetLineWidth();
         lss = gVirtualPS->GetLineStyle();
         lcs = gVirtualPS->GetLineColor();
      }
   }

   // Change the current line attributes to draw the hatches
   if (!gPad->IsBatch()) {
      GetPainter()->SetLineStyle(1);
      GetPainter()->SetLineWidth(Short_t(lw));
      GetPainter()->SetLineColor(GetPainter()->GetFillColor());
   }
   if (gVirtualPS) {
      gVirtualPS->SetLineStyle(1);
      gVirtualPS->SetLineWidth(Short_t(lw));
      gVirtualPS->SetLineColor(gVirtualPS->GetFillColor());
   }

   // Draw the hatches
   if (ang1[iAng1] != 5.) PaintHatches(dy, ang1[iAng1], nn, xx, yy);
   if (ang2[iAng2] != 5.) PaintHatches(dy, ang2[iAng2], nn, xx, yy);

   // Restore the line attributes
   if (!gPad->IsBatch()) {
      GetPainter()->SetLineStyle(lss);
      GetPainter()->SetLineWidth(lws);
      GetPainter()->SetLineColor(lcs);
   }
   if (gVirtualPS) {
      gVirtualPS->SetLineStyle(lss);
      gVirtualPS->SetLineWidth(lws);
      gVirtualPS->SetLineColor(lcs);
   }
}


//______________________________________________________________________________
void TPad::PaintHatches(Double_t dy, Double_t angle,
                        Int_t nn, Double_t *xx, Double_t *yy)
{
   // This routine draw hatches inclined with the
   // angle "angle" and spaced of "dy" in normalized device
   // coordinates in the surface defined by n,xx,yy.

   Int_t i, i1, i2, nbi, m, inv;
   Double_t ratiox, ratioy, ymin, ymax, yrot, ycur;
   const Double_t angr  = TMath::Pi()*(180-angle)/180.;
   const Double_t epsil = 0.0001;
   const Int_t maxnbi = 100;
   Double_t xli[maxnbi], xlh[2], ylh[2], xt1, xt2, yt1, yt2;
   Double_t ll, x, y, x1, x2, y1, y2, a, b, xi, xip, xin, yi, yip;

   Double_t rwxmin = gPad->GetX1();
   Double_t rwxmax = gPad->GetX2();
   Double_t rwymin = gPad->GetY1();
   Double_t rwymax = gPad->GetY2();
   ratiox = 1/(rwxmax-rwxmin);
   ratioy = 1/(rwymax-rwymin);

   Double_t sina = TMath::Sin(angr), sinb;
   Double_t cosa = TMath::Cos(angr), cosb;
   if (TMath::Abs(cosa) <= epsil) cosa=0.;
   if (TMath::Abs(sina) <= epsil) sina=0.;
   sinb = -sina;
   cosb = cosa;

   // Values needed to compute the hatches in TRUE normalized space (NDC)
   Int_t iw = gPad->GetWw();
   Int_t ih = gPad->GetWh();
   Double_t x1p,y1p,x2p,y2p;
   gPad->GetPadPar(x1p,y1p,x2p,y2p);
   iw  = (Int_t)(iw*x2p)-(Int_t)(iw*x1p);
   ih  = (Int_t)(ih*y2p)-(Int_t)(ih*y1p);
   Double_t wndc  = TMath::Min(1.,(Double_t)iw/(Double_t)ih);
   Double_t hndc  = TMath::Min(1.,(Double_t)ih/(Double_t)iw);

   // Search ymin and ymax
   ymin = 1.;
   ymax = 0.;
   for (i=1; i<=nn; i++) {
      x    = wndc*ratiox*(xx[i-1]-rwxmin);
      y    = hndc*ratioy*(yy[i-1]-rwymin);
      yrot = sina*x+cosa*y;
      if (yrot > ymax) ymax = yrot;
      if (yrot < ymin) ymin = yrot;
   }
   ymax = (Double_t)((Int_t)(ymax/dy))*dy;

   for (ycur=ymax; ycur>=ymin; ycur=ycur-dy) {
      nbi = 0;
      for (i=2; i<=nn+1; i++) {
         i2 = i;
         i1 = i-1;
         if (i == nn+1) i2=1;
         x1  = wndc*ratiox*(xx[i1-1]-rwxmin);
         y1  = hndc*ratioy*(yy[i1-1]-rwymin);
         x2  = wndc*ratiox*(xx[i2-1]-rwxmin);
         y2  = hndc*ratioy*(yy[i2-1]-rwymin);
         xt1 = cosa*x1-sina*y1;
         yt1 = sina*x1+cosa*y1;
         xt2 = cosa*x2-sina*y2;
         yt2 = sina*x2+cosa*y2;

         // Line segment parallel to oy
         if (xt1 == xt2) {
            if (yt1 < yt2) {
               yi  = yt1;
               yip = yt2;
            } else {
               yi  = yt2;
               yip = yt1;
            }
            if ((yi <= ycur) && (ycur < yip)) {
               nbi++;
               if (nbi >= maxnbi) return;
               xli[nbi-1] = xt1;
            }
            continue;
         }

         // Line segment parallel to ox
         if (yt1 == yt2) {
            if (yt1 == ycur) {
               nbi++;
               if (nbi >= maxnbi) return;
               xli[nbi-1] = xt1;
               nbi++;
               if (nbi >= maxnbi) return;
               xli[nbi-1] = xt2;
            }
            continue;
         }

         // Other line segment
         a = (yt1-yt2)/(xt1-xt2);
         b = (yt2*xt1-xt2*yt1)/(xt1-xt2);
         if (xt1 < xt2) {
            xi  = xt1;
            xip = xt2;
         } else {
            xi  = xt2;
            xip = xt1;
         }
         xin = (ycur-b)/a;
         if  ((xi <= xin) && (xin < xip) &&
              (TMath::Min(yt1,yt2) <= ycur) &&
              (ycur < TMath::Max(yt1,yt2))) {
            nbi++;
            if (nbi >= maxnbi) return;
            xli[nbi-1] = xin;
         }
      }

      // Sorting of the x coordinates intersections
      inv = 0;
      m   = nbi-1;
L30:
      for (i=1; i<=m; i++) {
         if (xli[i] < xli[i-1]) {
            inv++;
            ll       = xli[i-1];
            xli[i-1] = xli[i];
            xli[i]   = ll;
         }
      }
      m--;
      if (inv == 0) goto L50;
      inv = 0;
      goto L30;

      // Draw the hatches
L50:
      if (nbi%2 != 0) continue;

      for (i=1; i<=nbi; i=i+2) {
         // Rotate back the hatches
         xlh[0] = cosb*xli[i-1]-sinb*ycur;
         ylh[0] = sinb*xli[i-1]+cosb*ycur;
         xlh[1] = cosb*xli[i]  -sinb*ycur;
         ylh[1] = sinb*xli[i]  +cosb*ycur;
         // Convert hatches' positions from true NDC to WC
         xlh[0] = (xlh[0]/wndc)*(rwxmax-rwxmin)+rwxmin;
         ylh[0] = (ylh[0]/hndc)*(rwymax-rwymin)+rwymin;
         xlh[1] = (xlh[1]/wndc)*(rwxmax-rwxmin)+rwxmin;
         ylh[1] = (ylh[1]/hndc)*(rwymax-rwymin)+rwymin;
         gPad->PaintLine(xlh[0], ylh[0], xlh[1], ylh[1]);
      }
   }
}


//______________________________________________________________________________
void TPad::PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Paint line in CurrentPad World coordinates.

   Double_t x[2], y[2];
   x[0] = x1;   x[1] = x2;   y[0] = y1;   y[1] = y2;

   //If line is totally clipped, return
   if (TestBit(TGraph::kClipFrame)) {
      if (Clip(x,y,fUxmin,fUymin,fUxmax,fUymax) == 2) return;
   } else {
      if (Clip(x,y,fX1,fY1,fX2,fY2) == 2) return;
   }

   if (!gPad->IsBatch())
      GetPainter()->DrawLine(x[0], y[0], x[1], y[1]);

   if (gVirtualPS) {
      gVirtualPS->DrawPS(2, x, y);
   }

   Modified();
}


//______________________________________________________________________________
void TPad::PaintLineNDC(Double_t u1, Double_t v1,Double_t u2, Double_t v2)
{
   static Double_t xw[2], yw[2];
   if (!gPad->IsBatch())
      GetPainter()->DrawLineNDC(u1, v1, u2, v2);

   if (gVirtualPS) {
      xw[0] = fX1 + u1*(fX2 - fX1);
      xw[1] = fX1 + u2*(fX2 - fX1);
      yw[0] = fY1 + v1*(fY2 - fY1);
      yw[1] = fY1 + v2*(fY2 - fY1);
      gVirtualPS->DrawPS(2, xw, yw);
   }

   Modified();
}


//______________________________________________________________________________
void TPad::PaintLine3D(Float_t *p1, Float_t *p2)
{
   // Paint 3-D line in the CurrentPad.

   if (!fView) return;

   // convert from 3-D to 2-D pad coordinate system
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
   // Paint 3-D line in the CurrentPad.

   //take into account perspective view
   if (!fView) return;
   // convert from 3-D to 2-D pad coordinate system
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
void TPad::PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *)
{
   // Paint polyline in CurrentPad World coordinates.

   if (n < 2) return;

   Double_t xmin,xmax,ymin,ymax;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i, i1=-1,np=1;
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
      if (!gPad->IsBatch())
         GetPainter()->DrawPolyLine(np, &x[i1], &y[i1]);
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

   Modified();
}


//______________________________________________________________________________
void TPad::PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   // Paint polyline in CurrentPad World coordinates.
   //
   //  If option[0] == 'C' no clipping

   if (n < 2) return;

   Double_t xmin,xmax,ymin,ymax;
   Bool_t mustClip = kTRUE;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
      if (option && (option[0] == 'C')) mustClip = kFALSE;
   }

   Int_t i, i1=-1, np=1, iclip=0;

   for (i=0; i < n-1; i++) {
      Double_t x1=x[i];
      Double_t y1=y[i];
      Double_t x2=x[i+1];
      Double_t y2=y[i+1];
      if (mustClip) {
         iclip = Clip(&x[i],&y[i],xmin,ymin,xmax,ymax);
         if (iclip == 2) {
            i1 = -1;
            continue;
         }
      }
      np++;
      if (i1 < 0) i1 = i;
      if (iclip == 0 && i < n-2) continue;
      if (!gPad->IsBatch())
         GetPainter()->DrawPolyLine(np, &x[i1], &y[i1]);
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

   Modified();
}

//______________________________________________________________________________
void TPad::PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *)
{
   // Paint polyline in CurrentPad NDC coordinates.
   if (n <=0) return;

   if (!gPad->IsBatch())
      GetPainter()->DrawPolyLineNDC(n, x, y);

   if (gVirtualPS) {
      Double_t *xw = new Double_t[n];
      Double_t *yw = new Double_t[n];
      for (Int_t i=0; i<n; i++) {
         xw[i] = fX1 + x[i]*(fX2 - fX1);
         yw[i] = fY1 + y[i]*(fY2 - fY1);
      }
      gVirtualPS->DrawPS(n, xw, yw);
      delete [] xw;
      delete [] yw;
   }
   Modified();
}


//______________________________________________________________________________
void TPad::PaintPolyLine3D(Int_t n, Double_t *p)
{
   // Paint 3-D polyline in the CurrentPad.
   if (!fView) return;

   // Loop on each individual line
   for (Int_t i = 1; i < n; i++)
      PaintLine3D(&p[3*i-3], &p[3*i]);

   Modified();
}


//______________________________________________________________________________
void TPad::PaintPolyMarker(Int_t nn, Float_t *x, Float_t *y, Option_t *)
{
   // Paint polymarker in CurrentPad World coordinates.

   Int_t n = TMath::Abs(nn);
   Double_t xmin,xmax,ymin,ymax;
   if (nn > 0 || TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,i1=-1,np=0;
   for (i=0; i<n; i++) {
      if (x[i] >= xmin && x[i] <= xmax && y[i] >= ymin && y[i] <= ymax) {
         np++;
         if (i1 < 0) i1 = i;
         if (i < n-1) continue;
      }
      if (np == 0) continue;
      if (!gPad->IsBatch())
         GetPainter()->DrawPolyMarker(np, &x[i1], &y[i1]);
      if (gVirtualPS) {
         gVirtualPS->DrawPolyMarker(np, &x[i1], &y[i1]);
      }
      i1 = -1;
      np = 0;
   }
   Modified();
}


//______________________________________________________________________________
void TPad::PaintPolyMarker(Int_t nn, Double_t *x, Double_t *y, Option_t *)
{
   // Paint polymarker in CurrentPad World coordinates.

   Int_t n = TMath::Abs(nn);
   Double_t xmin,xmax,ymin,ymax;
   if (nn > 0 || TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }
   Int_t i,i1=-1,np=0;
   for (i=0; i<n; i++) {
      if (x[i] >= xmin && x[i] <= xmax && y[i] >= ymin && y[i] <= ymax) {
         np++;
         if (i1 < 0) i1 = i;
         if (i < n-1) continue;
      }
      if (np == 0) continue;
      if (!gPad->IsBatch())
         GetPainter()->DrawPolyMarker(np, &x[i1], &y[i1]);
      if (gVirtualPS) {
         gVirtualPS->DrawPolyMarker(np, &x[i1], &y[i1]);
      }
      i1 = -1;
      np = 0;
   }
   Modified();
}


//______________________________________________________________________________
void TPad::PaintText(Double_t x, Double_t y, const char *text)
{
   // Paint text in CurrentPad World coordinates.

   Modified();

   if (!gPad->IsBatch())
      GetPainter()->DrawText(x, y, text, TVirtualPadPainter::kClear);

   if (gVirtualPS) gVirtualPS->Text(x, y, text);
}


//______________________________________________________________________________
void TPad::PaintTextNDC(Double_t u, Double_t v, const char *text)
{
   // Paint text in CurrentPad NDC coordinates.

   Modified();

   if (!gPad->IsBatch())
      GetPainter()->DrawTextNDC(u, v, text, TVirtualPadPainter::kClear);

   if (gVirtualPS) {
      Double_t x = fX1 + u*(fX2 - fX1);
      Double_t y = fY1 + v*(fY2 - fY1);
      gVirtualPS->Text(x, y, text);
   }
}


//______________________________________________________________________________
TPad *TPad::Pick(Int_t px, Int_t py, TObjLink *&pickobj)
{
   // Search for an object at pixel position px,py.
   //
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
   // Search if point is in pad itself
   Double_t x = AbsPixeltoX(px);
   Double_t y = AbsPixeltoY(py);
   if (this != gPad->GetCanvas()) {
      if (!((x >= fX1 && x <= fX2) && (y >= fY1 && y <= fY2))) return 0;
   }

   // search for a primitive in this pad or its subpads
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

   //We can have 3d stuff in pad. If canvas prefers to draw
   //such stuff with OpenGL, the selection of 3d objects is
   //a gl viewer business so, in first cycle we do not
   //call DistancetoPrimitive for TAtt3D descendants.
   //In case of gl we first try to select 2d object first.

   while (lnk) {
      TObject *obj = lnk->GetObject();

      //If canvas prefers GL, all 3d objects must be drawn/selected by
      //gl viewer
      if (obj->InheritsFrom(TAtt3D::Class()) && fEmbeddedGL) {
         lnk = lnk->Prev();
         continue;
      }

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

   //if no primitive found, check if we have a TView
   //if yes, return the view except if you are in the lower or upper X range
   //of the pad.
   //In case canvas prefers gl, fView existance
   //automatically means viewer3d existance. (?)

   if (fView && !gotPrim) {
      Double_t dx = 0.05*(fUxmax-fUxmin);
      if ((x > fUxmin + dx) && (x < fUxmax-dx)) {

         if (fEmbeddedGL) {
            //No 2d stuff was selected, but we have gl-viewer. Let it select an object in
            //scene (or select itself). In any case it'll internally call
            //gPad->SetSelected(ptr) as, for example, hist painter does.
            py -= Int_t((1 - GetHNDC() - GetYlowNDC()) * GetWh());
            px -= Int_t(GetXlowNDC() * GetWw());
            fViewer3D->DistancetoPrimitive(px, py);
         }
         else
            dummyLink.SetObject(fView);
      }
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
   // Pop pad to the top of the stack.

   if (!fMother) return;
   if (!fPrimitives) fPrimitives = new TList;
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
   // Save Pad contents in a file in one of various formats.
   //
   //   if filename is "", the file produced is padname.ps
   //   if filename starts with a dot, the padname is added in front
   //   if filename contains .eps, an Encapsulated Postscript file is produced
   //   if filename contains .gif, a GIF file is produced
   //   if filename contains .gif+NN, an animated GIF file is produced
   //   if filename contains .C or .cxx, a C++ macro file is produced
   //   if filename contains .root, a Root file is produced
   //   if filename contains .xml,  a XML file is produced
   //
   //  See comments in TPad::SaveAs or the TPad::Print function below

   ((TPad*)this)->SaveAs(filename);
}


//______________________________________________________________________________
static Bool_t ContainsTImage(TList *li)
{
   // auxilary function. Returns kTRUE if list contains an object inherited
   // from TImage

   TIter next(li);
   TObject *obj;

   while ((obj = next())) {
      if (obj->InheritsFrom(TImage::Class())) {
         return kTRUE;
      } else if (obj->InheritsFrom(TPad::Class())) {
         if (ContainsTImage(((TPad*)obj)->GetListOfPrimitives())) {
            return kTRUE;
         }
      }
   }
   return kFALSE;
}


//______________________________________________________________________________
void TPad::Print(const char *filenam, Option_t *option)
{
   // Save Pad contents in a file in one of various formats.
   //
   //   if option  =  0   - as "ps"
   //               "ps"  - Postscript file is produced (see special cases below)
   //          "Portrait" - Postscript file is produced (Portrait)
   //         "Landscape" - Postscript file is produced (Landscape)
   //            "Title:" - The character strin after "Title:" becomes a table
   //                       of content entry.
   //               "eps" - an Encapsulated Postscript file is produced
   //           "Preview" - an Encapsulated Postscript file with preview is produced.
   //               "pdf" - a PDF file is produced
   //               "svg" - a SVG file is produced
   //               "gif" - a GIF file is produced
   //            "gif+NN" - an animated GIF file is produced, where NN is delay in 10ms units
   //               "xpm" - a XPM file is produced
   //               "png" - a PNG file is produced
   //               "jpg" - a JPEG file is produced.
   //                       NOTE: JPEG's lossy compression will make all sharp edges fuzzy.
   //              "tiff" - a TIFF file is produced
   //               "cxx" - a C++ macro file is produced
   //               "xml" - a XML file
   //              "root" - a ROOT binary file
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
   //  The "Preview" option allows to generate a preview (in the TIFF format) within
   //  the Encapsulated Postscript file. This preview can be used by programs like
   //  MSWord to visualize the picture on screen. The "Preview" option relies on the
   //  epstool command (http://www.cs.wisc.edu/~ghost/gsview/epstool.htm).
   //  Example:
   //     canvas->Print("example.eps","Preview");
   //
   //  To generate a Postscript file containing more than one picture, see
   //  class TPostScript.
   //
   //   Writing several canvases to the same Postscript or PDF file:
   //   ------------------------------------------------------------
   // if the Postscript or PDF file name finishes with "(", the file is not closed
   // if the Postscript or PDF file name finishes with ")" and the file has been opened
   // with "(", the file is closed. Example:
   //
   // {
   //    TCanvas c1("c1");
   //    h1.Draw();
   //    c1.Print("c1.ps("); //write canvas and keep the ps file open
   //    h2.Draw();
   //    c1.Print("c1.ps"); canvas is added to "c1.ps"
   //    h3.Draw();
   //    c1.Print("c1.ps)"); canvas is added to "c1.ps" and ps file is closed
   // }
   //
   //  In the previous example replacing "ps" by "pdf" will create a multi-pages PDF file.
   //
   //  Note that the following sequence writes the canvas to "c1.ps" and closes the ps file.:
   //    TCanvas c1("c1");
   //    h1.Draw();
   //    c1.Print("c1.ps");
   //
   //  The TCanvas::Print("file.ps(") mechanism is very useful, but it can be
   //  a little inconvenient to have the action of opening/closing a file
   //  being atomic with printing a page. Particularly if pages are being
   //  generated in some loop one needs to detect the special cases of first
   //  and last page and then munge the argument to Print() accordingly.
   //
   //  The "[" and "]" can be used instead of "(" and ")".  Example:
   //
   //    c1.Print("file.ps[");   // No actual print, just open file.ps
   //    for (int i=0; i<10; ++i) {
   //      // fill canvas for context i
   //      // ...
   //
   //      c1.Print("file.ps");  // actually print canvas to file
   //    }// end loop
   //    c1.Print("file.ps]");   // No actual print, just close.
   //
   // As before, the same macro is valid for PDF files.
   //
   // It is possible to print a pad into an animated GIF file by specifying the
   // file name as "myfile.gif+" or "myfile.gif+NN", where NN*10ms is delay
   // between the subimages' display. If NN is ommitted the delay between
   // subimages is zero. Each picture is added in the animation thanks to a loop
   // similar to the following one:
   //
   //    for (int i=0; i<10; ++i) {
   //      // fill canvas for context i
   //      // ...
   //
   //      c1.Print("file.gif+5");  // print canvas to GIF file with 50ms delays
   //    }// end loop
   //
   // The delay between each frame must be specified in each Print() statement.

   TString psname, fs1, fs2;
   char *filename;

   // "[" and "]" are special characters for ExpandPathName. When they are at the end
   // of the file name (see help) they must be removed before doing ExpandPathName.
   fs1 = filenam;
   if (fs1.EndsWith("[")) {
      fs1.Replace((fs1.Length()-1),1," ");
      fs2 = gSystem->ExpandPathName(fs1.Data());
      fs2.Replace((fs2.Length()-1),1,"[");
   } else if (fs1.EndsWith("]")) {
      fs1.Replace((fs1.Length()-1),1," ");
      fs2 = gSystem->ExpandPathName(fs1.Data());
      fs2.Replace((fs2.Length()-1),1,"]");
   } else {
      char* exppath = gSystem->ExpandPathName(fs1.Data());
      fs2 = exppath;
      delete [] exppath;
   }
   filename = (char*)fs2.Data();

   Int_t lenfil =  filename ? strlen(filename) : 0;
   const char *opt = option;
   Bool_t image = kFALSE;

   // Set the default option as "Postscript" (Should be a data member of TPad)
   const char *opt_default="ps";
   if( !opt ) opt = opt_default;

   if ( !lenfil )  {
      psname = GetName();
      psname += opt;
   } else {
      psname = filename;
   }

   // lines below protected against case like c1->SaveAs( "../ps/cs.ps" );
   if (psname.BeginsWith('.') && (psname.Contains('/') == 0)) {
      psname = GetName();
      psname.Append(filename);
      psname.Prepend("/");
      psname.Prepend(gEnv->GetValue("Canvas.PrintDirectory","."));
   }
   if (!gPad->IsBatch() && fCanvas)
      GetPainter()->SelectDrawable(GetCanvasID());

   // Save pad/canvas in alternative formats
   TImage::EImageFileTypes gtype = TImage::kUnknown;
   if (strstr(opt, "gif+")) {
      gtype = TImage::kAnimGif;
      image = kTRUE;
   } else if (strstr(opt, "gif")) {
      gtype = TImage::kGif;
      image = kTRUE;
   } else if (strstr(opt, "png")) {
      gtype = TImage::kPng;
      image = kTRUE;
   } else if (strstr(opt, "jpg")) {
      gtype = TImage::kJpeg;
      image = kTRUE;
   } else if (strstr(opt, "tiff")) {
      gtype = TImage::kTiff;
      image = kTRUE;
   } else if (strstr(opt, "xpm")) {
      gtype = TImage::kXpm;
      image = kTRUE;
   } else if (strstr(opt, "bmp")) {
      gtype = TImage::kBmp;
      image = kTRUE;
   }

   Int_t wid = 0;
   if (!GetCanvas()) return;
   if (!gROOT->IsBatch() && image) {
      if ((gtype == TImage::kGif) && !ContainsTImage(fPrimitives)) {
         wid = (this == GetCanvas()) ? GetCanvas()->GetCanvasID() : GetPixmapID();
         Color_t hc = gPad->GetCanvas()->GetHighLightColor();
         gPad->GetCanvas()->SetHighLightColor(-1);
         gPad->Modified();
         gPad->Update();
         GetPainter()->SelectDrawable(wid);
         GetPainter()->SaveImage(this, psname.Data(), gtype);
         if (!gSystem->AccessPathName(psname.Data())) {
            Info("Print", "GIF file %s has been created", psname.Data());
         }
         gPad->GetCanvas()->SetHighLightColor(hc);
         return;
      }
      if (gtype != TImage::kUnknown) {
         Color_t hc = gPad->GetCanvas()->GetHighLightColor();
         gPad->GetCanvas()->SetHighLightColor(-1);
         gPad->Modified();
         gPad->Update();
         if (gVirtualX->InheritsFrom("TGQt")) {
            wid = (this == GetCanvas()) ? GetCanvas()->GetCanvasID() : GetPixmapID();
            gVirtualX->WritePixmap(wid,UtoPixel(1.),VtoPixel(0.),(char *)psname.Data());
         } else {
            Int_t saver = gErrorIgnoreLevel;
            gErrorIgnoreLevel = kFatal;
            gVirtualX->Update(1);
            gSystem->Sleep(30); // syncronize
            GetPainter()->SaveImage(this, psname, gtype);
            gErrorIgnoreLevel = saver;
         }
         if (!gSystem->AccessPathName(psname)) {
            Info("Print", "file %s has been created", psname.Data());
         }
         gPad->GetCanvas()->SetHighLightColor(hc);
      } else {
         Warning("Print", "Unsupported image format %s", psname.Data());
      }
      return;
   }

   //==============Save pad/canvas as a C++ script==============================
   if (strstr(opt,"cxx")) {
      GetCanvas()->SaveSource(psname, "");
      return;
   }

   //==============Save pad/canvas as a root file===============================
   if (strstr(opt,"root")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,psname.Data(),"");
      return;
   }

   //==============Save pad/canvas as a XML file================================
   if (strstr(opt,"xml")) {
      // Plugin XML driver
      if (gDirectory) gDirectory->SaveObjectAs(this,psname.Data(),"");
      return;
   }

   //==============Save pad/canvas as a SVG file================================
   if (strstr(opt,"svg")) {
      gVirtualPS = (TVirtualPS*)gROOT->GetListOfSpecials()->FindObject(psname);

      Bool_t noScreen = kFALSE;
      if (!GetCanvas()->IsBatch() && GetCanvas()->GetCanvasID() == -1) {
         noScreen = kTRUE;
         GetCanvas()->SetBatch(kTRUE);
      }

      TPad *padsav = (TPad*)gPad;
      cd();
      TVirtualPS *psave = gVirtualPS;

      if (!gVirtualPS) {
         // Plugin Postscript/SVG driver
         TPluginHandler *h;
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualPS", "svg"))) {
            if (h->LoadPlugin() == -1)
               return;
            h->ExecPlugin(0);
         }
      }

      // Create a new SVG file
      gVirtualPS->SetName(psname);
      gVirtualPS->Open(psname);
      gVirtualPS->SetBit(kPrintingPS);
      gVirtualPS->NewPage();
      Paint();
      if (noScreen)  GetCanvas()->SetBatch(kFALSE);

      if (!gSystem->AccessPathName(psname)) Info("Print", "SVG file %s has been created", psname.Data());

      delete gVirtualPS;
      gVirtualPS = psave;
      gVirtualPS = 0;
      padsav->cd();

      return;
   }

   //==============Save pad/canvas as a Postscript file=========================

   // in case we read directly from a Root file and the canvas
   // is not on the screen, set batch mode

   char *l;
   Bool_t mustOpen  = kTRUE;
   Bool_t mustClose = kTRUE;
   char *copen=0, *cclose=0, *copenb=0, *ccloseb=0;
   if (!image) {
      // The parenthesis mechanism is only valid for PS and PDF files.
      copen   = (char*)strstr(psname.Data(),"("); if (copen)   *copen   = 0;
      cclose  = (char*)strstr(psname.Data(),")"); if (cclose)  *cclose  = 0;
      copenb  = (char*)strstr(psname.Data(),"["); if (copenb)  *copenb  = 0;
      ccloseb = (char*)strstr(psname.Data(),"]"); if (ccloseb) *ccloseb = 0;
   }
   gVirtualPS = (TVirtualPS*)gROOT->GetListOfSpecials()->FindObject(psname);
   if (gVirtualPS) {mustOpen = kFALSE; mustClose = kFALSE;}
   if (copen  || copenb)  mustClose = kFALSE;
   if (cclose || ccloseb) mustClose = kTRUE;

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
   if (strstr(opt,"eps"))       pstype = 113;
   if (strstr(opt,"Preview"))   pstype = 113;
   TPad *padsav = (TPad*)gPad;
   cd();
   TVirtualPS *psave = gVirtualPS;

   if (!gVirtualPS || mustOpen) {
      // Plugin Postscript driver
      TPluginHandler *h;
      if (strstr(opt,"pdf") || strstr(opt,"Title:")) {
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualPS", "pdf"))) {
            if (h->LoadPlugin() == -1) return;
            h->ExecPlugin(0);
         }
      } else if (image) {
         // Plugin TImageDump driver
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualPS", "image"))) {
            if (h->LoadPlugin() == -1) return;
            h->ExecPlugin(0);
         }
      } else {
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualPS", "ps"))) {
            if (h->LoadPlugin() == -1) return;
            h->ExecPlugin(0);
         }
      }

      // Create a new Postscript, PDF or image file
      gVirtualPS->SetName(psname);
      l = (char*)strstr(opt,"Title:");
      if (l) {
         gVirtualPS->SetTitle(&opt[6]);
         //Please fix this bug, we may overwrite an input argument
         strcpy(l,"pdf");
      }
      gVirtualPS->Open(psname,pstype);
      gVirtualPS->SetBit(kPrintingPS);
      if (!copenb) {
         if (!strstr(opt,"pdf") || image) gVirtualPS->NewPage();
         Paint();
      }
      if (noScreen) GetCanvas()->SetBatch(kFALSE);
      if (!gSystem->AccessPathName(psname)) Info("Print", "%s file %s has been created", opt, psname.Data());

      if (mustClose) {
         gROOT->GetListOfSpecials()->Remove(gVirtualPS);
         delete gVirtualPS;
         gVirtualPS = psave;
      } else {
         gROOT->GetListOfSpecials()->Add(gVirtualPS);
         gVirtualPS = 0;
      }
   } else {
      // Append to existing Postscript, PDF or GIF file
      if (!ccloseb) {
         gVirtualPS->NewPage();
         Paint();
      }
      l = (char*)strstr(opt,"Title:");
      if (l) {
         gVirtualPS->SetTitle(&opt[6]);
         //Please fix this bug, we may overwrite an input argument
         strcpy(l,"pdf");
      }
      Info("Print", "Current canvas added to %s file %s", opt, psname.Data());
      if (mustClose) {
         gROOT->GetListOfSpecials()->Remove(gVirtualPS);
         delete gVirtualPS;
         gVirtualPS = 0;
      } else {
         gVirtualPS = 0;
      }
   }

   if (strstr(opt,"Preview")) gSystem->Exec(Form("epstool --quiet -t6p %s %s",psname.Data(),psname.Data()));

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

   if (gPad == this)
      GetPainter()->InvalidateCS();

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
   // Recursively remove object from a pad and its subpads.

   if (obj == fCanvas->GetSelected()) fCanvas->SetSelected(0);
   if (obj == fCanvas->GetClickSelected()) fCanvas->SetClickSelected(0);
   if (obj == fView) fView = 0;
   if (!fPrimitives) return;
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
   //  By default, if the pad has the options gridx or/and gridy activated,
   //  the grid is not drawn by this function.
   //  if option="g" is specified, this will force the drawing of the grid
   //  on top of the picture

   // get first histogram in the list of primitives
   TString opt = option;
   opt.ToLower();

   TPad *padsav = (TPad*)gPad;
   cd();

   if (!fPrimitives) fPrimitives = new TList;
   TIter next(fPrimitives);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TH1::Class())) {
         TH1 *hobj = (TH1*)obj;
         if (opt.Contains("g")) hobj->DrawCopy("sameaxig");
         else                   hobj->DrawCopy("sameaxis");
         return;
      }
      if (obj->InheritsFrom(TMultiGraph::Class())) {
         TMultiGraph *mg = (TMultiGraph*)obj;
         if (mg) mg->GetHistogram()->DrawCopy("sameaxis");
         return;
      }
      if (obj->InheritsFrom(TGraph::Class())) {
         TGraph *g = (TGraph*)obj;
         if (g) g->GetHistogram()->DrawCopy("sameaxis");
         return;
      }
      if (obj->InheritsFrom(THStack::Class())) {
         THStack *hs = (THStack*)obj;
         if (hs) hs->GetHistogram()->DrawCopy("sameaxis");
         return;
      }
   }

   if (padsav) padsav->cd();
}


//______________________________________________________________________________
void TPad::ResizePad(Option_t *option)
{
   // Compute pad conversion coefficients.
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

   // Recompute subpad positions in case pad has been moved/resized
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

   // Linear X axis
   Double_t rounding = 0.00005;
   Double_t xrange  = fX2 - fX1;
   fXtoAbsPixelk = rounding + pxlow - pxrange*fX1/xrange;      //origin at left
   fXtoPixelk = rounding +  -pxrange*fX1/xrange;
   fXtoPixel  = pxrange/xrange;
   fAbsPixeltoXk = fX1 - pxlow*xrange/pxrange;
   fPixeltoXk = fX1;
   fPixeltoX  = xrange/pxrange;
   // Linear Y axis
   Double_t yrange  = fY2 - fY1;
   fYtoAbsPixelk = rounding + pylow - pyrange*fY1/yrange;      //origin at top
   fYtoPixelk = rounding +  -pyrange - pyrange*fY1/yrange;
   fYtoPixel  = pyrange/yrange;
   fAbsPixeltoYk = fY1 - pylow*yrange/pyrange;
   fPixeltoYk = fY1;
   fPixeltoY  = yrange/pyrange;

   // Coefficients to convert from pad NDC coordinates to pixel coordinates

   fUtoAbsPixelk = rounding + pxlow;
   fUtoPixelk = rounding;
   fUtoPixel  = pxrange;
   fVtoAbsPixelk = rounding + pylow;
   fVtoPixelk = -pyrange;
   fVtoPixel  = pyrange;

   // Coefficients to convert from canvas pixels to pad world coordinates

   // Resize all subpads
   TObject *obj;
   if (!fPrimitives) fPrimitives = new TList;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class()))
         ((TPad*)obj)->ResizePad(option);
   }

   // Reset all current sizes
   if (gPad->IsBatch())
      fPixmapID = 0;
   else {
      GetPainter()->SetLineWidth(-1);
      GetPainter()->SetTextSize(-1);

      // create or re-create off-screen pixmap
      if (fPixmapID) {
         int w = TMath::Abs(XtoPixel(fX2) - XtoPixel(fX1));
         int h = TMath::Abs(YtoPixel(fY2) - YtoPixel(fY1));
         //protection in case of wrong pad parameters.
         //without this protection, the OpenPixmap or ResizePixmap crashes with
         //the message "Error in <RootX11ErrorHandler>: BadValue (integer parameter out of range for operation)"
         //resulting in a frozen xterm
         if (!(TMath::Finite(fX1)) || !(TMath::Finite(fX2))
             || !(TMath::Finite(fY1)) || !(TMath::Finite(fY2)))
            Warning("ResizePad", "Inf/NaN propagated to the pad. Check drawn objects.");
         if (w <= 0 || w > 10000) {
            Warning("ResizePad", "%s width changed from %d to %d\n",GetName(),w,10);
            w = 10;
         }
         if (h <= 0 || h > 10000) {
            Warning("ResizePad", "%s height changed from %d to %d\n",GetName(),h,10);
            h = 10;
         }
         if (fPixmapID == -1) {      // this case is handled via the ctor
            fPixmapID = GetPainter()->CreateDrawable(w, h);
         } else {
            if (gVirtualX->ResizePixmap(fPixmapID, w, h)) {
               Modified(kTRUE);
            }
         }
      }
   }
   if (fView) {
      TPad *padsav  = (TPad*)gPad;
      if (padsav == this) {
         fView->ResizePad();
      } else {
         cd();
         fView->ResizePad();
         padsav->cd();
      }
   }
}


//______________________________________________________________________________
void TPad::SaveAs(const char *filename, Option_t * /*option*/) const
{
   // Save Pad contents in a file in one of various formats.
   //
   //   if filename is "", the file produced is padname.ps
   //   if filename starts with a dot, the padname is added in front
   //   if filename contains .eps, an Encapsulated Postscript file is produced
   //   if filename contains .pdf, a PDF file is produced
   //   if filename contains .svg, a SVG file is produced
   //   if filename contains .gif, a GIF file is produced
   //   if filename contains .gif+NN, an  animated GIF file is produced
   //   if filename contains .xpm, a XPM file is produced
   //   if filename contains .png, a PNG file is produced
   //   if filename contains .jpg, a JPEG file is produced
   //     NOTE: JPEG's lossy compression will make all sharp edges fuzzy.
   //   if filename contains .tiff, a TIFF file is produced
   //   if filename contains .C or .cxx, a C++ macro file is produced
   //   if filename contains .root, a Root file is produced
   //   if filename contains .xml, a XML file is produced
   //
   //   See comments in TPad::Print for the Postscript formats

   TString psname;
   Int_t lenfil =  filename ? strlen(filename) : 0;

   if (!lenfil)  { psname = GetName(); psname.Append(".ps"); }
   else            psname = filename;

   // lines below protected against case like c1->SaveAs( "../ps/cs.ps" );
   if (psname.BeginsWith('.') && (psname.Contains('/') == 0)) {
      psname = GetName();
      psname.Append(filename);
      psname.Prepend("/");
      psname.Prepend(gEnv->GetValue("Canvas.PrintDirectory","."));
   }

   if (psname.EndsWith(".gif"))
      ((TPad*)this)->Print(psname,"gif");
   else if (psname.Contains(".gif+"))
      ((TPad*)this)->Print(psname,"gif+");
   else if (psname.EndsWith(".C") || psname.EndsWith(".cxx") || psname.EndsWith(".cpp"))
      ((TPad*)this)->Print(psname,"cxx");
   else if (psname.EndsWith(".root"))
      ((TPad*)this)->Print(psname,"root");
   else if (psname.EndsWith(".xml"))
      ((TPad*)this)->Print(psname,"xml");
   else if (psname.EndsWith(".eps"))
      ((TPad*)this)->Print(psname,"eps");
   else if (psname.EndsWith(".pdf"))
      ((TPad*)this)->Print(psname,"pdf");
   else if (psname.EndsWith(".pdf["))
      ((TPad*)this)->Print(psname,"pdf");
   else if (psname.EndsWith(".pdf]"))
      ((TPad*)this)->Print(psname,"pdf");
   else if (psname.EndsWith(".pdf("))
      ((TPad*)this)->Print(psname,"pdf");
   else if (psname.EndsWith(".pdf)"))
      ((TPad*)this)->Print(psname,"pdf");
   else if (psname.EndsWith(".svg"))
      ((TPad*)this)->Print(psname,"svg");
   else if (psname.EndsWith(".xpm"))
      ((TPad*)this)->Print(psname,"xpm");
   else if (psname.EndsWith(".png"))
      ((TPad*)this)->Print(psname,"png");
   else if (psname.EndsWith(".jpg"))
      ((TPad*)this)->Print(psname,"jpg");
   else if (psname.EndsWith(".jpeg"))
      ((TPad*)this)->Print(psname,"jpg");
   else if (psname.EndsWith(".bmp"))
      ((TPad*)this)->Print(psname,"bmp");
   else if (psname.EndsWith(".tiff"))
      ((TPad*)this)->Print(psname,"tiff");
   else
      ((TPad*)this)->Print(psname,"ps");
}


//______________________________________________________________________________
void TPad::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   // Save primitives in this pad on the C++ source file out.

   TPad *padsav = (TPad*)gPad;
   gPad = this;
   char quote='"';
   char lcname[10];
   const char *cname = GetName();
   Int_t nch = strlen(cname);
   if (nch < 10) {
      strlcpy(lcname,cname,10);
      for (Int_t k=1;k<=nch;k++) {if (lcname[nch-k] == ' ') lcname[nch-k] = 0;}
      if (lcname[0] == 0) {
         if (this == gPad->GetCanvas()) {strlcpy(lcname,"c1",10);  nch = 2;}
         else                           {strlcpy(lcname,"pad",10); nch = 3;}
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
      out<<"   TView *view = TView::CreateView(1);"<<endl;
      out<<"   view->SetRange("<<rmin[0]<<","<<rmin[1]<<","<<rmin[2]<<","
                               <<rmax[0]<<","<<rmax[1]<<","<<rmax[2]<<");"<<endl;
   }
   if (GetFillColor() != 19) {
      if (GetFillColor() > 228) {
         TColor::SaveColor(out, GetFillColor());
         out<<"   "<<cname<<"->SetFillColor(ci);" << endl;
      } else
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
      out<<"   "<<cname<<"->SetTickx("<<GetTickx()<<");"<<endl;
   }
   if (GetTicky()) {
      out<<"   "<<cname<<"->SetTicky("<<GetTicky()<<");"<<endl;
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
      if (GetFrameFillColor() > 228) {
         TColor::SaveColor(out, GetFrameFillColor());
         out<<"   "<<cname<<"->SetFrameFillColor(ci);" << endl;
      } else
         out<<"   "<<cname<<"->SetFrameFillColor("<<GetFrameFillColor()<<");"<<endl;
   }
   if (GetFrameFillStyle() != 1001) {
      out<<"   "<<cname<<"->SetFrameFillStyle("<<GetFrameFillStyle()<<");"<<endl;
   }
   if (GetFrameLineStyle() != 1) {
      out<<"   "<<cname<<"->SetFrameLineStyle("<<GetFrameLineStyle()<<");"<<endl;
   }
   if (GetFrameLineColor() != 1) {
      if (GetFrameLineColor() > 228) {
         TColor::SaveColor(out, GetFrameLineColor());
         out<<"   "<<cname<<"->SetFrameLineColor(ci);" << endl;
      } else
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
         if (frame->GetFillColor() > 228) {
            TColor::SaveColor(out, frame->GetFillColor());
            out<<"   "<<cname<<"->SetFrameFillColor(ci);" << endl;
         } else
            out<<"   "<<cname<<"->SetFrameFillColor("<<frame->GetFillColor()<<");"<<endl;
      }
      if (frame->GetFillStyle() != 1001) {
         out<<"   "<<cname<<"->SetFrameFillStyle("<<frame->GetFillStyle()<<");"<<endl;
      }
      if (frame->GetLineStyle() != 1) {
         out<<"   "<<cname<<"->SetFrameLineStyle("<<frame->GetLineStyle()<<");"<<endl;
      }
      if (frame->GetLineColor() != 1) {
         if (frame->GetLineColor() > 228) {
            TColor::SaveColor(out, frame->GetLineColor());
            out<<"   "<<cname<<"->SetFrameLineColor(ci);" << endl;
         } else
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
   // If a pad is not editable:
   // - one cannot modify the pad and its objects via the mouse.
   // - one cannot add new objects to the pad

   fEditable = mode;

   TObject *obj;
   if (!fPrimitives) fPrimitives = new TList;
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
   // Set Lin/Log scale for X
   //   value = 0 X scale will be linear
   //   value = 1 X scale will be logarithmic (base 10)
   //   value > 1 reserved for possible support of base e or other

   fLogx = value;
   delete fView; fView=0;
   Modified();
}


//______________________________________________________________________________
void TPad::SetLogy(Int_t value)
{
   // Set Lin/Log scale for Y
   //   value = 0 Y scale will be linear
   //   value = 1 Y scale will be logarithmic (base 10)
   //   value > 1 reserved for possible support of base e or other

   fLogy = value;
   delete fView; fView=0;
   Modified();
}


//______________________________________________________________________________
void TPad::SetLogz(Int_t value)
{
   // Set Lin/Log scale for Z

   fLogz = value;
   delete fView; fView=0;
   Modified();
}


//______________________________________________________________________________
void TPad::SetPad(Double_t xlow, Double_t ylow, Double_t xup, Double_t yup)
{
   // Set canvas range for pad and resize the pad. If the aspect ratio
   // was fixed before the call it will be un-fixed.

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
   // Set all pad parameters.

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
void TPad::SetView(TView *view)
{
   // Set the current TView. Delete previous view if view=0

   if (!view) delete fView;
   fView = view;
}

//______________________________________________________________________________
void TPad::SetAttFillPS(Color_t color, Style_t style)
{
   // Set postscript fill area attributes.

   if (gVirtualPS) {
      gVirtualPS->SetFillColor(color);
      gVirtualPS->SetFillStyle(style);
   }
}


//______________________________________________________________________________
void TPad::SetAttLinePS(Color_t color, Style_t style, Width_t lwidth)
{
   // Set postscript line attributes.

   if (gVirtualPS) {
      gVirtualPS->SetLineColor(color);
      gVirtualPS->SetLineStyle(style);
      gVirtualPS->SetLineWidth(lwidth);
   }
}


//______________________________________________________________________________
void TPad::SetAttMarkerPS(Color_t color, Style_t style, Size_t msize)
{
   // Set postscript marker attributes.

   if (gVirtualPS) {
      gVirtualPS->SetMarkerColor(color);
      gVirtualPS->SetMarkerStyle(style);
      gVirtualPS->SetMarkerSize(msize);
   }
}


//______________________________________________________________________________
void TPad::SetAttTextPS(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize)
{
   // Set postscript text attributes.

   if (gVirtualPS) {
      gVirtualPS->SetTextAlign(align);
      gVirtualPS->SetTextAngle(angle);
      gVirtualPS->SetTextColor(color);
      gVirtualPS->SetTextFont(font);
      if (font%10 > 2) {
         Float_t wh = (Float_t)gPad->XtoPixel(gPad->GetX2());
         Float_t hh = (Float_t)gPad->YtoPixel(gPad->GetY1());
         Float_t dy;
         if (wh < hh)  {
            dy = AbsPixeltoX(Int_t(tsize)) - AbsPixeltoX(0);
            tsize = dy/(fX2-fX1);
         } else {
            dy = AbsPixeltoY(0) - AbsPixeltoY(Int_t(tsize));
            tsize = dy/(fY2-fY1);
         }
      }
      gVirtualPS->SetTextSize(tsize);
   }
}


//______________________________________________________________________________
Bool_t TPad::HasCrosshair() const
{
   // Return kTRUE if the crosshair has been activated (via SetCrosshair).

   return (Bool_t)GetCrosshair();
}


//______________________________________________________________________________
Int_t TPad::GetCrosshair() const
{
   // Return the crosshair type (from the mother canvas)
   // crosshair type = 0 means no crosshair.

   if (this == (TPad*)fCanvas)
      return fCrosshair;
   return fCanvas ? fCanvas->GetCrosshair() : 0;
}


//______________________________________________________________________________
void TPad::SetCrosshair(Int_t crhair)
{
   // Set crosshair active/inactive.
   // If crhair != 0, a crosshair will be drawn in the pad and its subpads.
   // If the canvas crhair = 1 , the crosshair spans the full canvas.
   // If the canvas crhair > 1 , the crosshair spans only the pad.

   fCrosshair = crhair;
   fCrosshairPos = 0;

   if (this != (TPad*)fCanvas) fCanvas->SetCrosshair(crhair);
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


//______________________________________________________________________________
void TPad::SetVertical(Bool_t vert)
{
   // Set pad vertical (default) or horizontal

   if (vert) ResetBit(kHori);
   else      SetBit(kHori);
}


//_______________________________________________________________________
void TPad::Streamer(TBuffer &b)
{
   // Stream a class object.

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
         gReadLevel++;
         gROOT->SetReadingObject(kTRUE);

         b.ReadClassBuffer(TPad::Class(), this, v, R__s, R__c);

         //Set the kCanDelete bit in all objects in the pad such that when the pad
         //is deleted all objects in the pad are deleted too.
         TIter next(fPrimitives);
         while ((obj = next())) {
            obj->SetBit(kCanDelete);
         }

         fModified = kTRUE;
         fPadPointer = 0;
         gReadLevel--;
         if (gReadLevel == 0 && IsA() == TPad::Class()) ResizePad();
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
      if (gReadLevel == 0) fMother = (TPad*)gROOT->GetSelectedPad();
      else                fMother = (TPad*)gPad;
      if (!fMother) fMother = (TPad*)gPad;
      if (fMother)  fCanvas = fMother->GetCanvas();
      gPad      = fMother;
      fPixmapID = -1;      // -1 means pixmap will be created by ResizePad()
      //-------------------------
      // read objects and their drawing options
      //      b >> fPrimitives;
      gReadLevel++;
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
      gReadLevel--;
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
      if (gReadLevel == 0 && IsA() == TPad::Class()) ResizePad();
      b.CheckByteCount(R__s, R__c, TPad::IsA());
      //====end of old versions

   } else {
      b.WriteClassBuffer(TPad::Class(),this);
   }
}


//______________________________________________________________________________
void TPad::UseCurrentStyle()
{
   // Force a copy of current style for all objects in pad.

   if (gStyle->IsReading()) {
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
   } else {
      gStyle->SetPadColor(GetFillColor());
      gStyle->SetPadBottomMargin(GetBottomMargin());
      gStyle->SetPadTopMargin(GetTopMargin());
      gStyle->SetPadLeftMargin(GetLeftMargin());
      gStyle->SetPadRightMargin(GetRightMargin());
      gStyle->SetPadBorderSize(GetBorderSize());
      gStyle->SetPadBorderMode(GetBorderMode());
      gStyle->SetPadGridX(fGridx);
      gStyle->SetPadGridY(fGridy);
      gStyle->SetPadTickX(fTickx);
      gStyle->SetPadTickY(fTicky);
      gStyle->SetOptLogx (fLogx);
      gStyle->SetOptLogy (fLogy);
      gStyle->SetOptLogz (fLogz);
   }

   if (!fPrimitives) fPrimitives = new TList;
   TIter next(GetListOfPrimitives());
   TObject *obj;

   while ((obj = next())) {
      obj->UseCurrentStyle();
   }

   TPaveText *title  = (TPaveText*)FindObject("title");
   if (title) {
      if (gStyle->IsReading()) {
         title->SetFillColor(gStyle->GetTitleFillColor());
         title->SetTextFont(gStyle->GetTitleFont(""));
         title->SetTextColor(gStyle->GetTitleTextColor());
         title->SetBorderSize(gStyle->GetTitleBorderSize());
         if (!gStyle->GetOptTitle()) delete title;
      } else {
         gStyle->SetTitleFillColor(title->GetFillColor());
         gStyle->SetTitleFont(title->GetTextFont());
         gStyle->SetTitleTextColor(title->GetTextColor());
         gStyle->SetTitleBorderSize(title->GetBorderSize());
      }
   }
   if (fFrame) fFrame->UseCurrentStyle();

   if (gStyle->IsReading()) Modified();
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
   //                            // If a double-click with the mouse is executed
   //                            // in the pad or any key pressed, the function
   //                            // returns 0.
   //   c1.WaitPrimitive("ggg"); // Set the editor in mode "PolyLine/Graph"
   //                            // Create a polyline, then using the context
   //                            // menu item "SetName", change the name
   //                            // of the created TGraph to "ggg"
   //   c1.WaitPrimitive("TArc");// Set the editor in mode "Arc". Returns
   //                            // as soon as a TArc object is created.
   //   c1.WaitPrimitive("lat","Text"); // Set the editor in Text/Latex mode.
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

   if (!fPrimitives) fPrimitives = new TList;
   gSystem->ProcessEvents();
   TObject *oldlast = gPad->GetListOfPrimitives()->Last();
   TObject *obj = 0;
   Bool_t testlast = kFALSE;
   Bool_t hasname = strlen(pname) > 0;
   if (strlen(pname) == 0 && strlen(emode) == 0) testlast = kTRUE;
   if (testlast) gROOT->SetEditorMode();
   while (!gSystem->ProcessEvents() && gROOT->GetSelectedPad()) {
      if (gROOT->GetEditorMode() == 0) {
         if (hasname) {
            obj = FindObject(pname);
            if (obj) return obj;
         }
         if (testlast) {
            obj = gPad->GetListOfPrimitives()->Last();
            if (obj != oldlast) return obj;
            Int_t event = GetEvent();
            if (event == kButton1Double || event == kKeyPress) {
               //the following statement is required against other loop executions
               //before returning
               fCanvas->HandleInput((EEventType)-1,0,0);
               return 0;
            }
         }
      }
      gSystem->Sleep(10);
   }

   return 0;
}


//______________________________________________________________________________
TObject *TPad::CreateToolTip(const TBox *box, const char *text, Long_t delayms)
{
   // Create a tool tip and return its pointer.

   if (gPad->IsBatch()) return 0;
   // return new TGToolTip(box, text, delayms);
   return (TObject*)gROOT->ProcessLineFast(Form("new TGToolTip((TBox*)0x%lx,\"%s\",%d)",
                                           (Long_t)box,text,(Int_t)delayms));
}


//______________________________________________________________________________
void TPad::DeleteToolTip(TObject *tip)
{
   // Delete tool tip object.

   // delete tip;
   if (!tip) return;
   gROOT->ProcessLineFast(Form("delete (TGToolTip*)0x%lx", (Long_t)tip));
}


//______________________________________________________________________________
void TPad::ResetToolTip(TObject *tip)
{
   // Reset tool tip, i.e. within time specified in CreateToolTip the
   // tool tip will pop up.

   if (!tip) return;
   // tip->Reset(this);
   gROOT->ProcessLineFast(Form("((TGToolTip*)0x%lx)->Reset((TPad*)0x%lx)",
                          (Long_t)tip,(Long_t)this));
}


//______________________________________________________________________________
void TPad::CloseToolTip(TObject *tip)
{
   // Hide tool tip.

   if (!tip) return;
   // tip->Hide();
   gROOT->ProcessLineFast(Form("((TGToolTip*)0x%lx)->Hide()",(Long_t)tip));
}


//______________________________________________________________________________
void TPad::x3d(Option_t *type)
{
   // Depreciated: use TPad::GetViewer3D() instead

   ::Info("TPad::x3d()", "Fn is depreciated - use TPad::GetViewer3D() instead");

   // Default on GetViewer3D is pad - for x3d
   // it was x3d...
   if (!type || !type[0]) {
      type = "x3d";
   }
   GetViewer3D(type);
}


//______________________________________________________________________________
TVirtualViewer3D *TPad::GetViewer3D(Option_t *type)
{
   // Create/obtain handle to 3D viewer. Valid types are:
   //    'pad' - pad drawing via TViewer3DPad
   //    any others registered with plugin manager supporting TVirtualViewer3D
   // If an invalid/null type is requested then the current viewer is returned
   // (if any), otherwise a default 'pad' type is returned

   Bool_t validType = kFALSE;

   if ( (!type || !type[0] || (strstr(type, "gl") && !strstr(type, "ogl"))) && !fCanvas->UseGL())
      type = "pad";

   if (type && type[0]) {

      if (gPluginMgr->FindHandler("TVirtualViewer3D", type))
         validType = kTRUE;

   }

   // Invalid/null type requested?
   if (!validType) {
      // Return current viewer if there is one
      if (fViewer3D) {
         return fViewer3D;
      }
      // otherwise default to the pad
      else {
         type = "pad";
      }
   }

   // Ensure we can create the new viewer before removing any exisiting one
   TVirtualViewer3D *newViewer = 0;

   Bool_t createdExternal = kFALSE;

   // External viewers need to be created via plugin manager via interface...
   if (!strstr(type,"pad")) {
      newViewer = TVirtualViewer3D::Viewer3D(this,type);

      if (!newViewer) {
         Warning("TPad::CreateViewer3D", "Cannot create 3D viewer of type: %s", type);

         // Return the existing viewer
         return fViewer3D;
      }

      if (strstr(type, "gl") && !strstr(type, "ogl"))
         fEmbeddedGL = kTRUE, fCopyGLDevice = kTRUE, Modified();
      else
         createdExternal = kTRUE;

   } else
      newViewer = new TViewer3DPad(*this);

   // If we had a previous viewer destroy it now
   // In this case we do take responsibility for destorying viewer
   // c.f. ReleaseViewer3D
   delete fViewer3D;

   // Set and return new viewer
   fViewer3D = newViewer;

   // Ensure any new external viewer is painted
   // For internal TViewer3DPad type we assume this is being
   // create on demand due to a paint - so this is not required
   if (createdExternal) {
      Modified();
      Update();
   }

   return fViewer3D;
}

//______________________________________________________________________________
void TPad::ReleaseViewer3D(Option_t * /*type*/ )
{
   // Release current (external) viewer
   // TODO: By type
   fViewer3D = 0;

   // We would like to ensure the pad is repainted
   // when external viewer is closed down. However
   // a modify/paint call here will repaint the pad
   // before the external viewer window actually closes.
   // So the pad would have to be redraw twice over.
   // Currenltly we just have to live with the pad staying blank
   // any click in pad will refresh.
}


//______________________________________________________________________________
Int_t TPad::GetGLDevice()
{
   // Get GL device.
   return fGLDevice;
}

//______________________________________________________________________________
void TPad::RecordPave(const TObject *obj)
{
   // Emit RecordPave() signal.

   Emit("RecordPave(const TObject*)", (Long_t)obj);
}

//______________________________________________________________________________
void TPad::RecordLatex(const TObject *obj)
{
   // Emit RecordLatex() signal.

   Emit("RecordLatex(const TObject*)", (Long_t)obj);
}

//______________________________________________________________________________
TVirtualPadPainter *TPad::GetPainter()
{
   // Get pad painter from TCanvas.

   if (!fCanvas) return 0;
   return fCanvas->GetCanvasPainter();
}
