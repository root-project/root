// @(#)root/gpad:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <locale>
#include <memory>

#include "TROOT.h"
#include "TBuffer.h"
#include "TError.h"
#include "TMath.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
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
#include "TPaveStats.h"
#include "TGroupButton.h"
#include "TBrowser.h"
#include "TVirtualGL.h"
#include "TString.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TDataType.h"
#include "TFrame.h"
#include "TWbox.h"
#include "TExec.h"
#include "TDatime.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TCanvasImp.h"
#include "TPluginManager.h"
#include "TEnv.h"
#include "TImage.h"
#include "TViewer3DPad.h"
#include "TCreatePrimitives.h"
#include "TLegend.h"
#include "TAtt3D.h"
#include "TVirtualPadPainter.h"
#include "strlcpy.h"
#include "snprintf.h"

#include "TVirtualMutex.h"

static Int_t gReadLevel = 0;

Int_t TPad::fgMaxPickDistance = 5;

/** \class TPad
\ingroup gpad

The most important graphics class in the ROOT system.

A Pad is contained in a Canvas.

A Pad may contain other pads (unlimited pad hierarchy).

A pad is a linked list of primitives of any type (graphics objects,
histograms, detectors, tracks, etc.).

Adding a new element into a pad is in general performed by the Draw
member function of the object classes.

It is important to realize that the pad is a linked list of references
to the original object.
For example, in case of a histogram, the histogram.Draw() operation
only stores a reference to the histogram object and not a graphical
representation of this histogram.
When the mouse is used to change (say the bin content), the bin content
of the original histogram is changed.

The convention used in ROOT is that a Draw operation only adds
a reference to the object. The effective drawing is performed
when the canvas receives a signal to be painted.

\image html gpad_pad1.png

This signal is generally sent when typing carriage return in the
command input or when a graphical operation has been performed on one
of the pads of this canvas.
When a Canvas/Pad is repainted, the member function Paint for all
objects in the Pad linked list is invoked.

\image html gpad_pad2.png

When the mouse is moved on the Pad, The member function DistancetoPrimitive
is called for all the elements in the pad. DistancetoPrimitive returns
the distance in pixels to this object.

When the object is within the distance window, the member function
ExecuteEvent is called for this object.

In ExecuteEvent, move, changes can be performed on the object.

For examples of DistancetoPrimitive and ExecuteEvent functions,
see classes
~~~ {.cpp}
      TLine::DistancetoPrimitive, TLine::ExecuteEvent
      TBox::DistancetoPrimitive,  TBox::ExecuteEvent
      TH1::DistancetoPrimitive,   TH1::ExecuteEvent
~~~
A Pad supports linear and log scales coordinate systems.
The transformation coefficients are explained in TPad::ResizePad.
*/

////////////////////////////////////////////////////////////////////////////////
/// Pad default constructor.

TPad::TPad()
{
   fModified   = kTRUE;
   fTip        = nullptr;
   fPadPointer = nullptr;
   fPrimitives = nullptr;
   fExecs      = nullptr;
   fCanvas     = nullptr;
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
   fPadView3D  = nullptr;
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
   fYtoPixelk    = 0.;
   fXUpNDC       = 0.;
   fYUpNDC       = 0.;

   fFixedAspectRatio = kFALSE;
   fAspectRatio      = 0.;

   fNumPaletteColor = 0;
   fNextPaletteColor = 0;
   fCGnx = 0;
   fCGny = 0;

   fLogx  = 0;
   fLogy  = 0;
   fLogz  = 0;
   fGridx = false;
   fGridy = false;
   fTickx = 0;
   fTicky = 0;
   fFrame = nullptr;
   fView  = nullptr;

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

   fViewer3D = nullptr;
   SetBit(kMustCleanup);

   // the following line is temporarily disabled. It has side effects
   // when the pad is a TDrawPanelHist or a TFitPanel.
   // the line was supposed to fix a problem with DrawClonePad
   //   gROOT->SetSelectedPad(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Pad constructor.
///
///  A pad is a linked list of primitives.
///  A pad is contained in a canvas. It may contain other pads.
///  A pad has attributes. When a pad is created, the attributes
///  defined in the current style are copied to the pad attributes.
///
/// \param[in] name        pad name
/// \param[in] title       pad title
/// \param[in] xlow [0,1]  is the position of the bottom left point of the pad
///             expressed  in the mother pad reference system
/// \param[in] ylow [0,1]  is the Y position of this point.
/// \param[in] xup  [0,1]  is the x position of the top right point of the pad
///                        expressed in the mother pad reference system
/// \param[in] yup  [0,1]  is the Y position of this point.
/// \param[in] color       pad color
/// \param[in] bordersize  border size in pixels
/// \param[in] bordermode  border mode
///                        - bordermode = -1 box looks as it is behind the screen
///                        - bordermode = 0  no special effects
///                        - bordermode = 1  box looks as it is in front of the screen

TPad::TPad(const char *name, const char *title, Double_t xlow,
           Double_t ylow, Double_t xup, Double_t yup,
           Color_t color, Short_t bordersize, Short_t bordermode)
          : TVirtualPad(name,title,xlow,ylow,xup,yup,color,bordersize,bordermode)
{
   fModified   = kTRUE;
   fTip        = nullptr;
   fBorderSize = bordersize;
   fBorderMode = bordermode;
   if (gPad)   fCanvas = gPad->GetCanvas();
   else        fCanvas = (TCanvas*)this;
   fMother     = (TPad*)gPad;
   fPrimitives = new TList;
   fExecs      = new TList;
   fPadPointer = nullptr;
   fTheta      = 30;
   fPhi        = 30;
   fGridx      = gStyle->GetPadGridX();
   fGridy      = gStyle->GetPadGridY();
   fTickx      = gStyle->GetPadTickX();
   fTicky      = gStyle->GetPadTickY();
   fFrame      = nullptr;
   fView       = nullptr;
   fPadPaint   = 0;
   fPadView3D  = nullptr;
   fPixmapID   = -1;      // -1 means pixmap will be created by ResizePad()
   fCopyGLDevice = kFALSE;
   fEmbeddedGL = kFALSE;
   fNumber     = 0;
   fAbsCoord   = kFALSE;
   fEditable   = kTRUE;
   fCrosshair  = 0;
   fCrosshairPos = 0;

   fVtoAbsPixelk = 0.;
   fVtoPixelk    = 0.;
   fVtoPixel     = 0.;
   fAbsPixeltoXk = 0.;
   fPixeltoXk    = 0.;
   fPixeltoX     = 0;
   fAbsPixeltoYk = 0.;
   fPixeltoYk    = 0.;
   fPixeltoY     = 0.;
   fXlowNDC      = 0;
   fYlowNDC      = 0;
   fWNDC         = 1;
   fHNDC         = 1;
   fXUpNDC       = 0.;
   fYUpNDC       = 0.;
   fAbsXlowNDC   = 0.;
   fAbsYlowNDC   = 0.;
   fAbsWNDC      = 0.;
   fAbsHNDC      = 0.;
   fXtoAbsPixelk = 0.;
   fXtoPixelk    = 0.;
   fXtoPixel     = 0.;
   fYtoAbsPixelk = 0.;
   fYtoPixelk    = 0.;
   fYtoPixel     = 0.;
   fUtoAbsPixelk = 0.;
   fUtoPixelk    = 0.;
   fUtoPixel     = 0.;

   fUxmin = fUymin = fUxmax = fUymax = 0;
   fLogx = gStyle->GetOptLogx();
   fLogy = gStyle->GetOptLogy();
   fLogz = gStyle->GetOptLogz();

   fFixedAspectRatio = kFALSE;
   fAspectRatio      = 0.;

   fNumPaletteColor = 0;
   fNextPaletteColor = 0;
   fCGnx = 0;
   fCGny = 0;

   fViewer3D = nullptr;

   if (fCanvas) fGLDevice = fCanvas->GetGLDevice();
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

   TContext ctxt(kTRUE);

   Bool_t zombie = kFALSE;

   if ((xlow < 0) || (xlow > 1) || (ylow < 0) || (ylow > 1)) {
      Error("TPad", "illegal bottom left position: x=%f, y=%f", xlow, ylow);
      zombie = kTRUE;
   } else if ((xup < 0) || (xup > 1) || (yup < 0) || (yup > 1)) {
      Error("TPad", "illegal top right position: x=%f, y=%f", xup, yup);
      zombie = kTRUE;
   } else if (xup-xlow <= 0) {
      Error("TPad", "illegal width: %f", xup-xlow);
      zombie = kTRUE;
   } else if (yup-ylow <= 0) {
      Error("TPad", "illegal height: %f", yup-ylow);
      zombie = kTRUE;
   }

   if (zombie) {
      // error in creating pad occurred, make this pad a zombie
      MakeZombie();
      return;
   }


   fLogx = gStyle->GetOptLogx();
   fLogy = gStyle->GetOptLogy();
   fLogz = gStyle->GetOptLogz();

   fUxmin = fUymin = fUxmax = fUymax = 0;

   // Set pad parameters and Compute conversion coefficients
   SetPad(name, title, xlow, ylow, xup, yup, color, bordersize, bordermode);
   Range(0, 0, 1, 1);
   SetBit(kMustCleanup);
   SetBit(kCanDelete);
}


////////////////////////////////////////////////////////////////////////////////
/// Pad destructor.

TPad::~TPad()
{
   if (ROOT::Detail::HasBeenDeleted(this)) return;
   Close();
   CloseToolTip(fTip);
   DeleteToolTip(fTip);
   auto primitives = fPrimitives;
   // In some cases, fPrimitives has the kMustCleanup bit set which will lead
   // its destructor to call RecursiveRemove and since this pad is still
   // likely to be (indirectly) in the list of cleanups, we must set
   // fPrimitives to nullptr to avoid TPad::RecursiveRemove from calling
   // a member function of a partially destructed object.
   fPrimitives = nullptr;
   delete primitives;
   SafeDelete(fExecs);
   delete fViewer3D;

   // Required since we overload TObject::Hash.
   ROOT::CallRecursiveRemoveIfNeeded(*this);
   if (this == gPad)
      gPad = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Add an object to list of primitives with specified draw option
/// When \par modified set to kTRUE (default) pad will be marked as modified
/// Let avoid usage of gPad when drawing object(s) in canvas or in subpads.
///
/// ~~~{.cpp}
/// auto c1 = new TCanvas("c1","Canvas with subpoads", 600, 600);
/// c1->Divide(2,2);
///
/// for (Int_t n = 1; n <= 4; ++n) {
///    auto h1 = new TH1I(TString::Format("hist_%d",n), "Random hist", 100, -5, 5);
///    h1->FillRandom("gaus", 2000 + n*1000);
///    c1->GetPad(n)->Add(h1);
/// }
/// ~~~

void TPad::Add(TObject *obj, Option_t *opt, Bool_t modified)
{
   if (!obj)
      return;

   if (!fPrimitives)
      fPrimitives = new TList;

   obj->SetBit(kMustCleanup);

   fPrimitives->Add(obj, opt);

   if (modified)
      Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Add an object as first in list of primitives with specified draw option
/// When \par modified set to kTRUE (default) pad will be marked as modified
/// Let avoid usage of gPad when drawing object(s) in canvas or in subpads.

void TPad::AddFirst(TObject *obj, Option_t *opt, Bool_t modified)
{
   if (!obj)
      return;

   if (!fPrimitives)
      fPrimitives = new TList;

   obj->SetBit(kMustCleanup);

   fPrimitives->AddFirst(obj, opt);

   if (modified)
      Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new TExec object to the list of Execs.
///
/// When an event occurs in the pad (mouse click, etc) the list of C++ commands
/// in the list of Execs are executed via TPad::AutoExec.
///
/// When a pad event occurs (mouse move, click, etc) all the commands
/// contained in the fExecs list are executed in the order found in the list.
///
/// This facility is activated by default. It can be deactivated by using
/// the canvas "Option" menu.
///
///  The following examples of TExec commands are provided in the tutorials:
///  macros exec1.C and exec2.C.
///
/// ### Example1 of use of exec1.C
///
/// ~~~ {.cpp}
///  Root > TFile f("hsimple.root")
///  Root > hpx.Draw()
///  Root > c1.AddExec("ex1",".x exec1.C")
/// ~~~
///
/// At this point you can use the mouse to click on the contour of
/// the histogram hpx. When the mouse is clicked, the bin number and its
/// contents are printed.
///
/// ### Example2 of use of exec1.C
///
/// ~~~ {.cpp}
///  Root > TFile f("hsimple.root")
///  Root > hpxpy.Draw()
///  Root > c1.AddExec("ex2",".x exec2.C")
/// ~~~
///
/// When moving the mouse in the canvas, a second canvas shows the
/// projection along X of the bin corresponding to the Y position
/// of the mouse. The resulting histogram is fitted with a gaussian.
/// A "dynamic" line shows the current bin position in Y.
/// This more elaborated example can be used as a starting point
/// to develop more powerful interactive applications exploiting the C++
/// interpreter as a development engine.

void TPad::AddExec(const char *name, const char *command)
{
   if (!fExecs) fExecs = new TList;
   TExec *ex = new TExec(name,command);
   fExecs->Add(ex);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the list of Execs when a pad event occurs.

void TPad::AutoExec()
{
   if (GetCrosshair())
      DrawCrosshair();

   if (!fExecs)
      return;
   TIter next(fExecs);
   while (auto exec = (TExec*)next())
      exec->Exec();
}

////////////////////////////////////////////////////////////////////////////////
/// Browse pad.

void TPad::Browse(TBrowser *b)
{
   cd();
   if (fPrimitives) fPrimitives->Browse(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Build a legend from the graphical objects in the pad.
///
/// A simple method to build automatically a TLegend from the primitives in a TPad.
///
/// Only those deriving from TAttLine, TAttMarker and TAttFill are added, excluding
/// TPave and TFrame derived classes.
///
/// \return    The built TLegend
///
/// \param[in] x1, y1, x2, y2       The TLegend coordinates
/// \param[in] title                The legend title. By default it is " "
/// \param[in] option               The TLegend option
///
/// The caller program owns the returned TLegend.
///
/// If the pad contains some TMultiGraph or THStack the individual
/// graphs or histograms in them are added to the TLegend.
///
/// ### Automatic placement of the legend
/// If `x1` is equal to `x2` and `y1` is equal to `y2` the legend will be automatically
/// placed to avoid overlapping with the existing primitives already displayed.
/// `x1` is considered as the width of the legend and `y1` the height. By default
/// the legend is automatically placed with width = `x1`= `x2` = 0.3 and
/// height = `y1`= `y2` = 0.21.

TLegend *TPad::BuildLegend(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
                           const char* title, Option_t *option)
{
   TList *lop = GetListOfPrimitives();
   if (!lop) return nullptr;
   TList *lof = nullptr;
   TLegend *leg = nullptr;
   TObject *obj = nullptr;
   TIter next(lop);
   TString mes;
   TString opt;

   auto AddEntryFromListOfFunctions = [&]() {
      TIter nextobj(lof);
      while ((obj = nextobj())) {
         if (obj->InheritsFrom(TNamed::Class())) {
            if (strlen(obj->GetTitle()))
               mes = obj->GetTitle();
            else
               mes = obj->GetName();
         } else {
            mes = obj->ClassName();
         }
         leg->AddEntry(obj, mes.Data(), "lpf");
      }
   };

   while(auto o = next()) {
      if ((o->InheritsFrom(TAttLine::Class()) || o->InheritsFrom(TAttMarker::Class()) ||
          o->InheritsFrom(TAttFill::Class())) &&
         ( !(o->InheritsFrom(TFrame::Class())) && !(o->InheritsFrom(TPave::Class())) )) {
         if (!leg)
            leg = new TLegend(x1, y1, x2, y2, title);
         if (o->InheritsFrom(TNamed::Class()) && strlen(o->GetTitle()))
            mes = o->GetTitle();
         else if (strlen(o->GetName()))
            mes = o->GetName();
         else
            mes = o->ClassName();
         if (option && strlen(option)) {
            opt = option;
         } else {
            if (o->InheritsFrom(TAttLine::Class()))
               opt += "l";
            if (o->InheritsFrom(TAttMarker::Class()))
               opt += "p";
            if (o->InheritsFrom(TAttFill::Class()))
               opt += "f";
         }
         leg->AddEntry(o,mes.Data(), opt.Data());
         if (o->InheritsFrom(TH1::Class())) {
            lof = ((TH1 *)o)->GetListOfFunctions();
            AddEntryFromListOfFunctions();
         }
         if (o->InheritsFrom(TGraph::Class())) {
            lof = ((TGraph *)o)->GetListOfFunctions();
            AddEntryFromListOfFunctions();
         }
      } else if (o->InheritsFrom(TMultiGraph::Class())) {
         if (!leg)
            leg = new TLegend(x1, y1, x2, y2, title);
         TList * grlist = ((TMultiGraph *)o)->GetListOfGraphs();
         TIter nextgraph(grlist);
         TGraph *gr = nullptr;
         while ((obj = nextgraph())) {
            gr = (TGraph*) obj;
            if (strlen(gr->GetTitle()))
               mes = gr->GetTitle();
            else if (strlen(gr->GetName()))
               mes = gr->GetName();
            else
               mes = gr->ClassName();
            if (option && strlen(option))
               opt = option;
            else
               opt = "lpf";
            leg->AddEntry(obj, mes.Data(), opt);
         }
         lof = ((TMultiGraph *)o)->GetListOfFunctions();
         AddEntryFromListOfFunctions();
      } else if (o->InheritsFrom(THStack::Class())) {
         if (!leg)
            leg = new TLegend(x1, y1, x2, y2, title);
         TList * hlist = ((THStack *)o)->GetHists();
         TIter nexthist(hlist);
         while ((obj = nexthist())) {
            TH1 *hist = (TH1*) obj;
            if (strlen(hist->GetTitle()))
               mes = hist->GetTitle();
            else if (strlen(hist->GetName()))
               mes = hist->GetName();
            else
               mes = hist->ClassName();
            if (option && strlen(option))
               opt = option;
            else
               opt = "lpf";
            leg->AddEntry( obj, mes.Data(), opt );
         }
      }
      opt = "";
   }
   if (leg) {
      TContext ctxt(this, kTRUE);
      leg->Draw();
   } else {
      Info("BuildLegend", "No object(s) to build a TLegend.");
   }
   return leg;
}

////////////////////////////////////////////////////////////////////////////////
/// Set Current pad.
///
/// When a canvas/pad is divided via TPad::Divide, one can directly
/// set the current path to one of the subdivisions.
/// See TPad::Divide for the convention to number sub-pads.
///
/// Returns the new current pad, or 0 in case of failure.
///
/// For example:
/// ~~~ {.cpp}
///    c1.Divide(2,3); // create 6 pads (2 divisions along x, 3 along y).
/// ~~~
/// To set the current pad to the bottom right pad, do
/// ~~~ {.cpp}
///    c1.cd(6);
/// ~~~
///  Note1:  c1.cd() is equivalent to c1.cd(0) and sets the current pad
///          to c1 itself.
///
///  Note2:  after a statement like c1.cd(6), the global variable gPad
///          points to the current pad. One can use gPad to set attributes
///          of the current pad.
///
///  Note3:  One can get a pointer to one of the sub-pads of pad with:
///          TPad *subpad = (TPad*)pad->GetPad(subpadnumber);

TVirtualPad *TPad::cd(Int_t subpadnumber)
{
   if (!subpadnumber) {
      gPad = this;
      if (!gPad->IsBatch() && GetPainter()) GetPainter()->SelectDrawable(fPixmapID);
      if (!fPrimitives) fPrimitives = new TList;
      return gPad;
   }

   if (!fPrimitives) fPrimitives = new TList;
   TIter    next(fPrimitives);
   while (auto obj = next()) {
      if (obj->InheritsFrom(TPad::Class())) {
         Int_t n = ((TPad*)obj)->GetNumber();
         if (n == subpadnumber) {
            return ((TPad*)obj)->cd();
         }
      }
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete all pad primitives.
///
/// If the bit kClearAfterCR has been set for this pad, the Clear function
/// will execute only after having pressed a CarriageReturn
/// Set the bit with `mypad->SetBit(TPad::kClearAfterCR)`

void TPad::Clear(Option_t *option)
{
   if (!IsEditable()) return;

   R__LOCKGUARD(gROOTMutex);

   if (!fPadPaint) {
      SafeDelete(fView);
      if (fPrimitives) fPrimitives->Clear(option);
      if (fFrame) {
         if (! ROOT::Detail::HasBeenDeleted(fFrame)) delete fFrame;
         fFrame = nullptr;
      }
   }
   if (fCanvas) fCanvas->Cleared(this);

   cd();

   if (TestBit(kClearAfterCR)) {
      // Intentional do not use the return value of getchar,
      // we just want to get it and forget it
      getchar();
   }

   auto pp = GetPainter();
   // If pad painter uses PS, TPad::Clear() start new page
   if (pp && pp->GetPS())
      pp->NewPage();

   PaintBorder(GetFillColor(), kTRUE);
   fCrosshairPos = 0;
   fNumPaletteColor = 0;
   fCollideGrid.clear();
   fCGnx = 0;
   fCGny = 0;
   ResetBit(TGraph::kClipFrame);
}

////////////////////////////////////////////////////////////////////////////////
/// Clipping routine: Cohen Sutherland algorithm.
///
///  - If Clip ==2 the segment is outside the boundary.
///  - If Clip ==1 the segment has one point outside the boundary.
///  - If Clip ==0 the segment is inside the boundary.
///
/// \param[inout]  x[],y[]                       Segment coordinates (2 points)
/// \param[in]     xclipl,yclipb,xclipr,yclipt   Clipping boundary

Int_t TPad::Clip(Float_t *x, Float_t *y, Float_t xclipl, Float_t yclipb, Float_t xclipr, Float_t yclipt)
{
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

/// @copydoc TPad::Clip(Float_t*,Float_t*,Float_t,Float_t,Float_t,Float_t)

Int_t TPad::Clip(Double_t *x, Double_t *y, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt)
{
   const Double_t kP = 10000;
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

////////////////////////////////////////////////////////////////////////////////
/// Compute the endpoint codes for TPad::Clip.

Int_t TPad::ClippingCode(Double_t x, Double_t y, Double_t xcl1, Double_t ycl1, Double_t xcl2, Double_t ycl2)
{
   Int_t code = 0;
   if (x < xcl1) code = code | 0x1;
   if (x > xcl2) code = code | 0x2;
   if (y < ycl1) code = code | 0x4;
   if (y > ycl2) code = code | 0x8;
   return code;
}

////////////////////////////////////////////////////////////////////////////////
/// Clip polygon using the Sutherland-Hodgman algorithm.
///
/// \param[in]  n                            Number of points in the polygon to
///                                          be clipped
/// \param[in]  x,y                          Polygon x[n], y[n] do be clipped vertices
/// \param[in]  xclipl,yclipb,xclipr,yclipt  Clipping boundary
/// \param[out] nn                           Number of points in xc and yc
/// \param[out] xc,yc                        Clipped polygon vertices. The Int_t
///                                          returned by this function is
///                                          the number of points in the clipped
///                                          polygon. These vectors must
///                                          be allocated by the calling function.
///                                          A size of 2*n for each is
///                                          enough.
///
/// Sutherland and Hodgman's polygon-clipping algorithm uses a divide-and-conquer
/// strategy: It solves a series of simple and identical problems that, when
/// combined, solve the overall problem. The simple problem is to clip a polygon
/// against a single infinite clip edge. Four clip edges, each defining one boundary
/// of the clip rectangle, successively clip a polygon against a clip rectangle.
///
/// Steps of Sutherland-Hodgman's polygon-clipping algorithm:
///
/// * Polygons can be clipped against each edge of the window one at a time.
///   Windows/edge intersections, if any, are easy to find since the X or Y coordinates
///   are already known.
/// * Vertices which are kept after clipping against one window edge are saved for
///   clipping against the remaining edges.
/// * Note that the number of vertices usually changes and will often increases.
///
/// The clip boundary determines a visible and invisible region. The edges from
/// vertex i to vertex i+1 can be one of four types:
///
/// * Case 1 : Wholly inside visible region - save endpoint
/// * Case 2 : Exit visible region - save the intersection
/// * Case 3 : Wholly outside visible region - save nothing
/// * Case 4 : Enter visible region - save intersection and endpoint

Int_t TPad::ClipPolygon(Int_t n, Double_t *x, Double_t *y, Int_t nn, Double_t *xc, Double_t *yc, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt)
{
   if (n <= 0)
      return 0;

   Int_t nc, nc2;
   Double_t x1, y1, x2, y2, slope; // Segment to be clipped

   std::vector<Double_t> xc2(nn), yc2(nn);

   // Clip against the left boundary
   x1 = x[n - 1];
   y1 = y[n - 1];
   nc2 = 0;
   Int_t i;
   for (i = 0; i < n; i++) {
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
   if (nc2 > 0) {
      x1 = xc2[nc2 - 1];
      y1 = yc2[nc2 - 1];
   }
   nc = 0;
   for (i = 0; i < nc2; i++) {
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
   if (nc > 0) {
      x1 = xc[nc - 1];
      y1 = yc[nc - 1];
      nc2 = 0;
      for (i = 0; i < nc; i++) {
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
      if (nc2 > 0) {
         x1 = xc2[nc2 - 1];
         y1 = yc2[nc2 - 1];
      }
      nc = 0;
      for (i = 0; i < nc2; i++) {
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
   }

   if (nc < 3)
      nc = 0;
   return nc;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete all primitives in pad and pad itself.
/// Pad cannot be used anymore after this call.
/// Emits signal "Closed()".

void TPad::Close(Option_t *)
{
   if (ROOT::Detail::HasBeenDeleted(this)) return;
   if (!fMother) return;
   if (ROOT::Detail::HasBeenDeleted(fMother)) return;

   if (fPrimitives)
      fPrimitives->Clear();
   if (fView) {
      if (!ROOT::Detail::HasBeenDeleted(fView)) delete fView;
      fView = nullptr;
   }
   if (fFrame) {
      if (!ROOT::Detail::HasBeenDeleted(fFrame)) delete fFrame;
      fFrame = nullptr;
   }

   // emit signal
   if (IsA() != TCanvas::Class())
      Closed();

   if (fPixmapID != -1) {
      if (gPad) {
         if (!gPad->IsBatch() && GetPainter())
            GetPainter()->DestroyDrawable(fPixmapID);
      }
      fPixmapID = -1;

      if (!gROOT->GetListOfCanvases()) return;
      if (fMother == this) {
         gROOT->GetListOfCanvases()->Remove(this);
         return;   // in case of TCanvas
      }

      // remove from the mother's list of primitives
      if (fMother) {
         fMother->Remove(this, kFALSE); // do not produce modified

         if (gPad == this)
            fMother->cd();
      }
      if (fCanvas) {
         if (fCanvas->GetPadSave() == this)
            fCanvas->ClearPadSave();
         if (fCanvas->GetSelectedPad() == this)
            fCanvas->SetSelectedPad(nullptr);
         if (fCanvas->GetClickSelectedPad() == this)
            fCanvas->SetClickSelectedPad(nullptr);
      }
   }

   fMother = nullptr;
   if (gROOT->GetSelectedPad() == this)
      gROOT->SetSelectedPad(nullptr);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the pixmap of the pad to the canvas.

void TPad::CopyPixmap()
{
   int px, py;
   XYtoAbsPixel(fX1, fY2, px, py);

   if (fPixmapID != -1)
      if (auto pp = GetPainter())
         pp->CopyDrawable(fPixmapID, px, py);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the sub-pixmaps of the pad to the canvas.

void TPad::CopyPixmaps()
{
   if (!fPrimitives)
      fPrimitives = new TList;
   TIter next(GetListOfPrimitives());
   while (auto obj = next()) {
      if (auto pad = dynamic_cast<TPad*>(obj)) {
         pad->CopyPixmap();
         pad->CopyPixmaps();
      }
   }

   if (this == gPad)
      HighLight(GetHighLightColor());
}

////////////////////////////////////////////////////////////////////////////////
/// Remove TExec name from the list of Execs.

void TPad::DeleteExec(const char *name)
{
   if (!fExecs) fExecs = new TList;
   TExec *ex = (TExec*)fExecs->FindObject(name);
   if (!ex) return;
   fExecs->Remove(ex);
   delete ex;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to a box.
///
///  Compute the closest distance of approach from point px,py to the
///  edges of this pad.
///  The distance is computed in pixels units.

Int_t TPad::DistancetoPrimitive(Int_t px, Int_t py)
{
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
   if (py < pyl) dxl += pyl - py;
   if (py > pyt) dxl += py - pyt;
   Int_t dxt = TMath::Abs(px - pxt);
   if (py < pyl) dxt += pyl - py;
   if (py > pyt) dxt += py - pyt;
   Int_t dyl = TMath::Abs(py - pyl);
   if (px < pxl) dyl += pxl - px;
   if (px > pxt) dyl += px - pxt;
   Int_t dyt = TMath::Abs(py - pyt);
   if (px < pxl) dyt += pxl - px;
   if (px > pxt) dyt += px - pxt;

   Int_t distance = dxl;
   if (dxt < distance) distance = dxt;
   if (dyl < distance) distance = dyl;
   if (dyt < distance) distance = dyt;

   return distance - Int_t(0.5*fLineWidth);
}

////////////////////////////////////////////////////////////////////////////////
/// Automatic pad generation by division.
///
///  - The current canvas is divided in nx by ny equal divisions (pads).
///  - xmargin defines the horizontal spacing around each pad as a percentage of the canvas
///    width. Therefore, the distance between two adjacent pads along the x-axis is equal
///    to twice the xmargin value.
///  - ymargin defines the vertical spacing around each pad as a percentage of the canvas
///    height. Therefore, the distance between two adjacent pads along the y-axis is equal
///    to twice the ymargin value.
///  - color is the color of the new pads. If 0, color is the canvas color.
///
/// Pads are automatically named `canvasname_n` where `n` is the division number
/// starting from top left pad.
///
/// Example if canvasname=c1 , nx=2, ny=3:
///
/// \image html gpad_pad3.png
///
/// Once a pad is divided into sub-pads, one can set the current pad
/// to a subpad with a given division number as illustrated above
/// with TPad::cd(subpad_number).
///
/// For example, to set the current pad to c1_4, one can do:
/// ~~~ {.cpp}
///    c1->cd(4)
/// ~~~
/// __Note1:__  c1.cd() is equivalent to c1.cd(0) and sets the current pad
///             to c1 itself.
///
/// __Note2:__  after a statement like c1.cd(6), the global variable gPad
///             points to the current pad. One can use gPad to set attributes
///             of the current pad.
///
/// __Note3:__  in case xmargin < 0 or ymargin < 0, there is no space
///             between pads. The current pad margins are recomputed to
///             optimize the layout in order to have similar frames' areas.
///             See the following example:
///
/// ~~~ {.cpp}
///    void divpad(Int_t nx=3, Int_t ny=2) {
///       gStyle->SetOptStat(0);
///       auto C = new TCanvas();
///       C->SetMargin(0.3, 0.3, 0.3, 0.3);
///       C->Divide(nx,ny,-1);
///       Int_t number = 0;
///       auto h = new TH1F("","",100,-3.3,3.3);
///       h->GetXaxis()->SetLabelFont(43);
///       h->GetXaxis()->SetLabelSize(12);
///       h->GetYaxis()->SetLabelFont(43);
///       h->GetYaxis()->SetLabelSize(12);
///       h->GetYaxis()->SetNdivisions(505);
///       h->SetMaximum(30*nx*ny);
///       h->SetFillColor(42);
///       for (Int_t i=0;i<nx*ny;i++) {
///          number++;
///          C->cd(number);
///          h->FillRandom("gaus",1000);
///          h->DrawCopy();
///       }
///    }
/// ~~~

void TPad::Divide(Int_t nx, Int_t ny, Float_t xmargin, Float_t ymargin, Int_t color)
{
   if (!IsEditable()) return;

   if (gThreadXAR) {
      void *arr[7];
      arr[1] = this; arr[2] = (void*)&nx;arr[3] = (void*)& ny;
      arr[4] = (void*)&xmargin; arr[5] = (void *)& ymargin; arr[6] = (void *)&color;
      if ((*gThreadXAR)("PDCD", 7, arr, nullptr)) return;
   }

   TContext ctxt(kTRUE);

   cd();
   if (nx <= 0) nx = 1;
   if (ny <= 0) ny = 1;
   Int_t ix, iy;
   Double_t x1, y1, x2, y2, dx, dy;
   TPad *pad;
   TString name, title;
   Int_t n = 0;
   if (color == 0) color = GetFillColor();
   if (xmargin >= 0 && ymargin >= 0) {
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
            name.Form("%s_%d", GetName(), n);
            pad = new TPad(name.Data(), name.Data(), x1, y1, x2, y2, color);
            pad->SetNumber(n);
            pad->Draw();
         }
      }
   } else {
      // special case when xmargin < 0 or ymargin < 0
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
            name.Form("%s_%d", GetName(), number);
            title.Form("%s_%d", GetTitle(), number);
            pad = new TPad(name.Data(), title.Data(), x1, y1, x2, y2, color);
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
   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Divide the canvas according to ratios.
///
/// The current canvas is divided in nx by ny according to the width and height ratios.
/// If the ratios are not specified they are assumed to be equal.
///
/// Pads are automatically named `canvasname_n` where `n` is the division number
/// starting from top left pad.
///
/// Top and left margins can be defined.

void TPad::DivideRatios(Int_t nx, Int_t ny,
                        const std::vector<double>& widthRatios,
                        const std::vector<double>& heightRatios,
                        const double canvasTopMargin,
                        const double canvasLeftMargin
                        )
{
   cd();

   int wrs = widthRatios.size();
   int hrs = heightRatios.size();
   int nxl = TMath::Min(nx,wrs), nyl = TMath::Min(ny,hrs);

   if (wrs==0) nxl = nx;
   if (hrs==0) nyl = ny;

   int    pn = 1;
   double xr = 0.;
   double yr = 0.;
   double x  = 0.;
   double y  = 1.;
   double x1, y1, x2, y2;

   // Check the validity of the margins
   if (canvasTopMargin <0 || canvasTopMargin >1 ) {
      Error("DivideRatios", "The canvas top margin must be >= 0 and <= 1");
      return;
   } else {
      y = 1.- canvasTopMargin;
   }
   if (canvasLeftMargin <0 || canvasLeftMargin >1 ) {
      Error("DivideRatios", "The canvas left margin must be >= 0 and <= 1");
      return;
   }

   // Check the validity of the ratios
   double sumOfHeightRatios = canvasTopMargin;
   if (hrs) {
      for (int i=0; i<nyl; i++) {
         yr = heightRatios[i];
         sumOfHeightRatios = sumOfHeightRatios + yr;
         if (yr <0 || yr >1 ) {
            Error("DivideRatios", "Y ratios plus the top margin must be >= 0 and <= 1");
            return;
         }
      }
   }
   if (sumOfHeightRatios > 1.) {
      Error("DivideRatios", "The sum of Y ratios plus the top margin must be <= 1 %g",sumOfHeightRatios);
      return;
   }
   double sumOfWidthRatios = canvasLeftMargin;
   if (wrs) {
      for (int j=0; j<nxl; j++) {
         xr = widthRatios[j];
         sumOfWidthRatios = sumOfWidthRatios +xr;
         if (xr <0 || xr >1 ) {
            Error("DivideRatios", "X ratios must be >= 0 and <= 1");
            return;
         }
      }
   }
   if (sumOfWidthRatios > 1.) {
      Error("DivideRatios", "The sum of X ratios must be <= 1 %g ",sumOfWidthRatios);
      return;
   }

   // Create the pads according to the ratios
   for (int i=0; i<nyl; i++) {
      x = canvasLeftMargin;
      if (hrs) yr = heightRatios[i];
      else     yr = 1./nyl;
      for (int j=0; j<nxl; j++) {
         if (wrs) xr = widthRatios[j];
         else     xr = 1./nxl;
         x1 = TMath::Max(0., x);
         y1 = TMath::Max(0., y - yr);
         x2 = TMath::Min(1., x + xr);
         y2 = TMath::Min(1., y);
         auto pad = new TPad(TString::Format("%s_%d", GetName(), pn),
                             TString::Format("%s_%d", GetName(), pn),
                             x1, y1, x2 ,y2);
         pad->SetNumber(pn);
         pad->Draw();
         x = x + xr;
         pn++;
      }
      y = y - yr;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// "n" is the total number of sub-pads. The number of sub-pads along the X
/// and Y axis are computed according to the square root of n.

void TPad::DivideSquare(Int_t n, Float_t xmargin, Float_t ymargin, Int_t color)
{
   Int_t w = 1, h = 1;
   if (!fCanvas) {
      Error("DivideSquare", "No canvas associated with this pad.");
      return;
   }
   if (fCanvas->GetWindowWidth() > fCanvas->GetWindowHeight()) {
      w = TMath::Ceil(TMath::Sqrt(n));
      h = TMath::Floor(TMath::Sqrt(n));
      if (w*h < n) w++;
   } else {
      h = TMath::Ceil(TMath::Sqrt(n));
      w = TMath::Floor(TMath::Sqrt(n));
      if (w*h < n) h++;
   }

   Divide( w, h, xmargin, ymargin, color);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw Pad in Current pad (re-parent pad if necessary).

void TPad::Draw(Option_t *option)
{
   // if no canvas opened yet create a default canvas
   if (!gPad) {
      gROOT->MakeDefCanvas();
   }

   // pad cannot be in itself and it can only be in one other pad at a time
   if (!fPrimitives) fPrimitives = new TList;
   if (gPad != this) {
      if (fMother && !ROOT::Detail::HasBeenDeleted(fMother))
            fMother->Remove(this, kFALSE);
      TPad *oldMother = fMother;
      fCanvas = gPad->GetCanvas();
      //
      fMother = (TPad*)gPad;
      if (oldMother != fMother || fPixmapID == -1) ResizePad();
   }

   if (fCanvas && fCanvas->IsWeb()) {
      Modified();
      fCanvas->UpdateAsync();
   } else {
      Paint();
   }

   if (gPad->IsRetained() && gPad != this && fMother)
      fMother->Add(this, option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw class inheritance tree of the class to which obj belongs.
///
/// If a class B inherits from a class A, description of B is drawn
/// on the right side of description of A.
///
/// Member functions overridden by B are shown in class A with a blue line
/// crossing-out the corresponding member function.

void TPad::DrawClassObject(const TObject *classobj, Option_t *option)
{
   if (!classobj) return;
   char dname[256];
   const Int_t kMAXLEVELS = 10;
   TClass *clevel[kMAXLEVELS], *cl, *cll;
   TBaseClass *base, *cinherit;
   TText *ptext = nullptr;
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
      if (!base->GetClassPointer()) break;
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
      cinherit = nullptr;
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
         if (box) box->SetFillColor(17);
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
            snprintf(dname,256,"%s",d->GetName());
            Int_t ldname = 0;
            while (indx < dim ){
               ldname = strlen(dname);
               snprintf(&dname[ldname],256-ldname,"[%d]",d->GetMaxIndex(indx));
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

            ptext = pt->AddText(x,(y-v1)/dv,m->GetName());
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
         cll = nullptr;
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

////////////////////////////////////////////////////////////////////////////////
/// Function called to draw a crosshair in the canvas
///
/// Example:
/// ~~~ {.cpp}
/// Root > TFile f("hsimple.root");
/// Root > hpxpy.Draw();
/// Root > c1.SetCrosshair();
/// ~~~
/// When moving the mouse in the canvas, a crosshair is drawn
///
///  - if the canvas fCrosshair = 1 , the crosshair spans the full canvas
///  - if the canvas fCrosshair > 1 , the crosshair spans only the pad

void TPad::DrawCrosshair()
{
   if (!gPad || (gPad->GetEvent() == kMouseEnter))
      return;

   TPad *cpad = (TPad*)gPad;
   TCanvas *canvas = cpad->GetCanvas();
   // switch off double buffer and select canvas drawable
   canvas->FeedbackMode(kTRUE);

   auto pp = GetPainter();

   //erase old position and draw a line at current position
   Double_t umin, umax, vmin, vmax, u, v;
   Int_t px    = cpad->GetEventX();
   Int_t py    = cpad->GetEventY() + 1;
   if (canvas->GetCrosshair() > 1) {  //crosshair only in the current pad
      umin = GetAbsXlowNDC();
      umax = GetAbsXlowNDC() + GetAbsWNDC();
      vmin = GetAbsYlowNDC();
      vmax = GetAbsYlowNDC() + GetAbsHNDC();
   } else { //default; crosshair spans the full canvas
      umin = 0;
      umax = 1;
      vmin = 0;
      vmax = 1;
   }

   TContext ctxt(canvas);

   pp->SetAttLine({1,1,1});

   if ((fCrosshairPos != 0) && !pp->IsCocoa()) {
      // xor does not supported on Cocoa, implemented differently
      Int_t pxold = fCrosshairPos % 10000;
      Int_t pyold = fCrosshairPos / 10000;
      u = 1. * pxold / canvas->GetWw();
      v = 1. - 1. * pyold / canvas->GetWh();
      pp->DrawLineNDC(umin, v, umax, v);
      pp->DrawLineNDC(u, vmin, u, vmax);
   }

   if (cpad->GetEvent() == kButton1Down ||
       cpad->GetEvent() == kButton1Up   ||
       cpad->GetEvent() == kMouseLeave) {
      fCrosshairPos = 0;
      return;
   }

   u = 1. * px / canvas->GetWw();
   v = 1. - 1. * py / canvas->GetWh();
   pp->DrawLineNDC(umin, v, umax, v);
   pp->DrawLineNDC(u, vmin, u, vmax);

   fCrosshairPos = px + 10000*py;
}

////////////////////////////////////////////////////////////////////////////////
///  Draw an empty pad frame with X and Y axis.
///
///   \return   The pointer to the histogram used to draw the frame.
///
///   \param[in] xmin      X axis lower limit
///   \param[in] xmax      X axis upper limit
///   \param[in] ymin      Y axis lower limit
///   \param[in] ymax      Y axis upper limit
///   \param[in] title     Pad title.If title is of the form "stringt;stringx;stringy"
///                        the pad title is set to stringt, the x axis title to
///                        stringx, the y axis title to stringy.
///
/// #### Example:
///
/// Begin_Macro(source)
/// {
///    auto c = new TCanvas("c","c",200,10,500,300);
///
///    const Int_t n = 50;
///    auto g = new TGraph();
///    for (Int_t i=0;i<n;i++) g->SetPoint(i,i*0.1,100*sin(i*0.1+0.2));
///
///    auto frame = c->DrawFrame(0, -110, 2, 110);
///    frame->GetXaxis()->SetTitle("X axis");
///
///    g->Draw("L*");
/// }
/// End_Macro

TH1F *TPad::DrawFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, const char *title)
{
   if (!IsEditable())
      return nullptr;

   if (this != gPad) {
      Warning("DrawFrame", "Must be called for the current pad only");
      if (gPad) return gPad->DrawFrame(xmin,ymin,xmax,ymax,title);
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
      std::vector<Double_t> xbins(nbins+1);
      xbins[0] = xmin;
      for (Int_t i=1;i<=nbins;i++) {
         xbins[i] = TMath::Exp(xminl+i*dx);
      }
      hframe = new TH1F("hframe",title,nbins,xbins.data());
   } else {
      hframe = new TH1F("hframe",title,nbins,xmin,xmax);
   }
   hframe->SetBit(TH1::kNoStats);
   hframe->SetBit(kCanDelete);
   hframe->SetMinimum(ymin);
   hframe->SetMaximum(ymax);
   hframe->GetYaxis()->SetLimits(ymin,ymax);
   hframe->SetDirectory(nullptr);
   hframe->Draw(" ");
   Update();
   cd();
   return hframe;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function to Display Color Table in a pad.

void TPad::DrawColorTable()
{
   Int_t i, j;
   Int_t color;
   Double_t xlow, ylow, xup, yup, hs, ws;
   Double_t x1, y1, x2, y2;
   x1 = y1 = 0;
   x2 = y2 = 20;

   gPad->SetFillColor(0);
   gPad->Clear();
   gPad->Range(x1,y1,x2,y2);

   TText text(0,0,"");
   text.SetTextFont(61);
   text.SetTextSize(0.07);
   text.SetTextAlign(22);

   TBox box;

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
         box.SetFillStyle(1001);
         box.SetFillColor(color);
         box.DrawBox(xlow, ylow, xup, yup);
         box.SetFillStyle(0);
         box.SetLineColor(1);
         box.DrawBox(xlow, ylow, xup, yup);
         if (color == 1) text.SetTextColor(0);
         else            text.SetTextColor(1);
         text.DrawText(0.5*(xlow+xup), 0.5*(ylow+yup), TString::Format("%d",color).Data());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
/// This member function is called when a TPad object is clicked.
///
/// If the mouse is clicked in one of the 4 corners of the pad (pA,pB,pC,pD)
/// the pad is resized with the rubber rectangle.
///
/// If the mouse is clicked inside the pad, the pad is moved.
///
/// If the mouse is clicked on the 4 edges (pL,pR,pTop,pBot), the pad is scaled
/// parallel to this edge.
///
/// \image html gpad_pad4.png
///
/// Note that this function duplicates on purpose the functionality
/// already implemented in TBox::ExecuteEvent.
/// If somebody modifies this function, may be similar changes should also
/// be applied to TBox::ExecuteEvent.

void TPad::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   constexpr Int_t kMaxDiff = 5;
   constexpr Int_t kMinSize = 20;
   static Int_t px1, px2, py1, py2, dpx1, dpy2;
   static Int_t px1p, px2p, py1p, py2p;
   static enum { pNone, pA, pB, pC, pD, pTop, pL, pR, pBot, pINSIDE } mode = pNone;
   static Bool_t firstPaint = kFALSE;
   Bool_t opaque  = OpaqueMoving();
   Bool_t ropaque = OpaqueResizing();

   if (!IsEditable() && event != kMouseEnter) return;
   TVirtualPad  &parent = *GetMother();
   if (!parent.IsEditable()) return;

   HideToolTip(event);

   if (fXlowNDC < 0 && event != kButton1Down) return;
   if (fYlowNDC < 0 && event != kButton1Down) return;

   Int_t newcode = gROOT->GetEditorMode();
   if (newcode)
      mode = pNone;
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
   if (newcode)
      return;

   auto paint_or_set = [this, &parent](Bool_t paint)
   {
      auto x1 = AbsPixeltoX(px1);
      auto y1 = AbsPixeltoY(py1);
      auto x2 = AbsPixeltoX(px2);
      auto y2 = AbsPixeltoY(py2);
      if (!paint) {
         // Get parent corners pixels coordinates
         Int_t parentpx1 = fMother->XtoAbsPixel(parent.GetX1());
         Int_t parentpx2 = fMother->XtoAbsPixel(parent.GetX2());
         Int_t parentpy1 = fMother->YtoAbsPixel(parent.GetY1());
         Int_t parentpy2 = fMother->YtoAbsPixel(parent.GetY2());

         // Get pad new corners pixels coordinates
         Int_t apx1 = XtoAbsPixel(x1); if (apx1 < parentpx1) {apx1 = parentpx1; }
         Int_t apx2 = XtoAbsPixel(x2); if (apx2 > parentpx2) {apx2 = parentpx2; }
         Int_t apy1 = YtoAbsPixel(y1); if (apy1 > parentpy1) {apy1 = parentpy1; }
         Int_t apy2 = YtoAbsPixel(y2); if (apy2 < parentpy2) {apy2 = parentpy2; }

         // Compute new pad positions in the NDC space of parent
         fXlowNDC = Double_t(apx1 - parentpx1)/Double_t(parentpx2 - parentpx1);
         fYlowNDC = Double_t(apy1 - parentpy1)/Double_t(parentpy2 - parentpy1);
         fWNDC    = Double_t(apx2 - apx1)/Double_t(parentpx2 - parentpx1);
         fHNDC    = Double_t(apy2 - apy1)/Double_t(parentpy2 - parentpy1);
      } else if (firstPaint) {
         // first paint with original coordinates not required
         firstPaint = kFALSE;
      } else {
         auto pp = GetPainter();
         pp->SetAttLine({GetFillColor() > 0 ? GetFillColor() : (Color_t) 1, GetLineStyle(), 2});
         pp->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kHollow);
      }
   };

   Int_t prevpx1 = px1, prevpx2 = px2, prevpy1 = py1, prevpy2 = py2;

   // function check how to restore pad ratio
   auto adjustRatio = [this, &parent](int choise = 11) -> bool
   {
      if (!HasFixedAspectRatio())
         return true; // do nothing

      if (choise == 11) {
         Int_t dx = parent.UtoPixel(fAspectRatio * (py1 - py2) / parent.VtoPixel(0));
         Int_t npx1 = (px1 + px2) / 2 - dx / 2;
         Int_t npx2 = npx1 + dx;
         if ((npx1 >= px1p) && (npx2 <= px2p)) {
            px1 = npx1; px2 = npx2;
            return true;
         }
      } else {
         Int_t dy = parent.VtoPixel(1. - (0. + px2 - px1) / parent.UtoPixel(1.) / fAspectRatio);
         Int_t npy1 = py1;
         Int_t npy2 = py2;
         switch (choise) {
            case -1: npy2 = py1 - dy; break;
            case  0: npy2 = (py1 + py2) / 2 - dy / 2; npy1 = npy2 + dy; break;
            case  1: npy1 = py2 + dy; break;
         }
         if ((npy1 <= py1p) && (npy2 >= py2p)) {
            py1 = npy1; py2 = npy2;
            return true;
         }
      }

      return false; // fail to adjust ratio, need to restore values
   };

   switch (event) {

   case kMouseEnter:
      if (fTip)
         ResetToolTip(fTip);
      break;

   case kArrowKeyPress:
   case kButton1Down:

      fXUpNDC = fXlowNDC + fWNDC;
      fYUpNDC = fYlowNDC + fHNDC;

      // No break !!!

   case kMouseMotion:

      px1 = XtoAbsPixel(fX1);
      py1 = YtoAbsPixel(fY1);
      px2 = XtoAbsPixel(fX2);
      py2 = YtoAbsPixel(fY2);

      if (px1 > px2)
         std::swap(px1, px2);

      if (py1 < py2)
         std::swap(py1, py2);

      px1p = parent.XtoAbsPixel(parent.GetX1()) + parent.GetBorderSize();
      py1p = parent.YtoAbsPixel(parent.GetY1()) - parent.GetBorderSize();
      px2p = parent.XtoAbsPixel(parent.GetX2()) - parent.GetBorderSize();
      py2p = parent.YtoAbsPixel(parent.GetY2()) + parent.GetBorderSize();

      if (px1p > px2p)
         std::swap(px1p, px2p);

      if (py1p < py2p)
         std::swap(py1p, py2p);

      mode = pNone;
      if (TMath::Abs(px - px1) <= kMaxDiff && TMath::Abs(py - py2) <= kMaxDiff) {
         mode = pA;
         SetCursor(kTopLeft);
      } else if (TMath::Abs(px - px2) <= kMaxDiff && TMath::Abs(py - py2) <= kMaxDiff) {
         mode = pB;
         SetCursor(kTopRight);
      } else if (TMath::Abs(px - px2) <= kMaxDiff && TMath::Abs(py - py1) <= kMaxDiff) {
         mode = pC;
         SetCursor(kBottomRight);
      } else if (TMath::Abs(px - px1) <= kMaxDiff && TMath::Abs(py - py1) <= kMaxDiff) {
         mode = pD;
         SetCursor(kBottomLeft);
      } else if ((px > px1 + kMaxDiff && px < px2 - kMaxDiff) && TMath::Abs(py - py2) < kMaxDiff) {
         mode = pTop;
         SetCursor(kTopSide);
      } else if ((px > px1 + kMaxDiff && px < px2 - kMaxDiff) && TMath::Abs(py - py1) < kMaxDiff) {
         mode = pBot;
         SetCursor(kBottomSide);
      } else if ((py > py2 + kMaxDiff && py < py1 - kMaxDiff) && TMath::Abs(px - px1) < kMaxDiff) {
         mode = pL;
         SetCursor(kLeftSide);
      } else if ((py > py2 + kMaxDiff && py < py1 - kMaxDiff) && TMath::Abs(px - px2) < kMaxDiff) {
         mode = pR;
         SetCursor(kRightSide);
      } else if ((px > px1+kMaxDiff && px < px2-kMaxDiff) && (py > py2+kMaxDiff && py < py1-kMaxDiff)) {
         dpx1 = px - px1; // cursor position relative to top-left corner
         dpy2 = py - py2;
         mode = pINSIDE;
         if (event == kButton1Down)
            SetCursor(kMove);
         else
            SetCursor(kCross);
      }

      fResizing = (mode != pNone) && (mode != pINSIDE);

      firstPaint = kTRUE;

      if (mode == pNone)
         SetCursor(kCross);

      break;

   case kArrowKeyRelease:
   case kButton1Motion:

      if (TestBit(kCannotMove)) break;

      switch (mode) {
      case pNone:
         return;
      case pA:
         if (!ropaque) paint_or_set(kTRUE);
         px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
         py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
         if (!adjustRatio(-1)) {
            px1 = prevpx1;
            py2 = prevpy2;
         }
         paint_or_set(!ropaque);
         break;
      case pB:
         if (!ropaque) paint_or_set(kTRUE);
         px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
         py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
         if (!adjustRatio(-1)) {
            px2 = prevpx2;
            py2 = prevpy2;
         }
         paint_or_set(!ropaque);
         break;
      case pC:
         if (!ropaque) paint_or_set(kTRUE);
         px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
         py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
         if (!adjustRatio(1)) {
            px2 = prevpx2;
            py1 = prevpy1;
         }
         paint_or_set(!ropaque);
         break;
      case pD:
         if (!ropaque) paint_or_set(kTRUE);
         px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
         py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
         if (!adjustRatio(1)) {
            px1 = prevpx1;
            py1 = prevpy1;
         }
         paint_or_set(!ropaque);
         break;
      case pTop:
         if (!ropaque) paint_or_set(kTRUE);
         py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
         if (!adjustRatio(11))
            py2 = prevpy2;
         paint_or_set(!ropaque);
         break;
      case pBot:
         if (!ropaque) paint_or_set(kTRUE);
         py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
         if (!adjustRatio(11))
            py1 = prevpy1;
         paint_or_set(!ropaque);
         break;
      case pL:
         if (!ropaque) paint_or_set(kTRUE);
         px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
         if (!adjustRatio(0))
            px1 = prevpx1;
         paint_or_set(!ropaque);
         break;
      case pR:
         if (!ropaque) paint_or_set(kTRUE);
         px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
         if (!adjustRatio(0))
            px2 = prevpx2;
         paint_or_set(!ropaque);
         break;
      case pINSIDE:
         if (!opaque) paint_or_set(kTRUE);  // draw the old box
         px2 += px - dpx1 - px1;
         px1 = px - dpx1;
         py1 += py - dpy2 - py2;
         py2 = py - dpy2;
         if (px1 < px1p) { px2 += px1p - px1; px1 = px1p; }
         if (px2 > px2p) { px1 -= px2 - px2p; px2 = px2p; }
         if (py1 > py1p) { py2 -= py1 - py1p; py1 = py1p; }
         if (py2 < py2p) { py1 += py2p - py2; py2 = py2p; }
         paint_or_set(!opaque);  // draw the new box
         break;
      }

      if ((mode == pINSIDE && opaque) || (fResizing && ropaque)) {
         // Reset pad parameters and recompute conversion coefficients
         ResizePad();
         switch(mode) {
            case pINSIDE: gPad->ShowGuidelines(this, event); break;
            case pTop: gPad->ShowGuidelines(this, event, 't', true); break;
            case pBot: gPad->ShowGuidelines(this, event, 'b', true); break;
            case pL: gPad->ShowGuidelines(this, event, 'l', true); break;
            case pR: gPad->ShowGuidelines(this, event, 'r', true); break;
            case pA: gPad->ShowGuidelines(this, event, '1', true); break;
            case pB: gPad->ShowGuidelines(this, event, '2', true); break;
            case pC: gPad->ShowGuidelines(this, event, '3', true); break;
            case pD: gPad->ShowGuidelines(this, event, '4', true); break;
            default: break;
         }

         Modified(kTRUE);
      }

      break;

   case kButton1Up:

      if (opaque || ropaque)
         ShowGuidelines(this, event);

      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         fResizing = kFALSE;
         mode = pNone;
         break;
      }

      if ((mode == pINSIDE && !opaque) || (fResizing && !ropaque)) {
         paint_or_set(kFALSE);

         if (fResizing)
            Modified(kTRUE);

         // Reset pad parameters and recompute conversion coefficients
         ResizePad();

         // emit signal
         RangeChanged();
      }

      mode = pNone;
      fResizing = kFALSE;

      break;

   case kButton1Locate:

      ExecuteEvent(kButton1Down, px, py);

      while (true) {
         px = py = 0;
         event = GetCanvasImp()->RequestLocator(px, py);

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

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event for a TAxis object
/// (called by TAxis::ExecuteEvent.)
///  This member function is called when an axis is clicked with the locator
///
/// The axis range is set between the position where the mouse is pressed
/// and the position where it is released.
///
/// If the mouse position is outside the current axis range when it is released
/// the axis is unzoomed with the corresponding proportions.
///
/// Note that the mouse does not need to be in the pad or even canvas
/// when it is released.

void TPad::ExecuteEventAxis(Int_t event, Int_t px, Int_t py, TAxis *axis)
{
   if (!IsEditable()) return;
   if (!axis) return;
   SetCursor(kHand);

   TView *view = GetView();
   static Int_t axisNumber;
   static Double_t ratio1, ratio2;
   static Double_t px1old, py1old, px2old, py2old;
   Int_t nbd, inc, bin1, bin2, first, last;
   Double_t temp, xmin,xmax;
   Bool_t opaque  = gPad->OpaqueMoving();
   bool resetAxisRange = false;
   static std::unique_ptr<TBox> zoombox;
   Double_t zbx1=0,zbx2=0,zby1=0,zby2=0;

   // The CONT4 option, used to paint TH2, is a special case; it uses a 3D
   // drawing technique to paint a 2D plot.
   TString opt = axis->GetParent()->GetDrawOption();
   opt.ToLower();
   Bool_t kCont4 = kFALSE;
   if (strstr(opt,"cont4")) {
      view = nullptr;
      kCont4 = kTRUE;
   }

   auto pp = GetPainter();

   switch (event) {

   case kButton1Down:
      axisNumber = 1;
      if (!strcmp(axis->GetName(),"xaxis"))
         axisNumber = IsVertical() ? 1 : 2;
      if (!strcmp(axis->GetName(),"yaxis"))
         axisNumber = IsVertical() ? 2 : 1;
      if (!strcmp(axis->GetName(),"zaxis"))
         axisNumber = 3;
      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio1);
      } else {
         if (axisNumber == 1) {
            ratio1 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            px1old = GetUxmin()+ratio1*(GetUxmax() - GetUxmin());
            py1old = GetUymin();
            px2old = px1old;
            py2old = GetUymax();
         } else if (axisNumber == 2) {
            ratio1 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py1old = GetUymin()+ratio1*(GetUymax() - GetUymin());
            px1old = GetUxmin();
            px2old = GetUxmax();
            py2old = py1old;
         } else {
            ratio1 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py1old = GetUymin()+ratio1*(GetUymax() - GetUymin());
            px1old = GetUxmax();
            px2old = XtoAbsPixel(GetX2());
            py2old = py1old;
         }
         if (!opaque) {
            pp->DrawBox(px1old, py1old, px2old, py2old, TVirtualPadPainter::kHollow);
         } else {
            if (axisNumber == 1) {
               zbx1 = px1old;
               zbx2 = px2old;
               zby1 = GetUymin();
               zby2 = GetUymax();
            } else if (axisNumber == 2) {
               zbx1 = GetUxmin();
               zbx2 = GetUxmax();
               zby1 = py1old;
               zby2 = py2old;
            }
            if (GetLogx()) {
               zbx1 = TMath::Power(10,zbx1);
               zbx2 = TMath::Power(10,zbx2);
            }
            if (GetLogy()) {
               zby1 = TMath::Power(10,zby1);
               zby2 = TMath::Power(10,zby2);
            }
            zoombox = std::make_unique<TBox>(zbx1, zby1, zbx2, zby2);
            Int_t ci = TColor::GetColor("#7d7dff");
            TColor *zoomcolor = gROOT->GetColor(ci);
            if (!pp->IsSupportAlpha() || !zoomcolor)
               zoombox->SetFillStyle(3002);
            else
               zoomcolor->SetAlpha(0.5);
            zoombox->SetFillColor(ci);
            zoombox->Draw();
            gPad->Modified();
            gPad->Update();
         }
      }
      if (!opaque)
         pp->SetAttLine({-1, 1, 1});
      // No break !!!

   case kButton1Motion:
      if (view) {
         view->GetDistancetoAxis(axisNumber, px, py, ratio2);
      } else {
         if (!opaque)
            pp->DrawBox(px1old, py1old, px2old, py2old, TVirtualPadPainter::kHollow);
         if (axisNumber == 1) {
            ratio2 = (AbsPixeltoX(px) - GetUxmin())/(GetUxmax() - GetUxmin());
            px2old = GetUxmin()+ratio2*(GetUxmax() - GetUxmin());
         } else {
            ratio2 = (AbsPixeltoY(py) - GetUymin())/(GetUymax() - GetUymin());
            py2old = GetUymin()+ratio2*(GetUymax() - GetUymin());
         }
         if (!opaque) {
            pp->DrawBox(px1old, py1old, px2old, py2old, TVirtualPadPainter::kHollow);
         } else {
            if (axisNumber == 1) {
               zbx1 = px1old;
               zbx2 = px2old;
               zby1 = GetUymin();
               zby2 = GetUymax();
            } else if (axisNumber == 2) {
               zbx1 = GetUxmin();
               zbx2 = GetUxmax();
               zby1 = py1old;
               zby2 = py2old;
            }
            if (GetLogx()) {
               zbx1 = TMath::Power(10,zbx1);
               zbx2 = TMath::Power(10,zbx2);
            }
            if (GetLogy()) {
               zby1 = TMath::Power(10,zby1);
               zby2 = TMath::Power(10,zby2);
            }
            if (zoombox) {
               zoombox->SetX1(zbx1);
               zoombox->SetY1(zby1);
               zoombox->SetX2(zbx2);
               zoombox->SetY2(zby2);
            }
            gPad->Modified();
            gPad->Update();
         }
      }
   break;

   case kWheelUp:
      nbd  = (axis->GetLast()-axis->GetFirst());
      inc  = TMath::Max(nbd/100,1);
      bin1 = axis->GetFirst()+inc;
      bin2 = axis->GetLast()-inc;
      bin1 = TMath::Max(bin1, 1);
      bin2 = TMath::Min(bin2, axis->GetNbins());
      if (bin2>bin1) {
         axis->SetRange(bin1,bin2);
         gPad->Modified();
         gPad->Update();
      }
   break;

   case kWheelDown:
      nbd  = (axis->GetLast()-axis->GetFirst());
      inc  = TMath::Max(nbd/100,1);
      bin1 = axis->GetFirst()-inc;
      bin2 = axis->GetLast()+inc;
      bin1 = TMath::Max(bin1, 1);
      bin2 = TMath::Min(bin2, axis->GetNbins());
      resetAxisRange = (bin1 == 1 && axis->GetFirst() == 1 && bin2 == axis->GetNbins() && axis->GetLast() == axis->GetNbins());
      if (bin2>bin1) {
         axis->SetRange(bin1,bin2);
      }
      if (resetAxisRange)
         axis->ResetBit(TAxis::kAxisRange);
      if (bin2>bin1) {
         gPad->Modified();
         gPad->Update();
      }
   break;

   case kButton1Up:
      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         if (opaque && zoombox)
            zoombox.reset();
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
               if (newmin < zmin) newmin = hobj->GetBinContent(hobj->GetMinimumBin());
               if (newmax > zmax) newmax = hobj->GetBinContent(hobj->GetMaximumBin());
               if (GetLogz()){
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
               bin1 = TMath::Max(bin1, 1);
               bin2 = TMath::Min(bin2, axis->GetNbins());
               axis->SetRange(bin1, bin2);
            }
            delete view;
            SetView(nullptr);
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
            bin1 = TMath::Max(bin1, 1);
            bin2 = TMath::Min(bin2, axis->GetNbins());
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
                     bin1 = hobj->GetYaxis()->FindFixBin(xmin);
                     bin2 = hobj->GetYaxis()->FindFixBin(xmax);
                     hobj->GetYaxis()->SetRange(bin1,bin2);
                  }
               }
            }
            Modified(kTRUE);
         }
      }
      if (!opaque) {
         pp->SetAttLine({-1, 1, 1});
      } else {
         if (zoombox) {
            zoombox.reset();
            gPad->Modified();
            gPad->Update();
         }
      }
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Search if object named name is inside this pad or in pads inside this pad.
///
/// In case name is in several sub-pads the first one is returned.

TObject *TPad::FindObject(const char *name) const
{
   if (!fPrimitives) return nullptr;
   TObject *found = fPrimitives->FindObject(name);
   if (found) return found;
   TIter    next(GetListOfPrimitives());
   while (auto cur = next()) {
      if (cur->InheritsFrom(TPad::Class())) {
         found = ((TPad*)cur)->FindObject(name);
         if (found) return found;
      }
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Search if obj is in pad or in pads inside this pad.
///
/// In case obj is in several sub-pads the first one is returned.

TObject *TPad::FindObject(const TObject *obj) const
{
   if (!fPrimitives) return nullptr;
   TObject *found = fPrimitives->FindObject(obj);
   if (found) return found;
   TIter    next(GetListOfPrimitives());
   while (auto cur = next()) {
      if (cur->InheritsFrom(TPad::Class())) {
         found = ((TPad*)cur)->FindObject(obj);
         if (found) return found;
      }
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get canvas identifier.

Int_t TPad::GetCanvasID() const
{
   return fCanvas ? fCanvas->GetCanvasID() : -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Get canvas implementation pointer if any

TCanvasImp *TPad::GetCanvasImp() const
{
   return fCanvas ? fCanvas->GetCanvasImp() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get Event.

Int_t TPad::GetEvent() const
{
   return  fCanvas ? fCanvas->GetEvent() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get X event.

Int_t TPad::GetEventX() const
{
   return  fCanvas ? fCanvas->GetEventX() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get Y event.

Int_t TPad::GetEventY() const
{
   return  fCanvas ? fCanvas->GetEventY() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get virtual canvas.

TVirtualPad *TPad::GetVirtCanvas() const
{
   return  fCanvas ? (TVirtualPad*) fCanvas : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get highlight color.

Color_t TPad::GetHighLightColor() const
{
   return  fCanvas ? fCanvas->GetHighLightColor() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function (see also TPad::SetMaxPickDistance)

Int_t TPad::GetMaxPickDistance()
{
   return fgMaxPickDistance;
}

////////////////////////////////////////////////////////////////////////////////
/// Get selected.

TObject *TPad::GetSelected() const
{
   if (fCanvas == this) return nullptr;
   return  fCanvas ? fCanvas->GetSelected() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get selected pad.

TVirtualPad *TPad::GetSelectedPad() const
{
   if (fCanvas == this) return nullptr;
   return  fCanvas ? fCanvas->GetSelectedPad() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get save pad.

TVirtualPad *TPad::GetPadSave() const
{
   if (fCanvas == this) return nullptr;
   return  fCanvas ? fCanvas->GetPadSave() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get Wh.

UInt_t TPad::GetWh() const
{
   return  fCanvas ? fCanvas->GetWh() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get Ww.

UInt_t TPad::GetWw() const
{
   return  fCanvas ? fCanvas->GetWw() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Hide tool tip depending on the event type. Typically tool tips
/// are hidden when event is not a kMouseEnter and not a kMouseMotion
/// event.

void TPad::HideToolTip(Int_t event)
{
   if (event != kMouseEnter && event != kMouseMotion && fTip)
      gPad->CloseToolTip(fTip);
}

////////////////////////////////////////////////////////////////////////////////
/// Is pad in batch mode ?

Bool_t TPad::IsBatch() const
{
   return  fCanvas ? fCanvas->IsBatch() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is pad retained ?

Bool_t TPad::IsRetained() const
{
   return  fCanvas ? fCanvas->IsRetained() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is web ?
Bool_t TPad::IsWeb() const
{
   return fCanvas ? fCanvas->IsWeb() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is pad moving in opaque mode ?

Bool_t TPad::OpaqueMoving() const
{
   return  fCanvas ? fCanvas->OpaqueMoving() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Is pad resizing in opaque mode ?

Bool_t TPad::OpaqueResizing() const
{
   return  fCanvas ? fCanvas->OpaqueResizing() : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set pad in batch mode.

void TPad::SetBatch(Bool_t batch)
{
   if (fCanvas) fCanvas->SetBatch(batch);
}

////////////////////////////////////////////////////////////////////////////////
/// Set canvas size.

void TPad::SetCanvasSize(UInt_t ww, UInt_t wh)
{
   if (fCanvas) fCanvas->SetCanvasSize(ww,wh);
}

////////////////////////////////////////////////////////////////////////////////
/// Set cursor type.

void TPad::SetCursor(ECursor cursor)
{
   if (fCanvas) fCanvas->SetCursor(cursor);
}

////////////////////////////////////////////////////////////////////////////////
/// Set double buffer mode ON or OFF.

void TPad::SetDoubleBuffer(Int_t mode)
{
   if (fCanvas) fCanvas->SetDoubleBuffer(mode);
}

////////////////////////////////////////////////////////////////////////////////
/// Set selected.

void TPad::SetSelected(TObject *obj)
{
   if (fCanvas) fCanvas->SetSelected(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Update pad.

void TPad::Update()
{
   if (fCanvas) fCanvas->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Asynchronous pad update.
/// In case of web-based canvas triggers update of the canvas on the client side,
/// but does not wait that real update is completed. Avoids blocking of caller thread.
/// Have to be used if called from other web-based widget to avoid logical dead-locks.
/// In case of normal canvas just canvas->Update() is performed.

void TPad::UpdateAsync()
{
   if (fCanvas) fCanvas->UpdateAsync();
}

////////////////////////////////////////////////////////////////////////////////
/// Get frame.

TFrame *TPad::GetFrame()
{
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
   } else {
      // Preexisting and now assigned to fFrame, let's make sure it is not
      // deleted twice (the bit might have been set in TPad::Streamer)
      fFrame->ResetBit(kCanDelete);
   }
   return fFrame;
}

////////////////////////////////////////////////////////////////////////////////
/// Get primitive.

TObject *TPad::GetPrimitive(const char *name) const
{
   if (!fPrimitives) return nullptr;
   TIter next(fPrimitives);
   TObject *found, *obj;
   while ((obj=next())) {
      if (!strcmp(name, obj->GetName())) return obj;
      if (obj->InheritsFrom(TPad::Class())) continue;
      found = obj->FindObject(name);
      if (found) return found;
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a pointer to subpadnumber of this pad.

TVirtualPad *TPad::GetPad(Int_t subpadnumber) const
{
   if (!subpadnumber) {
      return (TVirtualPad*)this;
   }

   TObject *obj;
   if (!fPrimitives) return nullptr;
   TIter    next(GetListOfPrimitives());
   while ((obj = next())) {
      if (obj->InheritsFrom(TVirtualPad::Class())) {
         TVirtualPad *pad = (TVirtualPad*)obj;
         if (pad->GetNumber() == subpadnumber) return pad;
      }
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Return lower and upper bounds of the pad in NDC coordinates.

void TPad::GetPadPar(Double_t &xlow, Double_t &ylow, Double_t &xup, Double_t &yup)
{
   xlow = fXlowNDC;
   ylow = fYlowNDC;
   xup  = fXlowNDC+fWNDC;
   yup  = fYlowNDC+fHNDC;
}

////////////////////////////////////////////////////////////////////////////////
/// Return pad world coordinates range.

void TPad::GetRange(Double_t &x1, Double_t &y1, Double_t &x2, Double_t &y2)
{
   x1 = fX1;
   y1 = fY1;
   x2 = fX2;
   y2 = fY2;
}

////////////////////////////////////////////////////////////////////////////////
/// Return pad axis coordinates range.

void TPad::GetRangeAxis(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax)
{
   xmin = fUxmin;
   ymin = fUymin;
   xmax = fUxmax;
   ymax = fUymax;
}

////////////////////////////////////////////////////////////////////////////////
/// Highlight pad.
/// do not highlight when printing on Postscript

void TPad::HighLight(Color_t color, Bool_t set)
{
   if (auto pp = GetPainter())
      if(!pp->IsNative())
         return;

   if (color <= 0)
      return;

   AbsCoordinates(kTRUE);

   // We do not want to have active(executable) buttons, etc highlighted
   // in this manner, unless we want to edit'em
   if (GetMother() && GetMother()->IsEditable() && !InheritsFrom(TButton::Class())) {
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
      if (GetBorderMode() > 0)
         PaintBorder(set ? -color : -GetFillColor(), kFALSE);
   }

   AbsCoordinates(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// List all primitives in pad.

void TPad::ls(Option_t *option) const
{
   TROOT::IndentLevel();
   std::cout <<IsA()->GetName()<<" fXlowNDC=" <<fXlowNDC<<" fYlowNDC="<<fYlowNDC<<" fWNDC="<<GetWNDC()<<" fHNDC="<<GetHNDC()
        <<" Name= "<<GetName()<<" Title= "<<GetTitle()<<" Option="<<option<<std::endl;
   TROOT::IncreaseDirLevel();
   if (!fPrimitives) return;
   fPrimitives->ls(option);
   TROOT::DecreaseDirLevel();
}

////////////////////////////////////////////////////////////////////////////////
/// Increment (i==1) or set (i>1) the number of autocolor in the pad.

Int_t TPad::IncrementPaletteColor(Int_t i, TString opt)
{
   if (opt.Index("pfc")>=0 || opt.Index("plc")>=0 || opt.Index("pmc")>=0) {
       if (i==1) fNumPaletteColor++;
       else      fNumPaletteColor = i;
       return    fNumPaletteColor;
   } else {
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get the next autocolor in the pad.

Int_t TPad::NextPaletteColor()
{
   Int_t i = 0;
   Int_t ncolors = gStyle->GetNumberOfColors();
   if (fNumPaletteColor>1) {
      i = fNextPaletteColor*(ncolors/(fNumPaletteColor-1));
      if (i>=ncolors) i = ncolors-1;
   }
   fNextPaletteColor++;
   if (fNextPaletteColor > fNumPaletteColor-1) fNextPaletteColor = 0;
   return gStyle->GetColorPalette(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialise the grid used to find empty space when adding a box (Legend) in a pad

void TPad::FillCollideGrid(TObject *oi)
{
   Int_t const cellSize = 10; // Size of an individual grid cell in pixels.

   fCGnx = GetWw()/cellSize;
   fCGny = GetWh()/cellSize;

   // Initialise the collide grid
   fCollideGrid.resize(fCGnx*fCGny);
   for (int i = 0; i < fCGnx; i++)
      for (int j = 0; j < fCGny; j++)
         fCollideGrid[i + j * fCGnx] = kTRUE;

   // Fill the collide grid
   TIter iter(GetListOfPrimitives());

   while(auto o = iter()) {
      if (o == oi)
         continue;
      if (o->InheritsFrom(TFrame::Class()))
         FillCollideGridTFrame(o);
      else if (o->InheritsFrom(TBox::Class()))
         FillCollideGridTBox(o);
      else if (o->InheritsFrom(TH1::Class()))
         FillCollideGridTH1(o);
      else if (o->InheritsFrom(TGraph::Class()))
         FillCollideGridTGraph(o);
      else if (o->InheritsFrom(TMultiGraph::Class())) {
         TIter nextgraph(((TMultiGraph *)o)->GetListOfGraphs());
         while (auto og = nextgraph())
            FillCollideGridTGraph(og);
      } else if (o->InheritsFrom(THStack::Class())) {
         TIter nexthist(((THStack *)o)->GetHists());
         while (auto oh = nexthist()) {
            if (oh->InheritsFrom(TH1::Class()))
               FillCollideGridTH1(oh);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if a box of size w and h collide some primitives in the pad at
/// position i,j

Bool_t TPad::Collide(Int_t i, Int_t j, Int_t w, Int_t h)
{
   for (int r = i; r < w + i; r++) {
      for (int c = j; c < h + j; c++) {
         if (!fCollideGrid[r + c * fCGnx])
            return kTRUE;
      }
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Place a box in NDC space
///
/// \return `true` if the box could be placed, `false` if not.
///
/// \param[in]  o               pointer to the box to be placed
/// \param[in]  w               box width to be placed
/// \param[in]  h               box height to be placed
/// \param[out] xl              x position of the bottom left corner of the placed box
/// \param[out] yb              y position of the bottom left corner of the placed box
/// \param[in]  option          l=left, r=right, t=top, b=bottom, w=within margins. Order determines
///                             priority for placement. Default is "lb" (prioritises horizontal over vertical)

Bool_t TPad::PlaceBox(TObject *o, Double_t w, Double_t h, Double_t &xl, Double_t &yb, Option_t* option)
{
   FillCollideGrid(o);

   Int_t iw = (int)(fCGnx*w);
   Int_t ih = (int)(fCGny*h);

   Int_t nxbeg = 0;
   Int_t nybeg = 0;
   Int_t nxend = fCGnx-iw-1;
   Int_t nyend = fCGny-ih-1;
   Int_t dx = 1;
   Int_t dy = 1;

   bool isFirstVertical = false;
   bool isFirstHorizontal = false;

   for (std::size_t i = 0; option[i] != '\0'; ++i) {
      char letter = std::tolower(option[i]);
      if (letter == 'w') {
         nxbeg += fCGnx*GetLeftMargin();
         nybeg += fCGny*GetBottomMargin();
         nxend -= fCGnx*GetRightMargin();
         nyend -= fCGny*GetTopMargin();
      } else if (letter == 't' || letter == 'b') {
         isFirstVertical = !isFirstHorizontal;
         // go from top to bottom instead of bottom to top
         dy = letter == 't' ? -1 : 1;
      } else if (letter == 'l' || letter == 'r') {
         isFirstHorizontal = !isFirstVertical;
         // go from right to left instead of left to right
         dx = letter == 'r' ? -1 : 1;
      }
   }

   if(dx < 0) std::swap(nxbeg, nxend);
   if(dy < 0) std::swap(nybeg, nyend);

   auto attemptPlacement = [&](Int_t i, Int_t j) {
      if (Collide(i, j, iw, ih)) {
         return false;
      } else {
         xl = (Double_t)(i) / (Double_t)(fCGnx);
         yb = (Double_t)(j) / (Double_t)(fCGny);
         return true;
      }
   };

   if(!isFirstVertical) {
      for (Int_t i = nxbeg; i != nxend; i += dx) {
         for (Int_t j = nybeg; j != nyend; j += dy) {
            if (attemptPlacement(i, j)) return true;
         }
      }
   } else {
      // prioritizing vertical over horizontal
      for (Int_t j = nybeg; j != nyend; j += dy) {
         for (Int_t i = nxbeg; i != nxend; i += dx) {
            if (attemptPlacement(i, j)) return true;
         }
      }
   }

   return kFALSE;
}

#define NotFree(i, j) fCollideGrid[TMath::Max(TMath::Min(i+j*fCGnx,fCGnx*fCGny),0)] = kFALSE;

////////////////////////////////////////////////////////////////////////////////
/// Mark as "not free" the cells along a line.

void TPad::LineNotFree(Int_t x1, Int_t x2, Int_t y1, Int_t y2)
{
   NotFree(x1, y1);
   NotFree(x2, y2);
   Int_t i, j, xt, yt;

   // horizontal lines
   if (y1==y2) {
      for (i=x1+1; i<x2; i++) NotFree(i,y1);
      return;
   }

   // vertical lines
   if (x1==x2) {
      for (i=y1+1; i<y2; i++) NotFree(x1,i);
      return;
   }

   // other lines
   if (TMath::Abs(x2-x1)>TMath::Abs(y2-y1)) {
      if (x1>x2) {
         xt = x1; x1 = x2; x2 = xt;
         yt = y1; y1 = y2; y2 = yt;
      }
      for (i=x1+1; i<x2; i++) {
         j = (Int_t)((Double_t)(y2-y1)*(Double_t)((i-x1)/(Double_t)(x2-x1))+y1);
         NotFree(i,j);
         NotFree(i,(j+1));
      }
   } else {
      if (y1>y2) {
         yt = y1; y1 = y2; y2 = yt;
         xt = x1; x1 = x2; x2 = xt;
      }
      for (j=y1+1; j<y2; j++) {
         i = (Int_t)((Double_t)(x2-x1)*(Double_t)((j-y1)/(Double_t)(y2-y1))+x1);
         NotFree(i,j);
         NotFree((i+1),j);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
void TPad::FillCollideGridTBox(TObject *o)
{
   TBox *b = (TBox *)o;
   if (fCGnx==0||fCGny==0) return;
   Double_t xs   = (fX2-fX1)/fCGnx;
   Double_t ys   = (fY2-fY1)/fCGny;

   Int_t x1 = (Int_t)((b->GetX1()-fX1)/xs);
   Int_t x2 = (Int_t)((b->GetX2()-fX1)/xs);
   Int_t y1 = (Int_t)((b->GetY1()-fY1)/ys);
   Int_t y2 = (Int_t)((b->GetY2()-fY1)/ys);
   for (int i = x1; i<=x2; i++) {
      for (int j = y1; j<=y2; j++) NotFree(i, j);
   }
}

////////////////////////////////////////////////////////////////////////////////
void TPad::FillCollideGridTFrame(TObject *o)
{
   TFrame *f = (TFrame *)o;
   if (fCGnx==0||fCGny==0) return;
   Double_t xs   = (fX2-fX1)/fCGnx;
   Double_t ys   = (fY2-fY1)/fCGny;

   Int_t x1 = (Int_t)((f->GetX1()-fX1)/xs);
   Int_t x2 = (Int_t)((f->GetX2()-fX1)/xs);
   Int_t y1 = (Int_t)((f->GetY1()-fY1)/ys);
   Int_t y2 = (Int_t)((f->GetY2()-fY1)/ys);
   Int_t i;

   for (i = x1; i<=x2; i++) {
      NotFree(i, y1);
      NotFree(i, (y1-1));
      NotFree(i, (y1-2));
   }
   for (i = y1; i<=y2; i++) {
      NotFree(x1, i);
      NotFree((x1-1), i);
      NotFree((x1-2), i);
   }
}

////////////////////////////////////////////////////////////////////////////////
void TPad::FillCollideGridTGraph(TObject *o)
{
   TGraph *g = (TGraph *)o;
   if (fCGnx==0||fCGny==0) return;
   Double_t xs   = (fX2-fX1)/fCGnx;
   Double_t ys   = (fY2-fY1)/fCGny;

   Int_t n = g->GetN();
   Int_t s = TMath::Max(n/10,1);
   Double_t x1, x2, y1, y2;
   for (Int_t i=s; i<n; i=i+s) {
      g->GetPoint(TMath::Max(0,i-s),x1,y1);
      g->GetPoint(i  ,x2,y2);
      if (fLogx) {
         if (x1 > 0) x1 = TMath::Log10(x1);
         else        x1 = fUxmin;
         if (x2 > 0) x2 = TMath::Log10(x2);
         else        x2 = fUxmin;
      }
      if (fLogy) {
         if (y1 > 0) y1 = TMath::Log10(y1);
         else        y1 = fUymin;
         if (y2 > 0) y2 = TMath::Log10(y2);
         else        y2 = fUymin;
      }
      LineNotFree((int)((x1-fX1)/xs), (int)((x2-fX1)/xs),
                  (int)((y1-fY1)/ys), (int)((y2-fY1)/ys));
   }
}

////////////////////////////////////////////////////////////////////////////////
void TPad::FillCollideGridTH1(TObject *o)
{
   TH1 *h = (TH1 *)o;
   if (fCGnx==0||fCGny==0) return;
   if (o->InheritsFrom(TH2::Class())) return;
   if (o->InheritsFrom(TH3::Class())) return;

   TString name = h->GetName();
   if (name.Index("hframe") >= 0) return;

   Double_t xs   = (fX2-fX1)/fCGnx;
   Double_t ys   = (fY2-fY1)/fCGny;

   bool haserrors = false;
   TString drawOption = h->GetDrawOption();
   drawOption.ToLower();
   drawOption.ReplaceAll("same","");

   if (drawOption.Index("hist") < 0) {
      if (drawOption.Index("e") >= 0) haserrors = true;
   }

   Int_t nx = h->GetNbinsX();
   Int_t  x1, y1, y2;
   Int_t i, j;
   Double_t x1l, y1l, y2l;

   for (i = 1; i<nx; i++) {
      if (haserrors) {
         x1l = h->GetBinCenter(i);
         if (fLogx) {
            if (x1l > 0) x1l = TMath::Log10(x1l);
            else         x1l = fUxmin;
         }
         x1 = (Int_t)((x1l-fX1)/xs);
         y1l = h->GetBinContent(i)-h->GetBinErrorLow(i);
         if (fLogy) {
            if (y1l > 0) y1l = TMath::Log10(y1l);
            else         y1l = fUymin;
         }
         y1 = (Int_t)((y1l-fY1)/ys);
         y2l = h->GetBinContent(i)+h->GetBinErrorUp(i);
         if (fLogy) {
            if (y2l > 0) y2l = TMath::Log10(y2l);
            else         y2l = fUymin;
         }
         y2 = (Int_t)((y2l-fY1)/ys);
         for (j=y1; j<y2; j++) {
            NotFree(x1, j);
         }
      }
      x1l = h->GetBinLowEdge(i);
      if (fLogx) {
         if (x1l > 0) x1l = TMath::Log10(x1l);
         else         x1l = fUxmin;
      }
      x1 = (Int_t)((x1l-fX1)/xs);
      y1l = h->GetBinContent(i);
      if (fLogy) {
         if (y1l > 0) y1l = TMath::Log10(y1l);
         else         y1l = fUymin;
      }
      y1 = (Int_t)((y1l-fY1)/ys);
      NotFree(x1, y1);
      x1l = h->GetBinLowEdge(i)+h->GetBinWidth(i);
      if (fLogx) {
         if (x1l > 0) x1l = TMath::Log10(x1l);
         else         x1l = fUxmin;
      }
      x1 = (int)((x1l-fX1)/xs);
      NotFree(x1, y1);
   }

   // Extra objects in the list of function
   TPaveStats *ps = (TPaveStats*)h->GetListOfFunctions()->FindObject("stats");
   if (ps) FillCollideGridTBox(ps);
}

////////////////////////////////////////////////////////////////////////////////
/// This method draws the collide grid on top of the canvas. This is used for
/// debugging only. At some point it will be removed.

void TPad::DrawCollideGrid()
{
   if (fCGnx==0||fCGny==0) return;

   TContext ctxt(this, kTRUE);

   TBox box;
   box.SetFillColorAlpha(kRed,0.5);

   Double_t xs   = (fX2-fX1)/fCGnx;
   Double_t ys   = (fY2-fY1)/fCGny;

   Double_t X1L, X2L, Y1L, Y2L;
   Double_t t = 0.15;
   Double_t Y1, Y2;
   Double_t X1 = fX1;
   Double_t X2 = X1+xs;

   for (int i = 0; i<fCGnx; i++) {
      Y1 = fY1;
      Y2 = Y1+ys;
      for (int j = 0; j<fCGny; j++) {
         if (GetLogx()) {
            X1L = TMath::Power(10,X1);
            X2L = TMath::Power(10,X2);
         } else {
            X1L = X1;
            X2L = X2;
         }
         if (GetLogy()) {
            Y1L = TMath::Power(10,Y1);
            Y2L = TMath::Power(10,Y2);
         } else {
            Y1L = Y1;
            Y2L = Y2;
         }
         if (!fCollideGrid[i + j*fCGnx]) {
            box.SetFillColorAlpha(kBlack,t);
            box.DrawBox(X1L, Y1L, X2L, Y2L);
         } else {
            box.SetFillColorAlpha(kRed,t);
            box.DrawBox(X1L, Y1L, X2L, Y2L);
         }
         Y1 = Y2;
         Y2 = Y1+ys;
         if (t==0.15) t = 0.1;
         else         t = 0.15;
      }
      X1 = X2;
      X2 = X1+xs;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Short cut to call Modified() and Update() in a single call.
/// On Mac with Cocoa, it performs an additional ProcessEvents().

void TPad::ModifiedUpdate()
{
   Modified();
   Update();
#ifdef R__HAS_COCOA
   gSystem->ProcessEvents();
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// Convert x from pad to X.

Double_t TPad::PadtoX(Double_t x) const
{
   if (fLogx && x < 50) return Double_t(TMath::Exp(2.302585092994*x));
   return x;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert y from pad to Y.

Double_t TPad::PadtoY(Double_t y) const
{
   if (fLogy && y < 50) return Double_t(TMath::Exp(2.302585092994*y));
   return y;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert x from X to pad.

Double_t TPad::XtoPad(Double_t x) const
{
   if (fLogx) {
      if (x > 0) x = TMath::Log10(x);
      else       x = fUxmin;
   }
   return x;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert y from Y to pad.

Double_t TPad::YtoPad(Double_t y) const
{
   if (fLogy) {
      if (y > 0) y = TMath::Log10(y);
      else       y = fUymin;
   }
   return y;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint all primitives in pad.

void TPad::Paint(Option_t * /*option*/)
{
   if (!fPrimitives)
      fPrimitives = new TList;
   if (fViewer3D && fViewer3D->CanLoopOnPrimitives()) {
      fViewer3D->PadPaint(this);
      Modified(kFALSE);
      if (GetGLDevice()!=-1 && gVirtualPS) {
         TContext ctxt(this, kFALSE);
         if (gGLManager) gGLManager->PrintViewer(GetViewer3D());
      }
      return;
   }

   TVirtualPadPainter *oldpp = nullptr;
   Bool_t replace_pp = kFALSE;

   if (fCanvas) {
      // check if special PS painter should be assigned to TCanvas
      replace_pp = fCanvas->EnsurePSPainter(kTRUE, oldpp);
      TColor::SetGrayscale(fCanvas->IsGrayscale());
   }

   Bool_t began3DScene = kFALSE;
   fPadPaint = 1;

   {
      TContext ctxt(this, kTRUE);

      PaintBorder(GetFillColor(), kTRUE);
      PaintDate();

      auto lnk = GetListOfPrimitives()->FirstLink();

      while (lnk) {
         TObject *obj = lnk->GetObject();

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
         lnk = lnk->Next();
      }

      if (fCanvas && (fCanvas->fHilightPadBorder == this))
         PaintBorder(-GetHighLightColor(), kTRUE);
   }

   fPadPaint = 0;
   Modified(kFALSE);

   if (replace_pp && fCanvas)
      fCanvas->EnsurePSPainter(kFALSE, oldpp);

   // Close the 3D scene if we opened it. This must be done after modified
   // flag is cleared, as some viewers will invoke another paint by marking pad modified again
   if (began3DScene) {
      fViewer3D->EndScene();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint the pad border.
/// Draw first  a box as a normal filled box

void TPad::PaintBorder(Color_t color, Bool_t /* tops */)
{
   auto pp = GetPainter();
   if (!pp)
      return;

   pp->OnPad(this);

   if (color >= 0) {

      Bool_t do_paint_box = kTRUE;

      Style_t style = GetFillStyle();
      if (!IsBatch() && (pp->IsCocoa() || (pp->IsNative() && (style > 3000) && (style < 3026))))
         pp->ClearDrawable();

      if ((style >= 4000) && (style <= 4100) && pp->IsNative()) {
         if (this == fMother) {
            style = 1001;
         } else if (pp->IsCocoa() || (fCanvas && fCanvas->UseGL())) {
            TColor *col = (style == 4000) ? nullptr : gROOT->GetColor(color);
            if (!col) {
               do_paint_box = kFALSE;
            } else {
               color = TColor::GetColor(col->GetRed(), col->GetGreen(), col->GetBlue(), (style - 4000) / 100.);
               style = 1001;
            }
         } else {
            // copy all pixmaps
            do_paint_box = kFALSE;
            int px, py;
            XYtoAbsPixel(fX1, fY2, px, py);
            if (fMother) {
               fMother->CopyBackgroundPixmap(px, py);
               CopyBackgroundPixmaps(fMother, this, px, py);
            }
            pp->SetOpacity(style - 4000);
         }
      } else if ((color == 10) && (style > 3000) && (style < 3100))
         color = 1;

      if (do_paint_box) {
         pp->SetAttFill({color, style});
         pp->SetAttLine(*this);
         PaintBox(fX1, fY1, fX2, fY2);
      }
   } else
      color = -color;

   if (IsTransparent() || (fBorderMode == 0))
      return;

   // then paint 3d frame (depending on bordermode)
   // Paint a 3D frame around the pad.

   TWbox box;
   box.SetFillColor(color);
   box.SetFillStyle(GetFillStyle());
   TAttLine::Copy(box);

   box.PaintBorderOn(this, fX1, fY1, fX2, fY2,
                     fBorderSize, fBorderMode,
                     InheritsFrom(TButton::Class()) && fBorderMode == -1 && TestBit(kFraming));
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a frame border with Postscript - no longer used

void TPad::PaintBorderPS(Double_t xl,Double_t yl,Double_t xt,Double_t yt,Int_t bmode,Int_t bsize,Int_t dark,Int_t light)
{
   if (!gVirtualPS) return;
   gVirtualPS->DrawFrame(xl, yl, xt, yt, bmode,bsize,dark,light);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint the current date and time if the option `Date` is set on via `gStyle->SetOptDate()`
/// Paint the current file name if the option `File` is set on via `gStyle->SetOptFile()`

void TPad::PaintDate()
{
   if (fCanvas == this) {
      if (gStyle->GetOptDate()) {
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
      if (gStyle->GetOptFile() && gFile) {
         TText tfile(1. - gStyle->GetDateX(),gStyle->GetDateY(),gFile->GetName());
         tfile.SetTextSize( gStyle->GetAttDate()->GetTextSize());
         tfile.SetTextFont( gStyle->GetAttDate()->GetTextFont());
         tfile.SetTextColor(gStyle->GetAttDate()->GetTextColor());
         tfile.SetTextAlign(31);
         tfile.SetTextAngle(gStyle->GetAttDate()->GetTextAngle());
         tfile.SetNDC();
         tfile.Paint();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint histogram/graph frame.

void TPad::PaintPadFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
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
   frame->Paint();
}

////////////////////////////////////////////////////////////////////////////////
/// Traverse pad hierarchy and (re)paint only modified pads.

void TPad::PaintModified()
{
   if (fViewer3D && fViewer3D->CanLoopOnPrimitives()) {
      if (IsModified()) {
         fViewer3D->PadPaint(this);
         Modified(kFALSE);
      }
      TList *pList = GetListOfPrimitives();
      auto lnk = pList ? pList->FirstLink() : nullptr;
      while (lnk) {
         auto obj = lnk->GetObject();
         if (obj->InheritsFrom(TPad::Class()))
            ((TPad*)obj)->PaintModified();
         lnk = lnk->Next();
      }
      return;
   }

   TVirtualPS *saveps = gVirtualPS;
   if (gVirtualPS && gVirtualPS->TestBit(kPrintingPS))
      gVirtualPS = nullptr;

   TVirtualPadPainter *oldpp = nullptr;
   Bool_t replace_pp = kFALSE;

   if (fCanvas) {
      // check if special PS painter should be assigned to TCanvas
      replace_pp = fCanvas->EnsurePSPainter(kTRUE, oldpp);
      TColor::SetGrayscale(fCanvas->IsGrayscale());
   }

   Bool_t began3DScene = kFALSE;
   fPadPaint = 1;
   {
      TContext ctxt(this, kTRUE);
      if (IsModified() || IsTransparent())
         PaintBorder(GetFillColor(), kTRUE);

      PaintDate();

      TList *pList = GetListOfPrimitives();
      auto lnk = pList ? pList->FirstLink() : nullptr;

      while (lnk) {
         TObject *obj = lnk->GetObject();
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
         lnk = lnk->Next();
      }
   }

   fPadPaint = 0;
   Modified(kFALSE);

   // This must be done after modified flag is cleared, as some
   // viewers will invoke another paint by marking pad modified again
   if (began3DScene && fViewer3D)
      fViewer3D->EndScene();

   if (replace_pp && fCanvas)
      fCanvas->EnsurePSPainter(kFALSE, oldpp);

   gVirtualPS = saveps;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint box in CurrentPad World coordinates.
///
///  - if option[0] = 's' the box is forced to be paint with style=0
///  - if option[0] = 'l' the box contour is drawn

void TPad::PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *option)
{
   auto pp = GetPainter();
   if (!pp)
      return;

   pp->OnPad(this);

   TAttFill att = pp->GetAttFill();

   Int_t style0 = -1111, style = att.GetFillStyle();
   Bool_t draw_border = kFALSE, draw_fill = kFALSE;
   if (option && *option == 's') {
      style0 = style;
      att.SetFillStyle(0);
      pp->SetAttFill(att);
      style = 0;
      draw_border = kTRUE;
   } else if (option && *option == 'l')
      draw_border = kTRUE;

   if (style >= 3100 && style < 4000) {
      Double_t xb[4] = {x1, x1, x2, x2};
      Double_t yb[4] = {y1, y2, y2, y1};
      PaintFillAreaHatches(4, xb, yb, style);
   } else if (pp->GetPS()) {
      draw_fill = kTRUE;
      if (style == 0)
         draw_border = kFALSE;
   } else if ((style > 0) && (style < 1000)) {
      draw_border = kTRUE;
   } else if ((style >= 1000) && (style < 2000)) {
      draw_fill = kTRUE;
   } else if (style > 3000 && style < 3100) {
      draw_fill = style < 3026;
   } else if (style >= 4000 && style <= 4100) {
      // only used by pad, ignored by all other objects
   } else if (style > 0)
      draw_border = kTRUE;

   if (draw_fill)
      pp->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kFilled);

   if (draw_border)
      pp->DrawBox(x1, y1, x2, y2, TVirtualPadPainter::kHollow);

   if (style0 != -1111) {
      att.SetFillStyle(style0);
      pp->SetAttFill(att);
   }

   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy pixmaps of pads laying below pad "stop" into pad "stop". This
/// gives the effect of pad "stop" being transparent.

void TPad::CopyBackgroundPixmaps(TPad *start, TPad *stop, Int_t x, Int_t y)
{
   if (!start) return;
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

////////////////////////////////////////////////////////////////////////////////
/// Copy pixmap of this pad as background of the current pad.

void TPad::CopyBackgroundPixmap(Int_t x, Int_t y)
{
   int px, py;
   XYtoAbsPixel(fX1, fY2, px, py);
   if (auto pp = GetPainter())
      pp->CopyDrawable(GetPixmapID(), px-x, py-y);
}

////////////////////////////////////////////////////////////////////////////////

void TPad::PaintFillArea(Int_t, Float_t *, Float_t *, Option_t *)
{
   Warning("TPad::PaintFillArea", "Float_t signature is obsolete. Use Double_t signature.");
}

////////////////////////////////////////////////////////////////////////////////
/// Paint fill area in CurrentPad World coordinates.

void TPad::PaintFillArea(Int_t nn, Double_t *xx, Double_t *yy, Option_t *)
{
   if (nn < 3)
      return;
   Double_t xmin,xmax,ymin,ymax;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
   }

   Int_t nc = 2*nn+1;
   std::vector<Double_t> x(nc, 0.);
   std::vector<Double_t> y(nc, 0.);

   Int_t n = ClipPolygon(nn, xx, yy, nc, x.data(), y.data(), xmin, ymin, xmax, ymax);
   if (!n)
      return;

   auto pp = GetPainter();
   if (!pp)
      return;

   pp->OnPad(this);

   // Paint the fill area with hatches
   Int_t fillstyle = pp->GetFillStyle();
   if (fillstyle >= 3100 && fillstyle < 4000)
      PaintFillAreaHatches(nn, x.data(), y.data(), fillstyle);
   else
      pp->DrawFillArea(n, x.data(), y.data());

   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint fill area in CurrentPad NDC coordinates.

void TPad::PaintFillAreaNDC(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   std::vector<Double_t> xw(n), yw(n);
   for (int i=0; i<n; i++) {
      xw[i] = fX1 + x[i]*(fX2 - fX1);
      yw[i] = fY1 + y[i]*(fY2 - fY1);
   }
   PaintFillArea(n, xw.data(), yw.data(), option);
}

////////////////////////////////////////////////////////////////////////////////
/// This function paints hatched fill area according to the FillStyle value
/// The convention for the Hatch is the following:
///
///     `FillStyle = 3ijk`
///
///  -  i (1-9) : specify the space between each hatch
///             1 = minimum  9 = maximum
///             the final spacing is i*GetHatchesSpacing(). The hatches spacing
///             is set by SetHatchesSpacing()
///  -  j (0-9) : specify angle between 0 and 90 degrees
///             * 0 = 0
///             * 1 = 10
///             * 2 = 20
///             * 3 = 30
///             * 4 = 45
///             * 5 = Not drawn
///             * 6 = 60
///             * 7 = 70
///             * 8 = 80
///             * 9 = 90
///  -  k (0-9) : specify angle between 90 and 180 degrees
///             * 0 = 180
///             * 1 = 170
///             * 2 = 160
///             * 3 = 150
///             * 4 = 135
///             * 5 = Not drawn
///             * 6 = 120
///             * 7 = 110
///             * 8 = 100
///             * 9 = 90

void TPad::PaintFillAreaHatches(Int_t nn, Double_t *xx, Double_t *yy, Int_t FillStyle)
{
   static Double_t ang1[10] = {  0., 10., 20., 30., 45.,5., 60., 70., 80., 89.99};
   static Double_t ang2[10] = {180.,170.,160.,150.,135.,5.,120.,110.,100., 89.99};

   Int_t fasi  = FillStyle % 1000;
   Int_t idSPA = fasi / 100;
   Int_t iAng2 = (fasi - 100 * idSPA) / 10;
   Int_t iAng1 = fasi % 10;
   Double_t dy = 0.003 * idSPA * gStyle->GetHatchesSpacing();
   Short_t lw = gStyle->GetHatchesLineWidth();
   auto pp = GetPainter();
   if (!pp)
      return;

   // Save the current line attributes and change to draw hatches
   TAttLine saveatt = pp->GetAttLine();

   pp->SetAttLine({ pp->GetAttFill().GetFillColor(), 1, lw });

   // Draw the hatches
   if (ang1[iAng1] != 5.)
      PaintHatches(dy, ang1[iAng1], nn, xx, yy);
   if (ang2[iAng2] != 5.)
      PaintHatches(dy, ang2[iAng2], nn, xx, yy);

   pp->SetAttLine(saveatt);
}

////////////////////////////////////////////////////////////////////////////////
/// This routine draw hatches inclined with the
/// angle "angle" and spaced of "dy" in normalized device
/// coordinates in the surface defined by n,xx,yy.

void TPad::PaintHatches(Double_t dy, Double_t angle,
                        Int_t nn, Double_t *xx, Double_t *yy)
{
   Int_t i, i1, i2, nbi;
   Double_t ratiox, ratioy, ymin, ymax, yrot, ycur;
   const Double_t angr  = TMath::Pi()*(180.-angle)/180.;
   const Double_t epsil = 0.0001;

   std::vector<Double_t> xli;
   std::vector<Double_t> yli;

   Double_t xt1, xt2, yt1, yt2;
   Double_t x, y, x1, x2, y1, y2, a, b, xi, xip, xin, yi, yip;

   Double_t rwxmin = gPad->GetX1();
   Double_t rwxmax = gPad->GetX2();
   Double_t rwymin = gPad->GetY1();
   Double_t rwymax = gPad->GetY2();
   ratiox = 1./(rwxmax-rwxmin);
   ratioy = 1./(rwymax-rwymin);

   Double_t sina = TMath::Sin(angr), sinb;
   Double_t cosa = TMath::Cos(angr), cosb;
   if (TMath::Abs(cosa) <= epsil) cosa=0.;
   if (TMath::Abs(sina) <= epsil) sina=0.;
   sinb = -sina;
   cosb = cosa;

   // Values needed to compute the hatches in TRUE normalized space (NDC)
   Int_t iw = (Int_t)gPad->GetWw();
   Int_t ih = (Int_t)gPad->GetWh();
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

   Int_t yindx = (Int_t) (ymax/dy);

   while (dy * yindx >= ymin) {
      ycur = dy * yindx--;
      nbi = 0;

      xli.clear();
      yli.clear();

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
               xli.push_back(xt1);
            }
            continue;
         }

         // Line segment parallel to ox
         if (yt1 == yt2) {
            if (yt1 == ycur) {
               nbi++;
               xli.push_back(xt1);
               nbi++;
               xli.push_back(xt2);
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

         if ((xi <= xin) && (xin < xip) &&
             (TMath::Min(yt1,yt2) <= ycur) &&
             (ycur < TMath::Max(yt1,yt2))) {
            nbi++;
            xli.push_back(xin);
         }
      }

      // Sorting of the x coordinates intersections
      std::sort(xli.begin(), xli.end());

      // Draw the hatches
      if ((nbi%2 != 0) || (nbi == 0))
         continue;

      for (i=0; i<nbi; ++i) {
         // Rotate back the hatches - first calculate y coordinate
         Double_t ytmp = sinb*xli[i] + cosb*ycur;
         xli[i] = cosb*xli[i] - sinb*ycur;
         // Convert hatches' positions from true NDC to WC to handle cliping
         xli[i] = (xli[i]/wndc)*(rwxmax-rwxmin)+rwxmin;
         ytmp   = (ytmp/hndc)*(rwymax-rwymin)+rwymin;
         yli.push_back(ytmp);
      }

      gPad->PaintSegments(nbi/2, xli.data(), yli.data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint line in CurrentPad World coordinates.

void TPad::PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   Double_t x[2] = {x1, x2}, y[2] = {y1, y2};

   //If line is totally clipped, return
   if (TestBit(TGraph::kClipFrame)) {
      if (Clip(x,y,fUxmin,fUymin,fUxmax,fUymax) == 2) return;
   } else {
      if (Clip(x,y,fX1,fY1,fX2,fY2) == 2) return;
   }

   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawLine(x[0], y[0], x[1], y[1]);
   }

   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint line in normalized coordinates.

void TPad::PaintLineNDC(Double_t u1, Double_t v1,Double_t u2, Double_t v2)
{
   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawLineNDC(u1, v1, u2, v2);
   }

   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint 3-D line in the CurrentPad.

void TPad::PaintLine3D(Float_t *p1, Float_t *p2)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Paint 3-D line in the CurrentPad.

void TPad::PaintLine3D(Double_t *p1, Double_t *p2)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Paint 3-D marker in the CurrentPad.

void TPad::PaintMarker3D(Double_t x, Double_t y, Double_t z)
{
   if (!fView) return;

   Double_t rmin[3], rmax[3];
   fView->GetRange(rmin, rmax);

   // convert from 3-D to 2-D pad coordinate system
   Double_t xpad[3];
   Double_t temp[3];
   temp[0] = x;
   temp[1] = y;
   temp[2] = z;
   if (x<rmin[0] || x>rmax[0]) return;
   if (y<rmin[1] || y>rmax[1]) return;
   if (z<rmin[2] || z>rmax[2]) return;
   fView->WCtoNDC(temp, &xpad[0]);
   PaintPolyMarker(1, &xpad[0], &xpad[1]);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in CurrentPad World coordinates.

void TPad::PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *)
{
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
      if (i1 < 0)
         i1 = i;
      if (iclip == 0 && i < n-2)
         continue;
      if (auto pp = GetPainter()) {
         pp->OnPad(this);
         pp->DrawPolyLine(np, &x[i1], &y[i1]);
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

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in CurrentPad World coordinates.
///
///  If option[0] == 'C' no clipping

void TPad::PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   if (n < 2) return;

   Double_t xmin,xmax,ymin,ymax;
   Bool_t mustClip = kTRUE;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
      if (option && (option[0] == 'C')) mustClip = kFALSE;
   }

   Int_t i, i1=-1, np = 1, iclip = 0;

   for (i=0; i < n-1; i++) {
      Double_t x1 = x[i];
      Double_t y1 = y[i];
      Double_t x2 = x[i+1];
      Double_t y2 = y[i+1];
      if (mustClip) {
         iclip = Clip(&x[i],&y[i],xmin,ymin,xmax,ymax);
         if (iclip == 2) {
            i1 = -1;
            continue;
         }
      }
      np++;
      if (i1 < 0)
         i1 = i;
      if (iclip == 0 && i < n-2)
         continue;
      if (auto pp = GetPainter()) {
         pp->OnPad(this);
         pp->DrawPolyLine(np, &x[i1], &y[i1]);
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

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in CurrentPad NDC coordinates.

void TPad::PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *)
{
   if (n <= 0)
      return;

   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawPolyLineNDC(n, x, y);
   }

   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint 3-D polyline in the CurrentPad.

void TPad::PaintPolyLine3D(Int_t n, Double_t *p)
{
   if (!fView) return;

   // Loop on each individual line
   for (Int_t i = 1; i < n; i++)
      PaintLine3D(&p[3*i-3], &p[3*i]);

   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker in CurrentPad World coordinates.

void TPad::PaintPolyMarker(Int_t nn, Float_t *x, Float_t *y, Option_t *)
{
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
      if (np == 0)
         continue;
      if (auto pp = GetPainter()) {
         pp->OnPad(this);
         pp->DrawPolyMarker(np, &x[i1], &y[i1]);
      }
      i1 = -1;
      np = 0;
   }
   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker in CurrentPad World coordinates.

void TPad::PaintPolyMarker(Int_t nn, Double_t *x, Double_t *y, Option_t *)
{
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
      if (np == 0)
         continue;
      if (auto pp = GetPainter()) {
         pp->OnPad(this);
         pp->DrawPolyMarker(np, &x[i1], &y[i1]);
      }
      i1 = -1;
      np = 0;
   }
   Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Paint N individual segments
/// Provided arrays should have 2*n elements
/// IMPORTANT! Provided arrays can be modified after function call!

void TPad::PaintSegments(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   if (n < 1)
      return;

   Double_t xmin,xmax,ymin,ymax;
   Bool_t mustClip = kTRUE, isAny = kFALSE;
   if (TestBit(TGraph::kClipFrame)) {
      xmin = fUxmin; ymin = fUymin; xmax = fUxmax; ymax = fUymax;
   } else {
      xmin = fX1; ymin = fY1; xmax = fX2; ymax = fY2;
      if (option && *option == 'C') mustClip = kFALSE;
   }

   if (!mustClip)
      isAny = kTRUE;
   else {
      for (Int_t i = 0; i < 2*n; i+=2) {
         Int_t iclip = Clip(&x[i],&y[i],xmin,ymin,xmax,ymax);
         if (iclip == 2)
            x[i] = y[i] = x[i+1] = y[i+1] = 0;
         else
            isAny = kTRUE;
      }
   }

   if (isAny)
      if (auto pp = GetPainter()) {
         pp->OnPad(this);
         pp->DrawSegments(n, x, y);
      }

   Modified();
}


////////////////////////////////////////////////////////////////////////////////
/// Paint N individual segments in NDC coordinates
/// Provided arrays should have 2*n elements
/// IMPORTANT! Provided arrays can be modified after function call!

void TPad::PaintSegmentsNDC(Int_t n, Double_t *u, Double_t *v)
{
   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawSegmentsNDC(n, u, v);
   }

   Modified();
}


////////////////////////////////////////////////////////////////////////////////
/// Paint text in CurrentPad World coordinates.

void TPad::PaintText(Double_t x, Double_t y, const char *text)
{
   Modified();

   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawText(x, y, text, TVirtualPadPainter::kClear);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text in CurrentPad World coordinates.

void TPad::PaintText(Double_t x, Double_t y, const wchar_t *text)
{
   Modified();

   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawText(x, y, text, TVirtualPadPainter::kClear);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text with URL in CurrentPad World coordinates.

void TPad::PaintTextUrl(Double_t x, Double_t y, const char *text, const char *url)
{
   Modified();

   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawTextUrl(x, y, text, url);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text in CurrentPad NDC coordinates.

void TPad::PaintTextNDC(Double_t u, Double_t v, const char *text)
{
   Modified();

   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawTextNDC(u, v, text, TVirtualPadPainter::kClear);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text in CurrentPad NDC coordinates.

void TPad::PaintTextNDC(Double_t u, Double_t v, const wchar_t *text)
{
   Modified();

   if (auto pp = GetPainter()) {
      pp->OnPad(this);
      pp->DrawTextNDC(u, v, text, TVirtualPadPainter::kClear);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Search for an object at pixel position px,py.
///
///  Check if point is in this pad.
///
///  If yes, check if it is in one of the sub-pads
///
///  If found in the pad, compute closest distance of approach
///  to each primitive.
///
///  If one distance of approach is found to be within the limit Distancemaximum
///  the corresponding primitive is selected and the routine returns.

TPad *TPad::Pick(Int_t px, Int_t py, TObjLink *&pickobj)
{
   //the two following statements are necessary under NT (multithreaded)
   //when a TCanvas object is being created and a thread calling TPad::Pick
   //before the TPad constructor has completed in the other thread
   if (!gPad) return nullptr; //Andy Haas
   if (!GetListOfPrimitives()) return nullptr; //Andy Haas

   Int_t dist;
   // Search if point is in pad itself
   Double_t x = AbsPixeltoX(px);
   Double_t y = AbsPixeltoY(py);
   if (this != gPad->GetCanvas()) {
      if (!((x >= fX1 && x <= fX2) && (y >= fY1 && y <= fY2))) return nullptr;
   }

   // search for a primitive in this pad or its sub-pads
   static TObjOptLink dummyLink(nullptr,"");  //place holder for when no link available

   TContext ctxt(this, kFALSE); // since no drawing will be done, don't use cd() for efficiency reasons

   TPad *pick   = nullptr;
   TPad *picked = this;
   pickobj      = nullptr;
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
   //In case canvas prefers gl, fView existence
   //automatically means viewer3d existence. (?)

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
      if (!button->IsEditable()) pickobj = nullptr;
   }

   if (TestBit(kCannotPick)) {

      if (picked == this) {
         // cannot pick pad itself!
         picked = nullptr;
      }

   }

   return picked;
}

////////////////////////////////////////////////////////////////////////////////
/// Pop pad to the top of the stack.

void TPad::Pop()
{
   if (!fMother || ROOT::Detail::HasBeenDeleted(fMother) || !fMother->GetListOfPrimitives())
      return;
   if (!fPrimitives)
      fPrimitives = new TList;
   if (this == fMother->GetListOfPrimitives()->Last())
      return;

   TListIter next(fMother->GetListOfPrimitives());
   while (auto obj = next())
      if (obj == this) {
         TString opt = next.GetOption();
         fMother->Remove(this, kFALSE); // do not issue modified
         fMother->Add(this, opt.Data());
         return;
      }
}

////////////////////////////////////////////////////////////////////////////////
/// This method is equivalent to `SaveAs("filename")`. See TPad::SaveAs for details.

void TPad::Print(const char *filename) const
{
   ((TPad*)this)->SaveAs(filename);
}

////////////////////////////////////////////////////////////////////////////////
/// Auxiliary function. Returns kTRUE if list contains an object inherited
/// from TImage

static Bool_t ContainsTImage(TList *li)
{
   TIter next(li);

   while (auto obj = next()) {
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

////////////////////////////////////////////////////////////////////////////////
/// Save Canvas contents in a file in one of various formats.
///
/// \anchor TPadPrint
/// option can be:
///
///  -         `ps`:  a Postscript file is produced (default). [See special cases](\ref TPadPrintPS).
///    -   `Portrait`:  Postscript file is produced (Portrait)
///    -  `Landscape`:  Postscript file is produced (Landscape)
///  -        `eps`:  an Encapsulated Postscript file is produced
///    -    `Preview`:  an [Encapsulated Postscript file with preview](\ref TPadPrintPreview) is produced.
///  -        `pdf`:  a PDF file is produced NOTE: TMathText will be converted to TLatex; q.e.d., symbols only available in TMathText will not render properly.
///    -     `Title:`:  The character string after `Title:` becomes a table
///                    of content entry (for PDF files).
///    - `EmbedFonts`:  a [PDF file with embedded fonts](\ref TPadPrintEmbedFonts) is generated.
///  -        `svg`:  a SVG file is produced
///  -        `tex`:  a TeX file is produced
///    - `Standalone`:  a [standalone TeX file](\ref TPadPrintStandalone) is produced.
///  -        `gif`:  a GIF file is produced
///  -     `gif+NN`:  an animated GIF file is produced, where NN is delay in 10ms units NOTE: See other variants for looping animation in TASImage::WriteImage
///  -        `xpm`:  a XPM file is produced
///  -        `png`:  a PNG file is produced
///  -        `jpg`:  a JPEG file is produced. NOTE: JPEG's lossy compression will make all sharp edges fuzzy.
///  -       `tiff`:  a TIFF file is produced
///  -        `cxx`:  a C++ macro file is produced
///  -        `xml`:  a XML file
///  -       `json`:  a JSON file
///  -       `root`:  a ROOT binary file
///
///     `filename` = 0 - filename  is defined by `GetName` and its
///                      extension is defined with the option
///
/// When Postscript output is selected (`ps`, `eps`), the canvas is saved
/// to `filename.ps` or `filename.eps`. The aspect ratio of the canvas is preserved
/// on the Postscript file. When the "ps" option is selected, the Postscript
/// page will be landscape format if the canvas is in landscape format, otherwise
/// portrait format is selected.
///
/// The physical size of the Postscript page is the one selected in the
/// current style. This size can be modified via TStyle::SetPaperSize.
///
///   Examples:
/// ~~~ {.cpp}
///      gStyle->SetPaperSize(TStyle::kA4);  //default
///      gStyle->SetPaperSize(TStyle::kUSLetter);
/// ~~~
/// where TStyle::kA4 and TStyle::kUSLetter are defined in the enum
/// EPaperSize in TStyle.h
///
/// An alternative is to call:
/// ~~~ {.cpp}
///        gStyle->SetPaperSize(20,26);  same as kA4
/// or     gStyle->SetPaperSize(20,24);  same as kUSLetter
/// ~~~
///   The above numbers take into account some margins and are in centimeters.
///
/// \anchor TPadPrintPreview
/// ### The "Preview" option
///
/// The "Preview" option allows to generate a preview (in the TIFF format) within
/// the Encapsulated Postscript file. This preview can be used by programs like
/// MSWord to visualize the picture on screen. The "Preview" option relies on the
/// ["epstool" command](http://www.cs.wisc.edu/~ghost/gsview/epstool.htm).
///
/// Example:
/// ~~~ {.cpp}
///     canvas->Print("example.eps","Preview");
/// ~~~
///
/// \anchor TPadPrintEmbedFonts
/// ### The "EmbedFonts" option
///
/// The "EmbedFonts" option allows to embed the fonts used in a PDF file inside
/// that file. This option relies on the ["gs" command](https://ghostscript.com).
///
/// Example:
/// ~~~ {.cpp}
///     canvas->Print("example.pdf","EmbedFonts");
/// ~~~
///
/// \anchor TPadPrintStandalone
/// ### The "Standalone" option
/// The "Standalone" option allows to generate a TeX file ready to be processed by
/// tools like `pdflatex`.
///
/// Example:
/// ~~~ {.cpp}
///     canvas->Print("example.tex","Standalone");
/// ~~~
///
/// \anchor TPadPrintPS
/// ### Writing several canvases to the same Postscript or PDF file:
///
///  - if the Postscript or PDF file name finishes with "(", the file is not closed
///  - if the Postscript or PDF file name finishes with ")" and the file has been opened
///    with "(", the file is closed.
///
/// Example:
/// ~~~ {.cpp}
/// {
///    TCanvas c1("c1");
///    h1.Draw();
///    c1.Print("c1.ps("); //write canvas and keep the ps file open
///    h2.Draw();
///    c1.Print("c1.ps"); canvas is added to "c1.ps"
///    h3.Draw();
///    c1.Print("c1.ps)"); canvas is added to "c1.ps" and ps file is closed
/// }
/// ~~~
/// In the previous example replacing "ps" by "pdf" will create a multi-pages PDF file.
///
/// Note that the following sequence writes the canvas to "c1.ps" and closes the ps file.:
/// ~~~ {.cpp}
///    TCanvas c1("c1");
///    h1.Draw();
///    c1.Print("c1.ps");
/// ~~~
///  The `TCanvas::Print("file.ps(")` mechanism is very useful, but it can be
///  a little inconvenient to have the action of opening/closing a file
///  being atomic with printing a page. Particularly if pages are being
///  generated in some loop one needs to detect the special cases of first
///  and last page and then munge the argument to Print() accordingly.
///
///  The "[" and "]" can be used instead of "(" and ")" to open / close without
///  actual printing.
///
/// Example:
/// ~~~ {.cpp}
///    c1.Print("file.ps[");   // No actual print, just open file.ps
///    for (int i=0; i<10; ++i) {
///      // fill canvas for context i
///      // ...
///
///      c1.Print("file.ps");  // actually print canvas to file
///    }// end loop
///    c1.Print("file.ps]");   // No actual print, just close.
/// ~~~
/// As before, the same macro is valid for PDF files.
///
/// It is possible to print a canvas into an animated GIF file by specifying the
/// file name as "myfile.gif+" or "myfile.gif+NN", where NN*10ms is delay
/// between the subimages' display. If NN is omitted the delay between
/// subimages is zero. Each picture is added in the animation thanks to a loop
/// similar to the following one:
/// ~~~ {.cpp}
///    for (int i=0; i<10; ++i) {
///      // fill canvas for context i
///      // ...
///
///      c1.Print("file.gif+5");  // print canvas to GIF file with 50ms delays
///    }// end loop
/// ~~~
/// The delay between each frame must be specified in each Print() statement.
/// If the file "myfile.gif" already exists, the new frame are appended at
/// the end of the file. To avoid this, delete it first with `gSystem->Unlink(myfile.gif);`
/// If you want the gif file to repeat or loop forever, check TASImage::WriteImage documentation

void TPad::Print(const char *filename, Option_t *option)
{
   if (!GetCanvas())
      return;

   TString psname, fs1 = filename;

   // "[" and "]" are special characters for ExpandPathName. When they are at the end
   // of the file name (see help) they must be removed before doing ExpandPathName.
   if (fs1.EndsWith("[")) {
      fs1.Replace((fs1.Length()-1),1," ");
      gSystem->ExpandPathName(fs1);
      fs1.Replace((fs1.Length()-1),1,"[");
   } else if (fs1.EndsWith("]")) {
      fs1.Replace((fs1.Length()-1),1," ");
      gSystem->ExpandPathName(fs1);
      fs1.Replace((fs1.Length()-1),1,"]");
   } else {
      gSystem->ExpandPathName(fs1);
   }

   // Set the default option as "Postscript" (Should be a data member of TPad)
   const char *opt_default = "ps";

   TString opt = !option ? opt_default : option;
   Bool_t image = kFALSE;

   Bool_t title = kFALSE;
   if (strstr(opt,"Title:")) title = kTRUE;

   if (!fs1.Length())  {
      psname = GetName();
      psname += opt;
   } else {
      psname = fs1;
   }

   // lines below protected against case like c1->SaveAs( "../ps/cs.ps" );
   if (psname.BeginsWith('.') && (psname.Contains('/') == 0)) {
      psname = GetName();
      psname.Append(fs1);
      psname.Prepend("/");
      psname.Prepend(gEnv->GetValue("Canvas.PrintDirectory","."));
   }

   // Save pad/canvas in alternative formats
   TImage::EImageFileTypes gtype = TImage::kUnknown;
   if (!title && strstr(opt, "gif+")) {
      gtype = TImage::kAnimGif;
      image = kTRUE;
   } else if (!title && strstr(opt, "gif")) {
      gtype = TImage::kGif;
      image = kTRUE;
   } else if (!title && strstr(opt, "png")) {
      gtype = TImage::kPng;
      image = kTRUE;
   } else if (!title && strstr(opt, "jpg")) {
      gtype = TImage::kJpeg;
      image = kTRUE;
   } else if (!title && strstr(opt, "tiff")) {
      gtype = TImage::kTiff;
      image = kTRUE;
   } else if (!title && strstr(opt, "xpm")) {
      gtype = TImage::kXpm;
      image = kTRUE;
   } else if (!title && strstr(opt, "bmp")) {
      gtype = TImage::kBmp;
      image = kTRUE;
   }

   if (GetCanvas()->IsWeb() && GetPainter() &&
       (strstr(opt,"svg") || strstr(opt,"pdf") || (gtype == TImage::kJpeg) || (gtype == TImage::kPng))) {
      GetPainter()->SaveImage(this, psname.Data(), gtype);
      return;
   }

   if (!GetCanvas()->IsBatch() && GetPainter())
      GetPainter()->SelectDrawable(GetCanvasID());


   if (!gROOT->IsBatch() && image) {
      if ((gtype == TImage::kGif) && !ContainsTImage(fPrimitives)) {
         Int_t wid = (this == GetCanvas()) ? GetCanvas()->GetCanvasID() : GetPixmapID();
         Color_t hc = gPad->GetCanvas()->GetHighLightColor();
         gPad->GetCanvas()->SetHighLightColor(-1);
         gPad->Modified();
         gPad->Update();
         if (GetPainter()) {
            GetPainter()->SelectDrawable(wid);
            GetPainter()->SaveImage(this, psname.Data(), gtype);
         }
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
         gPad->GetCanvasImp()->UpdateDisplay(1, kTRUE);
         if (GetPainter())
            GetPainter()->SaveImage(this, psname, gtype);
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
   if (!title && strstr(opt,"cxx")) {
      GetCanvas()->SaveSource(psname, "");
      return;
   }

   //==============Save pad/canvas as a root file===============================
   if (!title && strstr(opt,"root")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,psname.Data(),"");
      return;
   }

   //==============Save pad/canvas as a XML file================================
   if (!title && strstr(opt,"xml")) {
      // Plugin XML driver
      if (gDirectory) gDirectory->SaveObjectAs(this,psname.Data(),"");
      return;
   }

   //==============Save pad/canvas as a JSON file================================
   if (!title && strstr(opt,"json")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,psname.Data(),"");
      return;
   }

   //==============Save pad/canvas as a SVG file================================
   if (!title && strstr(opt,"svg")) {
      gVirtualPS = (TVirtualPS*)gROOT->GetListOfSpecials()->FindObject(psname);

      Bool_t noScreen = kFALSE;
      if (!GetCanvas()->IsBatch() && GetCanvas()->GetCanvasID() == -1) {
         noScreen = kTRUE;
         GetCanvas()->SetBatch(kTRUE);
      }

      TContext ctxt(this, kTRUE);

      if (!gVirtualPS) {
         // Plugin Postscript/SVG driver
         if (auto h = gROOT->GetPluginManager()->FindHandler("TVirtualPS", "svg")) {
            if (h->LoadPlugin() == -1)
               return;
            h->ExecPlugin(0);
         }
      }

      // Create a new SVG file
      if (gVirtualPS) {
         gVirtualPS->SetName(psname);
         gVirtualPS->Open(psname);
         gVirtualPS->SetBit(kPrintingPS);
         gVirtualPS->NewPage();
      }

      Paint();
      if (noScreen)
         GetCanvas()->SetBatch(kFALSE);

      if (!gSystem->AccessPathName(psname))
         Info("Print", "SVG file %s has been created", psname.Data());

      delete gVirtualPS;
      gVirtualPS = nullptr;

      return;
   }

   //==============Save pad/canvas as a TeX file================================
   if (!title && (strstr(opt,"tex") || strstr(opt,"Standalone"))) {
      gVirtualPS = (TVirtualPS*)gROOT->GetListOfSpecials()->FindObject(psname);

      Bool_t noScreen = kFALSE;
      if (!GetCanvas()->IsBatch() && GetCanvas()->GetCanvasID() == -1) {
         noScreen = kTRUE;
         GetCanvas()->SetBatch(kTRUE);
      }

      TContext ctxt(this, kTRUE);

      if (!gVirtualPS) {
         // Plugin Postscript/SVG driver
         if (auto h = gROOT->GetPluginManager()->FindHandler("TVirtualPS", "tex")) {
            if (h->LoadPlugin() == -1)
               return;
            h->ExecPlugin(0);
         }
      }

      Bool_t standalone = kFALSE;
      if (strstr(opt,"Standalone")) standalone = kTRUE;

      // Create a new TeX file
      if (gVirtualPS) {
         gVirtualPS->SetName(psname);
         if (standalone) gVirtualPS->SetTitle("Standalone");
         gVirtualPS->Open(psname);
         gVirtualPS->SetBit(kPrintingPS);
         gVirtualPS->NewPage();
      }
      Paint();
      if (noScreen)  GetCanvas()->SetBatch(kFALSE);

      if (!gSystem->AccessPathName(psname)) {
         if (standalone) {
            Info("Print", "Standalone TeX file %s has been created", psname.Data());
         } else{
            Info("Print", "TeX file %s has been created", psname.Data());
         }
      }

      delete gVirtualPS;
      gVirtualPS = nullptr;

      return;
   }

   //==============Save pad/canvas as a Postscript file=========================

   // in case we read directly from a Root file and the canvas
   // is not on the screen, set batch mode

   Bool_t mustOpen  = kTRUE, mustClose = kTRUE,
          copen = kFALSE, cclose = kFALSE, copenb = kFALSE, ccloseb = kFALSE;
   if (!image) {
      // The parenthesis mechanism is only valid for PS and PDF files.
      copen   = psname.EndsWith("("); if (copen)   psname[psname.Length()-1] = 0;
      cclose  = psname.EndsWith(")"); if (cclose)  psname[psname.Length()-1] = 0;
      copenb  = psname.EndsWith("["); if (copenb)  psname[psname.Length()-1] = 0;
      ccloseb = psname.EndsWith("]"); if (ccloseb) psname[psname.Length()-1] = 0;
   }
   gVirtualPS = (TVirtualPS*)gROOT->GetListOfSpecials()->FindObject(psname);
   if (gVirtualPS) mustOpen = mustClose = kFALSE;
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

   TContext ctxt(this, kTRUE);
   TVirtualPS *psave = gVirtualPS;

   if (!gVirtualPS || mustOpen) {

      const char *pluginName = "ps"; // Plugin Postscript driver
      if (strstr(opt,"pdf") || title || strstr(opt,"EmbedFonts"))
         pluginName = "pdf";
      else if (image)
         pluginName = "image"; // Plugin TImageDump driver

      if (auto h = gROOT->GetPluginManager()->FindHandler("TVirtualPS", pluginName)) {
         if (h->LoadPlugin() == -1)
            return;
          h->ExecPlugin(0);
      }

      // Create a new Postscript, PDF or image file
      if (gVirtualPS)
         gVirtualPS->SetName(psname);
      const Ssiz_t titlePos = opt.Index("Title:");
      if (titlePos != kNPOS) {
         if (gVirtualPS)
            gVirtualPS->SetTitle(opt.Data()+titlePos+6);
         opt.Replace(titlePos,opt.Length(),"pdf");
      }
      if (gVirtualPS)
         gVirtualPS->Open(psname,pstype);
      if (gVirtualPS)
         gVirtualPS->SetBit(kPrintingPS);
      if (!copenb) {
         if (!strstr(opt,"pdf") || image) {
            if (gVirtualPS) gVirtualPS->NewPage();
         }
         Paint();
      }
      if (noScreen) GetCanvas()->SetBatch(kFALSE);

      if (mustClose) {
         gROOT->GetListOfSpecials()->Remove(gVirtualPS);
         delete gVirtualPS;
         gVirtualPS = psave;
      } else {
         gROOT->GetListOfSpecials()->Add(gVirtualPS);
         gVirtualPS = nullptr;
      }

      if (!gSystem->AccessPathName(psname)) {
         if (!copen) Info("Print", "%s file %s has been created", opt.Data(), psname.Data());
         else        Info("Print", "%s file %s has been created using the current canvas", opt.Data(), psname.Data());
      }
   } else {
      // Append to existing Postscript, PDF or GIF file
      if (!ccloseb) {
         gVirtualPS->NewPage();
         Paint();
      }
      const Ssiz_t titlePos = opt.Index("Title:");
      if (titlePos != kNPOS) {
         gVirtualPS->SetTitle(opt.Data()+titlePos+6);
         opt.Replace(titlePos,opt.Length(),"pdf");
      } else if (!ccloseb) {
         gVirtualPS->SetTitle("PDF");
      }
      if (mustClose) {
         if (cclose) Info("Print", "Current canvas added to %s file %s and file closed", opt.Data(), psname.Data());
         else        Info("Print", "%s file %s has been closed", opt.Data(), psname.Data());
         gROOT->GetListOfSpecials()->Remove(gVirtualPS);
         delete gVirtualPS;
         gVirtualPS = nullptr;
      } else {
         Info("Print", "Current canvas added to %s file %s", opt.Data(), psname.Data());
         gVirtualPS = nullptr;
      }
   }

   if (strstr(opt,"Preview"))
      gSystem->Exec(TString::Format("epstool --quiet -t6p %s %s", psname.Data(), psname.Data()).Data());
   if (strstr(opt,"EmbedFonts")) {
      gSystem->Exec(TString::Format("gs -quiet -dSAFER -dNOPLATFONTS -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dCompatibilityLevel=1.4 -dMaxSubsetPct=100 -dSubsetFonts=true -dEmbedAllFonts=true -sOutputFile=pdf_temp.pdf -f %s",
                          psname.Data()).Data());
      gSystem->Rename("pdf_temp.pdf", psname.Data());
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Set world coordinate system for the pad.
/// Emits signal "RangeChanged()", in the slot get the range
/// via GetRange().

void TPad::Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
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

   if (gPad == this && GetPainter())
      GetPainter()->InvalidateCS();

   // emit signal
   RangeChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Set axis coordinate system for the pad.
/// The axis coordinate system is a subset of the world coordinate system
/// xmin,ymin is the origin of the current coordinate system,
/// xmax is the end of the X axis, ymax is the end of the Y axis.
/// By default a margin of 10 per cent is left on all sides of the pad
/// Emits signal "RangeAxisChanged()", in the slot get the axis range
/// via GetRangeAxis().

void TPad::RangeAxis(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Recursively remove object from a pad and its sub-pads.

void TPad::RecursiveRemove(TObject *obj)
{
   if (fCanvas) {
     if (obj == fCanvas->GetSelected()) fCanvas->SetSelected(nullptr);
     if (obj == fCanvas->GetClickSelected()) fCanvas->SetClickSelected(nullptr);
   }
   if (obj == fView) fView = nullptr;
   if (!fPrimitives) return;
   Int_t nold = fPrimitives->GetSize();
   fPrimitives->RecursiveRemove(obj);
   if (nold != fPrimitives->GetSize()) fModified = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from list of primitives
/// When \par modified set to kTRUE (default) pad will be marked as modified - if object really removed
/// Returns result of GetListOfPrimitives()->Remove(obj) or nullptr if list of primitives not exists

TObject *TPad::Remove(TObject *obj, Bool_t modified)
{
   TObject *res = nullptr;
   if (fPrimitives)
      res = fPrimitives->Remove(obj);
   if (res && modified)
      Modified();
   return res;
}

////////////////////////////////////////////////////////////////////////////////
///  Redraw the frame axis.
///
///  Redrawing axis may be necessary in case of superimposed histograms
///  when one or more histograms have a fill color.
///
///  Instead of calling this function, it may be more convenient
///  to call directly `h1->Draw("sameaxis")` where h1 is the pointer
///  to the first histogram drawn in the pad.
///
///  By default, if the pad has the options gridx or/and gridy activated,
///  the grid is not drawn by this function.
///
///  If option="g" is specified, this will force the drawing of the grid
///  on top of the picture
///
///  To redraw the axis tick marks do:
/// ~~~ {.cpp}
///   gPad->RedrawAxis();
/// ~~~
///  To redraw the axis grid do:
/// ~~~ {.cpp}
///   gPad->RedrawAxis("G");
/// ~~~
///  To redraw the axis tick marks and the axis grid do:
/// ~~~ {.cpp}
///   gPad->RedrawAxis();
///   gPad->RedrawAxis("G");
/// ~~~
///
///  If option="f" is specified, this will force the drawing of the frame
/// around the plot.

void TPad::RedrawAxis(Option_t *option)
{
   TString opt = option;
   opt.ToLower();

   TContext ctxt(this, kTRUE);

   TH1 *hobj = nullptr;

   // Get the first histogram drawing the axis in the list of primitives
   if (!fPrimitives) fPrimitives = new TList;
   TIter next(fPrimitives);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TH1::Class())) {
         hobj = (TH1*)obj;
         break;
      }
      if (obj->InheritsFrom(TMultiGraph::Class())) {
         TMultiGraph *mg = (TMultiGraph*)obj;
         if (mg) hobj = mg->GetHistogram();
         break;
      }
      if (obj->InheritsFrom(TGraph::Class())) {
         TGraph *g = (TGraph*)obj;
         if (g) hobj = g->GetHistogram();
         break;
      }
      if (obj->InheritsFrom(THStack::Class())) {
         THStack *hs = (THStack*)obj;
         if (hs) hobj = hs->GetHistogram();
         break;
      }
   }

   if (hobj) {
      if (opt.Contains("g")) hobj->DrawCopy("sameaxig");
      else                   hobj->DrawCopy("sameaxis");
   }

   if (opt.Contains("f")) {
      auto b = new TBox(gPad->GetUxmin(), gPad->GetUymin(),
                        gPad->GetUxmax(), gPad->GetUymax());
      b->SetFillStyle(0);
      b->SetLineStyle(gPad->GetFrameLineStyle());
      b->SetLineWidth(gPad->GetFrameLineWidth());
      b->SetLineColor(gPad->GetFrameLineColor());
      b->Draw();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Compute pad conversion coefficients.
///
/// ### Conversion from x to px
///
/// \f[\frac{x-xmin}{xrange} = \frac{px-pxlow}{pxrange}\f]
/// with:
/// \f[ xrange  = xmax-xmin \f]
/// \f[ pxrange = pxmax-pxmin \f]
///
/// \f[
/// \Rightarrow px = \frac{pxrange(x-xmin)}{xrange} + pxlow   = fXtoPixelk + fXtoPixel \times x
/// \f]
///
/// \f[
/// \Rightarrow fXtoPixelk = pxlow - pxrange \frac{xmin}{xrange}
/// \f]
/// \f[
/// fXtoPixel  = \frac{pxrange}{xrange}
/// \f]
/// where:
/// \f[
/// pxlow   = fAbsXlowNDC \times fCw
/// \f]
/// \f[
/// pxrange = fAbsWNDC \times fCw
/// \f]
///
/// ### Conversion from y to py
///
/// \f[\frac{y-ymin}{yrange} = \frac{py-pylow}{pyrange}\f]
/// with:
/// \f[ yrange  = ymax-ymin \f]
/// \f[ pyrange = pymax-pymin \f]
///
/// \f[
/// \Rightarrow py = \frac{pyrange(y-xmin)}{yrange} + pylow   = fYtoPixelk + fYtoPixel \times y
/// \f]
///
/// \f[
/// \Rightarrow fYtoPixelk = pylow - pyrange \frac{ymin}{yrange}
/// \f]
/// \f[
/// fYtoPixel  = \frac{pyrange}{yrange}
/// \f]
/// where:
/// \f[
/// pylow   = fAbsYlowNDC \times fCh
/// \f]
/// \f[
/// pyrange = fAbsHNDC \times fCh
/// \f]
///
/// ### Conversion from px to x
///
/// \f[
/// \Rightarrow  x = \frac{xrange(px-pxlow)}{pxrange}+ xmin  = fPixeltoXk + fPixeltoX \times px
/// \f]
///
/// \f[
/// \Rightarrow fPixeltoXk = xmin - pxlow \times\frac{xrange}{pxrange}
/// \f]
/// \f[
/// fPixeltoX  = \frac{xrange}{pxrange}
/// \f]
///
/// ### Conversion from py to y
///
/// \f[
/// \Rightarrow  y = \frac{yrange(py-pylow)}{pyrange}+ ymin  = fPixeltoYk + fPixeltoY \times py
/// \f]
///
/// \f[
/// \Rightarrow fPixeltoYk = ymin - pylow \times\frac{yrange}{pyrange}
/// \f]
/// \f[
/// fPixeltoY  = \frac{yrange}{pyrange}
/// \f]
///
/// ### Computation of the coefficients in case of LOG scales
///
/// #### Conversion from pixel coordinates to world coordinates
///
/// \f[
///  u = \frac{Log(x) - Log(xmin)}{Log(xmax) - Log(xmin)} = \frac{Log(x/xmin)}{Log(xmax/xmin)}  = \frac{px - pxlow}{pxrange}
/// \f]
///
/// \f[ \Rightarrow Log(\frac{x}{xmin}) = u \times Log(\frac{xmax}{xmin})   \f]
/// \f[ x = xmin \times e^{(u \times Log(\frac{xmax}{xmin})}                \f]
/// Let:
/// \f[ alfa = \frac{Log(\frac{xmax}{xmin})}{fAbsWNDC}                      \f]
///
/// \f[ x = xmin \times e^{(-alfa \times pxlow)} + e^{(alfa \times px)}     \f]
/// \f[ x = fPixeltoXk \times e^{(fPixeltoX \times px)}                     \f]
/// \f[ ==> fPixeltoXk = xmin \times e^{(-alfa*pxlow)}                      \f]
/// \f[ fPixeltoX  = alfa                                                   \f]
///
/// \f[
///  v = \frac{Log(y) - Log(ymin)}{Log(ymax) - Log(ymin)} = \frac{Log(y/ymin)}{Log(ymax/ymin)}  = \frac{py - pylow}{pyrange}
/// \f]
/// Let:
/// \f[ beta = Log(\frac{ymax}{ymin})                                       \f]
/// \f[ Log(\frac{y}{ymin}) = beta \times pylow - beta \times py            \f]
/// \f[ \frac{y}{ymin} = e^{(beta \times pylow - beta \times py)}           \f]
/// \f[ y = ymin \times e^{(beta \times pylow)} \times e^{(-beta \times py)}\f]
/// \f[ \Rightarrow y = fPixeltoYk \times e^{(fPixeltoY \times py)}         \f]
/// \f[ fPixeltoYk = ymin \times e^{(beta \times pylow)}                    \f]
/// \f[ fPixeltoY  = -beta                                                  \f]
///
/// #### Conversion from World coordinates to pixel coordinates
///
/// \f[ px = pxlow + u*pxrange \f]
/// \f[ = pxlow + Log(x/xmin)/alfa \f]
/// \f[ = pxlow -Log(xmin)/alfa  + Log(x)/alfa \f]
/// \f[ = fXtoPixelk + fXtoPixel*Log(x) \f]
/// \f[ \Rightarrow fXtoPixelk = pxlow -Log(xmin)/alfa \f]
/// \f[ \Rightarrow fXtoPixel  = 1/alfa \f]
///
/// \f[ py = pylow - Log(y/ymin)/beta \f]
/// \f[ = fYtoPixelk + fYtoPixel*Log(y) \f]
/// \f[ \Rightarrow fYtoPixelk = pylow - Log(ymin)/beta \f]
/// \f[ fYtoPixel  = 1/beta  \f]

void TPad::ResizePad(Option_t *option)
{

   if (!gPad) {
      Error("ResizePad", "Cannot resize pad. No current pad available.");
      return;
   }
   if (gPad->GetWw()==0.0||gPad->GetWh()==0.0) {
      Warning("ResizePad", "gPad has at least one zero dimension.");
      return;
   }
   if (fX1==fX2||fY1==fY2) {
      Warning("ResizePad", "The pad has at least one zero dimension.");
      return;
   }
   // Recompute subpad positions in case pad has been moved/resized
   TPad *parent = fMother;
   if (this == gPad->GetCanvas()) {
      fAbsXlowNDC  = fXlowNDC;
      fAbsYlowNDC  = fYlowNDC;
      fAbsWNDC     = fWNDC;
      fAbsHNDC     = fHNDC;
   }
   else {
      if (parent->GetAbsHNDC()==0.0||parent->GetAbsWNDC()==0.0||fHNDC==0.0||fWNDC==0.0) {
         Warning("ResizePad", "The parent pad has at least one zero dimension.");
         return;
      }
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
   Double_t rounding = 0.; // was used before to adjust somehow wrong int trunctation by coordiantes transformation
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

   // Resize all sub-pads
   if (!fPrimitives)
      fPrimitives = new TList;
   TIter next(GetListOfPrimitives());
   while (auto obj = next()) {
      if (obj->InheritsFrom(TPad::Class()))
         ((TPad *)obj)->ResizePad(option);
   }

   // Reset all current sizes
   if (gPad->IsBatch())
      fPixmapID = 0;
   else if (auto pp = GetPainter()) {
      if (pp->IsNative()) {
         // TODO: check if this is necessary
         auto attl = pp->GetAttLine();
         attl.SetLineColor(-1);
         pp->SetAttLine(attl);
         auto attt = pp->GetAttText();
         attt.SetTextSize(-1);
         pp->SetAttText(attt);
         // create or re-create off-screen pixmap
         if (fPixmapID) {
            int w = TMath::Abs(XtoPixel(fX2) - XtoPixel(fX1));
            int h = TMath::Abs(YtoPixel(fY2) - YtoPixel(fY1));
            //protection in case of wrong pad parameters.
            //without this protection, the OpenPixmap or ResizePixmap crashes with
            //the message "Error in <RootX11ErrorHandler>: BadValue (integer parameter out of range for operation)"
            //resulting in a frozen xterm
            if (!TMath::Finite(fX1) || !TMath::Finite(fX2) || !TMath::Finite(fY1) || !TMath::Finite(fY2) ||
                TMath::IsNaN(fX1) || TMath::IsNaN(fX2) || TMath::IsNaN(fY1) || TMath::IsNaN(fY2))
               Warning("ResizePad", "Inf/NaN propagated to the pad. Check drawn objects.");
            if (w <= 0 || w > 10000) {
               Warning("ResizePad", "%s width changed from %d to %d\n",GetName(),w,10);
               w = 10;
            }
            if (h <= 0 || h > 10000) {
               Warning("ResizePad", "%s height changed from %d to %d\n",GetName(),h,10);
               h = 10;
            }
            if (fPixmapID == -1)       // this case is handled via the ctor
               fPixmapID = pp->CreateDrawable(w, h);
            else if (pp->ResizeDrawable(fPixmapID, w, h)) {
               Resized();
               Modified(kTRUE);
            }
         }
      }
   }
   if (fView) {
      if (gPad == this) {
         fView->ResizePad();
      } else {
         TContext ctxt(this, kTRUE);
         fView->ResizePad();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save the pad content in a file.
///
/// The file's format used to save the pad  is determined by the `filename` extension:
///
///  - if `filename` is empty, the file produced is `padname.ps`
///  - if `filename` starts with a dot, the padname is added in front
///  - if `filename` ends with `.ps`, a Postscript file is produced
///  - if `filename` ends with `.eps`, an Encapsulated Postscript file is produced
///  - if `filename` ends with `.pdf`, a PDF file is produced NOTE: TMathText will be converted to TLatex; q.e.d., symbols only available in TMathText will not render properly.
///  - if `filename` ends with `.svg`, a SVG file is produced
///  - if `filename` ends with `.tex`, a TeX file is produced
///  - if `filename` ends with `.gif`, a GIF file is produced
///  - if `filename` ends with `.gif+NN`, an  animated GIF file is produced See comments in TASImage::WriteImage for meaning of NN and other .gif sufix variants
///  - if `filename` ends with `.xpm`, a XPM file is produced
///  - if `filename` ends with `.png`, a PNG file is produced
///  - if `filename` ends with `.bmp`, a BMP file is produced
///  - if `filename` ends with `.jpg` or `.jpeg` a JPEG file is produced NOTE: JPEG's lossy compression will make all sharp edges fuzzy.
///  - if `filename` ends with `.tiff`, a TIFF file is produced
///  - if `filename` ends with `.C`, `.cxx`,`.cpp` or `.cc`, a C++ macro file is produced
///  - if `filename` ends with `.root`, a Root file is produced
///  - if `filename` ends with `.xml`, a XML file is produced
///  - if `filename` ends with `.json`, a JSON file is produced
///
/// \remarks
///  - The parameter `option` is not used.
///  - This method calls [TPad::Print(const char *filename, Option_t *option)](\ref TPadPrint)
///    the value of `option` is determined by the `filename` extension.
///  - Postscript and PDF formats allow to have [several pictures in one file](\ref TPadPrintPS).

void TPad::SaveAs(const char *filename, Option_t * /*option*/) const
{
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
   else if (psname.EndsWith(".C") || psname.EndsWith(".cxx") || psname.EndsWith(".cpp") || psname.EndsWith(".cc"))
      ((TPad*)this)->Print(psname,"cxx");
   else if (psname.EndsWith(".root"))
      ((TPad*)this)->Print(psname,"root");
   else if (psname.EndsWith(".xml"))
      ((TPad*)this)->Print(psname,"xml");
   else if (psname.EndsWith(".json"))
      ((TPad*)this)->Print(psname,"json");
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
   else if (psname.EndsWith(".tex"))
      ((TPad*)this)->Print(psname,"tex");
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

////////////////////////////////////////////////////////////////////////////////
/// Save primitives in this pad on the C++ source file out.

void TPad::SavePrimitive(std::ostream &out, Option_t * option /*= ""*/)
{
   TContext ctxt(this, kFALSE); // not interactive

   TString padName = GetName();

   // check for space in the pad name
   auto p = padName.Index(" ");
   if (p != kNPOS)
      padName.Resize(p);

   TString opt = option;
   if (!opt.Contains("toplevel")) {
      static Int_t pcounter = 0;
      padName += TString::Format("__%d", pcounter++);
      padName = gInterpreter->MapCppName(padName);
   }

   const char *pname = padName.Data();
   const char *cname = padName.Data();

   if (padName.Length() == 0) {
      pname = "unnamed";
      if (this == gPad->GetCanvas())
         cname = "c1";
      else
         cname = "pad";
   }

   //   Write pad parameters
   if (this != gPad->GetCanvas()) {
      out << "   \n";
      out << "// ------------>Primitives in pad: " << GetName() << "\n";
      out << "   TPad *" << cname << " = new TPad(\"" << GetName() << "\", \""
          << TString(GetTitle()).ReplaceSpecialCppChars() << "\", " << fXlowNDC << ", " << fYlowNDC << ", "
          << fXlowNDC + fWNDC << ", " << fYlowNDC + fHNDC << ");\n";
      out << "   " << cname << "->Draw();\n";
      out << "   " << cname << "->cd();\n";
   }
   out << "   " << cname << "->Range(" << fX1 << "," << fY1 << "," << fX2 << "," << fY2 << ");\n";
   TView *view = GetView();
   if (view) {
      Double_t rmin[3], rmax[3];
      view->GetRange(rmin, rmax);
      out << "   TView::CreateView(1)->SetRange(" << rmin[0] << ", " << rmin[1] << ", " << rmin[2] << ", " << rmax[0]
          << ", " << rmax[1] << ", " << rmax[2] << ");\n";
   }

   SaveFillAttributes(out, cname, -1, -1);

   if (GetBorderMode() != 1)
      out << "   " << cname << "->SetBorderMode(" << GetBorderMode() << ");\n";
   if (GetBorderSize() != 4)
      out << "   " << cname << "->SetBorderSize(" << GetBorderSize() << ");\n";
   if (GetLogx())
      out << "   " << cname << "->SetLogx();\n";
   if (GetLogy())
      out << "   " << cname << "->SetLogy();\n";
   if (GetLogz())
      out << "   " << cname << "->SetLogz();\n";
   if (GetGridx())
      out << "   " << cname << "->SetGridx();\n";
   if (GetGridy())
      out << "   " << cname << "->SetGridy();\n";
   if (GetTickx())
      out << "   " << cname << "->SetTickx(" << GetTickx() << ");\n";
   if (GetTicky())
      out << "   " << cname << "->SetTicky(" << GetTicky() << ");\n";
   if (GetTheta() != 30)
      out << "   " << cname << "->SetTheta(" << GetTheta() << ");\n";
   if (GetPhi() != 30)
      out << "   " << cname << "->SetPhi(" << GetPhi() << ");\n";
   if (TMath::Abs(fLeftMargin - 0.1) > 0.01)
      out << "   " << cname << "->SetLeftMargin(" << GetLeftMargin() << ");\n";
   if (TMath::Abs(fRightMargin - 0.1) > 0.01)
      out << "   " << cname << "->SetRightMargin(" << GetRightMargin() << ");\n";
   if (TMath::Abs(fTopMargin - 0.1) > 0.01)
      out << "   " << cname << "->SetTopMargin(" << GetTopMargin() << ");\n";
   if (TMath::Abs(fBottomMargin - 0.1) > 0.01)
      out << "   " << cname << "->SetBottomMargin(" << GetBottomMargin() << ");\n";

   if (GetFrameFillColor() != GetFillColor())
      out << "   " << cname << "->SetFrameFillColor(" << TColor::SavePrimitiveColor(GetFrameFillColor()) << ");\n";
   if (GetFrameFillStyle() != 1001)
      out << "   " << cname << "->SetFrameFillStyle(" << GetFrameFillStyle() << ");\n";
   if (GetFrameLineStyle() != 1)
      out << "   " << cname << "->SetFrameLineStyle(" << GetFrameLineStyle() << ");\n";
   if (GetFrameLineColor() != 1)
      out << "   " << cname << "->SetFrameLineColor(" << TColor::SavePrimitiveColor(GetFrameLineColor()) << ");\n";
   if (GetFrameLineWidth() != 1)
      out << "   " << cname << "->SetFrameLineWidth(" << GetFrameLineWidth() << ");\n";
   if (GetFrameBorderMode() != 0)
      out << "   " << cname << "->SetFrameBorderMode(" << GetFrameBorderMode() << ");\n";
   if (GetFrameBorderSize() != 1)
      out << "   " << cname << "->SetFrameBorderSize(" << GetFrameBorderSize() << ");\n";

   TFrame *frame = fFrame;
   if (!frame)
      frame = (TFrame *)GetPrimitive("TFrame");
   if (frame) {
      if (frame->GetFillColor() != GetFillColor())
         out << "   " << cname << "->SetFrameFillColor(" << TColor::SavePrimitiveColor(frame->GetFillColor()) << ");\n";
      if (frame->GetFillStyle() != 1001)
         out << "   " << cname << "->SetFrameFillStyle(" << frame->GetFillStyle() << ");\n";
      if (frame->GetLineStyle() != 1)
         out << "   " << cname << "->SetFrameLineStyle(" << frame->GetLineStyle() << ");\n";
      if (frame->GetLineColor() != 1)
         out << "   " << cname << "->SetFrameLineColor(" << TColor::SavePrimitiveColor(frame->GetLineColor()) << ");\n";
      if (frame->GetLineWidth() != 1)
         out << "   " << cname << "->SetFrameLineWidth(" << frame->GetLineWidth() << ");\n";
      if (frame->GetBorderMode() != 0)
         out << "   " << cname << "->SetFrameBorderMode(" << frame->GetBorderMode() << ");\n";
      if (frame->GetBorderSize() != 1)
         out << "   " << cname << "->SetFrameBorderSize(" << frame->GetBorderSize() << ");\n";
   }

   TIter next(GetListOfPrimitives());

   while (auto obj = next()) {
      obj->SavePrimitive(out, (Option_t *)next.GetOption());
      if (obj->InheritsFrom(TPad::Class())) {
         if (opt.Contains("toplevel"))
            out << "   " << pname << "->cd();\n";
         else
            out << "   " << cname << "->cd();\n";
      }
   }
   out << "   " << cname << "->Modified();\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Fix pad aspect ratio to current value if fixed is true.

void TPad::SetFixedAspectRatio(Bool_t fixed)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set pad editable yes/no
/// If a pad is not editable:
/// - one cannot modify the pad and its objects via the mouse.
/// - one cannot add new objects to the pad

void TPad::SetEditable(Bool_t mode)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Override TAttFill::FillStyle for TPad because we want to handle style=0
/// as style 4000.

void TPad::SetFillStyle(Style_t fstyle)
{
   if (fstyle == 0) fstyle = 4000;
   TAttFill::SetFillStyle(fstyle);
}

////////////////////////////////////////////////////////////////////////////////
/// Set Lin/Log scale for X
///  - value = 0 X scale will be linear
///  - value = 1 X scale will be logarithmic (base 10)
///  - value > 1 reserved for possible support of base e or other

void TPad::SetLogx(Int_t value)
{
   fLogx = value;
   delete fView; fView = nullptr;
   Modified();
   RangeAxisChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Set Lin/Log scale for Y
///  - value = 0 Y scale will be linear
///  - value = 1 Y scale will be logarithmic (base 10)
///  - value > 1 reserved for possible support of base e or other

void TPad::SetLogy(Int_t value)
{
   fLogy = value;
   delete fView; fView = nullptr;
   Modified();
   RangeAxisChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Set Lin/Log scale for Z

void TPad::SetLogz(Int_t value)
{
   fLogz = value;
   delete fView; fView = nullptr;
   Modified();
   RangeAxisChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Set canvas range for pad and resize the pad. If the aspect ratio
/// was fixed before the call it will be un-fixed.

void TPad::SetPad(Double_t xlow, Double_t ylow, Double_t xup, Double_t yup)
{
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

   // Check if the new pad position is valid.
   if ((xlow < 0) || (xlow > 1) || (ylow < 0) || (ylow > 1)) {
      Error("TPad", "illegal bottom left position: x=%f, y=%f", xlow, ylow);
      return;
   }
   if ((xup < 0) || (xup > 1) || (yup < 0) || (yup > 1)) {
      Error("TPad", "illegal top right position: x=%f, y=%f", xup, yup);
      return;
   }
   if (xup-xlow <= 0) {
      Error("TPad", "illegal width: %f", xup-xlow);
      return;
   }
   if (yup-ylow <= 0) {
      Error("TPad", "illegal height: %f", yup-ylow);
      return;
   }

   fXlowNDC = xlow;
   fYlowNDC = ylow;
   fXUpNDC  = xup;
   fYUpNDC  = yup;
   fWNDC    = xup - xlow;
   fHNDC    = yup - ylow;

   SetFixedAspectRatio(kFALSE);

   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Set all pad parameters.

void TPad::SetPad(const char *name, const char *title,
                  Double_t xlow, Double_t ylow, Double_t xup, Double_t yup,
                  Color_t color, Short_t bordersize, Short_t bordermode)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set the current TView. Delete previous view if view=0

void TPad::SetView(TView *view)
{
   if (!view) delete fView;
   fView = view;
}

////////////////////////////////////////////////////////////////////////////////
/// Set postscript fill area attributes.
///
/// DEPRECATED!!! No longer used by ROOT, kept only for backward compatibility

void TPad::SetAttFillPS(Color_t color, Style_t style)
{
   if (auto pp = GetPainter())
      if (pp->GetPS())
         pp->SetAttFill({color, style});
}

////////////////////////////////////////////////////////////////////////////////
/// Set postscript line attributes.
///
/// DEPRECATED!!! No longer used by ROOT, kept only for backward compatibility

void TPad::SetAttLinePS(Color_t color, Style_t style, Width_t lwidth)
{
   if (auto pp = GetPainter())
      if (pp->GetPS())
         pp->SetAttLine({color, style, lwidth});
}

////////////////////////////////////////////////////////////////////////////////
/// Set postscript marker attributes.
///
/// DEPRECATED!!! No longer used by ROOT, kept only for backward compatibility

void TPad::SetAttMarkerPS(Color_t color, Style_t style, Size_t msize)
{
   if (auto pp = GetPainter())
      if (pp->GetPS())
         pp->SetAttMarker({color, style, msize});
}

////////////////////////////////////////////////////////////////////////////////
/// Set postscript text attributes.
///
/// DEPRECATED!!! No longer used by ROOT, kept only for backward compatibility

void TPad::SetAttTextPS(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize)
{
   if (auto pp = GetPainter())
      if (pp->GetPS()) {
         if (font % 10 > 2) {
            Float_t wh = (Float_t) XtoPixel(GetX2());
            Float_t hh = (Float_t) YtoPixel(GetY1());
            Float_t dy;
            if (wh < hh)  {
               dy = AbsPixeltoX(Int_t(tsize)) - AbsPixeltoX(0);
               tsize = dy/(GetX2()-GetX1());
            } else {
               dy = AbsPixeltoY(0) - AbsPixeltoY(Int_t(tsize));
               tsize = dy/(GetY2()-GetY1());
            }
         }
         pp->SetAttText({align, angle, color, font, tsize});
      }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw Arrows to indicated equal distances of Objects with given BBoxes.
/// Used by ShowGuidelines

void TPad::DrawDist(Rectangle_t aBBox, Rectangle_t bBBox, char mode)
{
   Int_t lineColor = TColor::GetColor(239, 202, 0);
   Int_t x1,x2,y1,y2;
   x1 = x2 = y1 = y2 = 0;
   if (mode == 'x') {
      if (aBBox.fX<bBBox.fX) {
         x1 = aBBox.fX+aBBox.fWidth;
         x2 = bBBox.fX;
      }
      else {
         x1 = bBBox.fX+bBBox.fWidth;
         x2 = aBBox.fX;
      }

      if ((aBBox.fY > bBBox.fY) && (aBBox.fY + aBBox.fHeight < bBBox.fY + bBBox.fHeight))
                                    y1 = y2 = aBBox.fY + TMath::Nint(0.5*(Double_t)(aBBox.fHeight))+1;
      else if ((bBBox.fY > aBBox.fY) && (bBBox.fY + bBBox.fHeight < aBBox.fY + aBBox.fHeight))
                                    y1 = y2 = bBBox.fY + TMath::Nint(0.5*(Double_t)(bBBox.fHeight))+1;
      else if (aBBox.fY>bBBox.fY)   y1 = y2 = aBBox.fY-TMath::Nint(0.5*(Double_t)(aBBox.fY-(bBBox.fY+bBBox.fHeight)));
      else                          y1 = y2 = bBBox.fY-TMath::Nint(0.5*(Double_t)(bBBox.fY-(aBBox.fY+aBBox.fHeight)));
   }
   else if (mode == 'y') {
      if (aBBox.fY<bBBox.fY) {
         y1 = aBBox.fY+aBBox.fHeight;
         y2 = bBBox.fY;
      }
      else {
         y1 = bBBox.fY+bBBox.fHeight;
         y2 = aBBox.fY;
      }
      if ((aBBox.fX > bBBox.fX) && (aBBox.fX + aBBox.fWidth < bBBox.fX + bBBox.fWidth))
                                    x1 = x2 = aBBox.fX + TMath::Nint(0.5*(Double_t)(aBBox.fWidth))+1;
      else if ((bBBox.fX > aBBox.fX) && (bBBox.fX + bBBox.fWidth < aBBox.fX + aBBox.fWidth))
                                    x1 = x2 = bBBox.fX + TMath::Nint(0.5*(Double_t)(bBBox.fWidth))+1;
      else if (aBBox.fX>bBBox.fX)   x1 = x2 = aBBox.fX+TMath::Nint(0.5*(Double_t)(bBBox.fX+bBBox.fWidth-aBBox.fX));
      else                          x1 = x2 = bBBox.fX+TMath::Nint(0.5*(Double_t)(aBBox.fX+aBBox.fWidth-bBBox.fX));
   }

   TArrow *A = new TArrow(gPad->PixeltoX(x1), gPad->PixeltoY(y1-gPad->VtoPixel(0)), gPad->PixeltoX(x2), gPad->PixeltoY(y2-gPad->VtoPixel(0)), 0.01, "<|>");
   A->SetBit(kCanDelete);
   A->SetFillColor(lineColor);
   A->SetLineWidth(1);
   A->SetLineColor(lineColor);
   A->Draw();

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// struct used by ShowGuidelines to store the distance Field between objects
/// in the canvas.

struct dField {
   TAttBBox2D *fa;
   TAttBBox2D *fb;
   Int_t fdist;
   char fdir;


   dField()
      : fa(nullptr), fb(nullptr), fdist(0), fdir(' ')
   {}

   dField(TAttBBox2D *a, TAttBBox2D *b, Int_t dist, char direction)
      : fa(a), fb(b), fdist(dist), fdir(direction)
   {}
};

////////////////////////////////////////////////////////////////////////////////
/// Shows lines to indicate if a TAttBBox2D object is aligned to
/// the center or to another object, shows distance arrows if two
/// objects on screen have the same distance to another object
/// Call from primitive in Execute Event, in ButtonMotion after
/// the new coordinates have been set, to 'stick'
/// once when button is up to delete lines
///
/// modes: t (Top), b (bottom), l (left), r (right), i (inside)
/// in resize modes (t,b,l,r) only size arrows are sticky
///
/// in mode, the function gets the point on the element that is clicked to
/// move (i) or resize (all others). The expected values are:
/// \image html gpad_pad5.png

void TPad::ShowGuidelines(TObject *object, const Int_t event, const char mode, const bool cling )
{
   // When the object is moved with arrow or when the ShowGuideLines flag
   // is off we do show guide lines.
   if ((event == kArrowKeyRelease) || (event == kArrowKeyPress) ||
       !gEnv->GetValue("Canvas.ShowGuideLines", 0)) return;

   std::vector<dField> curDist;
   std::vector<dField> otherDist;
   Int_t pMX, pMY;
   Double_t MX, MY;
   Int_t threshold;
   TList *prims;
   UInt_t n;
   Rectangle_t aBBox, bBBox;
   aBBox = bBBox = Rectangle_t();
   TLine *L;
   TArrow *A;
   Int_t dSizeArrow = 12;   // distance of arrows indicating same size from BBox in px
   Bool_t movedX, movedY;   // make sure the current object is moved just once
   movedX = movedY = false;
   Bool_t resize = false;   // indicates resize mode
   Bool_t log = gPad->GetLogx() || gPad->GetLogy();
   if (mode != 'i') resize = true;

   TPad *is_pad = dynamic_cast<TPad *>( object );

   TContext ctxt(kTRUE);

   if (is_pad && is_pad->GetMother())
      is_pad->GetMother()->cd();

   static TPad *tmpGuideLinePad = nullptr;

   //delete all existing Guidelines and create new invisible pad
   if (tmpGuideLinePad) {
      ctxt.PadDeleted(tmpGuideLinePad);
      auto guidePadClicked = (object == tmpGuideLinePad); // in case of funny button click combination.
      tmpGuideLinePad->Delete();
      tmpGuideLinePad = nullptr;
      if (guidePadClicked) return;
   }

   // Get Primitives
   prims = gPad->GetListOfPrimitives();
   n     = TMath::Min(15,prims->GetSize());
   Int_t lineColor = TColor::GetColor(239, 202, 0);

   TAttBBox2D *cur = dynamic_cast<TAttBBox2D *>( object );
   if (cur) {
      //create invisible TPad above gPad
      if (!tmpGuideLinePad){
         tmpGuideLinePad = new TPad("tmpGuideLinePad", "tmpGuideLinePad", 0, 0, 1, 1);
         Double_t x1, y1, x2, y2;
         gPad->GetRange(x1, y1, x2, y2);
         tmpGuideLinePad->Range(x1, y1, x2, y2);
         tmpGuideLinePad->SetFillStyle(0);
         tmpGuideLinePad->SetFillColor(0);
         tmpGuideLinePad->Draw();
         tmpGuideLinePad->cd();
         gPad->GetRange(x1, y1, x2, y2);
      }
      if (cling && !log) threshold = 7;
      else threshold = 1;

      Rectangle_t BBox = cur->GetBBox();
      TPoint center = cur->GetBBoxCenter();

      otherDist.clear();
      curDist.clear();

      switch (event) {

      case kButton1Down:
      case kButton1Motion:
         MX  = gPad->GetX1() + 0.5 * (gPad->GetX2()-gPad->GetX1());
         MY  = gPad->GetY1() + 0.5 * (gPad->GetY2()-gPad->GetY1());
         pMX = gPad->XtoPixel(MX);
         pMY = gPad->YtoPixel(MY);
         // Middlelines
         if (TMath::Abs(pMX-center.GetX())<threshold) {
            if (cling && (!resize)) {
               cur->SetBBoxCenterX(pMX);
               center = cur->GetBBoxCenter();
               BBox = cur->GetBBox();
               center = cur->GetBBoxCenter();
            }
            L = new TLine(MX, gPad->GetY1(), MX, gPad->GetY2());
            L->SetBit(kCanDelete);
            L->SetLineColor(lineColor);
            L->Draw();
         }
         if (TMath::Abs(pMY-center.GetY())<threshold) {
            if (cling && (!resize)) {
               cur->SetBBoxCenterY(pMY);
               center = cur->GetBBoxCenter();
               BBox = cur->GetBBox();
               center = cur->GetBBoxCenter();
            }
            L = new TLine(gPad->GetX1(), MY, gPad->GetX2(), MY);
            L->SetBit(kCanDelete);
            L->SetLineColor(lineColor);
            L->Draw();
         }
         // Alignment to other objects
         for (UInt_t i = 0; i<n; i++) {
            TAttBBox2D *other = dynamic_cast<TAttBBox2D *>( prims->At(i) );
            if (other) {
               if (other != cur) {
                  TPoint centerOther = other->GetBBoxCenter();
                  if (TMath::Abs(center.GetX()-centerOther.GetX())<threshold) {
                     if (cling && (!resize)) {
                        cur->SetBBoxCenterX(centerOther.GetX());
                        BBox   = cur->GetBBox();
                        center = cur->GetBBoxCenter();
                     }
                     L = new TLine(gPad->PixeltoX(centerOther.GetX()), gPad->PixeltoY(center.GetY()-gPad->VtoPixel(0)),
                                   gPad->PixeltoX(centerOther.GetX()), gPad->PixeltoY(centerOther.GetY()-gPad->VtoPixel(0)));
                     L->SetLineColor(lineColor);
                     L->Draw();
                     L->SetBit(kCanDelete);
                  }
                  if (TMath::Abs(center.GetY()-centerOther.GetY())<threshold) {
                     if (cling && (!resize)) {
                        cur->SetBBoxCenterY(centerOther.GetY());
                        BBox   = cur->GetBBox();
                        center = cur->GetBBoxCenter();
                     }
                     L = new TLine(gPad->PixeltoX(center.GetX()), gPad->PixeltoY(centerOther.GetY()-gPad->VtoPixel(0)),
                                   gPad->PixeltoX(centerOther.GetX()), gPad->PixeltoY(centerOther.GetY()-gPad->VtoPixel(0)));
                     L->SetBit(kCanDelete);
                     L->SetLineColor(lineColor);
                     L->Draw();
                  }
               }
            }
         }
         // Get Distances between objects
         for (UInt_t i = 0; i<n; i++) {
            TAttBBox2D *a = dynamic_cast<TAttBBox2D *>( prims->At(i) );
            if (a) {
               aBBox = a->GetBBox();
               for (UInt_t j = i+1; j<n; j++) {
                  TAttBBox2D *b = dynamic_cast<TAttBBox2D *>( prims->At(j) );
                  if (b) {
                     bBBox = b->GetBBox();

                     //only when bounding boxes overlap in x or y direction
                     if (((aBBox.fX<bBBox.fX) && (bBBox.fX-aBBox.fX<=aBBox.fWidth))||((aBBox.fX>bBBox.fX) && (aBBox.fX-bBBox.fX<=bBBox.fWidth))){ //BBoxes overlap in x direction
                        if ((aBBox.fY+aBBox.fHeight<bBBox.fY)||(bBBox.fY+bBBox.fHeight<aBBox.fY)) {//No overlap in Y-direction required
                           dField abDist = dField();
                           if (aBBox.fY>bBBox.fY) abDist = dField(a, b, TMath::Abs(aBBox.fY-(bBBox.fY+bBBox.fHeight)), 'y');
                           else                   abDist = dField(a, b, TMath::Abs(bBBox.fY-(aBBox.fY+aBBox.fHeight)), 'y');
                           if ((b != cur)&&(a != cur)) otherDist.push_back(abDist);
                           else curDist.push_back(abDist);
                        }
                     } else if (((aBBox.fY<bBBox.fY) && (bBBox.fY-aBBox.fY<=aBBox.fHeight))||((aBBox.fY>bBBox.fY) && (aBBox.fY-bBBox.fY<=bBBox.fHeight))) { //BBoxes overlap in y direction
                        if ((aBBox.fX+aBBox.fWidth<bBBox.fX)||(bBBox.fX+bBBox.fWidth<aBBox.fX)) {//No overlap in x-direction required
                           dField abDist = dField();
                           if (aBBox.fX>bBBox.fX) abDist = dField(a, b, TMath::Abs(aBBox.fX-(bBBox.fX+bBBox.fWidth)), 'x');
                           else                   abDist = dField(a, b, TMath::Abs(bBBox.fX-(aBBox.fX+aBBox.fWidth)), 'x');
                           if ((b != cur)&&(a != cur)) otherDist.push_back(abDist);
                           else                        curDist.push_back(abDist);
                        }
                     }
                  }
               }
            }
         }
         // Show equal distances
         for (UInt_t i = 0; i<curDist.size(); i++) {
            for (UInt_t j = 0; j<otherDist.size(); j++) {
               if ((curDist[i].fdir == otherDist[j].fdir) && (otherDist[j].fdir=='x') && (TMath::Abs(curDist[i].fdist-otherDist[j].fdist)<threshold)) {
                  if (cling && (!movedX) && (!resize)) {
                     if ((cur->GetBBoxCenter().fX < curDist[i].fb->GetBBoxCenter().fX)||(cur->GetBBoxCenter().fX < curDist[i].fa->GetBBoxCenter().fX))
                           cur->SetBBoxCenterX(cur->GetBBoxCenter().fX - otherDist[j].fdist + curDist[i].fdist);
                     else  cur->SetBBoxCenterX(cur->GetBBoxCenter().fX + otherDist[j].fdist - curDist[i].fdist);
                     movedX = true;
                  }
                  DrawDist(curDist[i].fa->GetBBox(), curDist[i].fb->GetBBox(), 'x');
                  DrawDist(otherDist[j].fa->GetBBox(), otherDist[j].fb->GetBBox(), 'x');
               }
               if ((curDist[i].fdir == otherDist[j].fdir) && (otherDist[j].fdir=='y') && (TMath::Abs(curDist[i].fdist-otherDist[j].fdist)<threshold)) {
                  if (cling && (!movedY) && (!resize)) {
                     if ((cur->GetBBoxCenter().fY < curDist[i].fb->GetBBoxCenter().fY)||(cur->GetBBoxCenter().fY < curDist[i].fa->GetBBoxCenter().fY))
                           cur->SetBBoxCenterY(cur->GetBBoxCenter().fY - otherDist[j].fdist + curDist[i].fdist);
                     else  cur->SetBBoxCenterY(cur->GetBBoxCenter().fY + otherDist[j].fdist - curDist[i].fdist);
                     movedY = true;
                  }
                  DrawDist(curDist[i].fa->GetBBox(), curDist[i].fb->GetBBox(), 'y');
                  DrawDist(otherDist[j].fa->GetBBox(), otherDist[j].fb->GetBBox(), 'y');
               }
            }
            for (UInt_t j = i; j<curDist.size(); j++) {
               if (i!=j) {
                  if ((curDist[i].fdir == curDist[j].fdir) && (curDist[j].fdir=='x') && (TMath::Abs(curDist[i].fdist-curDist[j].fdist)<threshold)) {
                     if (cling && (!movedX) && (!resize)) {
                        if ((cur->GetBBoxCenter().fX < curDist[i].fb->GetBBoxCenter().fX)||(cur->GetBBoxCenter().fX < curDist[i].fa->GetBBoxCenter().fX))
                              cur->SetBBoxCenterX(cur->GetBBoxCenter().fX - floor(0.5*(curDist[j].fdist - curDist[i].fdist)));
                        else  cur->SetBBoxCenterX(cur->GetBBoxCenter().fX + floor(0.5*(curDist[j].fdist - curDist[i].fdist)));
                     }
                     DrawDist(curDist[i].fa->GetBBox(), curDist[i].fb->GetBBox(), 'x');
                     DrawDist(curDist[j].fa->GetBBox(), curDist[j].fb->GetBBox(), 'x');
                  }

                  if ((curDist[i].fdir == curDist[j].fdir) && (curDist[j].fdir=='y') && (TMath::Abs(curDist[i].fdist-curDist[j].fdist)<threshold)) {
                     if (cling && (!movedY) && (!resize)) {
                        if ((cur->GetBBoxCenter().fY < curDist[i].fb->GetBBoxCenter().fY)||(cur->GetBBoxCenter().fY < curDist[i].fa->GetBBoxCenter().fY))
                              cur->SetBBoxCenterY(cur->GetBBoxCenter().fY - floor(0.5*(curDist[j].fdist - curDist[i].fdist)));
                        else  cur->SetBBoxCenterY(cur->GetBBoxCenter().fY + floor(0.5*(curDist[j].fdist - curDist[i].fdist)));
                     }
                     DrawDist(curDist[i].fa->GetBBox(), curDist[i].fb->GetBBox(), 'y');
                     DrawDist(curDist[j].fa->GetBBox(), curDist[j].fb->GetBBox(), 'y');
                  }
               }
            }
         }
         if (resize) {
            // Show equal Sizes
            for (UInt_t i = 0; i<n; i++) {
               TAttBBox2D *a = dynamic_cast<TAttBBox2D *>( prims->At(i) );
               if (a && (cur != a)) {
                  aBBox = a->GetBBox();

                  if ((TMath::Abs(aBBox.fWidth - BBox.fWidth)<threshold) && (mode != 't') && (mode != 'b')) {
                     if (cling) {
                        if (mode == 'l') cur->SetBBoxX1(BBox.fX + BBox.fWidth - aBBox.fWidth);
                        if (mode == 'r') cur->SetBBoxX2(BBox.fX + aBBox.fWidth);
                        if ((mode == '1')||(mode == '4')) cur->SetBBoxX1(BBox.fX + BBox.fWidth - aBBox.fWidth);
                        if ((mode == '2')||(mode == '3')) cur->SetBBoxX2(BBox.fX + aBBox.fWidth);
                        BBox = cur->GetBBox();
                     }

                     A = new TArrow(gPad->PixeltoX(aBBox.fX), gPad->PixeltoY(aBBox.fY-dSizeArrow-gPad->VtoPixel(0)),
                                    gPad->PixeltoX(aBBox.fX+aBBox.fWidth), gPad->PixeltoY(aBBox.fY-dSizeArrow-gPad->VtoPixel(0)), 0.01, "<|>");
                     A->SetBit(kCanDelete);
                     A->SetLineColor(lineColor);
                     A->SetFillColor(lineColor);
                     A->Draw();

                     A = new TArrow(gPad->PixeltoX(BBox.fX), gPad->PixeltoY(BBox.fY-dSizeArrow-gPad->VtoPixel(0)),
                                    gPad->PixeltoX(BBox.fX+BBox.fWidth), gPad->PixeltoY(BBox.fY-dSizeArrow-gPad->VtoPixel(0)), 0.01, "<|>");
                     A->SetBit(kCanDelete);
                     A->SetLineColor(lineColor);
                     A->SetFillColor(lineColor);
                     A->Draw();
                  }
                  if ((TMath::Abs(aBBox.fHeight - BBox.fHeight)<threshold) && (mode != 'r') && (mode != 'l')) {
                     if (cling) {
                        if (mode == 't') cur->SetBBoxY1(BBox.fY + BBox.fHeight - aBBox.fHeight);
                        if (mode == 'b') cur->SetBBoxY2(BBox.fY + aBBox.fHeight);
                        if ((mode == '1')||(mode == '2')) cur->SetBBoxY1(BBox.fY + BBox.fHeight - aBBox.fHeight);
                        if ((mode == '3')||(mode == '4')) cur->SetBBoxY2(BBox.fY + aBBox.fHeight);
                        BBox = cur->GetBBox();
                     }
                     A = new TArrow(gPad->PixeltoX(aBBox.fX-dSizeArrow), gPad->PixeltoY(aBBox.fY-gPad->VtoPixel(0)),
                                    gPad->PixeltoX(aBBox.fX-dSizeArrow), gPad->PixeltoY(aBBox.fY+aBBox.fHeight-gPad->VtoPixel(0)), 0.01, "<|>");
                     A->SetBit(kCanDelete);
                     A->SetLineColor(lineColor);
                     A->SetFillColor(lineColor);
                     A->Draw();

                     A = new TArrow(gPad->PixeltoX(BBox.fX-dSizeArrow), gPad->PixeltoY(BBox.fY-gPad->VtoPixel(0)),
                                   gPad->PixeltoX(BBox.fX-dSizeArrow), gPad->PixeltoY(BBox.fY+BBox.fHeight-gPad->VtoPixel(0)), 0.01, "<|>");
                     A->SetBit(kCanDelete);
                     A->SetLineColor(lineColor);
                     A->SetFillColor(lineColor);
                     A->Draw();
                  }
               }
            }
         }

         break;

      case kButton1Up:
         if (tmpGuideLinePad) {
            // All the arrows and lines in that pad are also deleted because
            // they all have the bit kCanDelete on.
            tmpGuideLinePad->Delete();
            tmpGuideLinePad = nullptr;
         }
         break;
      }
   }

   gPad->Modified(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Return kTRUE if the crosshair has been activated (via SetCrosshair).

Bool_t TPad::HasCrosshair() const
{
   return (Bool_t)GetCrosshair();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the crosshair type (from the mother canvas)
/// crosshair type = 0 means no crosshair.

Int_t TPad::GetCrosshair() const
{
   if (this == (TPad*)fCanvas)
      return fCrosshair;
   return fCanvas ? fCanvas->GetCrosshair() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set crosshair active/inactive.
///  - If crhair != 0, a crosshair will be drawn in the pad and its sub-pads.
///  - If the canvas crhair = 1 , the crosshair spans the full canvas.
///  - If the canvas crhair > 1 , the crosshair spans only the pad.

void TPad::SetCrosshair(Int_t crhair)
{
   if (!fCanvas) return;
   fCrosshair = crhair;
   fCrosshairPos = 0;

   if (this != (TPad*)fCanvas) fCanvas->SetCrosshair(crhair);
}

////////////////////////////////////////////////////////////////////////////////
/// static function to set the maximum Pick Distance fgMaxPickDistance
/// This parameter is used in TPad::Pick to select an object if
/// its DistancetoPrimitive returns a value < fgMaxPickDistance
/// The default value is 5 pixels. Setting a smaller value will make
/// picking more precise but also more difficult

void TPad::SetMaxPickDistance(Int_t maxPick)
{
   fgMaxPickDistance = maxPick;
}

////////////////////////////////////////////////////////////////////////////////
/// Set tool tip text associated with this pad. The delay is in
/// milliseconds (minimum 250). To remove tool tip call method with
/// text = 0.

void TPad::SetToolTipText(const char *text, Long_t delayms)
{
   if (fTip) {
      DeleteToolTip(fTip);
      fTip = nullptr;
   }

   if (text && strlen(text))
      fTip = CreateToolTip((TBox*)nullptr, text, delayms);
}

////////////////////////////////////////////////////////////////////////////////
/// Set pad vertical (default) or horizontal

void TPad::SetVertical(Bool_t vert)
{
   if (vert) ResetBit(kHori);
   else      SetBit(kHori);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a class object.

void TPad::Streamer(TBuffer &b)
{
   UInt_t R__s, R__c;
   Int_t nch, nobjects;
   Float_t single;
   TObject *obj;
   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 5) {
         if (!gPad)
            gPad = new TCanvas(GetName());
         fMother = (TPad*)gPad;
         fCanvas = fMother->GetCanvas();
         TContext ctxt(this, kFALSE);
         fPixmapID = -1;      // -1 means pixmap will be created by ResizePad()
         gReadLevel++;
         gROOT->SetReadingObject(kTRUE);

         b.ReadClassBuffer(TPad::Class(), this, v, R__s, R__c);

         //Set the kCanDelete bit in all objects in the pad such that when the pad
         //is deleted all objects in the pad are deleted too.
         //Also set must cleanup bit which normally set for all primitives add to pad,
         // but may be reset in IO like TH1::Streamer does
         TIter next(fPrimitives);
         while ((obj = next())) {
            obj->SetBit(kCanDelete);
            obj->SetBit(kMustCleanup);
         }

         fModified = kTRUE;
         fPadPointer = nullptr;
         gReadLevel--;
         if (gReadLevel == 0 && IsA() == TPad::Class()) ResizePad();
         gROOT->SetReadingObject(kFALSE);
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

      if (!gPad)
         gPad = new TCanvas(GetName());
      if (gReadLevel == 0)
         fMother = (TPad *)gROOT->GetSelectedPad();
      else
         fMother = (TPad *)gPad;
      if (!fMother)
         fMother = (TPad *)gPad;
      if (fMother)
         fCanvas = fMother->GetCanvas();
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
         TContext ctxt(this, kFALSE);
         char drawoption[64];
         for (Int_t i = 0; i < nobjects; i++) {
            b >> obj;
            b >> nch;
            b.ReadFastArray(drawoption,nch);
            fPrimitives->AddLast(obj, drawoption);
            gPad = this; // gPad may be modified in b >> obj if obj is a pad
         }
      }
      gReadLevel--;
      gROOT->SetReadingObject(kFALSE);
      //////////////////////////////////////////////////////////////////////////

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
      fPadPointer = nullptr;
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

////////////////////////////////////////////////////////////////////////////////
/// Force a copy of current style for all objects in pad.

void TPad::UseCurrentStyle()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Loop and sleep until a primitive with name=pname is found in the pad.
///
/// If emode is given, the editor is automatically set to emode, ie
/// it is not required to have the editor control bar.
///
/// The possible values for emode are:
///  - emode = "" (default). User will select the mode via the editor bar
///  - emode = "Arc", "Line", "Arrow", "Button", "Diamond", "Ellipse",
///  - emode = "Pad","pave", "PaveLabel","PaveText", "PavesText",
///  - emode = "PolyLine", "CurlyLine", "CurlyArc", "Text", "Marker", "CutG"
///
/// If emode is specified and it is not valid, "PolyLine" is assumed. If emode
/// is not specified or ="", an attempt is to use pname[1...]
///
/// for example if pname="TArc", emode="Arc" will be assumed.
/// When this function is called within a macro, the macro execution
/// is suspended until a primitive corresponding to the arguments
/// is found in the pad.
///
/// If CRTL/C is typed in the pad, the function returns 0.
///
/// While this function is executing, one can use the mouse, interact
/// with the graphics pads, use the Inspector, Browser, TreeViewer, etc.
///
/// Examples:
/// ~~~ {.cpp}
///   c1.WaitPrimitive();      // Return the first created primitive
///                            // whatever it is.
///                            // If a double-click with the mouse is executed
///                            // in the pad or any key pressed, the function
///                            // returns 0.
///   c1.WaitPrimitive("ggg"); // Set the editor in mode "PolyLine/Graph"
///                            // Create a polyline, then using the context
///                            // menu item "SetName", change the name
///                            // of the created TGraph to "ggg"
///   c1.WaitPrimitive("TArc");// Set the editor in mode "Arc". Returns
///                            // as soon as a TArc object is created.
///   c1.WaitPrimitive("lat","Text"); // Set the editor in Text/Latex mode.
///                            // Create a text object, then Set its name to "lat"
/// ~~~
/// The following macro waits for 10 primitives of any type to be created.
///
/// ~~~ {.cpp}
///{
///   TCanvas c1("c1");
///   TObject *obj;
///   for (Int_t i=0;i<10;i++) {
///      obj = gPad->WaitPrimitive();
///      if (!obj) break;
///      printf("Loop i=%d, found objIsA=%s, name=%s\n",
///         i,obj->ClassName(),obj->GetName());
///   }
///}
/// ~~~
///
/// If ROOT runs in batch mode a call to this method does nothing.

TObject *TPad::WaitPrimitive(const char *pname, const char *emode)
{
   if (!gPad || IsWeb())
      return nullptr;

   if (emode && strlen(emode)) gROOT->SetEditorMode(emode);
   if (gROOT->GetEditorMode() == 0 && pname && strlen(pname) > 2) gROOT->SetEditorMode(&pname[1]);

   if (!fPrimitives) fPrimitives = new TList;
   gSystem->ProcessEvents();
   TObject *oldlast = gPad->GetListOfPrimitives() ? gPad->GetListOfPrimitives()->Last() : nullptr;
   TObject *obj = nullptr;
   Bool_t testlast = kFALSE;
   Bool_t hasname = pname && (strlen(pname) > 0);
   if ((!pname || !pname[0]) && (!emode || !emode[0])) testlast = kTRUE;
   if (testlast) gROOT->SetEditorMode();
   while (!gSystem->ProcessEvents() && gROOT->GetSelectedPad() && gPad) {
      if (gROOT->GetEditorMode() == 0) {
         if (hasname) {
            obj = FindObject(pname);
            if (obj) return obj;
         }
         if (testlast) {
            if (!gPad->GetListOfPrimitives()) return nullptr;
            obj = gPad->GetListOfPrimitives()->Last();
            if (obj != oldlast) return obj;
            Int_t event = GetEvent();
            if (event == kButton1Double || event == kKeyPress) {
               //the following statement is required against other loop executions
               //before returning
               fCanvas->HandleInput((EEventType)-1,0,0);
               return nullptr;
            }
         }
      }
      gSystem->Sleep(10);
   }

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a tool tip and return its pointer.

TObject *TPad::CreateToolTip(const TBox *box, const char *text, Long_t delayms)
{
   if (gPad->IsBatch()) return nullptr;
   return (TObject*)gROOT->ProcessLineFast(TString::Format("new TGToolTip((TBox*)0x%zx,\"%s\",%d)",
                                           (size_t)box,text,(Int_t)delayms).Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Delete tool tip object.

void TPad::DeleteToolTip(TObject *tip)
{
   // delete tip;
   if (!tip) return;
   gROOT->ProcessLineFast(TString::Format("delete (TGToolTip*)0x%zx", (size_t)tip).Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Reset tool tip, i.e. within time specified in CreateToolTip the
/// tool tip will pop up.

void TPad::ResetToolTip(TObject *tip)
{
   if (!tip) return;
   // tip->Reset(this);
   gROOT->ProcessLineFast(TString::Format("((TGToolTip*)0x%zx)->Reset((TPad*)0x%zx)",
                          (size_t)tip,(size_t)this).Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Hide tool tip.

void TPad::CloseToolTip(TObject *tip)
{
   if (!tip) return;
   // tip->Hide();
   gROOT->ProcessLineFast(TString::Format("((TGToolTip*)0x%zx)->Hide()", (size_t)tip).Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Deprecated: use TPad::GetViewer3D() instead

void TPad::x3d(Option_t *type)
{
   ::Info("TPad::x3d()", "This function is deprecated. Use %s->GetViewer3D(\"x3d\") instead",this->GetName());

   // Default on GetViewer3D is pad - for x3d it was x3d...
   if (!type || !type[0]) {
      type = "x3d";
   }
   GetViewer3D(type);
}

////////////////////////////////////////////////////////////////////////////////
/// Create/obtain handle to 3D viewer. Valid types are:
///  - 'pad' - pad drawing via TViewer3DPad
/// any others registered with plugin manager supporting TVirtualViewer3D
/// If an invalid/null type is requested then the current viewer is returned
/// (if any), otherwise a default 'pad' type is returned

TVirtualViewer3D *TPad::GetViewer3D(Option_t *type)
{
   Bool_t validType = kFALSE;

   if ((!type || !*type || (strstr(type, "gl") && !strstr(type, "ogl"))) && (!fCanvas || !fCanvas->UseGL()))
      type = "pad";

   if (type && *type) {
      if (gPluginMgr->FindHandler("TVirtualViewer3D", type))
         validType = kTRUE;
   }

   // Invalid/null type requested?
   if (!validType) {
      // Return current viewer if there is one
      if (fViewer3D)
         return fViewer3D;
      // otherwise default to the pad
      else
         type = "pad";
   }

   // Ensure we can create the new viewer before removing any existing one
   TVirtualViewer3D *newViewer = nullptr;

   Bool_t createdExternal = kFALSE;

   // External viewers need to be created via plugin manager via interface...
   if (!strstr(type,"pad")) {
      newViewer = TVirtualViewer3D::Viewer3D(this, type);

      if (!newViewer) {
         Warning("GetViewer3D", "Cannot create 3D viewer of type: %s", type);
         // Return the existing viewer
         return fViewer3D;
      }

      if (strstr(type, "gl") && !strstr(type, "ogl")) {
         fEmbeddedGL = kTRUE;
         fCopyGLDevice = kTRUE;
         Modified();
      } else {
         createdExternal = kTRUE;
      }

   } else {
      newViewer = new TViewer3DPad(*this);
   }

   // If we had a previous viewer destroy it now
   // In this case we do take responsibility for destroying viewer
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

////////////////////////////////////////////////////////////////////////////////
/// Release current (external) viewer

void TPad::ReleaseViewer3D(Option_t * /*type*/ )
{
   fViewer3D = nullptr;

   // We would like to ensure the pad is repainted
   // when external viewer is closed down. However
   // a modify/paint call here will repaint the pad
   // before the external viewer window actually closes.
   // So the pad would have to be redraw twice over.
   // Currently we just have to live with the pad staying blank
   // any click in pad will refresh.
}

////////////////////////////////////////////////////////////////////////////////
/// Get GL device.

Int_t TPad::GetGLDevice()
{
   return fGLDevice;
}

////////////////////////////////////////////////////////////////////////////////
/// Emit RecordPave() signal.

void TPad::RecordPave(const TObject *obj)
{
   Emit("RecordPave(const TObject*)", (Longptr_t)obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Emit RecordLatex() signal.

void TPad::RecordLatex(const TObject *obj)
{
   Emit("RecordLatex(const TObject*)", (Longptr_t)obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Get pad painter from TCanvas.

TVirtualPadPainter *TPad::GetPainter()
{
   if (!fCanvas) return nullptr;
   return fCanvas->GetCanvasPainter();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the bounding Box of the Pad

Rectangle_t TPad::GetBBox()
{
   Rectangle_t BBox{0, 0, 0, 0};
   if (gPad) {
      BBox.fX = gPad->XtoPixel(fXlowNDC * (gPad->GetX2() - gPad->GetX1()) + gPad->GetX1());
      BBox.fY = gPad->YtoPixel((fYlowNDC + fHNDC) * (gPad->GetY2() - gPad->GetY1()) + gPad->GetY1());
      BBox.fWidth = gPad->XtoPixel((fXlowNDC + fWNDC) * (gPad->GetX2() - gPad->GetX1()) + gPad->GetX1()) -
                    gPad->XtoPixel(fXlowNDC * (gPad->GetX2() - gPad->GetX1()) + gPad->GetX1());
      BBox.fHeight = gPad->YtoPixel((fYlowNDC) * (gPad->GetY2() - gPad->GetY1()) + gPad->GetY1()) -
                     gPad->YtoPixel((fYlowNDC + fHNDC) * (gPad->GetY2() - gPad->GetY1()) + gPad->GetY1());
   }
   return BBox;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the center of the Pad as TPoint in pixels

TPoint TPad::GetBBoxCenter()
{
   TPoint p(0, 0);
   if (gPad) {
      Double_t x = ((fXlowNDC + 0.5 * fWNDC) * (gPad->GetX2() - gPad->GetX1())) + gPad->GetX1();
      Double_t y = ((fYlowNDC + 0.5 * fHNDC) * (gPad->GetY2() - gPad->GetY1())) + gPad->GetY1();
      p.SetX(gPad->XtoPixel(x));
      p.SetY(gPad->YtoPixel(y));
   }
   return p;
}

////////////////////////////////////////////////////////////////////////////////
/// Set center of the Pad

void TPad::SetBBoxCenter(const TPoint &p)
{
   if (!gPad)
      return;
   fXlowNDC = (gPad->PixeltoX(p.GetX()) - gPad->GetX1()) / (gPad->GetX2() - gPad->GetX1()) - 0.5 * fWNDC;
   fYlowNDC =
      (gPad->PixeltoY(p.GetY() - gPad->VtoPixel(0)) - gPad->GetY1()) / (gPad->GetY2() - gPad->GetY1()) - 0.5 * fHNDC;
   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Set X coordinate of the center of the Pad

void TPad::SetBBoxCenterX(const Int_t x)
{
   if (!gPad)
      return;
   fXlowNDC = (gPad->PixeltoX(x) - gPad->GetX1())/(gPad->GetX2()-gPad->GetX1())-0.5*fWNDC;
   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Set Y coordinate of the center of the Pad

void TPad::SetBBoxCenterY(const Int_t y)
{
   if (!gPad)
      return;
   fYlowNDC = (gPad->PixeltoY(y - gPad->VtoPixel(0)) - gPad->GetY1()) / (gPad->GetY2() - gPad->GetY1()) - 0.5 * fHNDC;
   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Set lefthandside of BoundingBox to a value
/// (resize in x direction on left)

void TPad::SetBBoxX1(const Int_t x)
{
   if (!gPad)
      return;
   fXlowNDC = (gPad->PixeltoX(x) - gPad->GetX1()) / (gPad->GetX2() - gPad->GetX1());
   fWNDC = fXUpNDC - fXlowNDC;
   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Set right hand side of BoundingBox to a value
/// (resize in x direction on right)

void TPad::SetBBoxX2(const Int_t x)
{
   if (!gPad)
      return;
   fWNDC = (gPad->PixeltoX(x) - gPad->GetX1()) / (gPad->GetX2() - gPad->GetX1()) - fXlowNDC;
   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Set top of BoundingBox to a value (resize in y direction on top)

void TPad::SetBBoxY1(const Int_t y)
{
   if (!gPad)
      return;
   fHNDC = (gPad->PixeltoY(y - gPad->VtoPixel(0)) - gPad->GetY1()) / (gPad->GetY2() - gPad->GetY1()) - fYlowNDC;
   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Set bottom of BoundingBox to a value
/// (resize in y direction on bottom)

void TPad::SetBBoxY2(const Int_t y)
{
   if (!gPad)
      return;
   fYlowNDC = (gPad->PixeltoY(y - gPad->VtoPixel(0)) - gPad->GetY1()) / (gPad->GetY2() - gPad->GetY1());
   fHNDC = fYUpNDC - fYlowNDC;
   ResizePad();
}

////////////////////////////////////////////////////////////////////////////////
/// Mark pad modified
/// Will be repainted when TCanvas::Update() will be called next time

void TPad::Modified(Bool_t flag)
{
   if (!fModified && flag) Emit("Modified()");
   fModified = flag;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert absolute pixel into X/Y coordinates

void TPad::AbsPixeltoXY(Double_t xpixel, Double_t ypixel, Double_t &x, Double_t &y)
{
   x = AbsPixeltoX(xpixel);
   y = AbsPixeltoY(ypixel);
}


////////////////////////////////////////////////////////////////////////////////
/// Convert pixel to X coordinate

Double_t TPad::PixeltoX(Double_t px)
{
   if (fAbsCoord) return fAbsPixeltoXk + px*fPixeltoX;
   else           return fPixeltoXk    + px*fPixeltoX;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert pixel to Y coordinate

Double_t TPad::PixeltoY(Double_t py)
{
   if (fAbsCoord) return fAbsPixeltoYk + py*fPixeltoY;
   else           return fPixeltoYk    + py*fPixeltoY;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert pixel to X/Y coordinates

void TPad::PixeltoXY(Double_t xpixel, Double_t ypixel, Double_t &x, Double_t &y)
{
   x = PixeltoX(xpixel);
   y = PixeltoY(ypixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X/Y into absolute pixel coordinates - integer

void TPad::XYtoAbsPixel(Double_t x, Double_t y, Int_t &xpixel, Int_t &ypixel) const
{
   xpixel = XtoAbsPixel(x);
   ypixel = YtoAbsPixel(y);
}

////////////////////////////////////////////////////////////////////////////////
/// Check value for valid range for pixel values

Double_t pixel_boundary(Double_t v)
{
   return v < -kMaxPixel ? -kMaxPixel : (v > kMaxPixel ? kMaxPixel : v);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X/Y into absolute pixel coordinates - doble
/// Introduced to avoid pixel rounding problems

void TPad::XYtoAbsPixel(Double_t x, Double_t y, Double_t &xpixel, Double_t &ypixel) const
{
   xpixel = pixel_boundary(fXtoAbsPixelk + x*fXtoPixel);
   ypixel = pixel_boundary(fYtoAbsPixelk + y*fYtoPixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X/Y into pixel coordinates - integer

void TPad::XYtoPixel(Double_t x, Double_t y, Int_t &xpixel, Int_t &ypixel) const
{
   xpixel = XtoPixel(x);
   ypixel = YtoPixel(y);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X/Y into pixel coordinates - double

void TPad::XYtoPixel(Double_t x, Double_t y, Double_t &xpixel, Double_t &ypixel) const
{
   xpixel = pixel_boundary(fAbsCoord ? fXtoAbsPixelk + x*fXtoPixel : fXtoPixelk + x*fXtoPixel);
   ypixel = pixel_boundary(fAbsCoord ? fYtoAbsPixelk + y*fYtoPixel : fYtoPixelk + y*fYtoPixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X NDC to pixel

Int_t TPad::UtoPixel(Double_t u) const
{
   return TMath::Nint(pixel_boundary(fAbsCoord ? fUtoAbsPixelk + u*fUtoPixel : fUtoPixelk + u*fUtoPixel));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert Y NDC to pixel

Int_t TPad::VtoPixel(Double_t v) const
{
   return TMath::Nint(pixel_boundary(fAbsCoord ? fVtoAbsPixelk + v*fVtoPixel : fVtoPixelk + v*fVtoPixel));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X NDC to absolute pixel

Int_t TPad::UtoAbsPixel(Double_t u) const
{
   return TMath::Nint(fUtoAbsPixelk + u*fUtoPixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert Y NDC to absolute pixel

Int_t TPad::VtoAbsPixel(Double_t v) const
{
   return TMath::Nint(fVtoAbsPixelk + v*fVtoPixel);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X coordinate to absolute pixel

Int_t TPad::XtoAbsPixel(Double_t x) const
{
   return TMath::Nint(pixel_boundary(fXtoAbsPixelk + x*fXtoPixel));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert X coordinate to pixel

Int_t TPad::XtoPixel(Double_t x) const
{
   return TMath::Nint(pixel_boundary(fAbsCoord ? fXtoAbsPixelk + x*fXtoPixel : fXtoPixelk + x*fXtoPixel));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert Y coordinate to absolute pixel

Int_t TPad::YtoAbsPixel(Double_t y) const
{
   return TMath::Nint(pixel_boundary(fYtoAbsPixelk + y*fYtoPixel));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a vertical distance [y1,y2] to pixel

Int_t TPad::HtoAbsPixel(Double_t y1, Double_t y2) const
{
   double h1 = fYtoAbsPixelk + y1 * fYtoPixel;
   double h2 = fYtoAbsPixelk + y2 * fYtoPixel;
   return TMath::Nint(pixel_boundary(std::abs(h1 - h2)));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a horizontal distance [x1,x2] to pixel

Int_t TPad::WtoAbsPixel(Double_t x1, Double_t x2) const
{
   double w1 = fXtoAbsPixelk + x1 * fXtoPixel;
   double w2 = fXtoAbsPixelk + x2 * fXtoPixel;
   return TMath::Nint(pixel_boundary(std::abs(w1 - w2)));
}


////////////////////////////////////////////////////////////////////////////////
/// Convert Y coordinate to pixel

Int_t TPad::YtoPixel(Double_t y) const
{
   return TMath::Nint(pixel_boundary(fAbsCoord ? fYtoAbsPixelk + y*fYtoPixel : fYtoPixelk + y*fYtoPixel));
}
