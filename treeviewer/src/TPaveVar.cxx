// @(#)root/treeviewer:$Name$:$Id$
// Author: Rene Brun   08/12/98
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream.h>
#include <iostream.h>

#include "TROOT.h"
#include "Buttons.h"
#include "TPaveVar.h"

ClassImp(TPaveVar)

//______________________________________________________________________________
//  A PaveVar is a TPaveLabel  specialized to process
//  variables and cuts inside a TTreeViewer.
//   A TPaveVar object is used by the TTreeViewer to represent
//     - a TTree variable
//     - a selection expression.
//   A TPavevar has the same graphical representation as a TPaveLabel.
//   One can only resize the left or right borders (not top/bottom).
//   The context menu allows merging (and/or) overlapping TPaveVars.
//

//______________________________________________________________________________
TPaveVar::TPaveVar(): TPaveLabel()
{
//*-*-*-*-*-*-*-*-*-*-*PaveVar default constructor*-*-*-*-*-*-*-*-*-*-*-*-*

   fViewer  = 0;
//*-*                  =============================
}

//______________________________________________________________________________
TPaveVar::TPaveVar(Coord_t x1, Coord_t y1,Coord_t x2, Coord_t  y2, const char *label, TTreeViewer *viewer)
           :TPaveLabel(x1,y1,x2,y2,label,"br")
{
//*-*-*-*-*-*-*-*-*-*-*PaveVar normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

   fViewer  = viewer;
   SetName(label);
   SetTextSize(fViewer->GetTextSize());
   SetTextFont(fViewer->GetTextFont());
   SetTextColor(fViewer->GetTextColor());
}

//______________________________________________________________________________
TPaveVar::~TPaveVar()
{
//*-*-*-*-*-*-*-*-*-*-*PaveVar default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

}

//______________________________________________________________________________
TPaveVar::TPaveVar(const TPaveVar &PaveVar)
{
   ((TPaveVar&)PaveVar).Copy(*this);
}

//______________________________________________________________________________
void TPaveVar::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this PaveVar to PaveVar*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================================

   TPaveLabel::Copy(obj);
   ((TPaveVar&)obj).fViewer      = fViewer;
}
//______________________________________________________________________________
void TPaveVar::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a TPaveVar object is clicked.
//
//  If the mouse is clicked inside the box, the box is moved.
//
//  If the mouse is clicked on the 2 edges (L,R), the box is rscaled
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

   const Int_t kMaxDiff = 5;
   const Int_t kMinSize = 20;

   static Int_t px1, px2, py1, py2, pxl, pyl, pxt, pyt, pxold, pyold, pxdold, pydold;
   static Int_t px1p, px2p, py1p, py2p;
   static Bool_t L, R, INSIDE;
   Int_t  wx, wy;

   Bool_t doing_again = kFALSE;
   Bool_t opaque  = gPad->OpaqueMoving();
   Bool_t ropaque = gPad->OpaqueResizing();

   TVirtualPad  *parent = gPad;
   char *action;

   HideToolTip(event);

again:

   switch (event) {

   case kMouseEnter:
      TBox::ExecuteEvent(event,px,py);
      break;

   case kButton1Double:
      px1 = -1; //used by kButton1Up
      py1 = strlen(GetName()) + 10;
      if (TestBit(kBranchObject)) break;
      action = new char[py1];
      sprintf(action,"VarDraw:%s",GetName());
      gPad->SetCursor(kWatch);
      fViewer->ExecuteDraw(action);
      delete [] action;
      break;

   case kButton1Down:

      pxdold = px;
      pydold = py;
      gVirtualX->SetLineColor(-1);
      TAttLine::Modify();  //Change line attributes only if necessary
      if (GetFillColor())
         gVirtualX->SetLineColor(GetFillColor());
      else
         gVirtualX->SetLineColor(1);
      gVirtualX->SetLineWidth(2);

      // No break !!!

   case kMouseMotion:

      px1 = gPad->XtoAbsPixel(GetX1());
      py1 = gPad->YtoAbsPixel(GetY1());
      px2 = gPad->XtoAbsPixel(GetX2());
      py2 = gPad->YtoAbsPixel(GetY2());

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

      L = R = INSIDE = kFALSE;

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxl) < kMaxDiff) {             // left edge
         pxold = pxl; pyold = pyl; L = kTRUE;
         gPad->SetCursor(kLeftSide);
      }

      if ((py > pyl+kMaxDiff && py < pyt-kMaxDiff) &&
          TMath::Abs(px - pxt) < kMaxDiff) {             // right edge
          pxold = pxt; pyold = pyt; R = kTRUE;
          gPad->SetCursor(kRightSide);
      }

      if ((px > pxl+kMaxDiff && px < pxt-kMaxDiff) &&
          (py > pyl+kMaxDiff && py < pyt-kMaxDiff)) {    // inside box
         pxold = px; pyold = py; INSIDE = kTRUE;
         if (event == kButton1Down)
            gPad->SetCursor(kMove);
         else
            gPad->SetCursor(kCross);
      }

      fResizing = kFALSE;
      if (L || R)
         fResizing = kTRUE;

      if (!L && !R && !INSIDE)
         gPad->SetCursor(kCross);

      break;

   case kButton1Motion:

      wx = wy = 0;

      if (L) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px1 += px - pxold;
         if (px1 > px2-kMinSize) { px1 = px2-kMinSize; wx = px1; }
         if (px1 < px1p) { px1 = px1p; wx = px1; }
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      }
      if (R) {
         if (!ropaque) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         px2 += px - pxold;
         if (px2 < px1+kMinSize) { px2 = px1+kMinSize; wx = px2; }
         if (px2 > px2p) { px2 = px2p; wx = px2; }
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

      if ((INSIDE && opaque) || (fResizing && ropaque)) {
         event = kButton1Up;
         doing_again = kTRUE;
         goto again;
      }

      break;

   case kButton1Up:

      if (px1 < 0 ) break;
      if (px == pxdold && py == pydold) break;
      if (L || R || INSIDE) {
         fX1 = gPad->AbsPixeltoX(px1);
         fY1 = gPad->AbsPixeltoY(py1);
         fX2 = gPad->AbsPixeltoX(px2);
         fY2 = gPad->AbsPixeltoY(py2);
      }

      if (INSIDE) {
         // if it was not a pad that was moved then it must have been
         // a box or something like that so we have to redraw the pad
         gPad->Modified(kTRUE);
         if (!doing_again) gPad->SetCursor(kCross);
      }

      if (L || R)
         gPad->Modified(kTRUE);

      // In case pave coordinates have been modified, recompute NDC coordinates
      {
         Float_t dpx  = gPad->GetX2() - gPad->GetX1();
         Float_t dpy  = gPad->GetY2() - gPad->GetY1();
         Float_t xp1  = gPad->GetX1();
         Float_t yp1  = gPad->GetY1();
         fX1NDC = (fX1-xp1)/dpx;
         fY1NDC = (fY1-yp1)/dpy;
         fX2NDC = (fX2-xp1)/dpx;
         fY2NDC = (fY2-yp1)/dpy;
      }
      gVirtualX->SetLineColor(-1);
      gVirtualX->SetLineWidth(-1);
   }
}

//______________________________________________________________________________
void TPaveVar::Merge(Option_t *option)
{
//  Merge all TPaveVars overlapping with this TPaveVar on the TTeeeViewer
//     option = "AND" all paves are ANDed in a new TPaveVar
//     option = "OR"  all paves are ORed in a new TPaveVar

   if (fViewer == 0) return;
   //Scan all TPaveVars, looking for the ones overlapping with this
   TString opt = option;
   opt.ToUpper();
   TIter next(fViewer->GetListOfPrimitives());
   TPaveVar *pl;
   TObject *obj;
   char *name = new char[2000];
   sprintf(name,"(%s)",GetLabel());
   while ((obj=next())) {
      if (obj->InheritsFrom(TPaveVar::Class())) {
         pl = (TPaveVar*)obj;
         if (pl == this) continue;
         // check if at least one corner overlaps with this
         if ((pl->GetX1() > fX1 && pl->GetX1() < fX2) ||
             (pl->GetX2() > fX1 && pl->GetX2() < fX2)) {
              if ((pl->GetY1() > fY1 && pl->GetY1() < fY2) ||
                  (pl->GetY2() > fY1 && pl->GetY2() < fY2)) {
                   if (opt.Contains("AND")) strcat(name,"&&");
                   if (opt.Contains("OR"))  strcat(name,"||");
                   strcat(name,"(");
                   strcat(name,pl->GetLabel());
                   strcat(name,")");
              }
         }
      }
   }
   pl = fViewer->CreateNewVar(name);
   delete [] name;
}

//______________________________________________________________________________
void TPaveVar::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TPaveVar::Class())) {
       out<<"   ";
   } else {
       out<<"   TPaveVar *";
   }
   out<<"pvar = new TPaveVar("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<quote<<fLabel<<quote<<","<<quote<<fOption<<quote<<");"<<endl;

   SaveFillAttributes(out,"pvar",0,1001);
   SaveLineAttributes(out,"pvar",1,1,1);
   SaveTextAttributes(out,"pvar",22,0,1,62,0);

   out<<"   pvar->Draw();"<<endl;
}
