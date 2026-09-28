// @(#)root/graf:$Id$
// Author: Sergey Linev   18/09/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBoxInteractive.h"

#include "TMath.h"

/** \class TBoxInterctive
 *
 * Internal interactive class to implement moving of box-like objects
 * in the ExecuteEvent
 */

///////////////////////////////////////////////////////////////////////////////
/// Constructor.

TBoxInteractive::TBoxInteractive(Bool_t bx, Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   isBox = bx;
   oldX1 = x1;
   oldY1 = y1;
   oldX2 = x2;
   oldY2 = y2;
}

///////////////////////////////////////////////////////////////////////////////
/// Calculate pixel coordinates

void TBoxInteractive::CalcPixelCoord(const TVirtualPad &parent, Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   px1 = parent.XtoAbsPixel(isBox ? parent.XtoPad(x1) : x1);
   py1 = parent.YtoAbsPixel(isBox ? parent.YtoPad(y1) : y1);
   px2 = parent.XtoAbsPixel(isBox ? parent.XtoPad(x2) : x2);
   py2 = parent.YtoAbsPixel(isBox ? parent.YtoPad(y2) : y2);
   if (px1 > px2)
      std::swap(px1, px2);
   if (py1 < py2)
      std::swap(py1, py2);
}

///////////////////////////////////////////////////////////////////////////////
/// Select corner or side for the inetarctive action

Bool_t TBoxInteractive::SelectCorner(Int_t px, Int_t py, Bool_t canX, Bool_t canY)
{
   constexpr Int_t kMaxDiff = 7;

   if (TMath::Abs(px - px1) <= kMaxDiff && TMath::Abs(py - py2) <= kMaxDiff) {
      fMode = canX && canY ? pA : (canX ? pL : pTop);
   } else if (TMath::Abs(px - px2) <= kMaxDiff && TMath::Abs(py - py2) <= kMaxDiff) {
      fMode = canX && canY ? pB : (canX ? pR : pTop);
   } else if (TMath::Abs(px - px2) <= kMaxDiff && TMath::Abs(py - py1) <= kMaxDiff) {
      fMode = canX && canY ? pC : (canX ? pR : pBot);
   } else if (TMath::Abs(px - px1) <= kMaxDiff && TMath::Abs(py - py1) <= kMaxDiff) {
      fMode = canX && canY ? pD : (canX ? pL : pBot);
   } else if ((px > px1 + kMaxDiff && px < px2 - kMaxDiff) && TMath::Abs(py - py2) < kMaxDiff) {
      fMode = canY ? pTop : pNone;
   } else if ((px > px1 + kMaxDiff && px < px2 - kMaxDiff) && TMath::Abs(py - py1) < kMaxDiff) {
      fMode = canY ? pBot : pNone;
   } else if ((py > py2 + kMaxDiff && py < py1 - kMaxDiff) && TMath::Abs(px - px1) < kMaxDiff) {
      fMode = canX ? pL : pNone;
   } else if ((py > py2 + kMaxDiff && py < py1 - kMaxDiff) && TMath::Abs(px - px2) < kMaxDiff) {
      fMode = canX ? pR : pNone;
   } else if ((px > px1 + kMaxDiff && px < px2 - kMaxDiff) && (py > py2 + kMaxDiff && py < py1 - kMaxDiff)) {
      dpx1 = px - px1; // cursor position relative to top-left corner
      dpy2 = py - py2;
      fMode = pINSIDE;
   } else {
      fMode = pNone;
   }
   return fMode != pNone;
}

///////////////////////////////////////////////////////////////////////////////
/// Select diamond corner - or inside area
/// Returns true if any mode was selected
/// Can be reused for diamond and ellipse

Bool_t TBoxInteractive::SelectDiamondCorner(Int_t px, Int_t py, Bool_t testInside)
{
   constexpr Int_t kMaxDiff = 7;

   Double_t xd[4] = {(px1 + px2) / 2., 1. * px1, (px1 + px2) / 2., 1. * px2};
   Double_t yd[4] = {1. * py2, (py1 + py2) / 2., 1. * py1, (py1 + py2) / 2.};
   EMode modes[4] = {pTop, pL, pBot, pR};

   fMode = pNone;

   for (Int_t i = 0; i < 4; ++i)
      if ((TMath::Abs(px - xd[i]) < kMaxDiff) && (TMath::Abs(py - yd[i]) < kMaxDiff)) {
         fMode = modes[i];
         break;
      }

   // use Double_t signature of IsInside because of rounding problems with Int_t
   if ((fMode == pNone) && (!testInside || TMath::IsInside((Double_t)px, (Double_t)py, 4, xd, yd))) {
      fMode = pINSIDE;
      dpx1 = px - px1; // cursor position relative to top-left corner
      dpy2 = py - py2;
   }

   return fMode != pNone;
}

///////////////////////////////////////////////////////////////////////////////
/// Return true if doing resize

Bool_t TBoxInteractive::IsResizing() const
{
    return fMode != pNone && fMode != pINSIDE;
}

///////////////////////////////////////////////////////////////////////////////
/// Return selected opaque mode

Bool_t TBoxInteractive::IsOpaque(const TVirtualPad &parent) const
{
   return IsResizing() ? parent.OpaqueResizing() : parent.OpaqueMoving();
}

///////////////////////////////////////////////////////////////////////////////
/// Process change of moue position during dragging

Bool_t TBoxInteractive::ProcessMouseMove(const TVirtualPad &parent, Int_t px, Int_t py, Bool_t canX, Bool_t canY, Double_t aspectRatio)
{
   constexpr Int_t kMinSize = 20;

   Int_t px1p = parent.XtoAbsPixel(parent.GetX1()) + parent.GetBorderSize();
   Int_t py1p = parent.YtoAbsPixel(parent.GetY1()) - parent.GetBorderSize();
   Int_t px2p = parent.XtoAbsPixel(parent.GetX2()) - parent.GetBorderSize();
   Int_t py2p = parent.YtoAbsPixel(parent.GetY2()) + parent.GetBorderSize();
   if (px1p > px2p)
      std::swap(px1p, px2p);
   if (py1p < py2p)
      std::swap(py1p, py2p);

   enum { noAdjust = 1111, adjWidth = 11, adjBot = -1, adjHeight = 0, adjTop = 1 } choise = noAdjust;
   Int_t prevpx1 = px1, prevpx2 = px2,
         prevpy1 = py1, prevpy2 = py2;

   switch (fMode) {
   case pNone: return kFALSE;
   case pA:
      choise = adjBot;
      px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
      py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
      break;
   case pB:
      choise = adjBot;
      px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
      py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
      break;
   case pC:
      choise = adjTop;
      px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
      py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
      break;
   case pD:
      choise = adjTop;
      px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
      py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
      break;
   case pTop:
      choise = adjWidth;
      py2 = TMath::Max(py2p, TMath::Min(py, py1 - kMinSize));
      break;
   case pBot:
      choise = adjWidth;
      py1 = TMath::Min(py1p, TMath::Max(py, py2 + kMinSize));
      break;
   case pL:
      choise = adjHeight;
      px1 = TMath::Max(px1p, TMath::Min(px, px2 - kMinSize));
      break;
   case pR:
      choise = adjHeight;
      px2 = TMath::Min(px2p, TMath::Max(px, px1 + kMinSize));
      break;
   case pINSIDE:
      if (canX) {
         px2 += px - dpx1 - px1;
         px1 = px - dpx1;
         if (px1 < px1p) {
            px2 += px1p - px1;
            px1 = px1p;
         }
         if (px2 > px2p) {
            px1 -= px2 - px2p;
            px2 = px2p;
         }
      }
      if (canY) {
         py1 += py - dpy2 - py2;
         py2 = py - dpy2;
         if (py1 > py1p) {
            py2 -= py1 - py1p;
            py1 = py1p;
         }
         if (py2 < py2p) {
            py1 += py2p - py2;
            py2 = py2p;
         }
      }
      break;
   }

   if (!aspectRatio || (choise == noAdjust))
      return kTRUE;

   if (choise == adjWidth) {
      Int_t dx = parent.UtoPixel(aspectRatio * (py1 - py2) / parent.VtoPixel(0));
      Int_t npx1 = (px1 + px2) / 2 - dx / 2, npx2 = npx1 + dx;
      if ((npx1 >= px1p) && (npx2 <= px2p)) {
         px1 = npx1;
         px2 = npx2;
         return kTRUE;
      }
   } else {
      Int_t dy = parent.VtoPixel(1. - (0. + px2 - px1) / parent.UtoPixel(1.) / aspectRatio);
      Int_t npy1 = py1, npy2 = py2;
      if (choise == adjBot)
         npy2 = py1 - dy;
      else if (choise == adjTop)
         npy1 = py2 + dy;
      else if (choise == adjHeight) {
         npy2 = (py1 + py2) / 2 - dy / 2;
         npy1 = npy2 + dy;
      }
      if ((npy1 <= py1p) && (npy2 >= py2p)) {
         py1 = npy1;
         py2 = npy2;
         return kTRUE;
      }
   }
   px1 = prevpx1;
   px2 = prevpx2;
   py1 = prevpy1;
   py2 = prevpy2;

   return kFALSE;
}

///////////////////////////////////////////////////////////////////////////////
/// Paint object outline.

void TBoxInteractive::PaintOutline(TVirtualPad &parent)
{
   // "i" is interactive painting, "l" is line, "box" is id
   parent.PaintBox(newX1, newY1, newX2, newY2, "ilbox");
}

///////////////////////////////////////////////////////////////////////////////
/// Apply pixel coordinates changes back to box coordinates

void TBoxInteractive::ApplyChanges(TVirtualPad &parent)
{
   newX1 = parent.AbsPixeltoX(px1);
   newY1 = parent.AbsPixeltoY(py1);
   newX2 = parent.AbsPixeltoX(px2);
   newY2 = parent.AbsPixeltoY(py2);

   if (!IsOpaque(parent)) {
      TAttLine(kBlack, 1, 1).ModifyOn(parent);
      PaintOutline(parent);
      parent.UpdateAsync();
   }

   if (isBox) {
      newX1 = parent.PadtoX(newX1);
      newX2 = parent.PadtoX(newX2);
      newY1 = parent.PadtoY(newY1);
      newY2 = parent.PadtoY(newY2);
   }
};

///////////////////////////////////////////////////////////////////////////////
/// Set cursor for current mode

void TBoxInteractive::SetCursor(TVirtualPad &parent, Bool_t is_down) const
{
   switch (fMode) {
   case pNone: parent.SetCursor(kCross); break;
   case pA: parent.SetCursor(kTopLeft); break;
   case pB: parent.SetCursor(kTopRight); break;
   case pC: parent.SetCursor(kBottomRight); break;
   case pD: parent.SetCursor(kBottomLeft); break;
   case pTop: parent.SetCursor(kTopSide); break;
   case pL: parent.SetCursor(kLeftSide); break;
   case pR: parent.SetCursor(kRightSide); break;
   case pBot: parent.SetCursor(kBottomSide); break;
   case pINSIDE: parent.SetCursor(is_down ? kMove : kCross); break;
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Return character for selecting guide action

char TBoxInteractive::GetGuideChar() const
{
   switch (fMode) {
   case pINSIDE: return 'i';
   case pTop: return 't';
   case pBot: return 'b';
   case pL: return 'l';
   case pR: return 'r';
   case pA: return '1';
   case pB: return '2';
   case pC: return '3';
   case pD: return '4';
   default: return 0; // not involved
   }
}
