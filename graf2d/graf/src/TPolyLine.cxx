// @(#)root/graf:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include <vector>
#include "TROOT.h"
#include "TBuffer.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TVirtualPadPainter.h"
#include "TAttMarker.h"
#include "TPolyLine.h"


/** \class TPolyLine
\ingroup BasicGraphics

Defined by an array on N points in a 2-D space.

One can draw the contour of the polyline or/and its fill area.
Example:
Begin_Macro(source)
{
   Double_t x[5] = {.2,.7,.6,.25,.2};
   Double_t y[5] = {.5,.1,.9,.7,.5};
   TPolyLine *pline = new TPolyLine(5,x,y);
   pline->SetFillColor(38);
   pline->SetLineColor(2);
   pline->SetLineWidth(4);
   pline->Draw("f");
   pline->Draw();
}
End_Macro
*/

////////////////////////////////////////////////////////////////////////////////
/// PolyLine default constructor.

TPolyLine::TPolyLine()
{
}

////////////////////////////////////////////////////////////////////////////////
/// PolyLine normal constructor without initialisation.
/// Allocates n points.

TPolyLine::TPolyLine(Int_t n, Option_t *option)
      :TObject(), TAttLine(), TAttFill()
{
   fOption = option;
   if (n <= 0)
      return;

   fN = n;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
}

////////////////////////////////////////////////////////////////////////////////
/// PolyLine normal constructor (single precision).
/// Makes n points with (x, y) coordinates from x and y.

TPolyLine::TPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option)
      :TObject(), TAttLine(), TAttFill()
{
   fOption = option;
   if (n <= 0)
      return;

   fN = n;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
   if (!x || !y) return;
   for (Int_t i = 0; i < fN; i++) {
      fX[i] = x[i];
      fY[i] = y[i];
   }
   fLastPoint = fN-1;
}

////////////////////////////////////////////////////////////////////////////////
/// PolyLine normal constructor (double precision).
/// Makes n points with (x, y) coordinates from x and y.

TPolyLine::TPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
      :TObject(), TAttLine(), TAttFill()
{
   fOption = option;
   if (n <= 0)
      return;
   fN = n;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
   if (!x || !y) return;
   for (Int_t i=0; i<fN;i++) {
      fX[i] = x[i];
      fY[i] = y[i];
   }
   fLastPoint = fN-1;
}

////////////////////////////////////////////////////////////////////////////////
///assignment operator

TPolyLine& TPolyLine::operator=(const TPolyLine& pl)
{
   if(this != &pl)
      pl.TPolyLine::Copy(*this);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// PolyLine default destructor.

TPolyLine::~TPolyLine()
{
   if (fX) delete [] fX;
   if (fY) delete [] fY;
}

////////////////////////////////////////////////////////////////////////////////
/// PolyLine copy constructor.

TPolyLine::TPolyLine(const TPolyLine &polyline) : TObject(polyline), TAttLine(polyline), TAttFill(polyline)
{
   polyline.TPolyLine::Copy(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy this polyline to polyline.

void TPolyLine::Copy(TObject &obj) const
{
   TObject::Copy(obj);
   TAttLine::Copy(((TPolyLine&)obj));
   TAttFill::Copy(((TPolyLine&)obj));
   ((TPolyLine&)obj).fN = fN;
   delete [] ((TPolyLine&)obj).fX;
   delete [] ((TPolyLine&)obj).fY;
   if (fN > 0) {
      ((TPolyLine&)obj).fX = new Double_t[fN];
      ((TPolyLine&)obj).fY = new Double_t[fN];
      for (Int_t i = 0; i < fN; i++) {
         ((TPolyLine &)obj).fX[i] = fX[i];
         ((TPolyLine &)obj).fY[i] = fY[i];
      }
   } else {
      ((TPolyLine&)obj).fX = nullptr;
      ((TPolyLine&)obj).fY = nullptr;
   }
   ((TPolyLine&)obj).fOption = fOption;
   ((TPolyLine&)obj).fLastPoint = fLastPoint;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns closest distance in pixels from point (px, py) to a polyline.
///
/// First looks for distances to the points of the polyline.  Stops search
/// and returns if a vertex of the polyline is found to be closer than 10
/// pixels.  Thus the return value may depend on the ordering of points
/// in the polyline.
///
/// Then looks for distances to the lines of the polyline.  There is no
/// arbitrary cutoff; any distance may be found.
///
/// Finally checks whether (px, py) is inside a closed and filled polyline.
/// (Must be EXACTLY closed.  "Filled" means fill color and fill style are
/// both non-zero.) If so, returns zero.
///
/// Returns 9999 if the polyline has no points.

Int_t TPolyLine::DistancetoPrimitive(Int_t px, Int_t py)
{
   const Int_t big = 9999;
   if (!gPad) return big;
   const Int_t kMaxDiff = 10;

   // check if point is near one of the points
   Int_t i, pxp, pyp, d;
   Int_t distance = big;
   if (Size() <= 0) return distance;

   for (i=0;i<Size();i++) {
      pxp = gPad->XtoAbsPixel(gPad->XtoPad(fX[i]));
      pyp = gPad->YtoAbsPixel(gPad->YtoPad(fY[i]));
      d   = TMath::Abs(pxp-px) + TMath::Abs(pyp-py);
      if (d < distance) distance = d;
   }
   if (distance < kMaxDiff) return distance;

   // check if point is near one of the connecting lines
   for (i=0;i<Size()-1;i++) {
      d = DistancetoLine(px, py, gPad->XtoPad(fX[i]), gPad->YtoPad(fY[i]), gPad->XtoPad(fX[i+1]), gPad->YtoPad(fY[i+1]));
      if (d < distance) distance = d;
   }

   // in case of a closed and filled polyline, check if we are inside
   if (fFillColor && fFillStyle && fX[0] == fX[fLastPoint] && fY[0] == fY[fLastPoint]) {
      if (TMath::IsInside(gPad->AbsPixeltoX(px),gPad->AbsPixeltoY(py),fLastPoint+1,fX,fY)) distance = 0;
   }
   return distance;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this polyline with its current attributes.

void TPolyLine::Draw(Option_t *option)
{
   AppendPad(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this polyline with new coordinates.

TPolyLine *TPolyLine::DrawPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   TPolyLine *newpolyline = new TPolyLine(n,x,y);
   TAttLine::Copy(*newpolyline);
   TAttFill::Copy(*newpolyline);
   newpolyline->fOption = fOption;
   newpolyline->SetBit(kCanDelete);
   newpolyline->AppendPad(option);
   return newpolyline;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
///  This member function is called when a polyline is clicked with the locator
///
///  If Left button clicked on one of the line end points, this point
///     follows the cursor until button is released.
///
///  if Middle button clicked, the line is moved parallel to itself
///     until the button is released.

void TPolyLine::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (!gPad || !gPad->IsEditable()) return;

   auto &parent = *gPad;

   constexpr Int_t kMaxDiff = 10;
   Bool_t opaque  = parent.OpaqueMoving();
   static Int_t sdx, sdy, ipoint;
   static Bool_t first_move;

   Int_t np = Size();
   Bool_t is_last_same = (np > 1) && (fX[0] == fX[np-1]) && (fY[0] == fY[np-1]);
   if (is_last_same)
      np--;

   auto paint_hollow = [this,&parent,is_last_same] () {
      auto pp = parent.GetPainter();
      pp->SetAttLine({1,1,GetLineWidth()});
      Double_t *x = fX, *y = fY;
      if (TestBit(kPolyLineNDC)) {
         pp->DrawPolyLineNDC(Size(), x, y);
      } else {
         std::vector<Double_t> xx, yy;
         if (parent.GetLogx()) {
            xx.resize(Size());
            for (Int_t ix = 0; ix < Size(); ix++)
               xx[ix] = parent.XtoPad(x[ix]);
            x = xx.data();
         }
         if (parent.GetLogy()) {
            yy.resize(Size());
            for (Int_t iy = 0; iy < Size(); iy++)
               yy[iy] = parent.YtoPad(y[iy]);
            y = yy.data();
         }
         pp->DrawPolyLine(Size(), x, y);

         pp->SetAttMarker({1,25,1});
         pp->DrawPolyMarker(is_last_same ? Size()-1 : Size(), x, y);
      }
   };

   auto get_point = [this, &parent](Int_t i, Int_t &pntx, Int_t &pnty) {
      if (TestBit(kPolyLineNDC)) {
         pntx = parent.UtoAbsPixel(fX[i]);
         pnty = parent.VtoAbsPixel(fY[i]);
      } else {
         pntx = parent.XtoAbsPixel(parent.XtoPad(fX[i]));
         pnty = parent.YtoAbsPixel(parent.YtoPad(fY[i]));
      }
   };

   auto set_point = [this, &parent](Int_t i, Int_t pntx, Int_t pnty) {
      if (TestBit(kPolyLineNDC)) {
         Double_t ww = parent.GetWw();
         Double_t wndc = parent.GetAbsWNDC();
         Double_t wh = parent.GetWh();
         Double_t hndc = parent.GetAbsHNDC();
         fX[i] = ww > 0 && wndc > 0 ? (pntx / ww - parent.GetAbsXlowNDC()) / wndc : 0.;
         fY[i] = wh > 0 && hndc > 0 ? ((1. - pnty / wh) - parent.GetAbsYlowNDC()) / hndc : 0.;
      } else {
         fX[i] = parent.PadtoX(parent.AbsPixeltoX(pntx));
         fY[i] = parent.PadtoY(parent.AbsPixeltoY(pnty));
      }
   };

   switch (event) {

   case kArrowKeyPress:
   case kButton1Down:
      // No break !!!
   case kMouseMotion: {

      Int_t minDiff = kMaxDiff;
      ipoint = -1;
      for (Int_t i = 0; i < np; i++) {
         Int_t pxp, pyp;
         get_point(i, pxp, pyp);
         if (i == 0) {
            sdx = pxp - px;
            sdy = pyp - py;
         }
         Int_t d = TMath::Abs(pxp - px) + TMath::Abs(pyp - py);
         if (d < minDiff) {
            ipoint = i;
            minDiff = d;
            sdx = pxp - px;
            sdy = pyp - py;
         }
      }
      first_move = kTRUE;
      if (ipoint < 0)
         parent.SetCursor(kMove);
      else
         parent.SetCursor(kHand);
      break;
   }

   case kButton1Motion:
      if (!opaque && !first_move)
         paint_hollow();

      if (ipoint < 0) {
         Int_t pxp0, pyp0, pxp, pyp;
         // move all points
         for (Int_t i = 0; i < np; i++) {
            get_point(i, pxp, pyp);
            if (i == 0) {
               pxp0 = pxp;
               pyp0 = pyp;
            }
            set_point(i, px + sdx + pxp - pxp0, py + sdy + pyp - pyp0);
         }
      } else {
         // move only selected point
         set_point(ipoint, px + sdx, py + sdy);
      }
      if (is_last_same) {
         fX[np] = fX[0];
         fY[np] = fY[0];
      }

      first_move = kFALSE;
      if (!opaque)
         paint_hollow();
      else
         parent.ModifiedUpdate();
      break;

   case kButton1Up:
      if (!opaque)
         parent.ModifiedUpdate();
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// List this polyline with its attributes.
/// The option string is ignored.

void TPolyLine::ls(Option_t *) const
{
   TROOT::IndentLevel();
   printf("TPolyLine  N=%d\n",fN);
}

////////////////////////////////////////////////////////////////////////////////
/// Merge polylines in the collection in this polyline

Int_t TPolyLine::Merge(TCollection *li)
{
   if (!li) return 0;
   TIter next(li);

   //first loop to count the number of entries
   TPolyLine *pl;
   Int_t npoints = 0;
   while ((pl = (TPolyLine*)next())) {
      if (!pl->InheritsFrom(TPolyLine::Class())) {
         Error("Add","Attempt to add object of class: %s to a %s",pl->ClassName(),this->ClassName());
         return -1;
      }
      npoints += pl->Size();
   }

   //extend this polyline to hold npoints
   if (npoints > 1) SetPoint(npoints-1,0,0);

   //merge all polylines
   next.Reset();
   while ((pl = (TPolyLine*)next())) {
      Int_t np = pl->Size();
      Double_t *x = pl->GetX();
      Double_t *y = pl->GetY();
      for (Int_t i=0;i<np;i++) {
         SetPoint(i,x[i],y[i]);
      }
   }

   return npoints;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this polyline with its current attributes.

void TPolyLine::Paint(Option_t *option)
{
   if (TestBit(kPolyLineNDC))
      PaintPolyLineNDC(fLastPoint+1, fX, fY, option && *option ? option : fOption.Data());
   else
      PaintPolyLine(fLastPoint+1, fX, fY, option && *option ? option : fOption.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this polyline with new coordinates.
///
///  If option = 'f' or 'F' the fill area is drawn.
///  The default is to draw the lines only.

void TPolyLine::PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   if (!gPad || n <= 0) return;
   TAttLine::Modify();  //Change line attributes only if necessary
   TAttFill::Modify();  //Change fill area attributes only if necessary
   std::vector<Double_t> xx, yy;
   if (gPad->GetLogx()) {
      xx.resize(n);
      for (Int_t ix = 0; ix < n; ix++)
         xx[ix] = gPad->XtoPad(x[ix]);
      x = xx.data();
   }
   if (gPad->GetLogy()) {
      yy.resize(n);
      for (Int_t iy = 0; iy < n; iy++)
         yy[iy] = gPad->YtoPad(y[iy]);
      y = yy.data();
   }
   if (option && (*option == 'f' || *option == 'F'))
      gPad->PaintFillArea(n, x, y, option);
   else
      gPad->PaintPolyLine(n, x, y, option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this polyline with new coordinates in NDC.

void TPolyLine::PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   TAttLine::Modify(); // Change line attributes only if necessary
   TAttFill::Modify(); // Change fill area attributes only if necessary
   if (option && (*option == 'f' || *option == 'F'))
      gPad->PaintFillAreaNDC(n, x, y, option);
   else
      gPad->PaintPolyLineNDC(n, x, y, option);
}

////////////////////////////////////////////////////////////////////////////////
/// Dump this polyline with its attributes.
/// The option string is ignored.

void TPolyLine::Print(Option_t *) const
{
   printf("PolyLine  N=%d\n",fN);
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TPolyLine::SavePrimitive(std::ostream &out, Option_t *option)
{
   TString args;
   if (Size() > 0) {
      TString arrx = SavePrimitiveVector(out, "polyline", Size(), fX, kTRUE);
      TString arry = SavePrimitiveVector(out, "polyline", Size(), fY);
      args.Form("%d, %s.data(), %s.data(), ", Size(), arrx.Data(), arry.Data());
   } else {
      args.Form("%d, ", fN);
   }
   args.Append(TString::Format("\"%s\"", TString(fOption).ReplaceSpecialCppChars().Data()));

   SavePrimitiveConstructor(out, Class(), "polyline", args, Size() == 0);
   SaveFillAttributes(out, "polyline", -1, -1);
   SaveLineAttributes(out, "polyline", 1, 1, 1);

   if (!option || !strstr(option, "nodraw"))
      out << "   polyline->Draw(\"" << TString(option).ReplaceSpecialCppChars() << "\");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Set NDC mode on if isNDC = kTRUE, off otherwise

void TPolyLine::SetNDC(Bool_t isNDC)
{
   ResetBit(kPolyLineNDC);
   if (isNDC) SetBit(kPolyLineNDC);
}

////////////////////////////////////////////////////////////////////////////////
/// Set point following LastPoint to x, y.
/// Returns index of the point (new last point).

Int_t TPolyLine::SetNextPoint(Double_t x, Double_t y)
{
   fLastPoint++;
   SetPoint(fLastPoint, x, y);
   return fLastPoint;
}

////////////////////////////////////////////////////////////////////////////////
/// Set point number n to (x, y)
/// If n is greater than the current size, the arrays are automatically
/// extended.

void TPolyLine::SetPoint(Int_t n, Double_t x, Double_t y)
{
   if (n < 0) return;
   if (!fX || !fY || n >= fN) {
      // re-allocate the object
      Int_t newN = TMath::Max(2*fN,n+1);
      Double_t *savex = new Double_t [newN];
      Double_t *savey = new Double_t [newN];
      if (fX && fN){
         memcpy(savex,fX,fN*sizeof(Double_t));
         memset(&savex[fN],0,(newN-fN)*sizeof(Double_t));
         delete [] fX;
      }
      if (fY && fN){
         memcpy(savey,fY,fN*sizeof(Double_t));
         memset(&savey[fN],0,(newN-fN)*sizeof(Double_t));
         delete [] fY;
      }
      fX = savex;
      fY = savey;
      fN = newN;
   }
   fX[n] = x;
   fY[n] = y;
   fLastPoint = TMath::Max(fLastPoint, n);
}

////////////////////////////////////////////////////////////////////////////////
/// Resize this polyline to size n.
/// If n <= 0 the current arrays of points are deleted.
/// If n is greater than the current size, the new points are set to (0, 0)

void TPolyLine::SetPolyLine(Int_t n)
{
   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      delete [] fX;
      delete [] fY;
      fX = fY = nullptr;
      return;
   }
   if (n < fN) {
      fN = n;
      fLastPoint = n - 1;
   } else {
      SetPoint(n-1,0,0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set new values for this polyline (single precision).
///
/// If n <= 0 the current arrays of points are deleted.

void TPolyLine::SetPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option)
{
   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      delete [] fX;
      delete [] fY;
      fX = fY = nullptr;
      return;
   }
   fN =n;
   if (fX) delete [] fX;
   if (fY) delete [] fY;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
   for (Int_t i=0; i<fN;i++) {
      if (x) fX[i] = (Double_t)x[i];
      if (y) fY[i] = (Double_t)y[i];
   }
   fOption = option;
   fLastPoint = fN-1;
}

////////////////////////////////////////////////////////////////////////////////
/// Set new values for this polyline (double precision).
///
/// If n <= 0 the current arrays of points are deleted.

void TPolyLine::SetPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      delete [] fX;
      delete [] fY;
      fX = fY = nullptr;
      return;
   }
   fN =n;
   if (fX) delete [] fX;
   if (fY) delete [] fY;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
   for (Int_t i=0; i<fN;i++) {
      if (x) fX[i] = x[i];
      if (y) fY[i] = y[i];
   }
   fOption = option;
   fLastPoint = fN-1;
}

////////////////////////////////////////////////////////////////////////////////
/// Stream a class object.

void TPolyLine::Streamer(TBuffer &b)
{
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         b.ReadClassBuffer(TPolyLine::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(b);
      TAttLine::Streamer(b);
      TAttFill::Streamer(b);
      b >> fN;
      fX = new Double_t[fN];
      fY = new Double_t[fN];
      Float_t *x = new Float_t[fN];
      Float_t *y = new Float_t[fN];
      b.ReadFastArray(x,fN);
      b.ReadFastArray(y,fN);
      for (Int_t i=0;i<fN;i++) {
         fX[i] = x[i];
         fY[i] = y[i];
      }
      fOption.Streamer(b);
      b.CheckByteCount(R__s, R__c, TPolyLine::IsA());
      //====end of old versions

      delete [] x;
      delete [] y;
   } else {
      b.WriteClassBuffer(TPolyLine::Class(),this);
   }
}
