// @(#)root/graf:$Name:  $:$Id: TPolyLine.cxx,v 1.2 2000/06/13 11:16:56 brun Exp $
// Author: Rene Brun   12/12/94

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
#include "TVirtualPad.h"
#include "TPolyLine.h"

ClassImp(TPolyLine)

//______________________________________________________________________________
//
//  a PolyLine is defined by an array on N points in a 2-D space.

//______________________________________________________________________________
TPolyLine::TPolyLine(): TObject()
{
//*-*-*-*-*-*-*-*-*-*-*PolyLine default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================
   fX = 0;
   fY = 0;
}

//______________________________________________________________________________
TPolyLine::TPolyLine(Int_t n, Option_t *option)
      :TObject(), TAttLine(), TAttFill()
{
//*-*-*-**-*-*PolyLine normal constructor without initialisation*-*-*-*-*-*-*-*
//*-*         ==================================================
   fN = n;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
   fOption = option;
}

//______________________________________________________________________________
TPolyLine::TPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option)
      :TObject(), TAttLine(), TAttFill()
{
//*-*-*-*-*-*-*-*-*-*-*PolyLine normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================
   fN = n;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
   if (!x || !y) return;
   for (Int_t i=0; i<fN;i++) { fX[i] = x[i]; fY[i] = y[i];}
   fOption = option;
}

//______________________________________________________________________________
TPolyLine::TPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
      :TObject(), TAttLine(), TAttFill()
{
//*-*-*-*-*-*-*-*-*-*-*PolyLine normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================
   fN = n;
   fX = new Double_t[fN];
   fY = new Double_t[fN];
   if (!x || !y) return;
   for (Int_t i=0; i<fN;i++) { fX[i] = x[i]; fY[i] = y[i];}
   fOption = option;
}

//______________________________________________________________________________
TPolyLine::~TPolyLine()
{
//*-*-*-*-*-*-*-*-*-*-*PolyLine default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================
   if (fX) delete [] fX;
   if (fY) delete [] fY;
}

//______________________________________________________________________________
TPolyLine::TPolyLine(const TPolyLine &polyline)
{
   ((TPolyLine&)polyline).Copy(*this);
}

//______________________________________________________________________________
void TPolyLine::Copy(TObject &obj)
{
//*-*-*-*-*-*-*Copy this polyline to polyline*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*          ==============================

   TObject::Copy(obj);
   TAttLine::Copy(((TPolyLine&)obj));
   TAttFill::Copy(((TPolyLine&)obj));
   ((TPolyLine&)obj).fN = fN;
   ((TPolyLine&)obj).fX = new Double_t[fN];
   ((TPolyLine&)obj).fY = new Double_t[fN];
   for (Int_t i=0; i<fN;i++)  {((TPolyLine&)obj).fX[i] = fX[i]; ((TPolyLine&)obj).fY[i] = fY[i];}
   ((TPolyLine&)obj).fOption = fOption;
}

//______________________________________________________________________________
Int_t TPolyLine::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*Compute distance from point px,py to a polyline*-*-*-*-*-*
//*-*              ===============================================
//  Compute the closest distance of approach from point px,py to each segment
//  of the polyline.
//  Returns when the distance found is below DistanceMaximum.
//  The distance is computed in pixels units.
//
   const Int_t big = 9999;
   const Int_t kMaxDiff = 10;

//*-*- check if point is near one of the points
   Int_t i, pxp, pyp, d;
   Int_t distance = big;

   for (i=0;i<fN;i++) {
      pxp = gPad->XtoAbsPixel(gPad->XtoPad(fX[i]));
      pyp = gPad->YtoAbsPixel(gPad->YtoPad(fY[i]));
      d   = TMath::Abs(pxp-px) + TMath::Abs(pyp-py);
      if (d < distance) distance = d;
   }
   if (distance < kMaxDiff) return distance;

//*-*- check if point is near one of the connecting lines
   for (i=0;i<fN-1;i++) {
      d = DistancetoLine(px, py, gPad->XtoPad(fX[i]), gPad->YtoPad(fY[i]), gPad->XtoPad(fX[i+1]), gPad->YtoPad(fY[i+1]));
      if (d < distance) distance = d;
   }
   return distance;
}

//______________________________________________________________________________
void TPolyLine::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Draw this polyline with its current attributes*-*-*-*-*-*-*
//*-*              ==============================================

   AppendPad(option);

}

//______________________________________________________________________________
void TPolyLine::DrawPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
//*-*-*-*-*-*-*-*-*Draw this polyline with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*              ========================================
   TPolyLine *newpolyline = new TPolyLine();
   newpolyline->fN =n;
   newpolyline->fX = new Double_t[fN];
   newpolyline->fY = new Double_t[fN];
   for (Int_t i=0; i<fN;i++) { newpolyline->fX[i] = x[i]; newpolyline->fY[i] = y[i];}
   TAttLine::Copy(*newpolyline);
   TAttFill::Copy(*newpolyline);
   newpolyline->fOption = fOption;
   newpolyline->SetBit(kCanDelete);
   newpolyline->AppendPad(option);
}

//______________________________________________________________________________
void TPolyLine::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a polyline is clicked with the locator
//
//  If Left button clicked on one of the line end points, this point
//     follows the cursor until button is released.
//
//  if Middle button clicked, the line is moved parallel to itself
//     until the button is released.
//
   Int_t i, d;
   Double_t xmin, xmax, ymin, ymax, dx, dy, dxr, dyr;
   const Int_t kMaxDiff = 10;
   static Bool_t MIDDLE;
   static Int_t ipoint, pxp, pyp;
   static Int_t px1,px2,py1,py2;
   static Int_t pxold, pyold, px1old, py1old, px2old, py2old;
   static Int_t dpx, dpy;
   static Int_t *x=0, *y=0;

   if (!gPad->IsEditable()) return;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      TAttLine::Modify();  //Change line attributes only if necessary
      px1 = gPad->XtoAbsPixel(gPad->GetX1());
      py1 = gPad->YtoAbsPixel(gPad->GetY1());
      px2 = gPad->XtoAbsPixel(gPad->GetX2());
      py2 = gPad->YtoAbsPixel(gPad->GetY2());
      ipoint = -1;


      if (x || y) break;
      x = new Int_t[fN+1];
      y = new Int_t[fN+1];
      for (i=0;i<fN;i++) {
         pxp = gPad->XtoAbsPixel(gPad->XtoPad(fX[i]));
         pyp = gPad->YtoAbsPixel(gPad->YtoPad(fY[i]));
         gVirtualX->DrawLine(pxp-4, pyp-4, pxp+4,  pyp-4);
         gVirtualX->DrawLine(pxp+4, pyp-4, pxp+4,  pyp+4);
         gVirtualX->DrawLine(pxp+4, pyp+4, pxp-4,  pyp+4);
         gVirtualX->DrawLine(pxp-4, pyp+4, pxp-4,  pyp-4);
         x[i] = pxp;
         y[i] = pyp;
         d   = TMath::Abs(pxp-px) + TMath::Abs(pyp-py);
         if (d < kMaxDiff) ipoint =i;
      }
      dpx = 0;
      dpy = 0;
      pxold = px;
      pyold = py;
      if (ipoint < 0) return;
      if (ipoint == 0) {
         px1old = 0;
         py1old = 0;
         px2old = gPad->XtoAbsPixel(fX[1]);
         py2old = gPad->YtoAbsPixel(fY[1]);
      } else if (ipoint == fN-1) {
         px1old = gPad->XtoAbsPixel(gPad->XtoPad(fX[fN-2]));
         py1old = gPad->YtoAbsPixel(gPad->YtoPad(fY[fN-2]));
         px2old = 0;
         py2old = 0;
      } else {
         px1old = gPad->XtoAbsPixel(gPad->XtoPad(fX[ipoint-1]));
         py1old = gPad->YtoAbsPixel(gPad->YtoPad(fY[ipoint-1]));
         px2old = gPad->XtoAbsPixel(gPad->XtoPad(fX[ipoint+1]));
         py2old = gPad->YtoAbsPixel(gPad->YtoPad(fY[ipoint+1]));
      }
      pxold = gPad->XtoAbsPixel(gPad->XtoPad(fX[ipoint]));
      pyold = gPad->YtoAbsPixel(gPad->YtoPad(fY[ipoint]));

      break;


   case kMouseMotion:

      MIDDLE = kTRUE;
      for (i=0;i<fN;i++) {
         pxp = gPad->XtoAbsPixel(gPad->XtoPad(fX[i]));
         pyp = gPad->YtoAbsPixel(gPad->YtoPad(fY[i]));
         d   = TMath::Abs(pxp-px) + TMath::Abs(pyp-py);
         if (d < kMaxDiff) MIDDLE = kFALSE;
      }


//*-*- check if point is close to an axis
      if (MIDDLE) gPad->SetCursor(kMove);
      else gPad->SetCursor(kHand);
      break;

   case kButton1Motion:
      if (MIDDLE) {
         for(i=0;i<fN-1;i++) {
            gVirtualX->DrawLine(x[i]+dpx, y[i]+dpy, x[i+1]+dpx, y[i+1]+dpy);
            pxp = x[i]+dpx;
            pyp = y[i]+dpy;
            gVirtualX->DrawLine(pxp-4, pyp-4, pxp+4,  pyp-4);
            gVirtualX->DrawLine(pxp+4, pyp-4, pxp+4,  pyp+4);
            gVirtualX->DrawLine(pxp+4, pyp+4, pxp-4,  pyp+4);
            gVirtualX->DrawLine(pxp-4, pyp+4, pxp-4,  pyp-4);
         }
         pxp = x[fN-1]+dpx;
         pyp = y[fN-1]+dpy;
         gVirtualX->DrawLine(pxp-4, pyp-4, pxp+4,  pyp-4);
         gVirtualX->DrawLine(pxp+4, pyp-4, pxp+4,  pyp+4);
         gVirtualX->DrawLine(pxp+4, pyp+4, pxp-4,  pyp+4);
         gVirtualX->DrawLine(pxp-4, pyp+4, pxp-4,  pyp-4);
         dpx += px - pxold;
         dpy += py - pyold;
         pxold = px;
         pyold = py;
         for(i=0;i<fN-1;i++) {
            gVirtualX->DrawLine(x[i]+dpx, y[i]+dpy, x[i+1]+dpx, y[i+1]+dpy);
            pxp = x[i]+dpx;
            pyp = y[i]+dpy;
            gVirtualX->DrawLine(pxp-4, pyp-4, pxp+4,  pyp-4);
            gVirtualX->DrawLine(pxp+4, pyp-4, pxp+4,  pyp+4);
            gVirtualX->DrawLine(pxp+4, pyp+4, pxp-4,  pyp+4);
            gVirtualX->DrawLine(pxp-4, pyp+4, pxp-4,  pyp-4);
         }
         pxp = x[fN-1]+dpx;
         pyp = y[fN-1]+dpy;
         gVirtualX->DrawLine(pxp-4, pyp-4, pxp+4,  pyp-4);
         gVirtualX->DrawLine(pxp+4, pyp-4, pxp+4,  pyp+4);
         gVirtualX->DrawLine(pxp+4, pyp+4, pxp-4,  pyp+4);
         gVirtualX->DrawLine(pxp-4, pyp+4, pxp-4,  pyp-4);
      } else {
         if (px1old) gVirtualX->DrawLine(px1old, py1old, pxold,  pyold);
         if (px2old) gVirtualX->DrawLine(pxold,  pyold,  px2old, py2old);
         gVirtualX->DrawLine(pxold-4, pyold-4, pxold+4,  pyold-4);
         gVirtualX->DrawLine(pxold+4, pyold-4, pxold+4,  pyold+4);
         gVirtualX->DrawLine(pxold+4, pyold+4, pxold-4,  pyold+4);
         gVirtualX->DrawLine(pxold-4, pyold+4, pxold-4,  pyold-4);
         pxold = px;
         pxold = TMath::Max(pxold, px1);
         pxold = TMath::Min(pxold, px2);
         pyold = py;
         pyold = TMath::Max(pyold, py2);
         pyold = TMath::Min(pyold, py1);
         if (px1old) gVirtualX->DrawLine(px1old, py1old, pxold,  pyold);
         if (px2old) gVirtualX->DrawLine(pxold,  pyold,  px2old, py2old);
         gVirtualX->DrawLine(pxold-4, pyold-4, pxold+4,  pyold-4);
         gVirtualX->DrawLine(pxold+4, pyold-4, pxold+4,  pyold+4);
         gVirtualX->DrawLine(pxold+4, pyold+4, pxold-4,  pyold+4);
         gVirtualX->DrawLine(pxold-4, pyold+4, pxold-4,  pyold-4);
      }
      break;

   case kButton1Up:

//*-*- Compute x,y range
      xmin = gPad->GetUxmin();
      xmax = gPad->GetUxmax();
      ymin = gPad->GetUymin();
      ymax = gPad->GetUymax();
      dx   = xmax-xmin;
      dy   = ymax-ymin;
      dxr  = dx/(1 - gPad->GetLeftMargin() - gPad->GetRightMargin());
      dyr  = dy/(1 - gPad->GetBottomMargin() - gPad->GetTopMargin());

   // Range() could change the size of the pad pixmap and therefore should
   // be called before the other paint routines
         gPad->Range(xmin - dxr*gPad->GetLeftMargin(),
                      ymin - dyr*gPad->GetBottomMargin(),
                      xmax + dxr*gPad->GetRightMargin(),
                      ymax + dyr*gPad->GetTopMargin());
         gPad->RangeAxis(xmin, ymin, xmax, ymax);

      if (MIDDLE) {
         for(i=0;i<fN;i++) {
            fX[i] = gPad->PadtoX(gPad->AbsPixeltoX(x[i]+dpx));
            fY[i] = gPad->PadtoY(gPad->AbsPixeltoY(y[i]+dpy));
         }
      } else {
         fX[ipoint] = gPad->PadtoX(gPad->AbsPixeltoX(pxold));
         fY[ipoint] = gPad->PadtoY(gPad->AbsPixeltoY(pyold));
      }
      delete [] x; x = 0;
      delete [] y; y = 0;
      gPad->Modified(kTRUE);
      gVirtualX->SetLineColor(-1);

   }
}



//______________________________________________________________________________
void TPolyLine::ls(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*List this polyline with its attributes*-*-*-*-*-*-*-*-*
//*-*                ======================================
   TROOT::IndentLevel();
   printf("TPolyLine  N=%d\n",fN);
}

//______________________________________________________________________________
void TPolyLine::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Paint this polyline with its current attributes*-*-*-*-*-*-*
//*-*              ===============================================
   PaintPolyLine(fN, fX, fY, option);
}

//______________________________________________________________________________
void TPolyLine::PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
//*-*-*-*-*-*-*-*-*Draw this polyline with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*              =======================================
//
//  if option = 'f' or 'F' the fill area is drawn
//  default is to draw the lines only

   TAttLine::Modify();  //Change line attributes only if necessary
   TAttFill::Modify();  //Change fill area attributes only if necessary
   if (*option == 'f' || *option == 'F') gPad->PaintFillArea(n,x,y,option);
   else                                  gPad->PaintPolyLine(n,x,y,option);
}

//______________________________________________________________________________
void TPolyLine::PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
//*-*-*-*-*-*-*-*-*Draw this polyline with new coordinates in NDC*-*-*-*-*-*-*
//*-*              ==============================================

   TAttLine::Modify();  //Change line attributes only if necessary
   TAttFill::Modify();  //Change fill area attributes only if necessary
   gPad->PaintPolyLineNDC(n,x,y,option);
}

//______________________________________________________________________________
void TPolyLine::Print(Option_t *)
{
//*-*-*-*-*-*-*-*-*Dump this polyline with its attributes*-*-*-*-*-*-*-*-*-*
//*-*              ======================================

   printf("PolyLine  N=%d\n",fN);
}

//______________________________________________________________________________
void TPolyLine::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   out<<"   Double_t *dum = 0;"<<endl;
   if (gROOT->ClassSaved(TPolyLine::Class())) {
       out<<"   ";
   } else {
       out<<"   TPolyLine *";
   }
   out<<"pline = new TPolyLine("<<fN<<",dum,dum,"<<quote<<fOption<<quote<<");"<<endl;

   SaveFillAttributes(out,"pline",0,1001);
   SaveLineAttributes(out,"pline",1,1,1);

   for (Int_t i=0;i<fN;i++) {
      out<<"   pline->SetPoint("<<i<<","<<fX[i]<<","<<fY[i]<<");"<<endl;
   }
   out<<"   pline->Draw();"<<endl;
}

//______________________________________________________________________________
void TPolyLine::SetPoint(Int_t point, Double_t x, Double_t y)
{
//*-*-*-*-*-*-*-*-*-*-*-*Initialize one point of the polyline*-*-*-*-*-*-*-*-*
//*-*                    ====================================
       if (point < 0 || point >= fN) return;
       fX[point] = x;
       fY[point] = y;
}

//______________________________________________________________________________
void TPolyLine::SetPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Set new values for this polyline*-*-*-*-*-*-*-*-*-*
//*-*                    ================================
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
}

//_______________________________________________________________________
void TPolyLine::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   UInt_t R__s, R__c;
   if (b.IsReading()) {
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      TObject::Streamer(b);
      TAttLine::Streamer(b);
      TAttFill::Streamer(b);
      b >> fN;
      fX = new Double_t[fN];
      fY = new Double_t[fN];
      if (R__v < 2) {
         Float_t *x = new Float_t[fN];
         Float_t *y = new Float_t[fN];
         b.ReadFastArray(x,fN);
         b.ReadFastArray(y,fN);
         for (Int_t i=0;i<fN;i++) {
            fX[i] = x[i];
            fY[i] = y[i];
         }
      } else {
         b.ReadFastArray(fX,fN);
         b.ReadFastArray(fY,fN);
      }
      fOption.Streamer(b);
      b.CheckByteCount(R__s, R__c, TPolyLine::IsA());
   } else {
      R__c = b.WriteVersion(TPolyLine::IsA(), kTRUE);
      TObject::Streamer(b);
      TAttLine::Streamer(b);
      TAttFill::Streamer(b);
      b << fN;
      b.WriteFastArray(fX,fN);
      b.WriteFastArray(fY,fN);
      fOption.Streamer(b);
      b.SetByteCount(R__c, kTRUE);
   }
}
