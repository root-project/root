// @(#)root/graf:$Name:  $:$Id: TMarker.cxx,v 1.1.1.1 2000/05/16 17:00:49 rdm Exp $
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>
#include <fstream.h>
#include <iostream.h>

#include "TROOT.h"
#include "TVirtualPad.h"
#include "TMarker.h"
#include "TVirtualX.h"
#include "TMath.h"
#include "TPoint.h"
#include "TText.h"

ClassImp(TMarker)

//______________________________________________________________________________
//
// Manages Markers. Marker attributes are managed by TAttMarker.
// The list of standard ROOT markers is shown in this picture
//Begin_Html
/*
<img src="gif/markers.gif">
*/
//End_Html
//

//______________________________________________________________________________
TMarker::TMarker(): TObject(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*Marker default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==========================
   fX = 0;
   fY = 0;
}
//______________________________________________________________________________
TMarker::TMarker(Double_t x, Double_t y, Int_t marker)
      :TObject(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*Marker normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
   fX = x;
   fY = y;
   fMarkerStyle = marker;
}

//______________________________________________________________________________
TMarker::~TMarker()
{
//*-*-*-*-*-*-*-*-*-*-*Marker default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
}

//______________________________________________________________________________
TMarker::TMarker(const TMarker &marker)
{
   ((TMarker&)marker).Copy(*this);
}

//______________________________________________________________________________
void TMarker::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this marker to marker*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==========================

   TObject::Copy(obj);
   TAttMarker::Copy(((TMarker&)obj));
   ((TMarker&)obj).fX = fX;
   ((TMarker&)obj).fY = fY;
}

//______________________________________________________________________________
void TMarker::DisplayMarkerTypes()
{
//*-*-*-*-*-*-*-*-*-*-*Display the table of markers with their numbers*-*-*-*
//*-*                  ===============================================

  TMarker *marker = new TMarker();
  marker->SetMarkerSize(3);
  TText *text = new TText();
  text->SetTextFont(62);
  text->SetTextAlign(22);
  text->SetTextSize(0.1);
  char atext[] = "       ";
  Double_t x = 0;
  Double_t dx = 1/12.0;
  for (Int_t i=1;i<12;i++) {
     x += dx;
     sprintf(atext,"%d",i);
     marker->SetMarkerStyle(i);
     marker->DrawMarker(x,.35);
     text->DrawText(x,.17,atext);
     sprintf(atext,"%d",i+19);
     marker->SetMarkerStyle(i+19);
     marker->DrawMarker(x,.8);
     text->DrawText(x,.62,atext);
  }
  delete marker;
  delete text;
}

//______________________________________________________________________________
Int_t TMarker::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a marker*-*-*-*-*-*
//*-*                  ===========================================
//  Compute the closest distance of approach from point px,py to this marker.
//  The distance is computed in pixels units.
//

   const Int_t kMaxDiff = 10;
   Int_t pxm  = gPad->XtoAbsPixel(fX);
   Int_t pym  = gPad->YtoAbsPixel(fY);
   Int_t dist = (px-pxm)*(px-pxm) + (py-pym)*(py-pym);

   if (dist > kMaxDiff) return 9999;
   return dist;
}

//______________________________________________________________________________
void TMarker::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this marker with its current attributes*-*-*-*-*-*-*
//*-*                  ============================================

   AppendPad(option);

}

//______________________________________________________________________________
void TMarker::DrawMarker(Double_t x, Double_t y)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this marker with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  =====================================
   TMarker *newmarker = new TMarker(x, y, 1);
   TAttMarker::Copy(*newmarker);
   newmarker->SetBit(kCanDelete);
   newmarker->AppendPad();
}

//______________________________________________________________________________
void TMarker::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a marker is clicked with the locator
//
//  If Left button is clicked on a marker, the marker is moved to
//     a new position when the mouse button is released.
//

   TPoint p;
   static Int_t pxold, pyold;

   if (!gPad->IsEditable()) return;

   switch (event) {


   case kButton1Down:
      gVirtualX->SetTextColor(-1);  // invalidate current text color (use xor mode)
      TAttMarker::Modify();  //Change marker attributes only if necessary

      // No break !!!

   case kMouseMotion:
      pxold = px;  pyold = py;
      gPad->SetCursor(kMove);
      break;

   case kButton1Motion:
      p.fX = pxold; p.fY = pyold;
      gVirtualX->DrawPolyMarker(1, &p);
      p.fX = px; p.fY = py;
      gVirtualX->DrawPolyMarker(1, &p);
      pxold = px;  pyold = py;
      break;

   case kButton1Up:
      fX = gPad->AbsPixeltoX(px);
      fY = gPad->AbsPixeltoY(py);
      gPad->Modified(kTRUE);
      gVirtualX->SetTextColor(-1);
      break;
   }
}

//______________________________________________________________________________
void TMarker::ls(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*-*List this marker with its attributes*-*-*-*-*-*-*-*-*
//*-*                    ====================================
   IndentLevel();
   printf("Marker  X=%f Y=%f marker type=%d\n",fX,fY,fMarkerStyle);
}

//______________________________________________________________________________
void TMarker::Paint(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this marker with its current attributes*-*-*-*-*-*-*
//*-*                  =============================================
   PaintMarker(fX,fY);
}

//______________________________________________________________________________
void TMarker::PaintMarker(Double_t x, Double_t y)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this marker with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  =====================================

   TAttMarker::Modify();  //Change line attributes only if necessary
   gPad->PaintPolyMarker(1,&x,&y,"");
}

//______________________________________________________________________________
void TMarker::PaintMarkerNDC(Double_t, Double_t)
{
//*-*-*-*-*-*-*-*Draw this marker with new coordinates in NDC*-*-*-*-*-*-*-*-*-*
//*-*            ============================================

}

//______________________________________________________________________________
void TMarker::Print(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Dump this marker with its attributes*-*-*-*-*-*-*-*-*-*
//*-*                  ====================================

   printf("Marker  X=%f Y=%f",fX,fY);
   if (GetMarkerColor() != 1) printf(" Color=%d",GetMarkerColor());
   if (GetMarkerStyle() != 1) printf(" MarkerStyle=%d",GetMarkerStyle());
   if (GetMarkerSize()  != 1) printf(" MarkerSize=%f",GetMarkerSize());
   printf("\n");
}

//______________________________________________________________________________
void TMarker::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   if (gROOT->ClassSaved(TMarker::Class())) {
       out<<"   ";
   } else {
       out<<"   TMarker *";
   }
   out<<"marker = new TMarker("<<fX<<","<<fY<<","<<fMarkerStyle<<");"<<endl;

   SaveMarkerAttributes(out,"marker",1,1,1);

   out<<"   marker->Draw();"<<endl;
}

//______________________________________________________________________________
void TMarker::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMarker.

   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion();
      TObject::Streamer(R__b);
      TAttMarker::Streamer(R__b);
      if (R__v < 2) {
         Float_t x,y;
         R__b >> x;  fX = x;
         R__b >> y;  fY = y;
      } else {
         R__b >> fX;
         R__b >> fY;
      }
   } else {
      R__b.WriteVersion(TMarker::IsA());
      TObject::Streamer(R__b);
      TAttMarker::Streamer(R__b);
      R__b << fX;
      R__b << fY;
   }
}
