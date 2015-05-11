// @(#)root/graf:$Id$
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "TMarker.h"
#include "TVirtualX.h"
#include "TMath.h"
#include "TPoint.h"
#include "TText.h"
#include "TClass.h"
#include "TPoint.h"

ClassImp(TMarker)


//______________________________________________________________________________
//
// Manages Markers. Marker attributes are managed by TAttMarker.
//


//______________________________________________________________________________
TMarker::TMarker(): TObject(), TAttMarker()
{
   // Marker default constructor.

   fX = 0;
   fY = 0;
}


//______________________________________________________________________________
TMarker::TMarker(Double_t x, Double_t y, Int_t marker)
      :TObject(), TAttMarker()
{
   // Marker normal constructor.

   fX = x;
   fY = y;
   fMarkerStyle = marker;
}


//______________________________________________________________________________
TMarker::~TMarker()
{
   // Marker default destructor.
}


//______________________________________________________________________________
TMarker::TMarker(const TMarker &marker) : TObject(marker), TAttMarker(marker), TAttBBox2D(marker)
{
   // Marker copy constructor.

   fX = 0;
   fY = 0;
   ((TMarker&)marker).Copy(*this);
}


//______________________________________________________________________________
void TMarker::Copy(TObject &obj) const
{
   // Copy this marker to marker.

   TObject::Copy(obj);
   TAttMarker::Copy(((TMarker&)obj));
   ((TMarker&)obj).fX = fX;
   ((TMarker&)obj).fY = fY;
}


//______________________________________________________________________________
void TMarker::DisplayMarkerTypes()
{
   // Display the table of markers with their numbers.

   TMarker *marker = new TMarker();
   marker->SetMarkerSize(3);
   TText *text = new TText();
   text->SetTextFont(62);
   text->SetTextAlign(22);
   text->SetTextSize(0.1);
   char atext[] = "       ";
   Double_t x = 0;
   Double_t dx = 1/16.0;
   for (Int_t i=1;i<16;i++) {
      x += dx;
      snprintf(atext,7,"%d",i);
      marker->SetMarkerStyle(i);
      marker->DrawMarker(x,.35);
      text->DrawText(x,.17,atext);
      snprintf(atext,7,"%d",i+19);
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
   // Compute distance from point px,py to a marker.
   //
   //  Compute the closest distance of approach from point px,py to this marker.
   //  The distance is computed in pixels units.

   Int_t pxm, pym;
   if (TestBit(kMarkerNDC)) {
      pxm = gPad->UtoPixel(fX);
      pym = gPad->VtoPixel(fY);
   } else {
      pxm  = gPad->XtoAbsPixel(gPad->XtoPad(fX));
      pym  = gPad->YtoAbsPixel(gPad->YtoPad(fY));
   }
   Int_t dist = (Int_t)TMath::Sqrt((px-pxm)*(px-pxm) + (py-pym)*(py-pym));

   //marker size = 1 is about 8 pixels
   Int_t markerRadius = Int_t(4*fMarkerSize);
   if (dist <= markerRadius)   return 0;
   if (dist >  markerRadius+3) return 999;
   return dist;
}


//______________________________________________________________________________
void TMarker::Draw(Option_t *option)
{
   // Draw this marker with its current attributes.

   AppendPad(option);

}


//______________________________________________________________________________
void TMarker::DrawMarker(Double_t x, Double_t y)
{
   // Draw this marker with new coordinates.

   TMarker *newmarker = new TMarker(x, y, 1);
   TAttMarker::Copy(*newmarker);
   newmarker->SetBit(kCanDelete);
   newmarker->AppendPad();
}


//______________________________________________________________________________
void TMarker::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //  This member function is called when a marker is clicked with the locator
   //
   //  If Left button is clicked on a marker, the marker is moved to
   //  a new position when the mouse button is released.

   if (!gPad) return;

   TPoint p;
   static Int_t pxold, pyold;
   static Bool_t ndcsav;
   Double_t dpx, dpy, xp1,yp1;
   Bool_t opaque  = gPad->OpaqueMoving();

   if (!gPad->IsEditable()) return;

   switch (event) {

   case kButton1Down:
      ndcsav = TestBit(kMarkerNDC);
      if (!opaque) {
         gVirtualX->SetTextColor(-1);  // invalidate current text color (use xor mode)
         TAttMarker::Modify();  //Change marker attributes only if necessary
      }
      // No break !!!

   case kMouseMotion:
      pxold = px;  pyold = py;
      gPad->SetCursor(kMove);
      break;

   case kButton1Motion:
      p.fX = pxold; p.fY = pyold;
      if (!opaque) gVirtualX->DrawPolyMarker(1, &p);
      p.fX = px; p.fY = py;
      if (!opaque) gVirtualX->DrawPolyMarker(1, &p);
      pxold = px;  pyold = py;
      if (opaque) {
         if (ndcsav) this->SetNDC(kFALSE);
         this->SetX(gPad->PadtoX(gPad->AbsPixeltoX(px)));
         this->SetY(gPad->PadtoY(gPad->AbsPixeltoY(py)));
         gPad->ShowGuidelines(this, event, 'i', true);
         gPad->Modified(kTRUE);
         gPad->Update();
      }
      break;

   case kButton1Up:
      if (opaque) {
         if (ndcsav) {
            this->SetX((fX - gPad->GetX1())/(gPad->GetX2()-gPad->GetX1()));
            this->SetY((fY - gPad->GetY1())/(gPad->GetY2()-gPad->GetY1()));
            this->SetNDC();
         }
         gPad->ShowGuidelines(this, event);
      } else {
         if (TestBit(kMarkerNDC)) {
            dpx  = gPad->GetX2() - gPad->GetX1();
            dpy  = gPad->GetY2() - gPad->GetY1();
            xp1  = gPad->GetX1();
            yp1  = gPad->GetY1();
            fX = (gPad->AbsPixeltoX(pxold)-xp1)/dpx;
            fY = (gPad->AbsPixeltoY(pyold)-yp1)/dpy;
         } else {
            fX = gPad->PadtoX(gPad->AbsPixeltoX(px));
            fY = gPad->PadtoY(gPad->AbsPixeltoY(py));
         }
         gPad->Modified(kTRUE);
         gPad->Update();
         gVirtualX->SetTextColor(-1);
      }
      break;
   }
}


//______________________________________________________________________________
void TMarker::ls(Option_t *) const
{
   // List this marker with its attributes.

   TROOT::IndentLevel();
   printf("Marker  X=%f Y=%f marker type=%d\n",fX,fY,fMarkerStyle);
}


//______________________________________________________________________________
void TMarker::Paint(Option_t *)
{
   // Paint this marker with its current attributes.

   if (TestBit(kMarkerNDC)) {
      Double_t u = gPad->GetX1() + fX*(gPad->GetX2()-gPad->GetX1());
      Double_t v = gPad->GetY1() + fY*(gPad->GetY2()-gPad->GetY1());
      PaintMarker(u,v);
   } else {
      PaintMarker(gPad->XtoPad(fX),gPad->YtoPad(fY));
   }
}


//______________________________________________________________________________
void TMarker::PaintMarker(Double_t x, Double_t y)
{
   // Draw this marker with new coordinates.

   TAttMarker::Modify();  //Change line attributes only if necessary
   gPad->PaintPolyMarker(-1,&x,&y,"");
}


//______________________________________________________________________________
void TMarker::PaintMarkerNDC(Double_t, Double_t)
{
   // Draw this marker with new coordinates in NDC.
}


//______________________________________________________________________________
void TMarker::Print(Option_t *) const
{
   // Dump this marker with its attributes.

   printf("Marker  X=%f Y=%f",fX,fY);
   if (GetMarkerColor() != 1) printf(" Color=%d",GetMarkerColor());
   if (GetMarkerStyle() != 1) printf(" MarkerStyle=%d",GetMarkerStyle());
   if (GetMarkerSize()  != 1) printf(" MarkerSize=%f",GetMarkerSize());
   printf("\n");
}


//______________________________________________________________________________
void TMarker::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   if (gROOT->ClassSaved(TMarker::Class())) {
      out<<"   ";
   } else {
      out<<"   TMarker *";
   }
   out<<"marker = new TMarker("<<fX<<","<<fY<<","<<fMarkerStyle<<");"<<std::endl;

   SaveMarkerAttributes(out,"marker",1,1,1);

   out<<"   marker->Draw();"<<std::endl;
}


//______________________________________________________________________________
void TMarker::SetNDC(Bool_t isNDC)
{
   // Set NDC mode on if isNDC = kTRUE, off otherwise

   ResetBit(kMarkerNDC);
   if (isNDC) SetBit(kMarkerNDC);
}


//______________________________________________________________________________
void TMarker::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMarker.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TMarker::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      TAttMarker::Streamer(R__b);
      Float_t x,y;
      R__b >> x;  fX = x;
      R__b >> y;  fY = y;
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TMarker::Class(),this);
   }
}

//______________________________________________________________________________
Rectangle_t TMarker::GetBBox()
{
   // Return the bounding Box of the Line

   Double_t size = this->GetMarkerSize();

   Rectangle_t BBox;
   BBox.fX = gPad->XtoPixel(fX)+(Int_t)(2*size);
   BBox.fY = gPad->YtoPixel(fY)-(Int_t)(2*size);
   BBox.fWidth = 2*size;
   BBox.fHeight = 2*size;
   return (BBox);
}

//______________________________________________________________________________
TPoint TMarker::GetBBoxCenter()
{
   // Return the center of the BoundingBox as TPoint in pixels

   TPoint p;
   p.SetX(gPad->XtoPixel(fX));
   p.SetY(gPad->YtoPixel(fY));
   return(p);
}

//______________________________________________________________________________
void TMarker::SetBBoxCenter(const TPoint &p)
{
   // Set center of the BoundingBox

   fX = gPad->PixeltoX(p.GetX());
   fY = gPad->PixeltoY(p.GetY() - gPad->VtoPixel(0));
}

//______________________________________________________________________________
void TMarker::SetBBoxCenterX(const Int_t x)
{
   // Set X coordinate of the center of the BoundingBox

   fX = gPad->PixeltoX(x);
}

//______________________________________________________________________________
void TMarker::SetBBoxCenterY(const Int_t y)
{
   // Set Y coordinate of the center of the BoundingBox

   fY = gPad->PixeltoY(y - gPad->VtoPixel(0));
}

//______________________________________________________________________________
void TMarker::SetBBoxX1(const Int_t x)
{
   // Set lefthandside of BoundingBox to a value
   // (resize in x direction on left)

   Double_t size = this->GetMarkerSize();
   fX = gPad->PixeltoX(x + (Int_t)size);
}

//______________________________________________________________________________
void TMarker::SetBBoxX2(const Int_t x)
{
   // Set righthandside of BoundingBox to a value
   // (resize in x direction on right)

   Double_t size = this->GetMarkerSize();
   fX = gPad->PixeltoX(x - (Int_t)size);
}

//_______________________________________________________________________________
void TMarker::SetBBoxY1(const Int_t y)
{
   // Set top of BoundingBox to a value (resize in y direction on top)

   Double_t size = this->GetMarkerSize();
   fY = gPad->PixeltoY(y - (Int_t)size - gPad->VtoPixel(0));
}

//_______________________________________________________________________________
void TMarker::SetBBoxY2(const Int_t y)
{
   // Set bottom of BoundingBox to a value
   // (resize in y direction on bottom)

   Double_t size = this->GetMarkerSize();
   fY = gPad->PixeltoY(y + (Int_t)size - gPad->VtoPixel(0));
}
