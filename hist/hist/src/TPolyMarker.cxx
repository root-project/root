// @(#)root/hist:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "TPolyMarker.h"
#include "TClass.h"
#include "TMath.h"

ClassImp(TPolyMarker)


//______________________________________________________________________________
//
//  a PolyMarker is defined by an array on N points in a 2-D space.
// At each point x[i], y[i] a marker is drawn.
// Marker attributes are managed by TAttMarker.
// See TMarker for the list of possible marker types.
//


//______________________________________________________________________________
TPolyMarker::TPolyMarker(): TObject()
{
   // Default constructor.

   fN = 0;
   fX = fY = 0;
   fLastPoint = -1;
}


//______________________________________________________________________________
TPolyMarker::TPolyMarker(Int_t n, Option_t *option)
      :TObject(), TAttMarker()
{
   // Constructor.

   fOption = option;
   SetBit(kCanDelete);
   fLastPoint = -1;
   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      fX = fY = 0;
      return;
   }
   fN = n;
   fX = new Double_t [fN];
   fY = new Double_t [fN];
}


//______________________________________________________________________________
TPolyMarker::TPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option)
      :TObject(), TAttMarker()
{
   // Constructor.

   fOption = option;
   SetBit(kCanDelete);
   fLastPoint = -1;
   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      fX = fY = 0;
      return;
   }
   fN = n;
   fX = new Double_t [fN];
   fY = new Double_t [fN];
   if (!x || !y) return;
   for (Int_t i=0; i<fN;i++) { fX[i] = x[i]; fY[i] = y[i]; }
   fLastPoint = fN-1;
}


//______________________________________________________________________________
TPolyMarker::TPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option)
      :TObject(), TAttMarker()
{
   // Constructor.

   fOption = option;
   SetBit(kCanDelete);
   fLastPoint = -1;
   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      fX = fY = 0;
      return;
   }
   fN = n;
   fX = new Double_t [fN];
   fY = new Double_t [fN];
   if (!x || !y) return;
   for (Int_t i=0; i<fN;i++) { fX[i] = x[i]; fY[i] = y[i]; }
   fLastPoint = fN-1;
}

//______________________________________________________________________________
TPolyMarker& TPolyMarker::operator=(const TPolyMarker& pm)
{
   //assignment operator
   if(this!=&pm) {
      TObject::operator=(pm);
      TAttMarker::operator=(pm);
      fN=pm.fN;
      fLastPoint=pm.fLastPoint;
      fX=pm.fX;
      fY=pm.fY;
      fOption=pm.fOption;
   }
   return *this;
}

//______________________________________________________________________________
TPolyMarker::~TPolyMarker()
{
   // Desctructor.

   if (fX) delete [] fX;
   if (fY) delete [] fY;
   fLastPoint = -1;
}


//______________________________________________________________________________
TPolyMarker::TPolyMarker(const TPolyMarker &polymarker) : TObject(polymarker), TAttMarker(polymarker)
{
   // Copy constructor.

   fN = 0;
   fX = fY = 0;
   fLastPoint = -1;
   ((TPolyMarker&)polymarker).Copy(*this);
}


//______________________________________________________________________________
void TPolyMarker::Copy(TObject &obj) const
{

   // Copy.

   TObject::Copy(obj);
   TAttMarker::Copy(((TPolyMarker&)obj));
   ((TPolyMarker&)obj).fN = fN;
   if (fN > 0) {
      ((TPolyMarker&)obj).fX = new Double_t [fN];
      ((TPolyMarker&)obj).fY = new Double_t [fN];
      for (Int_t i=0; i<fN;i++) { ((TPolyMarker&)obj).fX[i] = fX[i], ((TPolyMarker&)obj).fY[i] = fY[i]; }
   } else {
      ((TPolyMarker&)obj).fX = 0;
      ((TPolyMarker&)obj).fY = 0;
   }
   ((TPolyMarker&)obj).fOption = fOption;
   ((TPolyMarker&)obj).fLastPoint = fLastPoint;
}


//______________________________________________________________________________
Int_t TPolyMarker::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a polymarker.
   //
   //  Compute the closest distance of approach from point px,py to each point
   //  of the polymarker.
   //  Returns when the distance found is below DistanceMaximum.
   //  The distance is computed in pixels units.

   const Int_t big = 9999;

   // check if point is near one of the points
   Int_t i, pxp, pyp, d;
   Int_t distance = big;

   for (i=0;i<Size();i++) {
      pxp = gPad->XtoAbsPixel(gPad->XtoPad(fX[i]));
      pyp = gPad->YtoAbsPixel(gPad->YtoPad(fY[i]));
      d   = TMath::Abs(pxp-px) + TMath::Abs(pyp-py);
      if (d < distance) distance = d;
   }
   return distance;
}


//______________________________________________________________________________
void TPolyMarker::Draw(Option_t *option)
{
   // Draw.

   AppendPad(option);
}


//______________________________________________________________________________
void TPolyMarker::DrawPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *)
{
   // Draw polymarker.

   TPolyMarker *newpolymarker = new TPolyMarker(n,x,y);
   TAttMarker::Copy(*newpolymarker);
   newpolymarker->fOption = fOption;
   newpolymarker->SetBit(kCanDelete);
   newpolymarker->AppendPad();
}


//______________________________________________________________________________
void TPolyMarker::ExecuteEvent(Int_t, Int_t, Int_t)
{
   // Execute action corresponding to one event.
   //
   //  This member function must be implemented to realize the action
   //  corresponding to the mouse click on the object in the window
}


//______________________________________________________________________________
void TPolyMarker::ls(Option_t *) const
{
   // ls.

   TROOT::IndentLevel();
   printf("TPolyMarker  N=%d\n",fN);
}


//______________________________________________________________________________
Int_t TPolyMarker::Merge(TCollection *li)
{
   // Merge polymarkers in the collection in this polymarker.

   if (!li) return 0;
   TIter next(li);

   //first loop to count the number of entries
   TPolyMarker *pm;
   Int_t npoints = 0;
   while ((pm = (TPolyMarker*)next())) {
      if (!pm->InheritsFrom(TPolyMarker::Class())) {
         Error("Add","Attempt to add object of class: %s to a %s",pm->ClassName(),this->ClassName());
         return -1;
      }
      npoints += pm->Size();
   }

   //extend this polymarker to hold npoints
   SetPoint(npoints-1,0,0);

   //merge all polymarkers
   next.Reset();
   while ((pm = (TPolyMarker*)next())) {
      Int_t np = pm->Size();
      Double_t *x = pm->GetX();
      Double_t *y = pm->GetY();
      for (Int_t i=0;i<np;i++) {
         SetPoint(i,x[i],y[i]);
      }
   }

   return npoints;
}


//______________________________________________________________________________
void TPolyMarker::Paint(Option_t *option)
{
   // Paint.

   PaintPolyMarker(fLastPoint+1, fX, fY, option);
}


//______________________________________________________________________________
void TPolyMarker::PaintPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   // Paint polymarker.

   if (n <= 0) return;
   TAttMarker::Modify();  //Change marker attributes only if necessary
   Double_t *xx = x;
   Double_t *yy = y;
   if (gPad->GetLogx()) {
      xx = new Double_t[n];
      for (Int_t ix=0;ix<n;ix++) xx[ix] = gPad->XtoPad(x[ix]);
   }
   if (gPad->GetLogy()) {
      yy = new Double_t[n];
      for (Int_t iy=0;iy<n;iy++) yy[iy] = gPad->YtoPad(y[iy]);
   }
   gPad->PaintPolyMarker(n,xx,yy,option);
   if (x != xx) delete [] xx;
   if (y != yy) delete [] yy;
}


//______________________________________________________________________________
void TPolyMarker::Print(Option_t *) const
{
   // Print polymarker.

   printf("TPolyMarker  N=%d\n",fN);
}


//______________________________________________________________________________
void TPolyMarker::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out.

   char quote = '"';
   out<<"   "<<endl;
   out<<"   Double_t *dum = 0;"<<endl;
   if (gROOT->ClassSaved(TPolyMarker::Class())) {
      out<<"   ";
   } else {
      out<<"   TPolyMarker *";
   }
   out<<"pmarker = new TPolyMarker("<<fN<<",dum,dum,"<<quote<<fOption<<quote<<");"<<endl;

   SaveMarkerAttributes(out,"pmarker",1,1,1);

   for (Int_t i=0;i<Size();i++) {
      out<<"   pmarker->SetPoint("<<i<<","<<fX[i]<<","<<fY[i]<<");"<<endl;
   }
   out<<"   pmarker->Draw("
      <<quote<<option<<quote<<");"<<endl;
}


//______________________________________________________________________________
Int_t TPolyMarker::SetNextPoint(Double_t x, Double_t y)
{
   // Set point following LastPoint to x, y.
   // Returns index of the point (new last point).

   fLastPoint++;
   SetPoint(fLastPoint, x, y);
   return fLastPoint;
}


//______________________________________________________________________________
void TPolyMarker::SetPoint(Int_t n, Double_t x, Double_t y)
{
   // Set point number n.
   // if n is greater than the current size, the arrays are automatically
   // extended

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
   fLastPoint = TMath::Max(fLastPoint,n);
}


//______________________________________________________________________________
void TPolyMarker::SetPolyMarker(Int_t n)
{
   // If n <= 0 the current arrays of points are deleted.

   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      delete [] fX;
      delete [] fY;
      fX = fY = 0;
      return;
   }
   SetPoint(n-1,0,0);
}


//______________________________________________________________________________
void TPolyMarker::SetPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option)
{
   // If n <= 0 the current arrays of points are deleted.

   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      delete [] fX;
      delete [] fY;
      fX = fY = 0;
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


//______________________________________________________________________________
void TPolyMarker::SetPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{
   // If n <= 0 the current arrays of points are deleted.

   if (n <= 0) {
      fN = 0;
      fLastPoint = -1;
      delete [] fX;
      delete [] fY;
      fX = fY = 0;
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


//_______________________________________________________________________
void TPolyMarker::Streamer(TBuffer &R__b)
{
   // Stream a class object.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TPolyMarker::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      TAttMarker::Streamer(R__b);
      R__b >> fN;
      fX = new Double_t[fN];
      fY = new Double_t[fN];
      Int_t i;
      Float_t xold,yold;
      for (i=0;i<fN;i++) {R__b >> xold; fX[i] = xold;}
      for (i=0;i<fN;i++) {R__b >> yold; fY[i] = yold;}
      fOption.Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, TPolyMarker::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TPolyMarker::Class(),this);
   }
}
