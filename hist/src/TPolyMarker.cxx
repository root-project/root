// @(#)root/hist:$Name:  $:$Id: TPolyMarker.cxx,v 1.9 2002/01/23 17:52:50 rdm Exp $
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
   fN = 0;
   fX = fY = 0;
   fLastPoint = -1;
}

//______________________________________________________________________________
TPolyMarker::TPolyMarker(Int_t n, Option_t *option)
      :TObject(), TAttMarker()
{

   fLastPoint = -1;
   fN = n;
   fX = new Double_t [fN];
   fY = new Double_t [fN];
   fOption = option;
   SetBit(kCanDelete);
}

//______________________________________________________________________________
TPolyMarker::TPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option)
      :TObject(), TAttMarker()
{

   fLastPoint = -1;
   fN = n;
   fX = new Double_t [fN];
   fY = new Double_t [fN];
   if (!x || !y) return;
   for (Int_t i=0; i<fN;i++) { fX[i] = x[i]; fY[i] = y[i]; }
   fLastPoint = fN-1;
   fOption = option;
   SetBit(kCanDelete);
}

//______________________________________________________________________________
TPolyMarker::TPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option)
      :TObject(), TAttMarker()
{

   fLastPoint = -1;
   fN = n;
   fX = new Double_t [fN];
   fY = new Double_t [fN];
   if (!x || !y) return;
   for (Int_t i=0; i<fN;i++) { fX[i] = x[i]; fY[i] = y[i]; }
   fLastPoint = fN-1;
   fOption = option;
   SetBit(kCanDelete);
}

//______________________________________________________________________________
TPolyMarker::~TPolyMarker()
{
   if (fX) delete [] fX;
   if (fY) delete [] fY;
   fLastPoint = -1;
}

//______________________________________________________________________________
TPolyMarker::TPolyMarker(const TPolyMarker &polymarker)
{
   ((TPolyMarker&)polymarker).Copy(*this);
}

//______________________________________________________________________________
void TPolyMarker::Copy(TObject &obj)
{

   TObject::Copy(obj);
   TAttMarker::Copy(((TPolyMarker&)obj));
   ((TPolyMarker&)obj).fN = fN;
   ((TPolyMarker&)obj).fX = new Double_t [fN];
   ((TPolyMarker&)obj).fY = new Double_t [fN];
   for (Int_t i=0; i<fN;i++) { ((TPolyMarker&)obj).fX[i] = fX[i], ((TPolyMarker&)obj).fY[i] = fY[i]; }
   ((TPolyMarker&)obj).fOption = fOption;
   ((TPolyMarker&)obj).fLastPoint = fLastPoint;
}

//______________________________________________________________________________
void TPolyMarker::Draw(Option_t *option)
{

   AppendPad(option);

}

//______________________________________________________________________________
void TPolyMarker::DrawPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *)
{
   TPolyMarker *newpolymarker = new TPolyMarker(n,x,y);
   TAttMarker::Copy(*newpolymarker);
   newpolymarker->fOption = fOption;
   newpolymarker->SetBit(kCanDelete);
   newpolymarker->AppendPad();
}

//______________________________________________________________________________
void TPolyMarker::ExecuteEvent(Int_t, Int_t, Int_t)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function must be implemented to realize the action
//  corresponding to the mouse click on the object in the window
//
}

//______________________________________________________________________________
void TPolyMarker::ls(Option_t *) const
{
   TROOT::IndentLevel();
   printf("TPolyMarker  N=%d\n",fN);
}

//______________________________________________________________________________
Int_t TPolyMarker::Merge(TCollection *list)
{
// Merge polymarkers in the collection in this polymarker

   if (!list) return 0;
   TIter next(list);

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
   pm->SetPoint(npoints-1,0,0);

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
   PaintPolyMarker(fLastPoint+1, fX, fY, option);
}

//______________________________________________________________________________
void TPolyMarker::PaintPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option)
{

   TAttMarker::Modify();  //Change marker attributes only if necessary
   gPad->PaintPolyMarker(n,x,y,option);

}

//______________________________________________________________________________
void TPolyMarker::Print(Option_t *) const
{

   printf("TPolyMarker  N=%d\n",fN);
}

//______________________________________________________________________________
void TPolyMarker::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save primitive as a C++ statement(s) on output stream out

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
   // set point number n
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
   SetPoint(n-1,0,0);
}

//______________________________________________________________________________
void TPolyMarker::SetPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option)
{
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
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TPolyMarker::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
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
      TPolyMarker::Class()->WriteBuffer(R__b,this);
   }
}
