// @(#)root/g3d:$Name:  $:$Id: TPolyMarker3D.cxx,v 1.16 2003/07/02 20:04:01 brun Exp $
// Author: Nenad Buncic   21/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TView.h"
#include "TPolyMarker3D.h"
#include "TVirtualPad.h"
#include "TH3.h"
#include "TRandom.h"
#include "TBuffer3D.h"
#include "TGeometry.h"

ClassImp(TPolyMarker3D)

const Int_t kDimension = 3;

//______________________________________________________________________________
// PolyMarker3D is a 3D polymarker. It has three constructors.
//
//   First one, without any parameters TPolyMarker3D(), we call 'default
// constructor' and it's used in a case that just an initialisation is
// needed (i.e. pointer declaration).
//
//       Example:
//                 TPolyMarker3D *pm = new TPolyMarker3D;
//
//
//   Second one, takes, usually, two parameters, n (number of points) and
// marker (marker style). Third parameter is optional.
//
//       Example:
//                 TPolyMarker3D (150, 1);
//
//
//   Third one takes, usually, three parameters, n (number of points), *p
// (pointer to an array of 3D points), and marker (marker style). Fourth
// parameter is optional.
//
//       Example:
//                 Float_t *ptr = new Float_t [150*3];
//                         ... ... ...
//                         ... ... ...
//                         ... ... ...
//
//                 TPolyMarker3D (150, ptr, 1);
//
//


//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D()
{
   // 3-D polymarker default constructor.

   fN = 0;
   fP = 0;
   fLastPoint = -1;
}

//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(Int_t n, Marker_t marker, Option_t *option)
{
   // 3-D polymarker normal constructor with initialization to 0.

   fLastPoint = -1;
   fN = n;
   fP = new Float_t [kDimension*fN];
   for (Int_t i = 0; i < kDimension*fN; i++)  fP[i] = 0;
   fOption = option;
   SetMarkerStyle(marker);
   SetBit(kCanDelete);
}

//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(Int_t n, Float_t *p, Marker_t marker,
                             Option_t *option)
{
   // 3-D polymarker constructor. Polymarker is initialized with p.

   fLastPoint = -1;
   fN = 0;
   fP = 0;
   if (n > 0) {
      fN = n;
      fP = new Float_t [kDimension*fN];
      if (p) {
         for (Int_t i = 0; i < kDimension*fN; i++)
            fP[i] = p[i];
         fLastPoint = fN-1;
     } else
         memset(fP,0,kDimension*fN*sizeof(Float_t));
   }
   SetMarkerStyle(marker);
   SetBit(kCanDelete);
   fOption = option;
}

//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(Int_t n, Double_t *p, Marker_t marker,
                             Option_t *option)
{
   // 3-D polymarker constructor. Polymarker is initialized with p
   // (cast to float).

   fLastPoint = -1;
   fN = 0;
   fP = 0;
   if (n > 0) {
      fN = n;
      fP = new Float_t [kDimension*fN];
      if (p) {
         for (Int_t i = 0; i < kDimension*fN; i++)
            fP[i] = (Float_t) p[i];
         fLastPoint = fN-1;
      } else
         memset(fP,0,kDimension*fN*sizeof(Float_t));
   }
   SetMarkerStyle(marker);
   SetBit(kCanDelete);
   fOption = option;
}

//______________________________________________________________________________
TPolyMarker3D::~TPolyMarker3D()
{
   // 3-D polymarker destructor.

   fN = 0;
   if (fP) delete [] fP;
   fLastPoint = -1;
}

//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(const TPolyMarker3D &polymarker) : TObject(polymarker), TAttMarker(polymarker), TAtt3D(polymarker)
{
   // 3-D polymarker copy ctor.

   fP = 0;
   ((TPolyMarker3D&)polymarker).Copy(*this);
}

//______________________________________________________________________________
void TPolyMarker3D::Copy(TObject &obj) const
{
   // Copy polymarker to polymarker obj.

   TObject::Copy(obj);
   ((TPolyMarker3D&)obj).fN = fN;
   ((TPolyMarker3D&)obj).fP = new Float_t [kDimension*fN];
   for (Int_t i = 0; i < kDimension*fN; i++)  ((TPolyMarker3D&)obj).fP[i] = fP[i];
   ((TPolyMarker3D&)obj).SetMarkerStyle(GetMarkerStyle());
   ((TPolyMarker3D&)obj).fOption = fOption;
   ((TPolyMarker3D&)obj).fLastPoint = fLastPoint;
}

//______________________________________________________________________________
Int_t TPolyMarker3D::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a 3-D polymarker.
   // Compute the closest distance of approach from point px,py to each segment
   // of the polymarker.
   // Returns when the distance found is below DistanceMaximum.
   // The distance is computed in pixels units.

   const Int_t inaxis = 7;
   Int_t dist = 9999;

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());

   // return if point is not in the user area
   if (px < puxmin - inaxis) return dist;
   if (py > puymin + inaxis) return dist;
   if (px > puxmax + inaxis) return dist;
   if (py < puymax - inaxis) return dist;

   TView *view = gPad->GetView();
   if (!view) return dist;
   Int_t i, dpoint;
   Float_t xndc[3];
   Int_t x1,y1;
   Double_t u,v;
   for (i=0;i<Size();i++) {
      view->WCtoNDC(&fP[3*i], xndc);
      u      = (Double_t)xndc[0];
      v      = (Double_t)xndc[1];
      if (u < gPad->GetUxmin() || u > gPad->GetUxmax()) continue;
      if (v < gPad->GetUymin() || v > gPad->GetUymax()) continue;
      x1     = gPad->XtoAbsPixel(u);
      y1     = gPad->YtoAbsPixel(v);
      dpoint = Int_t(TMath::Sqrt((((Double_t)px-x1)*((Double_t)px-x1)
                                + ((Double_t)py-y1)*((Double_t)py-y1))));
      if (dpoint < dist) dist = dpoint;
   }
   return dist;
}

//______________________________________________________________________________
void TPolyMarker3D::Draw(Option_t *option)
{
   // Draws 3-D polymarker with its current attributes.

   AppendPad(option);
}

//______________________________________________________________________________
void TPolyMarker3D::DrawPolyMarker(Int_t n, Float_t *p, Marker_t, Option_t *option)
{
   // Draw this 3-D polymartker with new coordinates. Creates a new
   // polymarker which will be adopted by the pad in which it is drawn.
   // Does not change the original polymarker (should be static method).

   TPolyMarker3D *newpolymarker = new TPolyMarker3D();
   newpolymarker->fN = n;
   newpolymarker->fP = new Float_t [kDimension*fN];
   for (Int_t i = 0; i < kDimension*fN; i++)  newpolymarker->fP[i] = p[i];
   newpolymarker->SetMarkerStyle(GetMarkerStyle());
   newpolymarker->fOption = fOption;
   newpolymarker->fLastPoint = fLastPoint;
   newpolymarker->SetBit(kCanDelete);
   newpolymarker->AppendPad(option);
}

//______________________________________________________________________________
void TPolyMarker3D::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.

   if (gPad->GetView())
      gPad->GetView()->ExecuteRotateView(event, px, py);
}

//______________________________________________________________________________
void TPolyMarker3D::ls(Option_t *option) const
{
   // List this 3-D polymarker.

   TROOT::IndentLevel();
   cout << "    TPolyMarker3D  N=" << Size() <<" Option="<<option<<endl;
}

//______________________________________________________________________________
Int_t TPolyMarker3D::Merge(TCollection *list)
{
// Merge polymarkers in the collection in this polymarker

   if (!list) return 0;
   TIter next(list);

   //first loop to count the number of entries
   TPolyMarker3D *pm;
   Int_t npoints = 0;
   while ((pm = (TPolyMarker3D*)next())) {
      if (!pm->InheritsFrom(TPolyMarker3D::Class())) {
         Error("Add","Attempt to add object of class: %s to a %s",pm->ClassName(),this->ClassName());
         return -1;
      }
      npoints += pm->Size();
   }

   //extend this polymarker to hold npoints
   pm->SetPoint(npoints-1,0,0,0);

   //merge all polymarkers
   next.Reset();
   while ((pm = (TPolyMarker3D*)next())) {
      Int_t np = pm->Size();
      Float_t *p = pm->GetP();
      for (Int_t i=0;i<np;i++) {
         SetPoint(i,p[3*i],p[3*i+1],p[3*i+2]);
      }
   }

   return npoints;
}

//______________________________________________________________________________
void TPolyMarker3D::Paint(Option_t *option)
{
   // Paint this 3-D polymarker.

   Int_t NbPnts = Size();
   TBuffer3D *buff = gPad->AllocateBuffer3D(3*NbPnts, 1, 0);
   if (!buff) return;

   buff->fType = TBuffer3D::kMARKER;
   buff->fId   = this;

   // Fill gPad->fBuffer3D. Points coordinates are in Master space
   buff->fNbPnts = NbPnts;
   buff->fNbSegs = 0;
   buff->fNbPols = 0;
   // In case of option "size" it is not necessary to fill the buffer
   if (buff->fOption == TBuffer3D::kSIZE) {
      buff->Paint(option);
      return;
   }
    
   for (Int_t i=0; i<3*NbPnts; i++) buff->fPnts[i] = (Double_t)fP[i];

   if (gGeometry) {   
      Double_t dlocal[3];
      Double_t dmaster[3];
      for (Int_t j=0; j<buff->fNbPnts; j++) {
         dlocal[0] = buff->fPnts[3*j];
         dlocal[1] = buff->fPnts[3*j+1];
         dlocal[2] = buff->fPnts[3*j+2];
         gGeometry->Local2Master(&dlocal[0],&dmaster[0]);
         buff->fPnts[3*j]   = dmaster[0];
         buff->fPnts[3*j+1] = dmaster[1];
         buff->fPnts[3*j+2] = dmaster[2];
      }
   }

   // Paint gPad->fBuffer3D
   buff->fSegs[0] = ((GetMarkerColor() % 8) - 1) * 4;
   if ( buff->fSegs[0]<0 ) buff->fSegs[0] = 0;
   TAttMarker::Modify();
   buff->Paint(option);
}

//______________________________________________________________________________
void TPolyMarker3D::PaintH3(TH1 *h, Option_t *option)
{
   // Paint 3-d histogram h with 3-d polymarkers.

   const Int_t kMaxEntry = 100000;
   Int_t in, bin, binx, biny, binz;

   TAxis *xaxis = h->GetXaxis();
   TAxis *yaxis = h->GetYaxis();
   TAxis *zaxis = h->GetZaxis();
   Int_t entry = 0;
   for (binz=zaxis->GetFirst();binz<=zaxis->GetLast();binz++) {
      for (biny=yaxis->GetFirst();biny<=yaxis->GetLast();biny++) {
         for (binx=xaxis->GetFirst();binx<=xaxis->GetLast();binx++) {
            bin = h->GetBin(binx,biny,binz);
            for (in=0;in<h->GetBinContent(bin);in++) {
               entry++;
            }
         }
      }
   }

   // if histogram has too many entries, rescale it
   // never draw more than kMaxEntry markers, otherwise this kills
   // the X server
   Double_t scale = 1.;
   if (entry > kMaxEntry) scale = kMaxEntry/Double_t(entry);

   //Create or modify 3-d view object
   TView *view = gPad->GetView();
   if (!view) {
      gPad->Range(-1,-1,1,1);
      view = new TView(1);
   }
   view->SetRange(xaxis->GetBinLowEdge(xaxis->GetFirst()),
                  yaxis->GetBinLowEdge(yaxis->GetFirst()),
                  zaxis->GetBinLowEdge(zaxis->GetFirst()),
                  xaxis->GetBinUpEdge(xaxis->GetLast()),
                  yaxis->GetBinUpEdge(yaxis->GetLast()),
                  zaxis->GetBinUpEdge(zaxis->GetLast()));

   if (entry == 0) return;
   Int_t nmk = TMath::Min(kMaxEntry,entry);
   TPolyMarker3D *pm3d    = new TPolyMarker3D(nmk);
   pm3d->SetMarkerStyle(h->GetMarkerStyle());
   pm3d->SetMarkerColor(h->GetMarkerColor());
   pm3d->SetMarkerSize(h->GetMarkerSize());
   gPad->Modified(kTRUE);

   entry = 0;
   Double_t x,y,z,xw,yw,zw,xp,yp,zp;
   Int_t ncounts;
   for (binz=zaxis->GetFirst();binz<=zaxis->GetLast();binz++) {
      z  = zaxis->GetBinLowEdge(binz);
      zw = zaxis->GetBinWidth(binz);
      for (biny=yaxis->GetFirst();biny<=yaxis->GetLast();biny++) {
         y  = yaxis->GetBinLowEdge(biny);
         yw = yaxis->GetBinWidth(biny);
         for (binx=xaxis->GetFirst();binx<=xaxis->GetLast();binx++) {
            x  = xaxis->GetBinLowEdge(binx);
            xw = xaxis->GetBinWidth(binx);
            bin = h->GetBin(binx,biny,binz);
            ncounts = Int_t(h->GetBinContent(bin)*scale+0.5);
            for (in=0;in<ncounts;in++) {
               xp = x + xw*gRandom->Rndm(in);
               yp = y + yw*gRandom->Rndm(in);
               zp = z + zw*gRandom->Rndm(in);
               pm3d->SetPoint(entry,xp,yp,zp);
               entry++;
            }
         }
      }
   }
   pm3d->Paint(option);
   delete pm3d;
}

//______________________________________________________________________________
void TPolyMarker3D::Print(Option_t *option) const
{
   // Print 3-D polymarker with its attributes on stdout.

   printf("    TPolyMarker3D N=%d, Option=%s\n",fN,option);
   TString opt = option;
   opt.ToLower();
   if (opt.Contains("all")) {
      for (Int_t i=0;i<Size();i++) {
         printf(" x[%d]=%g, y[%d]=%g, z[%d]=%g\n",i,fP[3*i],i,fP[3*i+1],i,fP[3*i+2]);
      }
   }
}

//______________________________________________________________________________
void TPolyMarker3D::SavePrimitive(ofstream &out, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream.

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TPolyMarker3D::Class())) {
      out<<"   ";
   } else {
      out<<"   TPolyMarker3D *";
   }
   out<<"pmarker3D = new TPolyMarker3D("<<fN<<","<<GetMarkerStyle()<<","<<quote<<fOption<<quote<<");"<<endl;

   SaveMarkerAttributes(out,"pmarker3D",1,1,1);

   for (Int_t i=0;i<Size();i++) {
      out<<"   pmarker3D->SetPoint("<<i<<","<<fP[3*i]<<","<<fP[3*i+1]<<","<<fP[3*i+2]<<");"<<endl;
   }
   out<<"   pmarker3D->Draw();"<<endl;
}

//______________________________________________________________________________
Int_t TPolyMarker3D::SetNextPoint(Double_t x, Double_t y, Double_t z)
{
   // Set point following LastPoint to x, y, z.
   // Returns index of the point (new last point).

   fLastPoint++;
   SetPoint(fLastPoint, x, y, z);
   return fLastPoint;
}

//______________________________________________________________________________
void TPolyMarker3D::SetPoint(Int_t n, Double_t x, Double_t y, Double_t z)
{
   // Set point n to x, y, z.
   // If n is more then the current TPolyMarker3D size (n > fN) then
   // the polymarker will be resized to contain at least n points.

   if (n < 0) return;
   if (!fP || n >= fN) {
      // re-allocate the object
      Int_t newN = TMath::Max(2*fN,n+1);
      Float_t *savepoint = new Float_t [kDimension*newN];
      if (fP && fN){
         memcpy(savepoint,fP,kDimension*fN*sizeof(Float_t));
         memset(&savepoint[kDimension*fN],0,(newN-fN)*sizeof(Float_t));
         delete [] fP;
      }
      fP = savepoint;
      fN = newN;
   }
   fP[kDimension*n  ] = x;
   fP[kDimension*n+1] = y;
   fP[kDimension*n+2] = z;
   fLastPoint = TMath::Max(fLastPoint,n);
}

//______________________________________________________________________________
void TPolyMarker3D::SetPolyMarker(Int_t n, Float_t *p, Marker_t marker, Option_t *option)
{
   // Re-initialize polymarker with n points from p. If p=0 initialize with 0.

   fN = n;
   if (fP) delete [] fP;
   fP = new Float_t [3*fN];
   for (Int_t i = 0; i < fN; i++) {
      if (p) {
         fP[3*i]   = p[3*i];
         fP[3*i+1] = p[3*i+1];
         fP[3*i+2] = p[3*i+2];
      } else {
         memset(fP,0,kDimension*fN*sizeof(Float_t));
      }
   }
   SetMarkerStyle(marker);
   fOption = option;
   fLastPoint = fN-1;
}

//______________________________________________________________________________
void TPolyMarker3D::SetPolyMarker(Int_t n, Double_t *p, Marker_t marker, Option_t *option)
{
   // Re-initialize polymarker with n points from p. If p=0 initialize with 0.

   fN = n;
   if (fP) delete [] fP;
   fP = new Float_t [3*fN];
   for (Int_t i = 0; i < fN; i++) {
      if (p) {
         fP[3*i]   = (Float_t) p[3*i];
         fP[3*i+1] = (Float_t) p[3*i+1];
         fP[3*i+2] = (Float_t) p[3*i+2];
      } else {
         memset(fP,0,kDimension*fN*sizeof(Float_t));
      }
   }
   SetMarkerStyle(marker);
   fOption = option;
   fLastPoint = fN-1;
}

//_______________________________________________________________________
void TPolyMarker3D::Streamer(TBuffer &b)
{
   // Stream a 3-D polymarker object.

   UInt_t R__s, R__c;
   if (b.IsReading()) {
      b.ReadVersion(&R__s, &R__c);
      TObject::Streamer(b);
      TAttMarker::Streamer(b);
      b >> fN;
      if (fN) {
         fP = new Float_t[kDimension*fN];
         b.ReadFastArray(fP,kDimension*fN);
      }
      fLastPoint = fN-1;
      fOption.Streamer(b);
      b.CheckByteCount(R__s, R__c, TPolyMarker3D::IsA());
   } else {
      R__c = b.WriteVersion(TPolyMarker3D::IsA(), kTRUE);
      TObject::Streamer(b);
      TAttMarker::Streamer(b);
      Int_t size = Size();
      b << size;
      if (size) b.WriteFastArray(fP, kDimension*size);
      fOption.Streamer(b);
      b.SetByteCount(R__c, kTRUE);
   }
}
