// @(#)root/g3d:$Name:  $:$Id: TPolyMarker3D.cxx,v 1.6 2001/03/23 13:25:04 brun Exp $
// Author: Nenad Buncic   21/08/95

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
#include "TView.h"
#include "TStyle.h"
#include "TPolyMarker3D.h"
#include "TPoint.h"
#include "TVirtualPad.h"
#include "TVirtualPS.h"
#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TPadView3D.h"
#include "TH1.h"
#include "TH3.h"
#include "TRandom.h"

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
  : TObject(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*-*PolyMarker3D default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

        fN = 0;
        fP = 0;
        fLastPoint = -1;
}


//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(Int_t n, Marker_t marker, Option_t *option)
  : TObject(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*PolyMarker3D normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================

   fLastPoint = -1;
   fN = n;
   fP = new Float_t [kDimension*fN];
   for (Int_t i = 0; i < kDimension*fN; i++)  fP[i] = 0;
   fOption = option;
   SetMarkerStyle(marker);
   SetBit(kCanDelete);
}


//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(Int_t n, Float_t *p, Marker_t marker, Option_t *option)
  : TObject(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*PolyMarker3D constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        ========================

   fLastPoint = -1;
   fN = 0;
   fP = 0;
   if (n > 0) {
     fN = n;
     fP = new Float_t [kDimension*fN];
     if (p) {
         for (Int_t i = 0; i < kDimension*fN; i++)  fP[i] = p[i];
         fLastPoint = fN-1;
     }
     else
         memset(fP,0,kDimension*fN*sizeof(Float_t));
   }
   SetMarkerStyle(marker);
   SetBit(kCanDelete);
   fOption = option;
}

//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(Int_t n, Double_t *p, Marker_t marker, Option_t *option)
  : TObject(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*PolyMarker3D constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        ========================

   fLastPoint = -1;
   fN = 0;
   fP = 0;
   if (n > 0) {
     fN = n;
     fP = new Float_t [kDimension*fN];
     if (p) {
         for (Int_t i = 0; i < kDimension*fN; i++)  fP[i] = p[i];
         fLastPoint = fN-1;
     }
     else
         memset(fP,0,kDimension*fN*sizeof(Float_t));
   }
   SetMarkerStyle(marker);
   SetBit(kCanDelete);
   fOption = option;
}


//______________________________________________________________________________
TPolyMarker3D::~TPolyMarker3D()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*PolyMarker3D destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          =======================

        fN = 0;
        if (fP) delete [] fP;
        fLastPoint = -1;
}


//______________________________________________________________________________
TPolyMarker3D::TPolyMarker3D(const TPolyMarker3D &polymarker)
{
   ((TPolyMarker3D&)polymarker).Copy(*this);
}

//______________________________________________________________________________
void TPolyMarker3D::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Copy polymarker*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                              ===============

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
//*-*-*-*-*-*-*Compute distance from point px,py to a 3-D polymarker*-*-*-*-*-*-*
//*-*          =====================================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each segment
//*-*  of the polyline.
//*-*  Returns when the distance found is below DistanceMaximum.
//*-*  The distance is computed in pixels units.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   const Int_t inaxis = 7;
   Int_t dist = 9999;

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());

//*-*- return if point is not in the user area
   if (px < puxmin - inaxis) return dist;
   if (py > puymin + inaxis) return dist;
   if (px > puxmax + inaxis) return dist;
   if (py < puymax - inaxis) return dist;

   TView *view = gPad->GetView();
   if (!view) return dist;
   Int_t i, dpoint;
   Float_t xndc[3];
   Int_t x1,y1;
   for (i=0;i<Size();i++) {
      view->WCtoNDC(&fP[3*i], xndc);
      x1     = gPad->XtoAbsPixel(xndc[0]);
      y1     = gPad->YtoAbsPixel(xndc[1]);
      dpoint = Int_t(TMath::Sqrt((((Double_t)px-x1)*((Double_t)px-x1) 
                                + ((Double_t)py-y1)*((Double_t)py-y1))));
      if (dpoint < dist) dist = dpoint;
   }
   return dist;
}


//______________________________________________________________________________
void TPolyMarker3D::Draw(Option_t *option)
{
//*-*-*-*-*-*-*Draws PolyMarker and adds it into the ListOfPrimitives*-*-*-*-*-*
//*-*          ======================================================

   AppendPad(option);

}


//______________________________________________________________________________
void TPolyMarker3D::DrawPolyMarker(Int_t n, Float_t *p, Marker_t, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Draws PolyMarker*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ================

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
//*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*-*-*-*-*-*-*
//*-*                =========================================
//*-*
//*-*  This member function must be implemented to realize the action
//*-*  corresponding to the mouse click on the object in the window
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

        if (gPad->GetView())
                gPad->GetView()->ExecuteRotateView(event, px, py);

}

//______________________________________________________________________________
void TPolyMarker3D::ls(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*-*List PolyMarker's contents*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==========================

   TROOT::IndentLevel();
   cout << "    TPolyMarker3D  N=" << Size() <<" Option="<<option<<endl;
}

//______________________________________________________________________________
void TPolyMarker3D::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Paint PolyMarker*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            ================
    TPadView3D *view3D = (TPadView3D*)gPad->GetView3D();
    if (view3D) view3D->PaintPolyMarker(this,option);
    //*-* Check if option is 'x3d'.      NOTE: This is a simple checking
    //                                         but since there is no other
    //                                         options yet, this works fine.

    if ((*option != 'x') && (*option != 'X')) {
        Marker_t marker = GetMarkerStyle();
        PaintPolyMarker(Size(), fP, marker, option);
    }
    else {
        Int_t size = Size();
        Int_t mode;
        Int_t i, j, k, n;

        X3DBuffer *buff = new X3DBuffer;
        if(!buff) return;

        if (size > 10000) mode = 1;         // One line marker    '-'
        else if (size > 3000) mode = 2;     // Two lines marker   '+'
        else mode = 3;                      // Three lines marker '*'

        buff->numSegs   = size*mode;
        buff->numPoints = buff->numSegs*2;
        buff->numPolys  = 0;         //NOTE: Because of different structure, our
        buff->polys     = NULL;      //      TPolyMarker3D can't use polygons


    //*-* Allocate memory for points *-*
        Double_t delta = 0.002;

        buff->points = new Float_t[buff->numPoints*3];
        if (buff->points) {
            for (i = 0; i < size; i++) {
                for (j = 0; j < mode; j++) {
                    for (k = 0; k < 2; k++) {
                        delta *= -1;
                        for (n = 0; n < 3; n++) {
                            buff->points[mode*6*i+6*j+3*k+n] =
                                fP[3*i+n] * (1 + (j == n ? delta : 0));
                        }
                    }
                }
            }
        }

        Int_t c = ((GetMarkerColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 8
        if (c < 0) c = 0;

    //*-* Allocate memory for segments *-*
        buff->segs = new Int_t[buff->numSegs*3];
        if (buff->segs) {
            for (i = 0; i < buff->numSegs; i++) {
                buff->segs[3*i  ] = c;
                buff->segs[3*i+1] = 2*i;
                buff->segs[3*i+2] = 2*i+1;
            }
        }

        if (buff->points && buff->segs)    //If everything seems to be OK ...
            FillX3DBuffer(buff);
        else {                            // ... something very bad was happened
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }

        if (buff->points)   delete [] buff->points;
        if (buff->segs)     delete [] buff->segs;
        if (buff->polys)    delete [] buff->polys;
        if (buff)           delete    buff;
    }
}

//______________________________________________________________________________
void TPolyMarker3D::PaintH3(TH1 *h, Option_t *option)
{
//     Paint 3-d histogram h with 3d polymarkers

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
void TPolyMarker3D::PaintPolyMarker(Int_t n, Float_t *p, Marker_t, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint polymarker in CurrentPad World coordinates*-*-*-*-*-*-*-*
//*-*              ================================================

   if (n <= 0) return;

   //Create temorary storage
   TPoint *pxy = new TPoint[n];
   Double_t *x  = new Double_t[n];
   Double_t *y  = new Double_t[n];
   Double_t xndc[3], temp[3];
   Float_t *ptr = p;

   TView *view = gPad->GetView();      //Get current 3-D view
   if(!view) return;                           //Check if `view` is valid

//*-*- convert points from world to pixel coordinates
   Int_t nin = 0;
   for (Int_t i = 0; i < n; i++) {
      for (Int_t j=0;j<3;j++) temp[j] = ptr[j];
      view->WCtoNDC(temp, xndc);
      ptr += 3;
      if (xndc[0] < gPad->GetX1() || xndc[0] > gPad->GetX2()) continue;
      if (xndc[1] < gPad->GetY1() || xndc[1] > gPad->GetY2()) continue;
      x[nin] = xndc[0];
      y[nin] = xndc[1];
      pxy[nin].fX = gPad->XtoPixel(x[nin]);
      pxy[nin].fY = gPad->YtoPixel(y[nin]);
      nin++;
   }

   TAttMarker::Modify();  //Change marker attributes only if necessary

//*-*- invoke the graphics subsystem
   if (!gPad->IsBatch()) gVirtualX->DrawPolyMarker(nin, pxy);


   if (gVirtualPS) {
      gVirtualPS->DrawPolyMarker(nin, x, y);
   }
   delete [] x;
   delete [] y;

   delete [] pxy;
}


//______________________________________________________________________________
void TPolyMarker3D::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Print PolyMarker Info*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                          =====================

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
    // Save primitive as a C++ statement(s) on output stream out

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
void TPolyMarker3D::SetPoint(Int_t n, Double_t x, Double_t y, Double_t z)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*Set point n to x, y, z*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                        ======================
//*-*  if n is more then the current TPolyMarker3D size (n > fN) - re-allocate this
//*-*

   if (n < 0) return;
   if (!fP || n >= fN) {
   // re-allocate the object
      Float_t *savepoint = new Float_t [kDimension*(n+1)];
      if (fP && fN){
         memcpy(savepoint,fP,kDimension*fN*sizeof(Float_t));
        delete [] fP;
      }
      fP = savepoint;
      fN = n+1;
   }
   fP[kDimension*n  ] = x;
   fP[kDimension*n+1] = y;
   fP[kDimension*n+2] = z;
   fLastPoint = TMath::Max(fLastPoint,n);
}

//______________________________________________________________________________
Int_t TPolyMarker3D::SetNextPoint(Double_t x, Double_t y, Double_t z)
{
//*-*-*-*-*-*-*-*-*-*-*-*Set "next" point to x, y, z *-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============================
//*-*     SetNextPoint:  returns the index this point has occupied
//*-*
   fLastPoint++;
   SetPoint(fLastPoint, x, y, z);
   return fLastPoint;
}

//______________________________________________________________________________
void TPolyMarker3D::SetPolyMarker(Int_t n, Float_t *p, Marker_t marker, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Loads n points from array p*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================

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
       fLastPoint = n-1;
}

//______________________________________________________________________________
void TPolyMarker3D::SetPolyMarker(Int_t n, Double_t *p, Marker_t marker, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Loads n points from array p*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===========================

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
       fLastPoint = n-1;
}

//______________________________________________________________________________
void TPolyMarker3D::Sizeof3D() const
{
//*-*-*-*-*-*Return total size of this 3-D shape with its attributes*-*-*-*-*-*-*
//*-*        =======================================================

    Int_t mode;
    Int_t size = Size();

    if (size > 10000) mode = 1;         // One line marker    '-'
    else if (size > 3000) mode = 2;     // Two lines marker   '+'
    else mode = 3;                      // Three lines marker '*'

    gSize3D.numSegs   += size*mode;
    gSize3D.numPoints += size*mode*2;
    gSize3D.numPolys  += 0;
}


//______________________________________________________________________________
void TPolyMarker3D::SizeofH3(TH1 *h)
{
//*-*-*-*-*-*Return total size of 3-D histogram h*-*-*-*-*-*-*
//*-*        ====================================

   // take into account the 4 polylines of the OutlinetoCube
   gSize3D.numSegs   += 4*3;
   gSize3D.numPoints += 4*4;

   if (h->GetEntries() <= 0) return;
   Int_t nx  = h->GetXaxis()->GetNbins();
   Int_t ny  = h->GetYaxis()->GetNbins();
   Int_t nz  = h->GetZaxis()->GetNbins();
   Int_t entry = 0;
   Int_t bin, binx, biny, binz;
   for (binz=1;binz<=nz;binz++) {
      for (biny=1;biny<=ny;biny++) {
         for (binx=1;binx<=nx;binx++) {
            bin = h->GetBin(binx,biny,binz);
            for (Int_t in=0;in<h->GetBinContent(bin);in++) {
               entry++;
            }
         }
      }
   }
    Int_t mode;
    if (entry > 10000) mode = 1;         // One line marker    '-'
    else if (entry > 3000) mode = 2;     // Two lines marker   '+'
    else mode = 3;                       // Three lines marker '*'
    gSize3D.numSegs   += entry*mode;
    gSize3D.numPoints += entry*mode*2;
    gSize3D.numPolys  += 0;
}


//_______________________________________________________________________
void TPolyMarker3D::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
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
