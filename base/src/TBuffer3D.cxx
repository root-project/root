// @(#)root/base:$Name:  $:$Id: TBuffer3D.cxx,v 1.00
// Author: Olivier Couet   05/05/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBuffer3D.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
#include "TView.h"

ClassImp(TBuffer3D)


//______________________________________________________________________________
TBuffer3D::TBuffer3D()
{
   fType     = -1;
   fOption   = kPAD;
   fId       = 0;
   fNbPnts   = 0;
   fNbSegs   = 0;
   fNbPols   = 0;
   fPnts     = 0;
   fSegs     = 0;
   fPols     = 0;
   fPntsSize = 0;
   fSegsSize = 0;
   fPolsSize = 0;
}


//______________________________________________________________________________
TBuffer3D::TBuffer3D(Int_t n1, Int_t n2, Int_t n3)
{
   fPntsSize = n1;
   fSegsSize = n2;
   fPolsSize = n3;

   fType     = -1;
   fOption   = kPAD;
   fId       = 0;

   fNbPnts   = 0;
   fNbSegs   = 0;
   fNbPols   = 0;

   fPnts     = 0;
   fSegs     = 0;
   fPols     = 0;
   if ( fPntsSize>0 ) fPnts = new Double_t[fPntsSize];
   if ( fSegsSize>0 ) fSegs = new Int_t[fSegsSize];
   if ( fPolsSize>0 ) fPols = new Int_t[fPolsSize];
}


//______________________________________________________________________________
TBuffer3D::~TBuffer3D()
{
   if (fPnts) delete [] fPnts;
   if (fSegs) delete [] fSegs;
   if (fPols) delete [] fPols;
}


//______________________________________________________________________________
void TBuffer3D::Paint(Option_t *option)
{
   Int_t i, i0, i1, i2;
   Double_t x0, y0, z0, x1, y1, z1;
   TVirtualViewer3D *viewer3D;
   TView *view;

   // Compute the shape range and update gPad->fView
   switch (fOption) {
      case kRANGE:
         x0 = x1 = fPnts[0];
         y0 = y1 = fPnts[1];
         z0 = z1 = fPnts[2];
         for (i=1; i<fNbPnts; i++) {
         i0 = 3*i; i1 = i0+1; i2 = i0+2;
            x0 = fPnts[i0] < x0 ? fPnts[i0] : x0;
            y0 = fPnts[i1] < y0 ? fPnts[i1] : y0;
            z0 = fPnts[i2] < z0 ? fPnts[i2] : z0;
            x1 = fPnts[i0] > x1 ? fPnts[i0] : x1;
            y1 = fPnts[i1] > y1 ? fPnts[i1] : y1;
            z1 = fPnts[i2] > z1 ? fPnts[i2] : z1;
         }
         view = gPad->GetView();
         if (view->GetAutoRange()) view->SetRange(x0,y0,z0,x1,y1,z1,2);
         break;
   
      // Update viewer
      case kSIZE:
      case kX3D:
      case kOGL:
         viewer3D = gPad->GetViewer3D();
         if (viewer3D) viewer3D->UpdateScene(option);
         break;

      // Paint this in gPad
      case kPAD:
      default:
         if ( fType==kMARKER ) {
            view = gPad->GetView();
            Double_t pndc[3], temp[3];
            for (i=0; i<fNbPnts; i++) {
               for ( i0=0; i0<3; i0++ ) temp[i0] = fPnts[3*i+i0];
               view->WCtoNDC(temp, pndc);
               gPad->PaintPolyMarker(1, &pndc[0], &pndc[1]);
            }
         } else {
            for (i=0; i<fNbSegs; i++) {
               i0 = 3*fSegs[3*i+1];
               Double_t *ptpoints_0 = &(fPnts[i0]);
               i0 = 3*fSegs[3*i+2];
               Double_t *ptpoints_3 = &(fPnts[i0]);
               gPad->PaintLine3D(ptpoints_0, ptpoints_3);
            }
         }
         break;
   }
}


//______________________________________________________________________________
void TBuffer3D::ReAllocate(Int_t n1, Int_t n2, Int_t n3)
{
   if (n1 > fPntsSize) {
      delete [] fPnts;
      fPntsSize = n1;
      fPnts     = new Double_t[fPntsSize];
   }
   if (n2 > fSegsSize) {
      delete [] fSegs;
      fSegsSize = n2;
      fSegs     = new Int_t[fSegsSize];
   }
   if (n3 > fPolsSize) {
      delete [] fPols;
      fPolsSize = n3;
      fPols     = new Int_t[fPolsSize];
   }
}
