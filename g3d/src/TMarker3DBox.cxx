// @(#)root/g3d:$Name:  $:$Id: TMarker3DBox.cxx,v 1.6 2002/01/24 11:39:27 rdm Exp $
// Author: "Valery fine"   31/10/97


/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TView.h"
#include "TMarker3DBox.h"
#include "TVirtualPad.h"
#include "TVirtualGL.h"
#include "TPadView3D.h"
#include "TH1.h"
#include "TH3.h"
#include "TFile.h"

ClassImp(TMarker3DBox)

//______________________________________________________________________________
// Marker3DBox is a special 3-D marker designed for event display.
// It has the following parameters:
//    fX;               X coordinate of the center of the box
//    fY;               Y coordinate of the center of the box
//    fZ;               Z coordinate of the center of the box
//    fDx;              half length in X
//    fDy;              half length in Y
//    fDz;              half length in Z
//    fTheta;           Angle of box z axis with respect to main Z axis
//    fPhi;             Angle of box x axis with respect to main Xaxis
//    fRefObject;       A reference to an object
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/Marker3DBox.gif"> </P> End_Html




//______________________________________________________________________________
TMarker3DBox::TMarker3DBox()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Marker3DBox  default constructor*-*-*-*-*-*-*-*-*-*
//*-*                      ================================

   fRefObject = 0;
   //just in case of a stupid call to this constructor
   fDx = 1;
   fDy = 1;
   fDz = 1;

   fX  = 0;
   fY  = 0;
   fZ  = 0;

   fTheta = 0;
   fPhi   = 0;
}


//______________________________________________________________________________
TMarker3DBox::TMarker3DBox( Float_t x, Float_t y, Float_t z,
                            Float_t dx, Float_t dy, Float_t dz,
                            Float_t theta, Float_t phi)
              :TAttLine(1,1,1), TAttFill(1,0)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Marker3DBox normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

    fDx = dx;
    fDy = dy;
    fDz = dz;

    fX  = x;
    fY  = y;
    fZ  = z;

    fTheta = theta;
    fPhi   = phi;
    fRefObject = 0;
}


//______________________________________________________________________________
TMarker3DBox::~TMarker3DBox()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Marker3DBox shape default destructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =============================

}

//______________________________________________________________________________
Int_t TMarker3DBox::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*Compute distance from point px,py to a Marker3DBox*-*-*-*-*
//*-*            ==================================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each corner
//*-*  point of the Marker3DBox.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   const Int_t numPoints = 8;
   Int_t dist = 9999;
   Float_t points[3*numPoints];

   TView *view = gPad->GetView();
   if (!view) return dist;
   const Int_t seg1[12] = {0,1,2,3,4,5,6,7,0,1,2,3};
   const Int_t seg2[12] = {1,2,3,0,5,6,7,4,4,5,6,7};

   SetPoints(points);

   Int_t i, i1, i2, dsegment;
   Double_t x1,y1,x2,y2;
   Float_t xndc[3];
   for (i = 0; i < 12; i++) {
      i1 = 3*seg1[i];
      view->WCtoNDC(&points[i1], xndc);
      x1 = xndc[0];
      y1 = xndc[1];

      i2 = 3*seg2[i];
      view->WCtoNDC(&points[i2], xndc);
      x2 = xndc[0];
      y2 = xndc[1];
      dsegment = DistancetoLine(px,py,x1,y1,x2,y2);
      if (dsegment < dist) dist = dsegment;
   }
   if (dist < 5) {
      gPad->SetCursor(kCross);
      if (fRefObject) {gPad->SetSelected(fRefObject); return 0;}
   }
   return dist;
}


//______________________________________________________________________________
void TMarker3DBox::ExecuteEvent(Int_t event, Int_t px, Int_t py)
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
void TMarker3DBox::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*Paint this 3-D box marker with its current attributes*-*-*-*
//*-*            =====================================================

   const Int_t numpoints = 8;

//*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   SetPoints(points);

   TPadView3D *view3D = (TPadView3D*)gPad->GetView3D();
   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (view3D && !rangeView) {
     gVirtualGL->SetLineAttr(GetLineColor(),GetLineWidth());
     PaintGLPoints(points);
   }

   Int_t c = ((GetLineColor() % 8) - 1) * 4;  // Basic colors: 0, 1, ... 7
   if (c < 0) c = 0;

//*-* Allocate memory for segments *-*

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = 8;
        buff->numSegs   = 12;
        buff->numPolys  = 6;
    }

//*-* Allocate memory for points *-*

    buff->points = points;
    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {
        buff->segs[ 0] = c;    buff->segs[ 1] = 0;    buff->segs[ 2] = 1;
        buff->segs[ 3] = c+1;  buff->segs[ 4] = 1;    buff->segs[ 5] = 2;
        buff->segs[ 6] = c+1;  buff->segs[ 7] = 2;    buff->segs[ 8] = 3;
        buff->segs[ 9] = c;    buff->segs[10] = 3;    buff->segs[11] = 0;
        buff->segs[12] = c+2;  buff->segs[13] = 4;    buff->segs[14] = 5;
        buff->segs[15] = c+2;  buff->segs[16] = 5;    buff->segs[17] = 6;
        buff->segs[18] = c+3;  buff->segs[19] = 6;    buff->segs[20] = 7;
        buff->segs[21] = c+3;  buff->segs[22] = 7;    buff->segs[23] = 4;
        buff->segs[24] = c;    buff->segs[25] = 0;    buff->segs[26] = 4;
        buff->segs[27] = c+2;  buff->segs[28] = 1;    buff->segs[29] = 5;
        buff->segs[30] = c+1;  buff->segs[31] = 2;    buff->segs[32] = 6;
        buff->segs[33] = c+3;  buff->segs[34] = 3;    buff->segs[35] = 7;
    }

//*-* Allocate memory for polygons *-*

    buff->polys = new Int_t[buff->numPolys*6];
    if (buff->polys) {
        buff->polys[ 0] = c;   buff->polys[ 1] = 4;  buff->polys[ 2] = 0;
        buff->polys[ 3] = 9;   buff->polys[ 4] = 4;  buff->polys[ 5] = 8;
        buff->polys[ 6] = c+1; buff->polys[ 7] = 4;  buff->polys[ 8] = 1;
        buff->polys[ 9] = 10;  buff->polys[10] = 5;  buff->polys[11] = 9;
        buff->polys[12] = c;   buff->polys[13] = 4;  buff->polys[14] = 2;
        buff->polys[15] = 11;  buff->polys[16] = 6;  buff->polys[17] = 10;
        buff->polys[18] = c+1; buff->polys[19] = 4;  buff->polys[20] = 3;
        buff->polys[21] = 8;   buff->polys[22] = 7;  buff->polys[23] = 11;
        buff->polys[24] = c+2; buff->polys[25] = 4;  buff->polys[26] = 0;
        buff->polys[27] = 3;   buff->polys[28] = 2;  buff->polys[29] = 1;
        buff->polys[30] = c+3; buff->polys[31] = 4;  buff->polys[32] = 4;
        buff->polys[33] = 5;   buff->polys[34] = 6;  buff->polys[35] = 7;
    }

    //*-* Paint in the pad
    PaintShape(buff,rangeView);

    if (strstr(option, "x3d")) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    delete [] points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;

}

//______________________________________________________________________________
void TMarker3DBox::PaintGLPoints(Float_t *vertex)
{
//*-*-*-*-*-*-*-*-*-*-*Direct copy of TBRIK::PaintGLPoints*-*-*-*-*-*-*-*-*-*
//*-*                  ===================================
    gVirtualGL->PaintBrik(vertex);
}

//______________________________________________________________________________
void TMarker3DBox::PaintH3(TH1 *h, Option_t *option)
{
//*-*-*-*-*-*-*-*Paint 3-d histogram h with marker3dboxes*-*-*-*
//*-*            ========================================

   Int_t bin,ix,iy,iz;
   Double_t xmin,xmax,ymin,ymax,zmin,zmax,wmin,wmax,w;
   TAxis *xaxis = h->GetXaxis();
   TAxis *yaxis = h->GetYaxis();
   TAxis *zaxis = h->GetZaxis();

   //compute min and max of all cells
   wmin = wmax = 0;
   for (iz=zaxis->GetFirst();iz<=zaxis->GetLast();iz++) {
      for (iy=yaxis->GetFirst();iy<=yaxis->GetLast();iy++) {
         for (ix=xaxis->GetFirst();ix<=xaxis->GetLast();ix++) {
            bin = h->GetBin(ix,iy,iz);
            w = h->GetBinContent(bin);
            if (w > wmax) wmax = w;
            if (w < wmin) wmin = w;
         }
      }
   }

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

   //Draw TMarker3DBox with size proportional to cell content
   TMarker3DBox m3;
   m3.SetRefObject(h);
   m3.SetDirection(0,0);
   m3.SetLineColor(h->GetMarkerColor());
   Double_t scale;
   for (ix=xaxis->GetFirst();ix<=xaxis->GetLast();ix++) {
      xmin = h->GetXaxis()->GetBinLowEdge(ix);
      xmax = xmin + h->GetXaxis()->GetBinWidth(ix);
      for (iy=yaxis->GetFirst();iy<=yaxis->GetLast();iy++) {
         ymin = h->GetYaxis()->GetBinLowEdge(iy);
         ymax = ymin + h->GetYaxis()->GetBinWidth(iy);
         for (iz=zaxis->GetFirst();iz<=zaxis->GetLast();iz++) {
            zmin = h->GetZaxis()->GetBinLowEdge(iz);
            zmax = zmin + h->GetZaxis()->GetBinWidth(iz);
            bin = h->GetBin(ix,iy,iz);
            w = h->GetBinContent(bin);
            if (w == 0) continue;
            scale = (w-wmin)/(wmax-wmin);
            m3.SetPosition(0.5*(xmin+xmax),0.5*(ymin+ymax),0.5*(zmin+zmax));
            m3.SetSize(scale*(xmax-xmin),scale*(ymax-ymin),scale*(zmax-zmin));
            m3.Paint(option);
         }
      }
   }
}

//______________________________________________________________________________
void TMarker3DBox::PaintShape(X3DBuffer *buff, Bool_t rangeView)
{
//*-*-*-*-*Paint 3-D marker in current pad with its current attributes*-*-*-*-*
//*-*      ==========================================================

    //*-* Paint in the pad

    Double_t points[6], x0, y0, z0, x1, y1, z1;
    const Int_t kExpandView = 2;
    int i0;

    x0 = y0 = z0 = x1 = y1 = z1 = buff->points[0];

    TAttLine::Modify();  //Change line attributes only if necessary
    TAttFill::Modify();  //Change fill area attributes only if necessary

    for (Int_t i = 0; i < buff->numSegs; i++) {
        i0 = 3*buff->segs[3*i+1];
        points[0] = buff->points[i0++];
        points[1] = buff->points[i0++];
        points[2] = buff->points[i0];

        i0 = 3*buff->segs[3*i+2];
        points[3] = buff->points[i0++];
        points[4] = buff->points[i0++];
        points[5] = buff->points[i0];

        x0 = points[0] < x0 ? points[0] : x0;
        y0 = points[1] < y0 ? points[1] : y0;
        z0 = points[2] < z0 ? points[2] : z0;
        x1 = points[0] > x1 ? points[0] : x1;
        y1 = points[1] > y1 ? points[1] : y1;
        z1 = points[2] > z1 ? points[2] : z1;

       if (!rangeView) gPad->PaintLine3D(&points[0], &points[3]);
    }
    if (rangeView){
      TView *view = gPad->GetView();
      if (view->GetAutoRange()) view->SetRange(x0,y0,z0,x1,y1,z1,kExpandView);
    }

}

//______________________________________________________________________________
void TMarker3DBox::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   out<<"   "<<endl;
   if (gROOT->ClassSaved(TMarker3DBox::Class())) {
       out<<"   ";
   } else {
       out<<"   TMarker3DBox *";
   }
   out<<"marker3DBox = new TMarker3DBox("<<fX<<","
                                         <<fY<<","
                                         <<fZ<<","
                                         <<fDx<<","
                                         <<fDy<<","
                                         <<fDz<<","
                                         <<fTheta<<","
                                         <<fPhi<<");"<<endl;

   SaveLineAttributes(out,"marker3DBox",1,1,1);
   SaveFillAttributes(out,"marker3DBox",1,0);

   out<<"   marker3DBox->Draw();"<<endl;
}

//______________________________________________________________________________
void TMarker3DBox::SetDirection(Float_t theta, Float_t phi)
{
  fTheta = theta;
  fPhi   = phi;
}

//______________________________________________________________________________
void TMarker3DBox::SetSize(Float_t dx, Float_t dy, Float_t dz)
{
  fDx = dx;
  fDy = dy;
  fDz = dz;
}

//______________________________________________________________________________
void TMarker3DBox::SetPosition(Float_t x, Float_t y, Float_t z)
{
  fX  = x;
  fY  = y;
  fZ  = z;
}

//______________________________________________________________________________
void TMarker3DBox::SetPoints(Float_t *buff)
{
//*-*-*-*-*-*-*-*-*-*-*Direct copy of TBRIK::SetPoints *-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================

    if (buff) {
        buff[ 0] = -fDx;    buff[ 1] = -fDy;    buff[ 2] = -fDz;
        buff[ 3] =  fDx;    buff[ 4] = -fDy;    buff[ 5] = -fDz;
        buff[ 6] =  fDx;    buff[ 7] =  fDy;    buff[ 8] = -fDz;
        buff[ 9] = -fDx;    buff[10] =  fDy;    buff[11] = -fDz;
        buff[12] = -fDx;    buff[13] = -fDy;    buff[14] =  fDz;
        buff[15] =  fDx;    buff[16] = -fDy;    buff[17] =  fDz;
        buff[18] =  fDx;    buff[19] =  fDy;    buff[20] =  fDz;
        buff[21] = -fDx;    buff[22] =  fDy;    buff[23] =  fDz;
    }

   Double_t x, y, z;
   const Double_t kPI = TMath::Pi();
   Double_t theta  = fTheta*kPI/180;
   Double_t phi    = fPhi*kPI/180;
   Double_t sinth = TMath::Sin(theta);
   Double_t costh = TMath::Cos(theta);
   Double_t sinfi = TMath::Sin(phi);
   Double_t cosfi = TMath::Cos(phi);

   //
   // Matrix to convert from fruit frame to master frame
   //

   Double_t M[9];
   M[0] =  costh * cosfi;       M[1] = -sinfi;          M[2] = sinth*cosfi;
   M[3] =  costh * sinfi;       M[4] =  cosfi;          M[5] = sinth*sinfi;
   M[6] = -sinth;               M[7] =  0;              M[8] = costh;
   for (Int_t i = 0; i < 8; i++) {
      x = buff[3*i];
      y = buff[3*i+1];
      z = buff[3*i+2];

      buff[3*i]   = fX + M[0] * x + M[1] * y + M[2] * z;
      buff[3*i+1] = fY + M[3] * x + M[4] * y + M[5] * z;
      buff[3*i+2] = fZ + M[6] * x + M[7] * y + M[8] * z;
   }

}

//______________________________________________________________________________
void TMarker3DBox::Sizeof3D() const
{
//*-*-*-*-*-*-*Return total X3D size of this marker*-*-*-*-*-*-*-*
//*-*          ====================================

    gSize3D.numPoints += 8;
    gSize3D.numSegs   += 12;
    gSize3D.numPolys  += 6;
}

//______________________________________________________________________________
void TMarker3DBox::SizeofH3(TH1 *h)
{
//*-*-*-*-*-*-*Return total X3D size of 3-D histogram h*-*-*-*-*-*-*-*
//*-*          ========================================

   // take into account the 4 polylines of the OutlinetoCube
   gSize3D.numSegs   += 4*3;
   gSize3D.numPoints += 4*4;

   Int_t ix,iy,iz;
   Int_t nbinsx = h->GetXaxis()->GetNbins();
   Int_t nbinsy = h->GetYaxis()->GetNbins();
   Int_t nbinsz = h->GetZaxis()->GetNbins();
   Int_t ncells = 0;
   for (ix=1;ix<=nbinsx;ix++) {
      for (iy=1;iy<=nbinsy;iy++) {
         for (iz=1;iz<=nbinsz;iz++) {
            if (((TH3F*)h)->GetBinContent(((TH3F*)h)->GetBin(ix,iy,iz)) != 0)
                ncells++;
         }
      }
   }
   gSize3D.numPoints += 8*ncells;
   gSize3D.numSegs   += 12*ncells;
   gSize3D.numPolys  += 6*ncells;
}

//______________________________________________________________________________
void TMarker3DBox::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMarker3DBox.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TMarker3DBox::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      TAttLine::Streamer(R__b);
      TAttFill::Streamer(R__b);
      TFile *file = (TFile*)R__b.GetParent();
      if (file) {
         if (file->GetVersion() > 22300) TAtt3D::Streamer(R__b);
      } else {
         TAtt3D::Streamer(R__b);
       }
      R__b >> fX;
      R__b >> fY;
      R__b >> fZ;
      R__b >> fDx;
      R__b >> fDy;
      R__b >> fDz;
      R__b >> fTheta;
      R__b >> fPhi;
      R__b >> fRefObject;
      R__b.CheckByteCount(R__s, R__c, TMarker3DBox::IsA());
      //====end of old versions

   } else {
      TMarker3DBox::Class()->WriteBuffer(R__b,this);
   }
}

