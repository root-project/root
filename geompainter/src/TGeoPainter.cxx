// Author: Andrei Gheata   05/03/02
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TView.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TPad.h"
#include "TVirtualGL.h"

#include "TGeoSphere.h"
#include "TGeoPcon.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoPainter.h"

ClassImp(TGeoPainter)

//______________________________________________________________________________
TGeoPainter::TGeoPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default constructor*-*-*-*-*-*-*-*-*
//*-*                  ====================================
   TVirtualGeoPainter::SetPainter(this);
   fNsegments = 20;
   if (gGeoManager) fGeo = gGeoManager;
   else Error("ctor", "No geometry loaded");
}
//______________________________________________________________________________
TGeoPainter::~TGeoPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default destructor*-*-*-*-*-*-*-*-*
//*-*                  ===================================
}
//______________________________________________________________________________
Int_t TGeoPainter::DistanceToPrimitive(Int_t px, Int_t py)
{
   return 0;
}
//______________________________________________________________________________
void TGeoPainter::Draw(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   gGeoManager->GetTopVolume()->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = new TView(1);
      view->SetAutoRange(kTRUE);
      gGeoManager->GetTopVolume()->Paint("range");
      view->SetAutoRange(kFALSE);
      if (has_pad) gPad->Update();
   }

}
//______________________________________________________________________________
void TGeoPainter::DrawOnly(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   gGeoManager->GetCurrentVolume()->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = new TView(1);
      view->SetAutoRange(kTRUE);
      gGeoManager->SetVisOption(TGeoManager::kGeoVisOnly);
      gGeoManager->GetCurrentVolume()->Paint("range");
      view->SetAutoRange(kFALSE);
      if (has_pad) gPad->Update();
   }
}
//______________________________________________________________________________
void TGeoPainter::DrawPanel()
{
}
//______________________________________________________________________________
void TGeoPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
}
//______________________________________________________________________________
char *TGeoPainter::GetObjectInfo(Int_t px, Int_t py) const
{
   return 0;
}
//______________________________________________________________________________
void TGeoPainter::Paint(Option_t *option)
{
// paint current geometry according to option
//   printf("TGeoPainter::Paint()\n");
   if (!fGeo) return;
   if (fGeo->GetVisOption()==2) {
      fGeo->GetCurrentNode()->Paint(option);
      return;
   }
   fGeo->CdTop();
   TGeoNode *top = fGeo->GetTopNode();
   top->Paint(option);
}
//______________________________________________________________________________
void TGeoPainter::PaintShape(X3DBuffer *buff, Bool_t rangeView)
{
//*-*-*-*-*Paint 3-D shape in current pad with its current attributes*-*-*-*-*
//*-*      ==========================================================
//
// rangeView = kTRUE - means no real painting
//                     just estimate the range
//                     of this shape only

    //*-* Paint in the pad
    //*-* Convert to the master system

    if (!buff) return;
    if (!gGeoManager) return;
    TGeoVolume *vol = gGeoManager->GetCurrentVolume();
    Float_t *point = &(buff->points[0]);
    Double_t dlocal[3];
    Double_t dmaster[3];
    if (gGeoManager) {
       for (Int_t j = 0; j < buff->numPoints; j++) {
           dlocal[0]=point[3*j]; dlocal[1]=point[3*j+1]; dlocal[2]=point[3*j+2];
           if (gGeoManager->IsExplodedView()) 
              gGeoManager->LocalToMasterBomb(&dlocal[0],&dmaster[0]);
           else   
              gGeoManager->LocalToMaster(&dlocal[0],&dmaster[0]);
//           printf("point %i : %g %g %g\n", j,dmaster[0],dmaster[1],dmaster[2]);
           point[3*j]=dmaster[0]; point[3*j+1]=dmaster[1]; point[3*j+2]=dmaster[2];
       }
    }
    
    Float_t x0, y0, z0, x1, y1, z1;
    const Int_t kExpandView = 2;
    int i0;

    x0 = x1 = buff->points[0];
    y0 = y1 = buff->points[1];
    z0 = z1 = buff->points[2];

    if (!rangeView) {
      ((TAttLine*)vol)->Modify();  //Change line attributes only if necessary
      ((TAttFill*)vol)->Modify();  //Change fill area attributes only if necessary
    }

    for (Int_t i = 0; i < buff->numSegs; i++) {
        i0 = 3*buff->segs[3*i+1];
        Float_t *ptpoints_0 = &(buff->points[i0]);
        i0 = 3*buff->segs[3*i+2];
        Float_t *ptpoints_3 = &(buff->points[i0]);
        if (!rangeView) gPad->PaintLine3D(ptpoints_0, ptpoints_3);
        else {
            x0 = ptpoints_0[0] < x0 ? ptpoints_0[0] : x0;
            y0 = ptpoints_0[1] < y0 ? ptpoints_0[1] : y0;
            z0 = ptpoints_0[2] < z0 ? ptpoints_0[2] : z0;
            x1 = ptpoints_3[0] > x1 ? ptpoints_3[0] : x1;
            y1 = ptpoints_3[1] > y1 ? ptpoints_3[1] : y1;
            z1 = ptpoints_3[2] > z1 ? ptpoints_3[2] : z1;
        }
    }
    if (rangeView)
    {
      TView *view = gPad->GetView();
      if (view->GetAutoRange()) view->SetRange(x0,y0,z0,x1,y1,z1,kExpandView);
    }
}
//______________________________________________________________________________
void TGeoPainter::PaintBox(TGeoVolume *vol, Option_t *option)
{
// paint any type of box with 8 vertices
   const Int_t numpoints = 8;

//*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   vol->GetShape()->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView  && gPad->GetView3D()) gVirtualGL->PaintBrik(points);

 //==  for (Int_t i = 0; i < numpoints; i++)
 //            gNode->Local2Master(&points[3*i],&points[3*i]);


   Int_t c = ((vol->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
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
void TGeoPainter::PaintTube(TGeoVolume *vol, Option_t *option)
{
// paint tubes
   Int_t i, j;
   Int_t n = fNsegments;
   const Int_t numpoints = 4*n;

//*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   vol->GetShape()->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points, n, 2);

//==   for (i = 0; i < numpoints; i++)
//==            gNode->Local2Master(&points[3*i],&points[3*i]);

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        if (strstr(option, "x3d"))  buff->numSegs   = n*8;
        else                        buff->numSegs   = n*6;
        buff->numPolys  = n*4;
    }


//*-* Allocate memory for points *-*

    buff->points = points;

    Int_t c = ((vol->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;

//*-* Allocate memory for segments *-*

    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {
        for (i = 0; i < 4; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c;
                buff->segs[(i*n+j)*3+1] = i*n+j;
                buff->segs[(i*n+j)*3+2] = i*n+j+1;
            }
            buff->segs[(i*n+j-1)*3+2] = i*n;
        }
        for (i = 4; i < 6; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c+1;
                buff->segs[(i*n+j)*3+1] = (i-4)*n+j;
                buff->segs[(i*n+j)*3+2] = (i-2)*n+j;
            }
        }
        if (strstr(option, "x3d")) {
           for (i = 6; i < 8; i++) {
              for (j = 0; j < n; j++) {
                 buff->segs[(i*n+j)*3  ] = c;
                 buff->segs[(i*n+j)*3+1] = 2*(i-6)*n+j;
                 buff->segs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
              }
           }
        }
    }
//*-* Allocate memory for polygons *-*

    Int_t indx = 0;

    buff->polys = new Int_t[buff->numPolys*6];
    if (buff->polys) {
        for (i = 0; i < 2; i++) {
            for (j = 0; j < n; j++) {
                indx = 6*(i*n+j);
                buff->polys[indx  ] = c;
                buff->polys[indx+1] = 4;
                buff->polys[indx+2] = i*n+j;
                buff->polys[indx+3] = (4+i)*n+j;
                buff->polys[indx+4] = (2+i)*n+j;
                buff->polys[indx+5] = (4+i)*n+j+1;
            }
            buff->polys[indx+5] = (4+i)*n;
        }
        for (i = 2; i < 4; i++) {
            for (j = 0; j < n; j++) {
                indx = 6*(i*n+j);
                buff->polys[indx  ] = c+i;
                buff->polys[indx+1] = 4;
                buff->polys[indx+2] = (i-2)*2*n+j;
                buff->polys[indx+3] = (4+i)*n+j;
                buff->polys[indx+4] = ((i-2)*2+1)*n+j;
                buff->polys[indx+5] = (4+i)*n+j+1;
            }
            buff->polys[indx+5] = (4+i)*n;
        }
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

    if (buff->points)   delete [] buff->points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}
//______________________________________________________________________________
void TGeoPainter::PaintTubs(TGeoVolume *vol, Option_t *option)
{
// paint tubes
   Int_t i, j;
   const Int_t n = fNsegments+1;
   const Int_t numpoints = 4*n;

   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   vol->GetShape()->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points,-n,2);

//==   for (i = 0; i < numpoints; i++)
//==            gNode->Local2Master(&points[3*i],&points[3*i]);


    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints =   numpoints;
        buff->numSegs   = 2*numpoints;
        buff->numPolys  =   numpoints-2;
    }

    buff->points = points;

    Int_t c = ((vol->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;

//*-* Allocate memory for segments *-*

    buff->segs = new Int_t[buff->numSegs*3];
    memset(buff->segs, 0, buff->numSegs*3*sizeof(Int_t));
    if (buff->segs) {
        for (i = 0; i < 4; i++) {
            for (j = 1; j < n; j++) {
                buff->segs[(i*n+j-1)*3  ] = c;
                buff->segs[(i*n+j-1)*3+1] = i*n+j-1;
                buff->segs[(i*n+j-1)*3+2] = i*n+j;
            }
        }
        for (i = 4; i < 6; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c+1;
                buff->segs[(i*n+j)*3+1] = (i-4)*n+j;
                buff->segs[(i*n+j)*3+2] = (i-2)*n+j;
            }
        }
        for (i = 6; i < 8; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c;
                buff->segs[(i*n+j)*3+1] = 2*(i-6)*n+j;
                buff->segs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
            }
        }
    }

//*-* Allocate memory for polygons *-*

    Int_t indx = 0;

    buff->polys = new Int_t[buff->numPolys*6];
    memset(buff->polys, 0, buff->numPolys*6*sizeof(Int_t));
    if (buff->polys) {
        for (i = 0; i < 2; i++) {
            for (j = 0; j < n-1; j++) {
                buff->polys[indx++] = c;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = i*n+j;
                buff->polys[indx++] = (4+i)*n+j;
                buff->polys[indx++] = (2+i)*n+j;
                buff->polys[indx++] = (4+i)*n+j+1;
            }
        }
        for (i = 2; i < 4; i++) {
            for (j = 0; j < n-1; j++) {
                buff->polys[indx++] = c+i;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = (i-2)*2*n+j;
                buff->polys[indx++] = (4+i)*n+j;
                buff->polys[indx++] = ((i-2)*2+1)*n+j;
                buff->polys[indx++] = (4+i)*n+j+1;
            }
        }
        buff->polys[indx++] = c+2;
        buff->polys[indx++] = 4;
        buff->polys[indx++] = 6*n;
        buff->polys[indx++] = 4*n;
        buff->polys[indx++] = 7*n;
        buff->polys[indx++] = 5*n;

        buff->polys[indx++] = c+2;
        buff->polys[indx++] = 4;
        buff->polys[indx++] = 7*n-1;
        buff->polys[indx++] = 5*n-1;
        buff->polys[indx++] = 8*n-1;
        buff->polys[indx++] = 6*n-1;
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
void TGeoPainter::PaintSphere(TGeoVolume *vol, Option_t *option)
{
// paint a sphere
   Int_t i, j;
   const Int_t n = ((TGeoSphere*)vol->GetShape())->GetNumberOfDivisions()+1;
   Double_t ph1 = ((TGeoSphere*)vol->GetShape())->GetPhi1();
   Double_t ph2 = ((TGeoSphere*)vol->GetShape())->GetPhi2();
   Int_t nz = ((TGeoSphere*)vol->GetShape())->GetNz()+1;
   if (nz < 2) return;
   Int_t numpoints = 2*n*nz;
   if (numpoints <= 0) return;
   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;
   vol->GetShape()->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points, -n, nz);

 //==  for (i = 0; i < numpoints; i++)
 //==          gNode->Local2Master(&points[3*i],&points[3*i]);

   Bool_t specialCase = kFALSE;

   if (TMath::Abs(TMath::Sin(2*(ph2 - ph1))) <= 0.01)  //mark this as a very special case, when
         specialCase = kTRUE;                                  //we have to draw this PCON like a TUBE

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        buff->numSegs   = 4*(nz*n-1+(specialCase == kTRUE));
        buff->numPolys  = 2*(nz*n-1+(specialCase == kTRUE));
    }

//*-* Allocate memory for points *-*

    buff->points = points;

    Int_t c = ((vol->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;

//*-* Allocate memory for segments *-*

    Int_t indx, indx2, k;
    indx = indx2 = 0;

    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {

        //inside & outside spheres, number of segments: 2*nz*(n-1)
        //             special case number of segments: 2*nz*n
        for (i = 0; i < nz*2; i++) {
            indx2 = i*n;
            for (j = 1; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2+j;
            }
            if (specialCase) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2;
            }
        }

        //bottom & top lines, number of segments: 2*n
        for (i = 0; i < 2; i++) {
            indx2 = i*(nz-1)*2*n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n+j;
            }
        }

        //inside & outside spheres, number of segments: 2*(nz-1)*n
        for (i = 0; i < (nz-1); i++) {

            //inside sphere
            indx2 = i*n*2;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+2;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
            //outside sphere
            indx2 = i*n*2+n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+3;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
        }

        //left & right sections, number of segments: 2*(nz-2)
        //          special case number of segments: 0
        if (!specialCase) {
            for (i = 1; i < (nz-1); i++) {
                for (j = 0; j < 2; j++) {
                    buff->segs[indx++] = c;
                    buff->segs[indx++] =  2*i    * n + j*(n-1);
                    buff->segs[indx++] = (2*i+1) * n + j*(n-1);
                }
            }
        }
    }


    Int_t m = n - 1 + (specialCase == kTRUE);

//*-* Allocate memory for polygons *-*

    indx = 0;

    buff->polys = new Int_t[buff->numPolys*6];

    if (buff->polys) {

        //bottom & top, number of polygons: 2*(n-1)
        // special case number of polygons: 2*n
        for (i = 0; i < 2; i++) {
            for (j = 0; j < n-1; j++) {
                buff->polys[indx++] = c+3;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = 2*nz*m+i*n+j;
                buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                buff->polys[indx++] = 2*nz*m+i*n+j+1;
                buff->polys[indx++] = i*(nz*2-2)*m+j;
            }
            if (specialCase) {
                buff->polys[indx++] = c+3;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = 2*nz*m+i*n+j;
                buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                buff->polys[indx++] = 2*nz*m+i*n;
                buff->polys[indx++] = i*(nz*2-2)*m+j;
            }
        }


        //inside & outside, number of polygons: (nz-1)*2*(n-1)
        for (k = 0; k < (nz-1); k++) {
            for (i = 0; i < 2; i++) {
                for (j = 0; j < n-1; j++) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j+1;
                }
                if (specialCase) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n;
                }
            }
        }


        //left & right sections, number of polygons: 2*(nz-1)
        //          special case number of polygons: 0
        if (!specialCase) {
            indx2 = nz*2*(n-1);
            for (k = 0; k < (nz-1); k++) {
                for (i = 0; i < 2; i++) {
                    buff->polys[indx++] = c+2;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = k==0 ? indx2+i*(n-1) : indx2+2*nz*n+2*(k-1)+i;
                    buff->polys[indx++] = indx2+2*(k+1)*n+i*(n-1);
                    buff->polys[indx++] = indx2+2*nz*n+2*k+i;
                    buff->polys[indx++] = indx2+(2*k+3)*n+i*(n-1);
                }
            }
            buff->polys[indx-8] = indx2+n;
            buff->polys[indx-2] = indx2+2*n-1;
        }
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
void TGeoPainter::PaintPcon(TGeoVolume *vol, Option_t *option)
{
// paint a pcon
   Int_t i, j;
   const Int_t n = fNsegments+1;
   Int_t nz = ((TGeoPcon*)vol->GetShape())->GetNz();
   if (nz < 2) return;
   Int_t numpoints =  nz*2*n;
   if (numpoints <= 0) return;
   Double_t dphi = ((TGeoPcon*)vol->GetShape())->GetDphi();
   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;
   vol->GetShape()->SetPoints(points);

   Bool_t rangeView = strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points, -n, nz);

 //==  for (i = 0; i < numpoints; i++)
 //==          gNode->Local2Master(&points[3*i],&points[3*i]);

   Bool_t specialCase = kFALSE;

   if (dphi == 360)           //mark this as a very special case, when
        specialCase = kTRUE;     //we have to draw this PCON like a TUBE

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        buff->numSegs   = 4*(nz*n-1+(specialCase == kTRUE));
        buff->numPolys  = 2*(nz*n-1+(specialCase == kTRUE));
    }

//*-* Allocate memory for points *-*

    buff->points = points;

    Int_t c = ((vol->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;

//*-* Allocate memory for segments *-*

    Int_t indx, indx2, k;
    indx = indx2 = 0;

    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {

        //inside & outside circles, number of segments: 2*nz*(n-1)
        //             special case number of segments: 2*nz*n
        for (i = 0; i < nz*2; i++) {
            indx2 = i*n;
            for (j = 1; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2+j;
            }
            if (specialCase) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2;
            }
        }

        //bottom & top lines, number of segments: 2*n
        for (i = 0; i < 2; i++) {
            indx2 = i*(nz-1)*2*n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n+j;
            }
        }

        //inside & outside cilindres, number of segments: 2*(nz-1)*n
        for (i = 0; i < (nz-1); i++) {

            //inside cilinder
            indx2 = i*n*2;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+2;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
            //outside cilinder
            indx2 = i*n*2+n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+3;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
        }

        //left & right sections, number of segments: 2*(nz-2)
        //          special case number of segments: 0
        if (!specialCase) {
            for (i = 1; i < (nz-1); i++) {
                for (j = 0; j < 2; j++) {
                    buff->segs[indx++] = c;
                    buff->segs[indx++] =  2*i    * n + j*(n-1);
                    buff->segs[indx++] = (2*i+1) * n + j*(n-1);
                }
            }
        }
    }


    Int_t m = n - 1 + (specialCase == kTRUE);

//*-* Allocate memory for polygons *-*

    indx = 0;

    buff->polys = new Int_t[buff->numPolys*6];

    if (buff->polys) {

        //bottom & top, number of polygons: 2*(n-1)
        // special case number of polygons: 2*n
        for (i = 0; i < 2; i++) {
            for (j = 0; j < n-1; j++) {
                buff->polys[indx++] = c+3;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = 2*nz*m+i*n+j;
                buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                buff->polys[indx++] = 2*nz*m+i*n+j+1;
                buff->polys[indx++] = i*(nz*2-2)*m+j;
            }
            if (specialCase) {
                buff->polys[indx++] = c+3;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = 2*nz*m+i*n+j;
                buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                buff->polys[indx++] = 2*nz*m+i*n;
                buff->polys[indx++] = i*(nz*2-2)*m+j;
            }
        }


        //inside & outside, number of polygons: (nz-1)*2*(n-1)
        for (k = 0; k < (nz-1); k++) {
            for (i = 0; i < 2; i++) {
                for (j = 0; j < n-1; j++) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j+1;
                }
                if (specialCase) {
                    buff->polys[indx++] = c+i;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = (2*k+i*1)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                    buff->polys[indx++] = (2*k+i*1+2)*m+j;
                    buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n;
                }
            }
        }


        //left & right sections, number of polygons: 2*(nz-1)
        //          special case number of polygons: 0
        if (!specialCase) {
            indx2 = nz*2*(n-1);
            for (k = 0; k < (nz-1); k++) {
                for (i = 0; i < 2; i++) {
                    buff->polys[indx++] = c+2;
                    buff->polys[indx++] = 4;
                    buff->polys[indx++] = k==0 ? indx2+i*(n-1) : indx2+2*nz*n+2*(k-1)+i;
                    buff->polys[indx++] = indx2+2*(k+1)*n+i*(n-1);
                    buff->polys[indx++] = indx2+2*nz*n+2*k+i;
                    buff->polys[indx++] = indx2+(2*k+3)*n+i*(n-1);
                }
            }
            buff->polys[indx-8] = indx2+n;
            buff->polys[indx-2] = indx2+2*n-1;
        }
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
void TGeoPainter::PaintStat(Int_t dostat, Option_t *option)
{
// paint stat
}
