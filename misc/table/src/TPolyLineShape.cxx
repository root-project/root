// @(#)root/table:$Id$
// Author: Valeri Fine 1999

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPolyLineShape.h"

#include "TPoints3D.h"
#include "TVolume.h"
#include "TVolumePosition.h"
#include "TTUBE.h"
#include "TBRIK.h"
#include "TSPHE.h"
#include "TView.h"
#include "TVirtualPad.h"
#include "TTablePadView3D.h"
#include "TPoint.h"
#include "TVirtualPS.h"
#include "TMath.h"

//////////////////////////////////////////////////////////////////////////////////////
//                                                                                  //
//                           STAR 3D geometry Object Model                          //
//                           =============================                          //
//                                                                                  //
//                           TDataSet            Legend:                            //
//                              |                  "classA"                         //
//                              |                     |     "classB" is derived from//
//                              v                     v       the "classA"          //
//                           TVolume               "classB"                         //
//                              ^                                                   //
//                              |                  "classQ"                         //
//                              |                      ^     "classQ" has a pointer //
//                            TShape                   |       to "classT"          //
//                              |                   "classT"                        //
//                              v                                                   //
//           -----------------------------------------------------                  //
//           |               |      |                     |      |                  //
//           |               |      |     .    .    .     |      |                  //
//           |               |      |                     |      |                  //
//           V               v      v                     v      v                  //
//      TPolyLineShape     TBRIK  TTUBE                 TPCON  TTRD1                //
//       |        ^                                                                 //
//       |        |       begin_html <a href="http://root.cern.ch/root/html/TShape.html#TShape:description">R  O  O  T        S  H  A  P  E  S</a>end_html                          //
//       V        |       (see begin_html <a href="http://wwwinfo.cern.ch/asdoc/geant_html3/node109.html#SECTION041000000000000000000000">GEANT 3.21 shapes</a>end_html as well)                           //
// St_PolyLine3D  |                                                                 //
//                |                                                                 //
//           TPoints3DABC                                                           //
//                |                                                                 //
//                |                                                                 //
//                v                                                                 //
//      --------------------------------------------------------                    //
//      |                 |                 |                  |                    //
//      |                 |                 |                  |                    //
//      |                 |                 |                  |                    //
//      V                 v                 v                  v                    //
//StHits3DPoints   StHelix3DPoints   TTable3Points          TPoints3D               //
//      ^                 ^                 ^                  ^                    //
//      |                 |                 |                  |                    //
//      |                 |                 |                  |                    //
//  StObjArray    StTrack / StHelixD  TTableSorter       flat floating              //
//                                          ^              point array              //
//                                          |        (see St_PolyLine3D as well)    //
//                                          |                                       //
//                                        TTable                                    //
//                                                                                  //
//                                                                                  //
//                     S  T  A  R    S  H  A  P  E  S                               //
//                     -------------------------------                              //
//                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////

ClassImp(TPolyLineShape)

//______________________________________________________________________________
TPolyLineShape::TPolyLineShape()
{
   //to be documented
   fShape = 0;
   fSmooth = kFALSE;
   fConnection= 0;
   fPoints=0;
   SetWidthFactor();
   fHasDrawn = kFALSE;
   fShapeType = kNULL;
   fSizeX3D   = 0;
   fPointFlag = kFALSE;
   fLineFlag  = kFALSE;
}

//______________________________________________________________________________
TPolyLineShape::TPolyLineShape(TPoints3DABC  *points,Option_t* option)
{
  //  fShape       = new TTUBE("tube","tube","void",0.5,0.5);
   fShape      = 0;
   fShapeType   = kNULL;
   fSmooth      = kFALSE;
   fConnection  = 0;
   fPoints      = points;
   fHasDrawn    = kFALSE;
   fSizeX3D     = 0;
   // Take in account the current node if any
   if (!fPoints) {
      Error("TPolyLineShape","No polyline is defined");
      return;
   }
   fPointFlag = strchr(option,'P')?kTRUE:kFALSE;
   fLineFlag  = strchr(option,'L')?kTRUE:kFALSE;

   SetWidthFactor();
   Create();
}

//______________________________________________________________________________
TPolyLineShape::~TPolyLineShape()
{
   //to be documented
   SafeDelete(fShape);
   SafeDelete(fSizeX3D);
}

//______________________________________________________________________________
void TPolyLineShape::Create()
{
   //to be documented
   if (!fConnection) SetConnection(kBrik);
}

//______________________________________________________________________________
Size3D *TPolyLineShape::CreateX3DSize(Bool_t marker)
{
   //to be documented
   if (!fSizeX3D) fSizeX3D = new Size3D;
   fSizeX3D->numPoints = 0;
   fSizeX3D->numSegs   = 0;
   fSizeX3D->numPolys  = 0;         //NOTE: Because of different structure, our
   if (fPoints) {
      Int_t size = fPoints->Size();
      if (marker) {
         Int_t mode;
         if (size > 10000) mode = 1;         // One line marker    '-'
         else if (size > 3000) mode = 2;     // Two lines marker   '+'
         else mode = 3;                      // Three lines marker '*'

         fSizeX3D->numSegs   = size*mode;
         fSizeX3D->numPoints = size*mode*2;
         fSizeX3D->numPolys  = 0;
      } else {
         fSizeX3D->numSegs   = size-1;
         fSizeX3D->numPoints = size;
      }
      fSizeX3D->numPolys  = 0;         //NOTE: Because of different structure, our
   }
   return fSizeX3D;
}

//______________________________________________________________________________
Int_t TPolyLineShape::SetConnection(EShapeTypes connection)
{
   //to be documented
   Float_t size = 0.5*GetWidthFactor()*GetLineWidth();

   if (fShapeType != connection) {
      SafeDelete(fConnection);
      fShapeType = connection;
      switch (fShapeType) {
         case  kSphere:
            SetConnection(new TSPHE("connection","sphere","void",0,size,0,90,0,360));
            break;
         default:
            SetConnection(new TBRIK("connection","brik","void",size,size,size));
            break;
      };
   }
   return 0;
}

//______________________________________________________________________________
Int_t TPolyLineShape::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*Compute distance from point px,py to a 3-D polyline*-*-*-*-*-*-*
//*-*            ===================================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each segment
//*-*  of the polyline.
//*-*  Returns when the distance found is below DistanceMaximum.
//*-*  The distance is computed in pixels units.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fPoints) {
      Int_t ret = fPoints->DistancetoPrimitive( px, py);
      if (ret == -1) ret = PointDistancetoPrimitive(px, py);
      return ret;
   }
   return 999999;
}

//______________________________________________________________________________
Int_t TPolyLineShape::PointDistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*Compute distance from point px,py to a 3-D points *-*-*-*-*-*-*
//*-*          =====================================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each segment
//*-*  of the polyline.
//*-*  Returns when the distance found is below DistanceMaximum.
//*-*  The distance is computed in pixels units.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   const Int_t inaxis = 7;
   Float_t dist = 999999;

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());

   TView *view = 0;
//*-*- return if point is not in the user area
   if (px < puxmin - inaxis) goto END;
   if (py > puymin + inaxis) goto END;
   if (px > puxmax + inaxis) goto END;
   if (py < puymax - inaxis) goto END;

   view = gPad->GetView();
   if (view) {
      Int_t i;
      Float_t dpoint;
      Float_t xndc[3];
      Int_t x1,y1;
      Int_t pointSize = fPoints->Size();
      for (i=0;i<pointSize;i++) {
         Float_t thisPoints[3];
         view->WCtoNDC(fPoints->GetXYZ(thisPoints,i), xndc);
         x1     = gPad->XtoAbsPixel(xndc[0]);
         y1     = gPad->YtoAbsPixel(xndc[1]);
         dpoint = (px-x1)*(px-x1) + (py-y1)*(py-y1);
         if (dpoint < dist) dist = dpoint;
      }
      dist = (TMath::Sqrt(dist));
   }
END:
   return Int_t(dist);
}

//______________________________________________________________________________
void TPolyLineShape::Draw(Option_t *)
{
   //to be documented
   Create();
   AppendPad();
}

//______________________________________________________________________________
void TPolyLineShape::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // forward the ExecuteEvent call to the decorated object
   if (fPoints) fPoints->ExecuteEvent(event,px, py);
}

//______________________________________________________________________________
Color_t TPolyLineShape::GetColorAttribute() const
{
   //to be documented
   return ((TPolyLineShape *)this)->GetLineColor();
}
//______________________________________________________________________________
const char  *TPolyLineShape::GetName()  const
{
   // forward the GetName call to the decorated object
   return fPoints ? fPoints->GetName() : TShape::GetName();
}
//______________________________________________________________________________
char  *TPolyLineShape::GetObjectInfo(Int_t px, Int_t py) const
{
   // forward the GetObjectInfo call to the decorated object
   return fPoints ? fPoints->GetObjectInfo(px, py) : TShape::GetObjectInfo(px,py);
}

//______________________________________________________________________________
Size_t TPolyLineShape::GetSizeAttribute() const
{
   //to be documented
   return ((TPolyLineShape *)this)->GetMarkerSize();
}

//______________________________________________________________________________
Style_t TPolyLineShape::GetStyleAttribute() const
{
   //to be documented
   return ((TPolyLineShape *)this)->GetLineStyle();
}

//______________________________________________________________________________
const char  *TPolyLineShape::GetTitle() const
{
   // forward the GetObjectInfo call to the decorated object
   return fPoints ? fPoints->GetTitle() : TShape::GetTitle();
}
//______________________________________________________________________________
void TPolyLineShape::PaintNode(Float_t *start,Float_t *end,Option_t *option)
{
  // Double_t *start - coordinate of the start point of the current segment
  // Double_t *end   - coordinate of the end   point of the current segment

  // Calculate the vector
   const Int_t kDimension = 3;
   Double_t vector[kDimension];
   Double_t nodeposition[kDimension];
   Int_t i=0;
   for (i=0;i<kDimension;i++) {
      vector[i]=end[i]-start[i];
      nodeposition[i]=0.5*(start[i]+end[i]);
   }
   Double_t length = TMath::Normalize(vector);

  // Calculate the rotation axis for Axis Oz

   Double_t oz[3]={0,0,1};
   Double_t rotate[3];

   Double_t sina = TMath::Normalize(TMath::Cross(vector,oz,rotate));
   Double_t cosa = Product(vector,oz);
   Double_t mrot[3][3];

   TShape *shape = fShape;
   if (!shape) shape = fConnection;

   Gyrot(rotate,cosa,sina,mrot);

   Float_t width = GetWidthFactor()*GetLineWidth();

   mrot[0][0] *= width;
   mrot[0][1] *= width;
   mrot[0][2] *= width;

   mrot[1][0] *= width;
   mrot[1][1] *= width;
   mrot[1][2] *= width;

   mrot[2][0] *= length;
   mrot[2][1] *= length;
   mrot[2][2] *= length;

   Color_t color = GetLineColor();

   TVolume node("SegmentNode","SegmentNode", shape);
   node.SetLineColor(color);
   if (!fShape) node.SetVisibility();
   node.SetLineColor(color);

   TRotMatrix matrix ("rotate","rotate",&mrot[0][0]);
   TVolumePosition position(&node,nodeposition[0],nodeposition[1]
                                 ,nodeposition[2],&matrix);

   if (!(fSmooth || fConnection))  {
      node.PaintNodePosition(option, &position);
      return;
   }

   // Add the connection

   memset(mrot,0,9*sizeof(Double_t));

   length = width/length;
   mrot[2][2] = length;
   mrot[0][0] = 1;
   mrot[1][1] = 1;

   TRotMatrix kneeMatrix("knee","knee",&mrot[0][0]);
   TVolume knee("ConnectionNode","ConnectionNode", fConnection);
   TVolumePosition kneePosition(&knee, 0, 0, 0.5, &kneeMatrix);
   knee.SetLineColor(color);
   node.Add(&knee,&kneePosition);

   node.PaintNodePosition(option, &position);
}

//______________________________________________________________________________
void TPolyLineShape::Paint(Option_t *opt)
{
   //to be documented
   if (!GetPoints()) return;

   Bool_t rangeView = opt && opt[0] && strcmp(opt,"range")==0 ? kTRUE : kFALSE;
   TTablePadView3D *view3D = 0;
   if (!rangeView  && (view3D = (TTablePadView3D*)gPad->GetView3D()) ) {
      TString mode;

      mode="";
      if (fLineFlag)  mode  = "L";
      if (fPointFlag) mode += "P";

      view3D->SetLineAttr(GetColorAttribute(), (Int_t)GetSizeAttribute());
      view3D->PaintPoints3D(GetPoints(), mode.Data());
   }
   if (opt && !strstr(opt, "x3d")) {
      if (fPointFlag) {
         SetMarkerColor(GetColorAttribute());
         SetMarkerSize(GetSizeAttribute());
         PaintPolyMarker(fPoints->Size());
      }
      if (fLineFlag) {
         SetLineColor(GetColorAttribute());
         SetLineWidth((Width_t)GetSizeAttribute());
         PaintPoints(fPoints->Size());
      }

   } else {
      if (fLineFlag) {
         CreateX3DSize(kFALSE); PaintX3DLine(opt);
      } else {
         CreateX3DSize(kTRUE);  PaintX3DMarker(opt);
      }
//     Paint3d(opt);
   }
}

//______________________________________________________________________________
void  TPolyLineShape::PaintPoints(Int_t n, Float_t *, Option_t *)
{
//*-*-*-*-*-*-*-*-*Draw this 3-D polyline with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*              ===========================================
   if (n < 2) return;

   TAttLine::Modify();  //Change line attributes only if necessary

//*-*- Loop on each individual line
   for (Int_t i=1;i<n;i++) {
      Float_t xyz[6];
      fPoints->GetXYZ(&xyz[0],i-1,2);
      gPad->PaintLine3D(xyz, &xyz[3]);
   }
}

//______________________________________________________________________________
void TPolyLineShape::PaintPolyMarker(Int_t n, Float_t *, Marker_t, Option_t *)
{
//*-*-*-*-*-*-*-*-*Paint polymarker in CurrentPad World coordinates*-*-*-*-*-*-*-*
//*-*              ================================================

   if (n <= 0) return;

   TView *view = gPad->GetView();      //Get current 3-D view
   if(!view) return;                   //Check if `view` is valid

   //Create temorary storage
   TPoint *pxy = new TPoint[n];
   Float_t *x  = new Float_t[n];
   Float_t *y  = new Float_t[n];
   Float_t xndc[3], ptr[3];

//*-*- convert points from world to pixel coordinates
   Int_t nin = 0;
   for (Int_t i = 0; i < n; i++) {
      fPoints->GetXYZ(ptr,i);
      view->WCtoNDC(ptr, xndc);
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
void TPolyLineShape::Paint3d(Option_t *opt)
{
   //to be documented
   if (!fPoints) return;

   Create();

   struct XYZ { Float_t fValues[3]; } *points;
   points  = (XYZ *)(fPoints->GetP());
   Int_t size      = fPoints->GetN()-1;

   for (Int_t i=0;i<size;i++)
      PaintNode((Float_t *)(points+i+1),(Float_t *)(points+i),opt);
   fHasDrawn = kTRUE;
}

//______________________________________________________________________________
void TPolyLineShape::PaintX3DLine(Option_t *)
{
   //to be documented
#ifndef WIN32
   Int_t size = 0;
   if (fPoints) size = fPoints->Size();
   if (!size) return;

   X3DBuffer *buff = new X3DBuffer;
   if (!buff) return;

   fSizeX3D->numPoints = buff->numPoints = size;
   fSizeX3D->numSegs   = buff->numSegs   = size-1;
   fSizeX3D->numPolys  = buff->numPolys  = 0;        //NOTE: Because of different structure, our

   buff->polys     = 0;     //      TPolyLine3D can't use polygons
   TPoints3D x3dPoints(size);
   buff->points    = fPoints->GetXYZ(x3dPoints.GetP(),0,size);

//        Int_t c = (((fAttributes?fAttributes->GetColorAttribute():0) % 8) - 1) * 4;     // Basic colors: 0, 1, ... 8
   Int_t c = ((GetColorAttribute() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 8
   if (c < 0) c = 0;

   //*-* Allocate memory for segments *-*
   buff->segs = new Int_t[buff->numSegs*3];
   if (buff->segs) {
      for (Int_t i = 0; i < buff->numSegs; i++) {
         buff->segs[3*i  ] = c;
         buff->segs[3*i+1] = i;
         buff->segs[3*i+2] = i+1;
      }
   }


   if (buff && buff->points && buff->segs) //If everything seems to be OK ...
      FillX3DBuffer(buff);
   else {                            // ... something very bad was happened
      gSize3D.numPoints -= buff->numPoints;
      gSize3D.numSegs   -= buff->numSegs;
      gSize3D.numPolys  -= buff->numPolys;
   }

   if (buff->segs)     delete [] buff->segs;
   if (buff->polys)    delete [] buff->polys;
   if (buff)           delete    buff;
#endif
}

//______________________________________________________________________________
void TPolyLineShape::PaintX3DMarker(Option_t *)
{
   //to be documented
#ifndef WIN32
   Int_t size = 0;
   if (fPoints) size = fPoints->Size();
   if (!size) return;
   Int_t mode;
   Int_t i, j, k, n;

   X3DBuffer *buff = new X3DBuffer;
   if(!buff) return;

   if (size > 10000) mode = 1;         // One line marker    '-'
   else if (size > 3000) mode = 2;     // Two lines marker   '+'
   else mode = 3;                      // Three lines marker '*'

   fSizeX3D->numSegs   = buff->numSegs   = size*mode;
   fSizeX3D->numPoints = buff->numPoints = buff->numSegs*2;
   fSizeX3D->numPolys  = buff->numPolys  = 0;         //NOTE: Because of different structure, our

   buff->polys     = 0;      //      TPolyMarker3D can't use polygons


    //*-* Allocate memory for points *-*
   Float_t delta = 0.002;

   buff->points = new Float_t[buff->numPoints*3];
   if (buff->points) {
      for (i = 0; i < size; i++) {
         for (j = 0; j < mode; j++) {
            for (k = 0; k < 2; k++) {
               delta *= -1;
               for (n = 0; n < 3; n++) {
                  Float_t xyz[3];
                  fPoints->GetXYZ(xyz,i);
                  buff->points[mode*6*i+6*j+3*k+n] =
                  xyz[n] * (1 + (j == n ? delta : 0));
               }
            }
         }
      }
   }

   Int_t c = ((GetColorAttribute() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 8
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
#endif
}

//______________________________________________________________________________
Float_t TPolyLineShape::Product(Float_t *v1, Float_t *v2,Int_t ndim)
{
   //to be documented
   Float_t p = 0;
   if (v1 && v2 && ndim > 0)
      for (Int_t i=0; i<ndim; i++) p+= v1[i]*v2[i];
   return p;
}

//______________________________________________________________________________
Double_t TPolyLineShape::Product(Double_t *v1, Double_t *v2,Int_t ndim)
{
   //to be documented
   Double_t p = 0;
   if (v1 && v2 && ndim > 0)
      for (Int_t i=0;i<ndim;i++) p+= v1[i]*v2[i];
   return p;
}

//______________________________________________________________________________
Double_t *TPolyLineShape::Gyrot(Double_t *dirc, Double_t cosang, Double_t sinang, Double_t trans[3][3])
{
//************************************************************************
//*                                                                      *
//*   call gyrot(dirc,angp,trans,ntrans)                       vp 880722 *
//*                                       revised              vp 921009 *
//*                                       revised (f->c++)     vf 981006 *
//*       routine for filling rotation transformation matrix             *
//*       from axis and rotation angle around                            *
//*                                                                      *
//*   arguments:                                                         *
//*       dirc    direct cosinuses (may be not normalised)               *
//*       cosang, sinang - cos and sin of the rotation angle             *
//*       tranz   rotation & shift matrix 3*3  (input/output)            *
//*    ---------------------------------------------------------------   *
//*  This code is provided by Victor Perevoztchikov                      *
//************************************************************************

   Double_t ax[3];

   memcpy(ax,dirc,3*sizeof(Double_t));
   TMath::Normalize(ax);

   Double_t ca  = cosang;
   Double_t sa  = sinang;
   Double_t ca1;

   if (ca < 0.5)
      ca1 = 1. - ca ;
   else
      ca1 = (sa*sa)/(1.+ca) ;

   Int_t j1 = 0;
   Int_t j2 = 0;
   for(j1 = 0; j1 < 3; j1++) {
      for(j2 = 0; j2 < 3; j2++)
         trans[j1][j2] = ca1*ax[j1]*ax[j2];
      trans[j1][j1]   += ca;
   }

   trans[0][1] = trans[0][1] - sa*ax[2];
   trans[1][0] = trans[1][0] + sa*ax[2];
   trans[0][2] = trans[0][2] + sa*ax[1];
   trans[2][0] = trans[2][0] - sa*ax[1];
   trans[1][2] = trans[1][2] - sa*ax[0];
   trans[2][1] = trans[2][1] + sa*ax[0];

   return (Double_t *)trans;

}

//______________________________________________________________________________
Color_t TPolyLineShape::SetColorAttribute(Color_t color)
{
   //to be documented
   Color_t currentColor = GetColorAttribute();
   if (color != currentColor) {
      SetLineColor(color);
      SetMarkerColor(color);
   }
   return currentColor;
}

//______________________________________________________________________________
Size_t TPolyLineShape::SetSizeAttribute(Size_t size)
{
   //to be documented
   Size_t currentSize = GetSizeAttribute();
   if (size != currentSize) {
      SetLineWidth(Width_t(size));
      SetMarkerSize(size);
   }
   return currentSize;
}

//______________________________________________________________________________
Style_t TPolyLineShape::SetStyleAttribute(Style_t style)
{
  // SetStyleAttribute(Style_t style) - set new style for this line
  // Returns:
  //          previous value of the line style
  //
   Style_t s = 0;
   s = GetStyleAttribute();
   SetLineStyle(style);
   SetMarkerStyle(style);
   return s;
}

//______________________________________________________________________________
void TPolyLineShape::SetShape(TShape *shape)
{
   //to be documented
   SafeDelete(fShape)
   fShape = shape;
}

//_______________________________________________________________________
Int_t TPolyLineShape::Size() const
{
   //to be documented
   return fPoints ? fPoints->Size():0;
}

//______________________________________________________________________________
void TPolyLineShape::Sizeof3D() const
{
//*-*-*-*-*-*-*Return total X3D size of this shape with its attributes*-*-*-*-*-*
//*-*          =======================================================
   TPolyLineShape *line = (TPolyLineShape *)this;
   if (fLineFlag )
      line->CreateX3DSize(kFALSE);
   else
      line->CreateX3DSize(kTRUE);
   if (fSizeX3D) {
      gSize3D.numPoints += fSizeX3D->numPoints;
      gSize3D.numSegs   += fSizeX3D->numSegs;
      gSize3D.numPolys  += fSizeX3D->numPolys;
   }
   else Error("Sizeof3D()","buffer size has not been defined yet");
}
