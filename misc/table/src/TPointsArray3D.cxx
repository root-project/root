// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   24/04/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"

#include "TPointsArray3D.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TClass.h"
#include "TROOT.h"
#include "TMath.h"

//______________________________________________________________________________
//
// TPointsArray3D is an abstract class of the array of 3-dimensional points.
// It has 4 different constructors.
//
// This class has no implementation for Paint, Draw, and SavePrimitive methods
//
//   First one, without any parameters TPointsArray3D(), we call 'default
// constructor' and it's used in a case that just an initialisation is
// needed (i.e. pointer declaration).
//
//       Example:
//                 TPointsArray3D *pl1 = new TPointsArray3D;
//
//
//   Second one is 'normal constructor' with, usually, one parameter
// n (number of points), and it just allocates a space for the points.
//
//       Example:
//                 TPointsArray3D pl1(150);
//
//
//   Third one allocates a space for the points, and also makes
// initialisation from the given array.
//
//       Example:
//                 TPointsArray3D pl1(150, pointerToAnArray);
//
//
//   Fourth one is, almost, similar to the constructor above, except
// initialisation is provided with three independent arrays (array of
// x coordinates, y coordinates and z coordinates).
//
//       Example:
//                 TPointsArray3D pl1(150, xArray, yArray, zArray);
//

ClassImp(TPointsArray3D)

//______________________________________________________________________________
TPointsArray3D::TPointsArray3D()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D PolyLine default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                      ================================

   fN = 0;
   fP = 0;
   fLastPoint = -1;
   fGLList = 0;
   fLastPoint = 0;
}


//______________________________________________________________________________
TPointsArray3D::TPointsArray3D(Int_t n, Option_t *option)
{
//*-*-*-*-*-*3-D PolyLine normal constructor without initialisation*-*-*-*-*-*-*
//*-*        ======================================================
//*-*  If n < 0 the default size (2 points) is set
//*-*
   fLastPoint = -1;
   if (n < 1) fN = 2;  // Set the default size for this object
   else fN = n;

   fP = new Float_t[3*fN];
   memset(fP,0,3*fN*sizeof(Float_t));
   fOption = option;

   fGLList = 0;
   fLastPoint = 0;
}

//______________________________________________________________________________
TPointsArray3D::TPointsArray3D(Int_t n, Float_t *p, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D Point3D normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================
//*-*  If n < 0 the default size (2 points) is set
//*-*

   if (n < 1) fN = 2;  // Set the default size for this object
   else fN = n;

   fP = new Float_t[3*fN];
   if (n > 0) {
      memcpy(fP,p,3*fN*sizeof(Float_t));
      fLastPoint = fN-1;
   } else {
      memset(fP,0,3*fN*sizeof(Float_t));
      fLastPoint = -1;
   }
   fOption = option;

   fGLList = 0;
   fLastPoint = 0;
}


//______________________________________________________________________________
TPointsArray3D::TPointsArray3D(Int_t n, Float_t *x, Float_t *y, Float_t *z, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D PolyLine normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================
//*-*  If n < 0 the default size (2 points) is set
//*-*

   fLastPoint = -1;
   if (n < 1) fN = 2;  // Set the default size for this object
   else fN = n;

   fP = new Float_t[3*fN];
   Int_t j = 0;
   if (n > 0) {
      for (Int_t i=0; i<n;i++) {
         fP[j++] = x[i];
         fP[j++] = y[i];
         fP[j++] = z[i];
      }
      fLastPoint = fN-1;
   } else {
      memset(fP,0,3*fN*sizeof(Float_t));
   }
   fOption = option;

   fGLList = 0;
   fLastPoint = 0;
}


//______________________________________________________________________________
TPointsArray3D::~TPointsArray3D()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D PolyLine default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================

   if (fP) delete [] fP;

}


//______________________________________________________________________________
TPointsArray3D::TPointsArray3D(const TPointsArray3D &point) : TPoints3DABC(point),
                                                              fN(point.fN),fP(0),fGLList(point.fGLList),fLastPoint(point.fLastPoint)
{
   //to be documented
   ((TPointsArray3D&)point).Copy(*this);
}


//______________________________________________________________________________
void TPointsArray3D::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Copy this TPointsArray3D to another *-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============================

   TObject::Copy(obj);
   ((TPointsArray3D&)obj).fN = fN;
   if (((TPointsArray3D&)obj).fP)
      delete [] ((TPointsArray3D&)obj).fP;
   ((TPointsArray3D&)obj).fP = new Float_t[3*fN];
   for (Int_t i=0; i<3*fN;i++)  {((TPointsArray3D&)obj).fP[i] = fP[i];}
   ((TPointsArray3D&)obj).fOption = fOption;
   ((TPointsArray3D&)obj).fLastPoint = fLastPoint;
}


//______________________________________________________________________________
Int_t TPointsArray3D::DistancetoPrimitive(Int_t px, Int_t py)
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
   Float_t dist = 9999;

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());

//*-*- return if point is not in the user area
   if (px < puxmin - inaxis) return Int_t (dist);
   if (py > puymin + inaxis) return Int_t (dist);
   if (px > puxmax + inaxis) return Int_t (dist);
   if (py < puymax - inaxis) return Int_t (dist);

   TView *view = gPad->GetView();
   if (!view) return Int_t(dist);
   Int_t i;
   Float_t dpoint;
   Float_t xndc[3];
   Int_t x1,y1;
   Int_t size = Size();
   for (i=0;i<size;i++) {
      view->WCtoNDC(&fP[3*i], xndc);
      x1     = gPad->XtoAbsPixel(xndc[0]);
      y1     = gPad->YtoAbsPixel(xndc[1]);
      dpoint = (px-x1)*(px-x1) + (py-y1)*(py-y1);
      if (dpoint < dist) dist = dpoint;
   }
   return Int_t(TMath::Sqrt(dist));
}


//______________________________________________________________________________
void TPointsArray3D::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*-*-*-*-*-*-*
//*-*                =========================================
   if (gPad->GetView())
      gPad->GetView()->ExecuteRotateView(event, px, py);
}

//______________________________________________________________________________
void TPointsArray3D::ls(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*List this 3-D polyline with its attributes*-*-*-*-*-*-*
//*-*                ==========================================

   TROOT::IndentLevel();
   std::cout << IsA()->GetName() << " N=" <<fN<<" Option="<<option<<std::endl;

}
//______________________________________________________________________________
void TPointsArray3D::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*Dump this 3-D polyline with its attributes*-*-*-*-*-*-*-*-*
//*-*                ==========================================

   std::cout <<"   " << IsA()->GetName() <<" Printing N=" <<fN<<" Option="<<option<<std::endl;
}
//______________________________________________________________________________
Int_t TPointsArray3D::SetLastPosition(Int_t idx)
{
   //to be documented
   fLastPoint = TMath::Min(idx,GetN()-1);
   return idx;
}

//______________________________________________________________________________
Int_t TPointsArray3D::SetPoint(Int_t n, Float_t x, Float_t y, Float_t z)
{
//*-*-*-*-*-*-*-*-*-*Initialize one point of the 3-D polyline*-*-*-*-*-*-*-*-*-*
//*-*                ========================================
//*-*  if n is more then the current TPointsArray3D size (n > fN) - re-allocate this
//*-*  The new size of the object will be fN += min(10,fN/4)
//*-*
//*-*  return the total number of points introduced
//*-*

   if (n < 0) return n;
   if (!fP || n >= fN) {
   // re-allocate the object
      Int_t step = TMath::Max(10, fN/4);
      Float_t *savepoint = new Float_t [3*(fN+step)];
      if (fP && fN){
         memcpy(savepoint,fP,3*fN*sizeof(Float_t));
         delete [] fP;
      }
      fP = savepoint;
      fN += step;
   }
   fP[3*n  ] = x;
   fP[3*n+1] = y;
   fP[3*n+2] = z;
   fLastPoint = TMath::Max(fLastPoint,n);
   return fLastPoint;
}

//______________________________________________________________________________
Int_t TPointsArray3D::SetPoints(Int_t n, Float_t *p, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Set new values for this 3-D polyline*-*-*-*-*-*-*-*-*-*-*
//*-*                  ====================================
//*-* return the total number of points introduced
//*-*

   if (n < 0) return n;
   fN = n;
   if (fP) delete [] fP;
   fP = new Float_t[3*fN];
   for (Int_t i=0; i<3*fN;i++) {
      if (p) fP[i] = p[i];
      else   memset(fP,0,3*fN*sizeof(Float_t));
   }
   fOption = option;
   fLastPoint = fN-1;
   return fLastPoint;
}

//_______________________________________________________________________
void TPointsArray3D::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      b.ReadVersion();  //Version_t v = b.ReadVersion();
      TObject::Streamer(b);
      b >> fN;
      if (fN) {
         fP = new Float_t[3*fN];
         b.ReadFastArray(fP,3*fN);
      }
      fOption.Streamer(b);
      fLastPoint = fN;
   } else {
      b.WriteVersion(TPointsArray3D::IsA());
      TObject::Streamer(b);
      Int_t size = Size();
      b << size;
      if (size) b.WriteFastArray(fP, 3*size);
      fOption.Streamer(b);
   }
}
