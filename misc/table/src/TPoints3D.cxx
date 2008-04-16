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

#include "TROOT.h"
#include "TClass.h"
#include "TPoints3D.h"
#include "TPointsArray3D.h"

//______________________________________________________________________________
//
// TPoints3D is an abstract class of the array of 3-dimensional points.
// It has 4 different constructors.
//
// This class has no implemenatation for Paint, Draw, and SavePrimitive methods
//
//   First one, without any parameters TPoints3D(), we call 'default
// constructor' and it's used in a case that just an initialisation is
// needed (i.e. pointer declaration).
//
//       Example:
//                 TPoints3D *pl1 = new TPoints3D;
//
//
//   Second one is 'normal constructor' with, usually, one parameter
// n (number of points), and it just allocates a space for the points.
//
//       Example:
//                 TPoints3D pl1(150);
//
//
//   Third one allocates a space for the points, and also makes
// initialisation from the given array.
//
//       Example:
//                 TPoints3D pl1(150, pointerToAnArray);
//
//
//   Fourth one is, almost, similar to the constructor above, except
// initialisation is provided with three independent arrays (array of
// x coordinates, y coordinates and z coordinates).
//
//       Example:
//                 TPoints3D pl1(150, xArray, yArray, zArray);
//

ClassImp(TPoints3D)

//______________________________________________________________________________
TPoints3D::TPoints3D(TPoints3DABC *points) : fPoints(points)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D PolyLine default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                      ================================
   DoOwner(kFALSE);
   fPoints = points;
   if (!fPoints) {
      fPoints = new TPointsArray3D;
      DoOwner();
   }
}

//______________________________________________________________________________
TPoints3D::TPoints3D(Int_t n, Option_t *option) : fPoints( new TPointsArray3D(n,option))
{
//*-*-*-*-*-*3-D PolyLine normal constructor without initialisation*-*-*-*-*-*-*
//*-*        ======================================================
//*-*  If n < 0 the default size (2 points) is set
//*-*
   DoOwner();
}

//______________________________________________________________________________
TPoints3D::TPoints3D(Int_t n, Float_t *p, Option_t *option) : fPoints(new TPointsArray3D(n,p,option))
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D Point3D normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================
//*-*  If n < 0 the default size (2 points) is set
//*-*
   DoOwner();
}


//______________________________________________________________________________
TPoints3D::TPoints3D(Int_t n, Float_t *x, Float_t *y, Float_t *z, Option_t *option)
                       : fPoints(new TPointsArray3D(n,x,y,z,option))
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D PolyLine normal constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================
//*-*  If n < 0 the default size (2 points) is set
//*-*
   DoOwner();
}


//______________________________________________________________________________
TPoints3D::~TPoints3D()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*3-D PolyLine default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ===============================
   Delete();
}
//______________________________________________________________________________
TPoints3D::TPoints3D(const TPoints3D &point) : TPoints3DABC(point)
{
   //to be documented
   ((TPoints3D&)point).Copy(*this);
}
//______________________________________________________________________________
void TPoints3D::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Copy this TPoints3D to another *-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============================

   TPoints3DABC::Copy(obj);
   TPoints3D &thatObject = (TPoints3D&)obj;
   thatObject.Delete();
   if (thatObject.IsOwner()) {
      thatObject.fPoints =  new TPoints3D(GetN(),GetP(),GetOption());
      (thatObject.fPoints)->SetLastPosition(GetLastPosition());
   }
   else
      thatObject.fPoints = fPoints;
}

//______________________________________________________________________________
void TPoints3D::Delete()
{
   // Delete only own object
   if (fPoints && IsOwner()) delete fPoints;
   fPoints = 0;
}

//______________________________________________________________________________
Bool_t TPoints3D::DoOwner(Bool_t done)
{
   //to be documented
   if (done) SetBit(kIsOwner);
   else ResetBit(kIsOwner);
   return IsOwner();
}

//______________________________________________________________________________
void TPoints3D::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*-*-*-*-*-*-*
//*-*                =========================================
   if (fPoints)
      fPoints->ExecuteEvent(event,px,py);
}

//______________________________________________________________________________
void TPoints3D::ls(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*List this 3-D polyline with its attributes*-*-*-*-*-*-*
//*-*                ==========================================

   TROOT::IndentLevel();
   cout << IsA()->GetName() << " N=" <<GetN()<<" Option="<<option<<endl;
//   IsOwner()?"Owner":"Not owner" << endl;
}

//______________________________________________________________________________
void TPoints3D::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*Dump this 3-D polyline with its attributes*-*-*-*-*-*-*-*-*
//*-*                ==========================================
   cout <<"   " << IsA()->GetName() <<" Printing N=" <<GetN()<<" Option="<<option<<endl;
//   IsOwner()?"Owner":"Not owner" << endl;
}

