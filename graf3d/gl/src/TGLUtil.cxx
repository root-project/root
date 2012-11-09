// @(#)root/gl:$Id$
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <string>
#include <map>

#include "THLimitsFinder.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "Riostream.h"
#include "TStyle.h"
#include "TGaxis.h"
#include "TColor.h"
#include "TError.h"
#include "TH1.h"
#include "TMath.h"
#include "TROOT.h"
#include "TClass.h"
#include "TEnv.h"

#include "TGLBoundingBox.h"
#include "TGLCamera.h"
#include "TGLPlotPainter.h"
#include "TGLIncludes.h"
#include "TGLQuadric.h"
#include "TGLUtil.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLVertex3                                                           //
//                                                                      //
// 3 component (x/y/z) vertex class                                     //
//                                                                      //
// This is part of collection of simple utility classes for GL only in  //
// TGLUtil.h/cxx. These provide const and non-const accessors Arr() &   //
// CArr() to a GL compatible internal field - so can be used directly   //
// with OpenGL C API calls - which TVector3 etc cannot (easily).        //
// They are not intended to be fully featured just provide minimum      //
// required.                                                            //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLVertex3);

//______________________________________________________________________________
TGLVertex3::TGLVertex3()
{
   // Construct a default (0.0, 0.0, 0.0) vertex
   Fill(0.0);
}

//______________________________________________________________________________
TGLVertex3::TGLVertex3(Double_t x, Double_t y, Double_t z)
{
   // Construct a vertex with components (x,y,z)
   Set(x,y,z);
}

//______________________________________________________________________________
TGLVertex3::TGLVertex3(Double_t* v)
{
   // Construct a vertex with components (v[0], v[1], v[2])
   Set(v[0], v[1], v[2]);
}

//______________________________________________________________________________
TGLVertex3::TGLVertex3(const TGLVertex3 & other)
{
   // Construct a vertex from 'other'
   Set(other);
}

//______________________________________________________________________________
TGLVertex3::~TGLVertex3()
{
   // Destroy vertex object
}

//______________________________________________________________________________
void TGLVertex3::Shift(TGLVector3 & shift)
{
   // Offset a vertex by vector 'shift'
   fVals[0] += shift[0];
   fVals[1] += shift[1];
   fVals[2] += shift[2];
}

//______________________________________________________________________________
void TGLVertex3::Shift(Double_t xDelta, Double_t yDelta, Double_t zDelta)
{
   // Offset a vertex by components (xDelta, yDelta, zDelta)
   fVals[0] += xDelta;
   fVals[1] += yDelta;
   fVals[2] += zDelta;
}

//______________________________________________________________________________
void TGLVertex3::Minimum(const TGLVertex3 & other)
{
   fVals[0] = TMath::Min(fVals[0], other.fVals[0]);
   fVals[1] = TMath::Min(fVals[1], other.fVals[1]);
   fVals[2] = TMath::Min(fVals[2], other.fVals[2]);
}

//______________________________________________________________________________
void TGLVertex3::Maximum(const TGLVertex3 & other)
{
   fVals[0] = TMath::Max(fVals[0], other.fVals[0]);
   fVals[1] = TMath::Max(fVals[1], other.fVals[1]);
   fVals[2] = TMath::Max(fVals[2], other.fVals[2]);
}

//______________________________________________________________________________
void TGLVertex3::Dump() const
{
   // Output vertex component values to std::cout
   std::cout << "(" << fVals[0] << "," << fVals[1] << "," << fVals[2] << ")" << std::endl;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLVector3                                                           //
//                                                                      //
// 3 component (x/y/z) vector class                                     //
//                                                                      //
// This is part of collection of utility classes for GL in TGLUtil.h/cxx//
// These provide const and non-const accessors Arr() / CArr() to a GL   //
// compatible internal field - so can be used directly with OpenGL C API//
// calls. They are not intended to be fully featured just provide       //
// minimum required.                                                    //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLVector3);

//______________________________________________________________________________
TGLVector3::TGLVector3() :
   TGLVertex3()
{
   // Construct a default (0.0, 0.0, 0.0) vector
}

//______________________________________________________________________________
TGLVector3::TGLVector3(Double_t x, Double_t y, Double_t z) :
   TGLVertex3(x, y, z)
{
   // Construct a vector with components (x,y,z)
}

//______________________________________________________________________________
TGLVector3::TGLVector3(const TGLVector3 & other) :
   TGLVertex3(other.fVals[0], other.fVals[1], other.fVals[2])
{
   // Construct a vector from components of 'other'
}

//______________________________________________________________________________
TGLVector3::TGLVector3(const Double_t *src) :
   TGLVertex3(src[0], src[1], src[2])
{
   // Construct a vector with components (src[0], src[1], src[2])
}

//______________________________________________________________________________
TGLVector3::~TGLVector3()
{
   // Destroy vector object
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLLine3                                                             //
//                                                                      //
// 3D space, fixed length, line class, with direction / length 'vector',//
// passing through point 'vertex'. Just wraps a TGLVector3 / TGLVertex3 //
// pair.                                                                //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLLine3);

//______________________________________________________________________________
TGLLine3::TGLLine3(const TGLVertex3 & start, const TGLVertex3 & end) :
   fVertex(start), fVector(end - start)
{
   // Construct 3D line running from 'start' to 'end'
}

//______________________________________________________________________________
TGLLine3::TGLLine3(const TGLVertex3 & start, const TGLVector3 & vect) :
   fVertex(start), fVector(vect)
{
   // Construct 3D line running from 'start', magnitude 'vect'
}

//______________________________________________________________________________
TGLLine3::~TGLLine3()
{
   // Destroy 3D line object
}

//______________________________________________________________________________
void TGLLine3::Set(const TGLVertex3 & start, const TGLVertex3 & end)
{
   // Set 3D line running from 'start' to 'end'

   fVertex = start;
   fVector = end - start;
}

//______________________________________________________________________________
void TGLLine3::Set(const TGLVertex3 & start, const TGLVector3 & vect)
{
   // Set 3D line running from start, magnitude 'vect'
   fVertex = start;
   fVector = vect;
}

//______________________________________________________________________________
void TGLLine3::Draw() const
{
   // Draw line in current basic GL color. Assume we are in the correct reference
   // frame
   glBegin(GL_LINE_LOOP);
   glVertex3dv(fVertex.CArr());
   glVertex3dv(End().CArr());
   glEnd();
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLRect                                                              //
//                                                                      //
// Viewport (pixel base) 2D rectangle class                             //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLRect);

//______________________________________________________________________________
TGLRect::TGLRect() :
      fX(0), fY(0), fWidth(0), fHeight(0)
{
   // Construct empty rect object, corner (0,0), width/height 0
}

//______________________________________________________________________________
TGLRect::TGLRect(Int_t x, Int_t y, Int_t width, Int_t height) :
      fX(x), fY(y), fWidth(width), fHeight(height)
{
   // Construct rect object, corner (x,y), dimensions 'width', 'height'
}

//______________________________________________________________________________
TGLRect::TGLRect(Int_t x, Int_t y, UInt_t width, UInt_t height) :
      fX(x), fY(y), fWidth(width), fHeight(height)
{
   // Construct rect object, corner (x,y), dimensions 'width', 'height'
}


//______________________________________________________________________________
TGLRect::~TGLRect()
{
   // Destroy rect object
}

//______________________________________________________________________________
void TGLRect::Expand(Int_t x, Int_t y)
{
   // Expand the rect to encompass point (x,y)
   Int_t delX = x - fX;
   Int_t delY = y - fY;

   if (delX > fWidth) {
      fWidth = delX;
   }
   if (delY > fHeight) {
      fHeight = delY;
   }

   if (delX < 0) {
      fX = x;
      fWidth += -delX;
   }
   if (delY < 0) {
      fY = y;
      fHeight += -delY;
   }
}

//______________________________________________________________________________
Int_t TGLRect::Diagonal() const
{
   // Return the diagonal of the rectangle.

   const Double_t w = static_cast<Double_t>(fWidth);
   const Double_t h = static_cast<Double_t>(fHeight);
   return TMath::Nint(TMath::Sqrt(w*w + h*h));
}

//______________________________________________________________________________
Rgl::EOverlap TGLRect::Overlap(const TGLRect & other) const
{
   // Return overlap result (kInside, kOutside, kPartial) of this
   // rect with 'other'

   using namespace Rgl;

   if ((fX <= other.fX) && (fX + fWidth  >= other.fX + other.fWidth) &&
       (fY <= other.fY) && (fY + fHeight >= other.fY + other.fHeight))
   {
      return kInside;
   }
   else if ((fX >= other.fX + static_cast<Int_t>(other.fWidth))  ||
            (fX + static_cast<Int_t>(fWidth) <= other.fX)        ||
            (fY >= other.fY + static_cast<Int_t>(other.fHeight)) ||
            (fY + static_cast<Int_t>(fHeight) <= other.fY))
   {
      return kOutside;
   }
   else
   {
      return kPartial;
   }
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLPlane                                                             //
//                                                                      //
// 3D plane class - of format Ax + By + Cz + D = 0                      //
//                                                                      //
// This is part of collection of simple utility classes for GL only in  //
// TGLUtil.h/cxx. These provide const and non-const accessors Arr() &   //
// CArr() to a GL compatible internal field - so can be used directly   //
// with OpenGL C API calls - which TVector3 etc cannot (easily).        //
// They are not intended to be fully featured just provide minimum      //
// required.                                                            //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLPlane);

//______________________________________________________________________________
TGLPlane::TGLPlane()
{
   // Construct a default plane of x + y + z = 0
   Set(1.0, 1.0, 1.0, 0.0);
}

//______________________________________________________________________________
TGLPlane::TGLPlane(const TGLPlane & other)
{
   // Construct plane from 'other'
   Set(other);
}

//______________________________________________________________________________
TGLPlane::TGLPlane(Double_t a, Double_t b, Double_t c, Double_t d)
{
   // Construct plane with equation a.x + b.y + c.z + d = 0
   // with optional normalisation
   Set(a, b, c, d);
}

//______________________________________________________________________________
TGLPlane::TGLPlane(Double_t eq[4])
{
   // Construct plane with equation eq[0].x + eq[1].y + eq[2].z + eq[3] = 0
   // with optional normalisation
   Set(eq);
}

//______________________________________________________________________________
TGLPlane::TGLPlane(const TGLVertex3 & p1, const TGLVertex3 & p2,
                   const TGLVertex3 & p3)
{
   // Construct plane passing through 3 supplied points
   // with optional normalisation
   Set(p1, p2, p3);
}

//______________________________________________________________________________
TGLPlane::TGLPlane(const TGLVector3 & v, const TGLVertex3 & p)
{
   // Construct plane with supplied normal vector, passing through point
   // with optional normalisation
   Set(v, p);
}

//______________________________________________________________________________
TGLPlane::~TGLPlane()
{
   // Destroy plane object
}

//______________________________________________________________________________
void TGLPlane::Normalise()
{
   // Normalise the plane.

   Double_t mag = sqrt(fVals[0]*fVals[0] + fVals[1]*fVals[1] + fVals[2]*fVals[2]);

   if (mag == 0.0 ) {
      Error("TGLPlane::Normalise", "trying to normalise plane with zero magnitude normal");
      return;
   }
   mag = 1.0 / mag;
   fVals[0] *= mag;
   fVals[1] *= mag;
   fVals[2] *= mag;
   fVals[3] *= mag;
}

//______________________________________________________________________________
void TGLPlane::Dump() const
{
   // Output plane equation to std::out

   std::cout.precision(6);
   std::cout << "Plane : " << fVals[0] << "x + " << fVals[1] << "y + " << fVals[2] << "z + " << fVals[3] << std::endl;
}

//______________________________________________________________________________
void TGLPlane::Set(const TGLPlane & other)
{
   // Assign from other.

   fVals[0] = other.fVals[0];
   fVals[1] = other.fVals[1];
   fVals[2] = other.fVals[2];
   fVals[3] = other.fVals[3];
}

//______________________________________________________________________________
void TGLPlane::Set(Double_t a, Double_t b, Double_t c, Double_t d)
{
   // Set by values.

   fVals[0] = a;
   fVals[1] = b;
   fVals[2] = c;
   fVals[3] = d;
   Normalise();
}

//______________________________________________________________________________
void TGLPlane::Set(Double_t eq[4])
{
   // Set by array values.

   fVals[0] = eq[0];
   fVals[1] = eq[1];
   fVals[2] = eq[2];
   fVals[3] = eq[3];
   Normalise();
}

//______________________________________________________________________________
void TGLPlane::Set(const TGLVector3 & norm, const TGLVertex3 & point)
{
   // Set plane from a normal vector and in-plane point pair

   fVals[0] = norm[0];
   fVals[1] = norm[1];
   fVals[2] = norm[2];
   fVals[3] = -(fVals[0]*point[0] + fVals[1]*point[1] + fVals[2]*point[2]);
   Normalise();
}

//______________________________________________________________________________
void TGLPlane::Set(const TGLVertex3 & p1, const TGLVertex3 & p2, const TGLVertex3 & p3)
{
   // Set plane by three points.

   TGLVector3 norm = Cross(p2 - p1, p3 - p1);
   Set(norm, p2);
}

//______________________________________________________________________________
void TGLPlane::Negate()
{
   // Negate the plane.

   fVals[0] = -fVals[0];
   fVals[1] = -fVals[1];
   fVals[2] = -fVals[2];
   fVals[3] = -fVals[3];
}

//______________________________________________________________________________
Double_t TGLPlane::DistanceTo(const TGLVertex3 & vertex) const
{
   // Distance from plane to vertex.

   return (fVals[0]*vertex[0] + fVals[1]*vertex[1] + fVals[2]*vertex[2] + fVals[3]);
}

//______________________________________________________________________________
TGLVertex3 TGLPlane::NearestOn(const TGLVertex3 & point) const
{
   // Return nearest point on plane.

   TGLVector3 o = Norm() * (Dot(Norm(), TGLVector3(point[0], point[1], point[2])) + D() / Dot(Norm(), Norm()));
   TGLVertex3 v = point - o;
   return v;
}

// Some free functions for plane intersections

//______________________________________________________________________________
std::pair<Bool_t, TGLLine3> Intersection(const TGLPlane & p1, const TGLPlane & p2)
{
   // Find 3D line interestion of this plane with 'other'. Returns a std::pair
   //
   // first (Bool_t)                   second (TGLLine3)
   // kTRUE - planes intersect         intersection line between planes
   // kFALSE - no intersect (parallel) undefined
   TGLVector3 lineDir = Cross(p1.Norm(), p2.Norm());

   if (lineDir.Mag() == 0.0) {
      return std::make_pair(kFALSE, TGLLine3(TGLVertex3(0.0, 0.0, 0.0),
                                             TGLVector3(0.0, 0.0, 0.0)));
   }
   TGLVertex3 linePoint = Cross((p1.Norm()*p2.D() - p2.Norm()*p1.D()), lineDir) /
                           Dot(lineDir, lineDir);
   return std::make_pair(kTRUE, TGLLine3(linePoint, lineDir));
}

//______________________________________________________________________________
std::pair<Bool_t, TGLVertex3> Intersection(const TGLPlane & p1, const TGLPlane & p2, const TGLPlane & p3)
{
   Double_t denom = Dot(p1.Norm(), Cross(p2.Norm(), p3.Norm()));
   if (denom == 0.0) {
      return std::make_pair(kFALSE, TGLVertex3(0.0, 0.0, 0.0));
   }
   TGLVector3 vect = ((Cross(p2.Norm(),p3.Norm())* -p1.D()) -
                      (Cross(p3.Norm(),p1.Norm())*p2.D()) -
                      (Cross(p1.Norm(),p2.Norm())*p3.D())) / denom;
   TGLVertex3 interVert(vect.X(), vect.Y(), vect.Z());
   return std::make_pair(kTRUE, interVert);
}

//______________________________________________________________________________
std::pair<Bool_t, TGLVertex3> Intersection(const TGLPlane & plane, const TGLLine3 & line, Bool_t extend)
{
   // Find intersection of 3D space 'line' with this plane. If 'extend' is kTRUE
   // then line extents can be extended (infinite length) to find intersection.
   // If 'extend' is kFALSE the fixed extents of line is respected.
   //
   // The return a std::pair
   //
   // first (Bool_t)                   second (TGLVertex3)
   // kTRUE - line/plane intersect     intersection vertex on plane
   // kFALSE - no line/plane intersect undefined
   //
   // If intersection is not found (first == kFALSE) & 'extend' was kTRUE (infinite line)
   // this implies line and plane are parallel. If 'extend' was kFALSE, then
   // either line parallel or insuffient length.
   Double_t denom = -(plane.A()*line.Vector().X() +
                      plane.B()*line.Vector().Y() +
                      plane.C()*line.Vector().Z());

   if (denom == 0.0) {
      return std::make_pair(kFALSE, TGLVertex3(0.0, 0.0, 0.0));
   }

   Double_t num = plane.A()*line.Start().X() + plane.B()*line.Start().Y() +
                  plane.C()*line.Start().Z() + plane.D();
   Double_t factor = num/denom;

   // If not extending (projecting) line is length from start enough to reach plane?
   if (!extend && (factor < 0.0 || factor > 1.0)) {
      return std::make_pair(kFALSE, TGLVertex3(0.0, 0.0, 0.0));
   }

   TGLVector3 toPlane = line.Vector() * factor;
   return std::make_pair(kTRUE, line.Start() + toPlane);
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLMatrix                                                            //
//                                                                      //
// 16 component (4x4) transform matrix - column MAJOR as per GL.        //
// Provides limited support for adjusting the translation, scale and    //
// rotation components.                                                 //
//                                                                      //
// This is part of collection of simple utility classes for GL only in  //
// TGLUtil.h/cxx. These provide const and non-const accessors Arr() &   //
// CArr() to a GL compatible internal field - so can be used directly   //
// with OpenGL C API calls - which TVector3 etc cannot (easily).        //
// They are not intended to be fully featured just provide minimum      //
// required.                                                            //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLMatrix);

//______________________________________________________________________________
TGLMatrix::TGLMatrix()
{
   // Construct default identity matrix:
   //
   // 1 0 0 0
   // 0 1 0 0
   // 0 0 1 0
   // 0 0 0 1
   SetIdentity();
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(Double_t x, Double_t y, Double_t z)
{
   // Construct matrix with translation components x,y,z:
   //
   // 1 0 0 x
   // 0 1 0 y
   // 0 0 1 z
   // 0 0 0 1
   SetIdentity();
   SetTranslation(x, y, z);
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(const TGLVertex3 & translation)
{
   // Construct matrix with translation components x,y,z:
   //
   // 1 0 0 translation.X()
   // 0 1 0 translation.Y()
   // 0 0 1 translation.Z()
   // 0 0 0 1
   SetIdentity();
   SetTranslation(translation);
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(const TGLVertex3 & origin, const TGLVector3 & zAxis)
{
   // Construct matrix which when applied puts local origin at
   // 'origin' and the local Z axis in direction 'z'. Both
   // 'origin' and 'zAxisVec' are expressed in the parent frame
   SetIdentity();

   TGLVector3 zAxisInt(zAxis);
   zAxisInt.Normalise();
   TGLVector3 arbAxis;

   if (TMath::Abs(zAxisInt.X()) <= TMath::Abs(zAxisInt.Y()) && TMath::Abs(zAxisInt.X()) <= TMath::Abs(zAxisInt.Z())) {
      arbAxis.Set(1.0, 0.0, 0.0);
   } else if (TMath::Abs(zAxisInt.Y()) <= TMath::Abs(zAxisInt.X()) && TMath::Abs(zAxisInt.Y()) <= TMath::Abs(zAxisInt.Z())) {
      arbAxis.Set(0.0, 1.0, 0.0);
   } else {
      arbAxis.Set(0.0, 0.0, 1.0);
   }

   Set(origin, zAxis, Cross(zAxisInt, arbAxis));
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(const TGLVertex3 & origin, const TGLVector3 & zAxis, const TGLVector3 & xAxis)
{
   // Construct matrix which when applied puts local origin at
   // 'origin' and the local Z axis in direction 'z'. Both
   // 'origin' and 'zAxisVec' are expressed in the parent frame
   SetIdentity();
   Set(origin, zAxis, xAxis);
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(const Double_t vals[16])
{
   // Construct matrix using the 16 Double_t 'vals' passed,
   // ordering is maintained - i.e. should be column major
   // as we are
   Set(vals);
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(const TGLMatrix & other)
{
   // Construct matrix from 'other'
   *this = other;
}

//______________________________________________________________________________
TGLMatrix::~TGLMatrix()
{
   // Destroy matirx object
}

//______________________________________________________________________________
void TGLMatrix::MultRight(const TGLMatrix & rhs)
{
   // Multiply with matrix rhs on right.

  Double_t  B[4];
  Double_t* C = fVals;
  for(int r=0; r<4; ++r, ++C)
  {
    const Double_t* T = rhs.fVals;
    for(int c=0; c<4; ++c, T+=4)
      B[c] = C[0]*T[0] + C[4]*T[1] + C[8]*T[2] + C[12]*T[3];
    C[0] = B[0]; C[4] = B[1]; C[8] = B[2]; C[12] = B[3];
  }
}

//______________________________________________________________________________
void TGLMatrix::MultLeft (const TGLMatrix & lhs)
{
   // Multiply with matrix lhs on left.

   Double_t  B[4];
   Double_t* C = fVals;
   for (int c=0; c<4; ++c, C+=4)
   {
      const Double_t* T = lhs.fVals;
      for(int r=0; r<4; ++r, ++T)
         B[r] = T[0]*C[0] + T[4]*C[1] + T[8]*C[2] + T[12]*C[3];
      C[0] = B[0]; C[1] = B[1]; C[2] = B[2]; C[3] = B[3];
   }
}

//______________________________________________________________________________
void TGLMatrix::Set(const TGLVertex3 & origin, const TGLVector3 & zAxis, const TGLVector3 & xAxis)
{
   // Set matrix which when applied puts local origin at
   // 'origin' and the local Z axis in direction 'z'. Both
   // 'origin' and 'z' are expressed in the parent frame
   TGLVector3 zAxisInt(zAxis);
   zAxisInt.Normalise();

   TGLVector3 xAxisInt(xAxis);
   xAxisInt.Normalise();
   TGLVector3 yAxisInt = Cross(zAxisInt, xAxisInt);

   fVals[0] = xAxisInt.X(); fVals[4] = yAxisInt.X(); fVals[8 ] = zAxisInt.X(); fVals[12] = origin.X();
   fVals[1] = xAxisInt.Y(); fVals[5] = yAxisInt.Y(); fVals[9 ] = zAxisInt.Y(); fVals[13] = origin.Y();
   fVals[2] = xAxisInt.Z(); fVals[6] = yAxisInt.Z(); fVals[10] = zAxisInt.Z(); fVals[14] = origin.Z();
   fVals[3] = 0.0;          fVals[7] = 0.0;          fVals[11] = 0.0;          fVals[15] = 1.0;
}

//______________________________________________________________________________
void TGLMatrix::Set(const Double_t vals[16])
{
   // Set matrix using the 16 Double_t 'vals' passed,
   // ordering is maintained - i.e. should be column major.

   for (UInt_t i=0; i < 16; i++) {
      fVals[i] = vals[i];
   }
}

//______________________________________________________________________________
void TGLMatrix::SetIdentity()
{
   // Set matrix to identity.

   fVals[0] = 1.0; fVals[4] = 0.0; fVals[8 ] = 0.0; fVals[12] = 0.0;
   fVals[1] = 0.0; fVals[5] = 1.0; fVals[9 ] = 0.0; fVals[13] = 0.0;
   fVals[2] = 0.0; fVals[6] = 0.0; fVals[10] = 1.0; fVals[14] = 0.0;
   fVals[3] = 0.0; fVals[7] = 0.0; fVals[11] = 0.0; fVals[15] = 1.0;
}

//______________________________________________________________________________
void TGLMatrix::SetTranslation(Double_t x, Double_t y, Double_t z)
{
   // Set matrix translation components x,y,z.

   SetTranslation(TGLVertex3(x,y,z));
}

//______________________________________________________________________________
void TGLMatrix::SetTranslation(const TGLVertex3 & translation)
{
   // Set matrix translation components x,y,z.

   fVals[12] = translation[0];
   fVals[13] = translation[1];
   fVals[14] = translation[2];
}

//______________________________________________________________________________
TGLVector3 TGLMatrix::GetTranslation() const
{
   // Return the translation component of matrix.

   return TGLVector3(fVals[12], fVals[13], fVals[14]);
}

//______________________________________________________________________________
void TGLMatrix::Translate(const TGLVector3 & vect)
{
   // Shift matrix translation components by 'vect' in parent frame.

   fVals[12] += vect[0];
   fVals[13] += vect[1];
   fVals[14] += vect[2];
}

//______________________________________________________________________________
void TGLMatrix::MoveLF(Int_t ai, Double_t amount)
{
   // Translate in local frame.
   // i1, i2 are axes indices: 1 ~ x, 2 ~ y, 3 ~ z.

   const Double_t *C = fVals + 4*--ai;
   fVals[12] += amount*C[0]; fVals[13] += amount*C[1]; fVals[14] += amount*C[2];
}

//______________________________________________________________________________
void TGLMatrix::Move3LF(Double_t x, Double_t y, Double_t z)
{
   // Translate in local frame along all base vectors simultaneously.

   fVals[12] += x*fVals[0] + y*fVals[4] + z*fVals[8];
   fVals[13] += x*fVals[1] + y*fVals[5] + z*fVals[9];
   fVals[14] += x*fVals[2] + y*fVals[6] + z*fVals[10];
}

//______________________________________________________________________________
void TGLMatrix::Scale(const TGLVector3 & scale)
{
   // Set matrix axis scales to 'scale'. Note - this really sets
   // the overall (total) scaling for each axis - it does NOT
   // apply compounded scale on top of existing one
   TGLVector3 currentScale = GetScale();

   // x
   if (currentScale[0] != 0.0) {
      fVals[0] *= scale[0]/currentScale[0];
      fVals[1] *= scale[0]/currentScale[0];
      fVals[2] *= scale[0]/currentScale[0];
   } else {
      Error("TGLMatrix::Scale()", "zero scale div by zero");
   }
   // y
   if (currentScale[1] != 0.0) {
      fVals[4] *= scale[1]/currentScale[1];
      fVals[5] *= scale[1]/currentScale[1];
      fVals[6] *= scale[1]/currentScale[1];
   } else {
      Error("TGLMatrix::Scale()", "zero scale div by zero");
   }
   // z
   if (currentScale[2] != 0.0) {
      fVals[8] *= scale[2]/currentScale[2];
      fVals[9] *= scale[2]/currentScale[2];
      fVals[10] *= scale[2]/currentScale[2];
   } else {
      Error("TGLMatrix::Scale()", "zero scale div by zero");
   }
}

//______________________________________________________________________________
void TGLMatrix::Rotate(const TGLVertex3 & pivot, const TGLVector3 & axis, Double_t angle)
{
   // Update martix so resulting transform has been rotated about 'pivot'
   // (in parent frame), round vector 'axis', through 'angle' (radians)
   // Equivalent to glRotate function, but with addition of translation
   // and compounded on top of existing.
   TGLVector3 nAxis = axis;
   nAxis.Normalise();
   Double_t x = nAxis.X();
   Double_t y = nAxis.Y();
   Double_t z = nAxis.Z();
   Double_t c = TMath::Cos(angle);
   Double_t s = TMath::Sin(angle);

   // Calculate local rotation, with pre-translation to local pivot origin
   TGLMatrix rotMat;
   rotMat[ 0] = x*x*(1-c) + c;   rotMat[ 4] = x*y*(1-c) - z*s; rotMat[ 8] = x*z*(1-c) + y*s; rotMat[12] = pivot[0];
   rotMat[ 1] = y*x*(1-c) + z*s; rotMat[ 5] = y*y*(1-c) + c;   rotMat[ 9] = y*z*(1-c) - x*s; rotMat[13] = pivot[1];
   rotMat[ 2] = x*z*(1-c) - y*s; rotMat[ 6] = y*z*(1-c) + x*s; rotMat[10] = z*z*(1-c) + c;   rotMat[14] = pivot[2];
   rotMat[ 3] = 0.0;             rotMat[ 7] = 0.0;             rotMat[11] = 0.0;             rotMat[15] = 1.0;
   TGLMatrix localToWorld(-pivot);

   // TODO: Ugly - should use quaternions to avoid compound rounding errors and
   // triple multiplication
   *this = rotMat * localToWorld * (*this);
}

//______________________________________________________________________________
void TGLMatrix::RotateLF(Int_t i1, Int_t i2, Double_t amount)
{
   // Rotate in local frame. Does optimised version of MultRight.
   // i1, i2 are axes indices: 1 ~ x, 2 ~ y, 3 ~ z.

   if(i1 == i2) return;
   const Double_t cos = TMath::Cos(amount), sin = TMath::Sin(amount);
   Double_t  b1, b2;
   Double_t* c = fVals;
   --i1 <<= 2; --i2 <<= 2; // column major
   for(int r=0; r<4; ++r, ++c) {
      b1 = cos*c[i1] + sin*c[i2];
      b2 = cos*c[i2] - sin*c[i1];
      c[i1] = b1; c[i2] = b2;
   }
}

//______________________________________________________________________________
void TGLMatrix::RotatePF(Int_t i1, Int_t i2, Double_t amount)
{
   // Rotate in parent frame. Does optimised version of MultLeft.

   if(i1 == i2) return;

   // Optimized version:
   const Double_t cos = TMath::Cos(amount), sin = TMath::Sin(amount);
   Double_t  b1, b2;
   Double_t* C = fVals;
   --i1; --i2;
   for(int c=0; c<4; ++c, C+=4) {
      b1 = cos*C[i1] - sin*C[i2];
      b2 = cos*C[i2] + sin*C[i1];
      C[i1] = b1; C[i2] = b2;
   }
}

//______________________________________________________________________________
void TGLMatrix::TransformVertex(TGLVertex3 & vertex) const
{
   // Transform passed 'vertex' by this matrix - converts local frame to parent
   TGLVertex3 orig = vertex;
   for (UInt_t i = 0; i < 3; i++) {
      vertex[i] = orig[0] * fVals[0+i] + orig[1] * fVals[4+i] +
                  orig[2] * fVals[8+i] + fVals[12+i];
   }
}

//______________________________________________________________________________
void TGLMatrix::Transpose3x3()
{
   // Transpose the top left 3x3 matrix component along major diagonal
   // Supported as currently incompatability between TGeo and GL matrix
   // layouts for this 3x3 only. To be resolved.

   // TODO: Move this fix to the TBuffer3D filling side and remove
   //
   // 0  4  8 12
   // 1  5  9 13
   // 2  6 10 14
   // 3  7 11 15

   Double_t temp = fVals[4];
   fVals[4] = fVals[1];
   fVals[1] = temp;
   temp = fVals[8];
   fVals[8] = fVals[2];
   fVals[2] = temp;
   temp = fVals[9];
   fVals[9] = fVals[6];
   fVals[6] = temp;
}

//______________________________________________________________________________
Double_t TGLMatrix::Invert()
{
   // Invert the matrix, returns determinant.
   // Copied from TMatrixFCramerInv.

   Double_t* M = fVals;

   const Double_t det2_12_01 = M[1]*M[6]  - M[5]*M[2];
   const Double_t det2_12_02 = M[1]*M[10] - M[9]*M[2];
   const Double_t det2_12_03 = M[1]*M[14] - M[13]*M[2];
   const Double_t det2_12_13 = M[5]*M[14] - M[13]*M[6];
   const Double_t det2_12_23 = M[9]*M[14] - M[13]*M[10];
   const Double_t det2_12_12 = M[5]*M[10] - M[9]*M[6];
   const Double_t det2_13_01 = M[1]*M[7]  - M[5]*M[3];
   const Double_t det2_13_02 = M[1]*M[11] - M[9]*M[3];
   const Double_t det2_13_03 = M[1]*M[15] - M[13]*M[3];
   const Double_t det2_13_12 = M[5]*M[11] - M[9]*M[7];
   const Double_t det2_13_13 = M[5]*M[15] - M[13]*M[7];
   const Double_t det2_13_23 = M[9]*M[15] - M[13]*M[11];
   const Double_t det2_23_01 = M[2]*M[7]  - M[6]*M[3];
   const Double_t det2_23_02 = M[2]*M[11] - M[10]*M[3];
   const Double_t det2_23_03 = M[2]*M[15] - M[14]*M[3];
   const Double_t det2_23_12 = M[6]*M[11] - M[10]*M[7];
   const Double_t det2_23_13 = M[6]*M[15] - M[14]*M[7];
   const Double_t det2_23_23 = M[10]*M[15] - M[14]*M[11];


   const Double_t det3_012_012 = M[0]*det2_12_12 - M[4]*det2_12_02 + M[8]*det2_12_01;
   const Double_t det3_012_013 = M[0]*det2_12_13 - M[4]*det2_12_03 + M[12]*det2_12_01;
   const Double_t det3_012_023 = M[0]*det2_12_23 - M[8]*det2_12_03 + M[12]*det2_12_02;
   const Double_t det3_012_123 = M[4]*det2_12_23 - M[8]*det2_12_13 + M[12]*det2_12_12;
   const Double_t det3_013_012 = M[0]*det2_13_12 - M[4]*det2_13_02 + M[8]*det2_13_01;
   const Double_t det3_013_013 = M[0]*det2_13_13 - M[4]*det2_13_03 + M[12]*det2_13_01;
   const Double_t det3_013_023 = M[0]*det2_13_23 - M[8]*det2_13_03 + M[12]*det2_13_02;
   const Double_t det3_013_123 = M[4]*det2_13_23 - M[8]*det2_13_13 + M[12]*det2_13_12;
   const Double_t det3_023_012 = M[0]*det2_23_12 - M[4]*det2_23_02 + M[8]*det2_23_01;
   const Double_t det3_023_013 = M[0]*det2_23_13 - M[4]*det2_23_03 + M[12]*det2_23_01;
   const Double_t det3_023_023 = M[0]*det2_23_23 - M[8]*det2_23_03 + M[12]*det2_23_02;
   const Double_t det3_023_123 = M[4]*det2_23_23 - M[8]*det2_23_13 + M[12]*det2_23_12;
   const Double_t det3_123_012 = M[1]*det2_23_12 - M[5]*det2_23_02 + M[9]*det2_23_01;
   const Double_t det3_123_013 = M[1]*det2_23_13 - M[5]*det2_23_03 + M[13]*det2_23_01;
   const Double_t det3_123_023 = M[1]*det2_23_23 - M[9]*det2_23_03 + M[13]*det2_23_02;
   const Double_t det3_123_123 = M[5]*det2_23_23 - M[9]*det2_23_13 + M[13]*det2_23_12;

   const Double_t det = M[0]*det3_123_123 - M[4]*det3_123_023 +
      M[8]*det3_123_013 - M[12]*det3_123_012;

   if(det == 0) {
      Warning("TGLMatrix::Invert", "matrix is singular.");
      return 0;
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   M[0]  = det3_123_123 * oneOverDet;
   M[4]  = det3_023_123 * mn1OverDet;
   M[8]  = det3_013_123 * oneOverDet;
   M[12] = det3_012_123 * mn1OverDet;

   M[1]  = det3_123_023 * mn1OverDet;
   M[5]  = det3_023_023 * oneOverDet;
   M[9]  = det3_013_023 * mn1OverDet;
   M[13] = det3_012_023 * oneOverDet;

   M[2]  = det3_123_013 * oneOverDet;
   M[6]  = det3_023_013 * mn1OverDet;
   M[10] = det3_013_013 * oneOverDet;
   M[14] = det3_012_013 * mn1OverDet;

   M[3]  = det3_123_012 * mn1OverDet;
   M[7]  = det3_023_012 * oneOverDet;
   M[11] = det3_013_012 * mn1OverDet;
   M[15] = det3_012_012 * oneOverDet;

   return det;
}

//______________________________________________________________________________
TGLVector3 TGLMatrix::Multiply(const TGLVector3& v, Double_t w) const
{
   // Multiply vector.
   const Double_t* M = fVals;
   TGLVector3 r;
   r.X() = M[0]*v[0] + M[4]*v[1] +  M[8]*v[2] + M[12]*w;
   r.Y() = M[1]*v[0] + M[5]*v[1] +  M[9]*v[2] + M[13]*w;
   r.Z() = M[2]*v[0] + M[6]*v[1] + M[10]*v[2] + M[14]*w;
   return r;
}

//______________________________________________________________________________
TGLVector3 TGLMatrix::Rotate(const TGLVector3& v) const
{
   // Rotate vector. Translation is not applied.
   const Double_t* M = fVals;
   TGLVector3 r;
   r.X() = M[0]*v[0] + M[4]*v[1] +  M[8]*v[2];
   r.Y() = M[1]*v[0] + M[5]*v[1] +  M[9]*v[2];
   r.Z() = M[2]*v[0] + M[6]*v[1] + M[10]*v[2];
   return r;
}

//______________________________________________________________________________
void TGLMatrix::MultiplyIP(TGLVector3& v, Double_t w) const
{
   // Multiply vector in-place.
   const Double_t* M = fVals;
   Double_t r[3] = { v[0], v[1], v[2] };
   v.X() = M[0]*r[0] + M[4]*r[1] +  M[8]*r[2] + M[12]*w;
   v.Y() = M[1]*r[0] + M[5]*r[1] +  M[9]*r[2] + M[13]*w;
   v.Z() = M[2]*r[0] + M[6]*r[1] + M[10]*r[2] + M[14]*w;
}

//______________________________________________________________________________
void TGLMatrix::RotateIP(TGLVector3& v) const
{
   // Rotate vector in-place. Translation is not applied.
   const Double_t* M = fVals;
   Double_t r[3] = { v[0], v[1], v[2] };
   v.X() = M[0]*r[0] + M[4]*r[1] +  M[8]*r[2];
   v.Y() = M[1]*r[0] + M[5]*r[1] +  M[9]*r[2];
   v.Z() = M[2]*r[0] + M[6]*r[1] + M[10]*r[2];
}

//______________________________________________________________________________
TGLVector3 TGLMatrix::GetScale() const
{
   // Get local axis scaling factors
   TGLVector3 x(fVals[0], fVals[1], fVals[2]);
   TGLVector3 y(fVals[4], fVals[5], fVals[6]);
   TGLVector3 z(fVals[8], fVals[9], fVals[10]);
   return TGLVector3(x.Mag(), y.Mag(), z.Mag());
}

//______________________________________________________________________________
Bool_t TGLMatrix::IsScalingForRender() const
{
   // Return true if matrix is to be considered a scaling matrix
   // for rendering.
   Double_t ss;
   ss = fVals[0]*fVals[0] + fVals[1]*fVals[1] + fVals[2]*fVals[2];
   if (ss < 0.8 || ss > 1.2) return kTRUE;
   ss = fVals[4]*fVals[4] + fVals[5]*fVals[5] + fVals[6]*fVals[6];
   if (ss < 0.8 || ss > 1.2) return kTRUE;
   ss = fVals[8]*fVals[8] + fVals[9]*fVals[9] + fVals[10]*fVals[10];
   if (ss < 0.8 || ss > 1.2) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TGLMatrix::Dump() const
{
   // Output 16 matrix components to std::cout
   //
   // 0  4   8  12
   // 1  5   9  13
   // 2  6  10  14
   // 3  7  11  15
   //
   std::cout.precision(6);
   for (Int_t x = 0; x < 4; x++) {
      std::cout << "[ ";
      for (Int_t y = 0; y < 4; y++) {
         std::cout << fVals[y*4 + x] << " ";
      }
      std::cout << "]" << std::endl;
   }
}


//==============================================================================
// TGLColor
//==============================================================================

//______________________________________________________________________________
//
// Class encapsulating color information in preferred GL format - an
// array of four unsigned bytes.
// Color index is also cached for easier interfacing with the
// traditional ROOT graphics.
//

ClassImp(TGLColor);

//______________________________________________________________________________
TGLColor::TGLColor()
{
   // Default constructor. Color is initialized to black.

   fRGBA[0] = fRGBA[1] = fRGBA[2] = 0;
   fRGBA[3] = 255;
   fIndex   = -1;
}

//______________________________________________________________________________
TGLColor::TGLColor(Int_t r, Int_t g, Int_t b, Int_t a)
{
   // Constructor from Int_t values.

   SetColor(r, g, b, a);
}

//______________________________________________________________________________
TGLColor::TGLColor(Float_t r, Float_t g, Float_t b, Float_t a)
{
   // Constructor from Float_t values.

   SetColor(r, g, b, a);
}

//______________________________________________________________________________
TGLColor::TGLColor(Color_t color_index, Char_t transparency)
{
   // Constructor from color-index and transparency.

   SetColor(color_index, transparency);
}

//______________________________________________________________________________
TGLColor::~TGLColor()
{
   // Dectructor.
}

//______________________________________________________________________________
TGLColor& TGLColor::operator=(const TGLColor& c)
{
   // Assignment operator.

   fRGBA[0] = c.fRGBA[0];
   fRGBA[1] = c.fRGBA[1];
   fRGBA[2] = c.fRGBA[2];
   fRGBA[3] = c.fRGBA[3];
   fIndex   = c.fIndex;
   return *this;
}

//______________________________________________________________________________
Color_t TGLColor::GetColorIndex() const
{
   // Returns color-index representing the color.

   if (fIndex == -1)
      fIndex = TColor::GetColor(fRGBA[0], fRGBA[1], fRGBA[2]);
   return fIndex;
}

//______________________________________________________________________________
Char_t TGLColor::GetTransparency() const
{
   // Returns transparecy value.

   return TMath::Nint(100.0*(1.0 - fRGBA[3]/255.0));
}

//______________________________________________________________________________
void TGLColor::SetColor(Int_t r, Int_t g, Int_t b, Int_t a)
{
   // Set color with Int_t values.

   fRGBA[0] = r;
   fRGBA[1] = g;
   fRGBA[2] = b;
   fRGBA[3] = a;
   fIndex   = -1;
}

//______________________________________________________________________________
void TGLColor::SetColor(Float_t r, Float_t g, Float_t b, Float_t a)
{
   // Set color with Float_t values.

   fRGBA[0] = (UChar_t)(255*r);
   fRGBA[1] = (UChar_t)(255*g);
   fRGBA[2] = (UChar_t)(255*b);
   fRGBA[3] = (UChar_t)(255*a);
   fIndex   = -1;
}

//______________________________________________________________________________
void TGLColor::SetColor(Color_t color_index)
{
   // Set color by color-index. Alpha is not changed.
   // If color_index is not valid, color is set to magenta.

   TColor* c = gROOT->GetColor(color_index);
   if (c)
   {
      fRGBA[0] = (UChar_t)(255*c->GetRed());
      fRGBA[1] = (UChar_t)(255*c->GetGreen());
      fRGBA[2] = (UChar_t)(255*c->GetBlue());
      fIndex   = color_index;
   }
   else
   {
      // Set to magenta.
      fRGBA[0] = 255;
      fRGBA[1] = 0;
      fRGBA[2] = 255;
      fIndex   = -1;
   }
}

//______________________________________________________________________________
void TGLColor::SetColor(Color_t color_index, Char_t transparency)
{
   // Set color by color-index and alpha from the transparency.
   // If color_index is not valid, color is set to magenta.

   UChar_t alpha = (255*(100 - transparency))/100;
   
   TColor* c = gROOT->GetColor(color_index);
   if (c)
   {
      fRGBA[0] = (UChar_t)(255*c->GetRed());
      fRGBA[1] = (UChar_t)(255*c->GetGreen());
      fRGBA[2] = (UChar_t)(255*c->GetBlue());
      fRGBA[3] = alpha;
      fIndex   = color_index;
   }
   else
   {
      // Set to magenta.
      fRGBA[0] = 255;
      fRGBA[1] = 0;
      fRGBA[2] = 255;
      fRGBA[3] = alpha;
      fIndex   = -1;
      return;
   }
}

//______________________________________________________________________________
void TGLColor::SetTransparency(Char_t transparency)
{
   // Set alpha from the transparency.

   fRGBA[3] = (255*(100 - transparency))/100;
}

//______________________________________________________________________________
TString TGLColor::AsString() const
{
   // Return string describing the color.

   return TString::Format("rgba:%02hhx/%02hhx/%02hhx/%02hhx",
                          fRGBA[0], fRGBA[1], fRGBA[2], fRGBA[3]);
}


//==============================================================================
// TGLColorSet
//==============================================================================

//______________________________________________________________________________
//
// Class encapsulating a set of colors used throughout standard rendering.
//

ClassImp(TGLColorSet);

//______________________________________________________________________________
TGLColorSet::TGLColorSet()
{
   // Constructor. Sets default for dark background.

   StdDarkBackground();
}

//______________________________________________________________________________
TGLColorSet::~TGLColorSet()
{
   // Destructor.
}

//______________________________________________________________________________
TGLColorSet& TGLColorSet::operator=(const TGLColorSet& s)
{
   // Assignment operator.

   fBackground = s.fBackground;
   fForeground = s.fForeground;
   fOutline    = s.fOutline;
   fMarkup     = s.fMarkup;
   for (Int_t i = 0; i < 5; ++i)
      fSelection[i] = s.fSelection[i];
   return *this;
}

//______________________________________________________________________________
void TGLColorSet::StdDarkBackground()
{
   // Set defaults for dark (black) background.

   fBackground .SetColor(0,   0,   0);
   fForeground .SetColor(255, 255, 255);
   fOutline    .SetColor(240, 255, 240);
   fMarkup     .SetColor(200, 200, 200);

   fSelection[0].SetColor(  0,   0,   0);
   fSelection[1].SetColor(255, 220, 220);
   fSelection[2].SetColor(255, 220, 220);
   fSelection[3].SetColor(200, 200, 255);
   fSelection[4].SetColor(200, 200, 255);
}

//______________________________________________________________________________
void TGLColorSet::StdLightBackground()
{
   // Set defaults for light (white) background.

   fBackground .SetColor(255, 255, 255);
   fForeground .SetColor(0,   0,   0);
   fOutline    .SetColor(0,   0,   0);
   fMarkup     .SetColor(55,  55,  55);

   fSelection[0].SetColor(0,   0,   0);
   fSelection[1].SetColor(200, 100, 100);
   fSelection[2].SetColor(200, 100, 100);
   fSelection[3].SetColor(100, 100, 200);
   fSelection[4].SetColor(100, 100, 200);
}


//==============================================================================
// TGLUtil
//==============================================================================

//______________________________________________________________________________
//
// Wrapper class for various misc static functions - error checking,
// draw helpers etc.
//

ClassImp(TGLUtil);

UInt_t TGLUtil::fgDefaultDrawQuality = 10;
UInt_t TGLUtil::fgDrawQuality        = fgDefaultDrawQuality;
UInt_t TGLUtil::fgColorLockCount     = 0;

Float_t TGLUtil::fgPointSize      = 1.0f;
Float_t TGLUtil::fgLineWidth      = 1.0f;
Float_t TGLUtil::fgPointSizeScale = 1.0f;
Float_t TGLUtil::fgLineWidthScale = 1.0f;

Float_t TGLUtil::fgScreenScalingFactor     = 1.0f;
Float_t TGLUtil::fgPointLineScalingFactor  = 1.0f;
Int_t   TGLUtil::fgPickingRadius           = 1;

const UChar_t TGLUtil::fgRed[4]    = { 230,   0,   0, 255 };
const UChar_t TGLUtil::fgGreen[4]  = {   0, 230,   0, 255 };
const UChar_t TGLUtil::fgBlue[4]   = {   0,   0, 230, 255 };
const UChar_t TGLUtil::fgYellow[4] = { 210, 210,   0, 255 };
const UChar_t TGLUtil::fgWhite[4]  = { 255, 255, 255, 255 };
const UChar_t TGLUtil::fgGrey[4]   = { 128, 128, 128, 100 };

#ifndef CALLBACK
#define CALLBACK
#endif

extern "C"
{
#if defined(__APPLE_CC__) && __APPLE_CC__ > 4000 && __APPLE_CC__ < 5450 && !defined(__INTEL_COMPILER)
    typedef GLvoid (*tessfuncptr_t)(...);
#elif defined(__mips) || defined(__linux__) || defined(__FreeBSD__) || defined( __OpenBSD__ ) || defined(__sun) || defined (__CYGWIN__) || defined (__APPLE__)
    typedef GLvoid (*tessfuncptr_t)();
#elif defined (WIN32)
    typedef GLvoid (CALLBACK *tessfuncptr_t)();
#else
    #error "Error - need to define type tessfuncptr_t for this platform/compiler"
#endif
}

namespace
{

class TGLTesselatorWrap
{
protected:

public:
   GLUtesselator *fTess;

   TGLTesselatorWrap(tessfuncptr_t vertex_func) : fTess(0)
   {
      fTess = gluNewTess();
      if (!fTess)
         throw std::bad_alloc();

      gluTessCallback(fTess, (GLenum)GLU_BEGIN,  (tessfuncptr_t) glBegin);
      gluTessCallback(fTess, (GLenum)GLU_END,    (tessfuncptr_t) glEnd);
      gluTessCallback(fTess, (GLenum)GLU_VERTEX, vertex_func);
   }

   virtual ~TGLTesselatorWrap()
   {
      if (fTess)
         gluDeleteTess(fTess);
   }
};

}

//______________________________________________________________________________
GLUtesselator* TGLUtil::GetDrawTesselator3fv()
{
   // Returns a tesselator for direct drawing when using 3-vertices with
   // single precision.

   static TGLTesselatorWrap singleton((tessfuncptr_t) glVertex3fv);

   return singleton.fTess;
}

//______________________________________________________________________________
GLUtesselator* TGLUtil::GetDrawTesselator4fv()
{
   // Returns a tesselator for direct drawing when using 4-vertices with
   // single precision.

   static TGLTesselatorWrap singleton((tessfuncptr_t) glVertex4fv);

   return singleton.fTess;
}

//______________________________________________________________________________
GLUtesselator* TGLUtil::GetDrawTesselator3dv()
{
   // Returns a tesselator for direct drawing when using 3-vertices with
   // double precision.

   static TGLTesselatorWrap singleton((tessfuncptr_t) glVertex3dv);

   return singleton.fTess;
}

//______________________________________________________________________________
GLUtesselator* TGLUtil::GetDrawTesselator4dv()
{
   // Returns a tesselator for direct drawing when using 4-vertices with
   // double precision.

   static TGLTesselatorWrap singleton((tessfuncptr_t) glVertex4dv);

   return singleton.fTess;
}

//______________________________________________________________________________
void TGLUtil::InitializeIfNeeded()
{
   // Initialize globals that require other libraries to be initialized.
   // This is called from TGLWidget creation function.

   static Bool_t init_done = kFALSE;
   if (init_done) return;
   init_done = kTRUE;

   fgScreenScalingFactor = gVirtualX->GetOpenGLScalingFactor();

   if (strcmp(gEnv->GetValue("OpenGL.PointLineScalingFactor", "native"), "native") == 0)
   {
      fgPointLineScalingFactor = fgScreenScalingFactor;
   }
   else
   {
      fgPointLineScalingFactor = gEnv->GetValue("OpenGL.PointLineScalingFactor", 1.0);
   }

   fgPickingRadius = TMath::Nint(gEnv->GetValue("OpenGL.PickingRadius", 3.0) * TMath::Sqrt(fgScreenScalingFactor));
}

//______________________________________________________________________________
UInt_t TGLUtil::GetDrawQuality()
{
   //static: get draw quality
   return fgDrawQuality;
}

//______________________________________________________________________________
void TGLUtil::SetDrawQuality(UInt_t dq)
{
   //static: set draw quality
   fgDrawQuality = dq;
}

//______________________________________________________________________________
void TGLUtil::ResetDrawQuality()
{
   //static: reset draw quality
   fgDrawQuality = fgDefaultDrawQuality;
}

//______________________________________________________________________________
UInt_t TGLUtil::GetDefaultDrawQuality()
{
   //static: get default draw quality
   return fgDefaultDrawQuality;
}

//______________________________________________________________________________
void TGLUtil::SetDefaultDrawQuality(UInt_t dq)
{
   //static: set default draw quality
   fgDefaultDrawQuality = dq;
}

//______________________________________________________________________________
Int_t TGLUtil::CheckError(const char * loc)
{
   // Check current GL error state, outputing details via ROOT
   // Error method if one
   GLenum errCode;
   const GLubyte *errString;

   if ((errCode = glGetError()) != GL_NO_ERROR) {
      errString = gluErrorString(errCode);
      if (loc) {
         Error(loc, "GL Error %s", (const char *)errString);
      } else {
         Error("TGLUtil::CheckError", "GL Error %s", (const char *)errString);
      }
   }
   return errCode;
}

/******************************************************************************/
// Color wrapping functions
/******************************************************************************/

//______________________________________________________________________________
UInt_t TGLUtil::LockColor()
{
   // Prevent further color changes.

   return ++fgColorLockCount;
}

//______________________________________________________________________________
UInt_t TGLUtil::UnlockColor()
{
   // Allow color changes.

   if (fgColorLockCount)
      --fgColorLockCount;
   else
      Error("TGLUtil::UnlockColor", "fgColorLockCount already 0.");
   return fgColorLockCount;
}

//______________________________________________________________________________
Bool_t TGLUtil::IsColorLocked()
{
   // Returns true if color lockcount is greater than 0.

   return fgColorLockCount > 0;
}

//______________________________________________________________________________
void TGLUtil::Color(const TGLColor& color)
{
   // Set color from TGLColor.

   if (fgColorLockCount == 0) glColor4ubv(color.CArr());
}

//______________________________________________________________________________
void TGLUtil::ColorAlpha(const TGLColor& color, UChar_t alpha)
{
   // Set color from TGLColor and alpha value.

   if (fgColorLockCount == 0)
   {
      glColor4ub(color.GetRed(), color.GetGreen(), color.GetBlue(), alpha);
   }
}

//______________________________________________________________________________
void TGLUtil::ColorAlpha(const TGLColor& color, Float_t alpha)
{
   // Set color from TGLColor and alpha value.

   if (fgColorLockCount == 0)
   {
      glColor4ub(color.GetRed(), color.GetGreen(), color.GetBlue(), (UChar_t)(255*alpha));
   }
}

//______________________________________________________________________________
void TGLUtil::ColorAlpha(Color_t color_index, Float_t alpha)
{
   // Set color from color_index and GL-style alpha (default 1).

   if (fgColorLockCount == 0) {
      if (color_index < 0)
         color_index = 1;
      TColor* c = gROOT->GetColor(color_index);
      if (c)
         glColor4f(c->GetRed(), c->GetGreen(), c->GetBlue(), alpha);
   }
}

//______________________________________________________________________________
void TGLUtil::ColorTransparency(Color_t color_index, Char_t transparency)
{
   // Set color from color_index and ROOT-style transparency (default 0).

   if (fgColorLockCount == 0) {
      if (color_index < 0)
         color_index = 1;
      TColor* c = gROOT->GetColor(color_index);
      if (c)
         glColor4f(c->GetRed(), c->GetGreen(), c->GetBlue(), 1.0f - 0.01f*transparency);
   }
}

//______________________________________________________________________________
void TGLUtil::Color3ub(UChar_t r, UChar_t g, UChar_t b)
{
   // Wrapper for glColor3ub.
   if (fgColorLockCount == 0) glColor3ub(r, g, b);
}

//______________________________________________________________________________
void TGLUtil::Color4ub(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Wrapper for glColor4ub.
   if (fgColorLockCount == 0) glColor4ub(r, g, b, a);
}

//______________________________________________________________________________
void TGLUtil::Color3ubv(const UChar_t* rgb)
{
   // Wrapper for glColor3ubv.
   if (fgColorLockCount == 0) glColor3ubv(rgb);
}

//______________________________________________________________________________
void TGLUtil::Color4ubv(const UChar_t* rgba)
{
   // Wrapper for glColor4ubv.
   if (fgColorLockCount == 0) glColor4ubv(rgba);
}

//______________________________________________________________________________
void TGLUtil::Color3f(Float_t r, Float_t g, Float_t b)
{
   // Wrapper for glColor3f.
   if (fgColorLockCount == 0) glColor3f(r, g, b);
}

//______________________________________________________________________________
void TGLUtil::Color4f(Float_t r, Float_t g, Float_t b, Float_t a)
{
   // Wrapper for glColor4f.
   if (fgColorLockCount == 0) glColor4f(r, g, b, a);
}

//______________________________________________________________________________
void TGLUtil::Color3fv(const Float_t* rgb)
{
   // Wrapper for glColor3fv.
   if (fgColorLockCount == 0) glColor3fv(rgb);
}

//______________________________________________________________________________
void TGLUtil::Color4fv(const Float_t* rgba)
{
   // Wrapper for glColor4fv.
   if (fgColorLockCount == 0) glColor4fv(rgba);
}

/******************************************************************************/
// Coordinate conversion and extra scaling (needed for osx retina)
/******************************************************************************/

//______________________________________________________________________________
void TGLUtil::PointToViewport(Int_t& x, Int_t& y)
{
   // Convert from point/screen coordinates to GL viewport coordinates.

   if (fgScreenScalingFactor != 1.0)
   {
      x = TMath::Nint(x * fgScreenScalingFactor);
      y = TMath::Nint(y * fgScreenScalingFactor);
   }
}

//______________________________________________________________________________
void TGLUtil::PointToViewport(Int_t& x, Int_t& y, Int_t& w, Int_t& h)
{
   // Convert from point/screen coordinates to GL viewport coordinates.

   if (fgScreenScalingFactor != 1.0)
   {
      x = TMath::Nint(x * fgScreenScalingFactor);
      y = TMath::Nint(y * fgScreenScalingFactor);
      w = TMath::Nint(w * fgScreenScalingFactor);
      h = TMath::Nint(h * fgScreenScalingFactor);
   }
}

//______________________________________________________________________________
Float_t TGLUtil::GetScreenScalingFactor()
{
   // Returns scaling factor between screen points and GL viewport pixels.
   // This is what is returned by gVirtualX->GetOpenGLScalingFactor() but is
   // cached here to avoid a virtual function call as it is used quite often in
   // TGLPhysical shape when drawing the selection highlight.

   return fgScreenScalingFactor;
}

//______________________________________________________________________________
Float_t TGLUtil::GetPointLineScalingFactor()
{
   // Return extra scaling factor for points and lines.
   // By default this is set to the same value as ScreenScalingFactor to keep
   // the same appearance. To override use rootrc entry, e.g.:
   // OpenGL.PointLineScalingFactor: 1.0
   
   return fgPointLineScalingFactor;
}

//______________________________________________________________________________
Int_t TGLUtil::GetPickingRadius()
{
   // Returns picking radius.

   return fgPickingRadius;
}

/******************************************************************************/
// Control for scaling of point-size and line-width.
/******************************************************************************/

//______________________________________________________________________________
Float_t TGLUtil::GetPointSizeScale()
{
   // Get global point-size scale.

   return fgPointSizeScale;
}

//______________________________________________________________________________
void TGLUtil::SetPointSizeScale(Float_t scale)
{
   // Set global point-size scale.

   fgPointSizeScale = scale;
}

//______________________________________________________________________________
Float_t TGLUtil::GetLineWidthScale()
{
   // Returns global line-width scale.

   return fgLineWidthScale;
}

//______________________________________________________________________________
void TGLUtil::SetLineWidthScale(Float_t scale)
{
   // Set global line-width scale.

   fgLineWidthScale = scale;
}

//______________________________________________________________________________
void TGLUtil::PointSize(Float_t point_size)
{
   // Set the point-size, taking the global scaling into account.
   // Wrapper for glPointSize.

   fgPointSize = point_size * fgPointSizeScale * fgPointLineScalingFactor;
   glPointSize(fgPointSize);
}

//______________________________________________________________________________
void TGLUtil::LineWidth(Float_t line_width)
{
   // Set the line-width, taking the global scaling into account.
   // Wrapper for glLineWidth.

   fgLineWidth = line_width * fgLineWidthScale * fgPointLineScalingFactor;
   glLineWidth(fgLineWidth);
}

//______________________________________________________________________________
Float_t TGLUtil::PointSize()
{
   // Get the point-size, taking the global scaling into account.

   return fgPointSize;
}

//______________________________________________________________________________
Float_t TGLUtil::LineWidth()
{
   // Get the line-width, taking the global scaling into account.

   return fgLineWidth;
}

/******************************************************************************/
// Rendering of polymarkers and lines from logical-shapes.
/******************************************************************************/

void TGLUtil::BeginExtendPickRegion(Float_t scale)
{
   // Extend pick region for large point-sizes or line-widths.

   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   Float_t pm[16];
   glGetFloatv(GL_PROJECTION_MATRIX, pm);
   for (Int_t i=0; i<=12; i+=4) {
      pm[i] *= scale; pm[i+1] *= scale;
   }
   glLoadMatrixf(pm);
   glMatrixMode(GL_MODELVIEW);
}

void TGLUtil::EndExtendPickRegion()
{
   // End extension of the pick region.

   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
}

//______________________________________________________________________________
void TGLUtil::RenderPolyMarkers(const TAttMarker& marker, Char_t transp,
                                Float_t* p, Int_t n,
                                Int_t pick_radius, Bool_t selection,
                                Bool_t sec_selection)
{
   // Render polymarkers at points specified by p-array.
   // Supports point and cross-like styles.

   if (n == 0) return;

   glPushAttrib(GL_ENABLE_BIT | GL_POINT_BIT | GL_LINE_BIT);

   glDisable(GL_LIGHTING);
   TGLUtil::ColorTransparency(marker.GetMarkerColor(), transp);

   Int_t s = marker.GetMarkerStyle();
   if (s == 2 || s == 3 || s == 5 || s == 28)
      RenderCrosses(marker, p, n, sec_selection);
   else
      RenderPoints(marker, p, n, pick_radius, selection, sec_selection);

   glPopAttrib();
}

//______________________________________________________________________________
void TGLUtil::RenderPolyMarkers(const TAttMarker &marker, const std::vector<Double_t> &points,
                                Double_t dX, Double_t dY, Double_t dZ)
{
   // Render polymarkers at points specified by p-array.
   // Supports point and cross-like styles.
   // Color is set externally. Lighting is disabled externally.

   const Int_t s = marker.GetMarkerStyle();
   if (s == 2 || s == 3 || s == 5 || s == 28)
      RenderCrosses(marker, points, dX, dY, dZ);
   else
      RenderPoints(marker, points);
}

//______________________________________________________________________________
void TGLUtil::RenderPoints(const TAttMarker& marker,
                           Float_t* op, Int_t n,
                           Int_t pick_radius, Bool_t selection,
                           Bool_t sec_selection)
{
   // Render markers as circular or square points.
   // Color is never changed.

   Int_t   style = marker.GetMarkerStyle();
   Float_t size  = 5*marker.GetMarkerSize();
   if (style == 4 || style == 20 || style == 24)
   {
      glEnable(GL_POINT_SMOOTH);
      if (style == 4 || style == 24) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
      }
   }
   else
   {
      glDisable(GL_POINT_SMOOTH);
      if      (style == 1) size = 1;
      else if (style == 6) size = 2;
      else if (style == 7) size = 3;
   }
   TGLUtil::PointSize(size);

   // During selection extend picking region for large point-sizes.
   Bool_t changePM = selection && PointSize() > pick_radius;
   if (changePM)
      BeginExtendPickRegion((Float_t) pick_radius / PointSize());

   Float_t* p = op;
   if (sec_selection)
   {
      glPushName(0);
      for (Int_t i=0; i<n; ++i, p+=3)
      {
         glLoadName(i);
         glBegin(GL_POINTS);
         glVertex3fv(p);
         glEnd();
      }
      glPopName();
   }
   else
   {
      glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
      glVertexPointer(3, GL_FLOAT, 0, p);
      glEnableClientState(GL_VERTEX_ARRAY);
      { // Circumvent bug in ATI's linux drivers.
         Int_t nleft = n;
         Int_t ndone = 0;
         const Int_t maxChunk = 8192;
         while (nleft > maxChunk)
         {
            glDrawArrays(GL_POINTS, ndone, maxChunk);
            nleft -= maxChunk;
            ndone += maxChunk;
         }
         glDrawArrays(GL_POINTS, ndone, nleft);
      }
      glPopClientAttrib();
   }

   if (changePM)
      EndExtendPickRegion();
}

//______________________________________________________________________________
void TGLUtil::RenderPoints(const TAttMarker& marker, const std::vector<Double_t> &points)
{
   // Render markers as circular or square points.
   // Color is never changed.
   const Int_t style = marker.GetMarkerStyle();
   Float_t size = 5 * marker.GetMarkerSize();

   if (style == 4 || style == 20 || style == 24)
   {
      glEnable(GL_POINT_SMOOTH);
      if (style == 4 || style == 24) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
      }
   }
   else
   {
      glDisable(GL_POINT_SMOOTH);
      if      (style == 1) size = 1;
      else if (style == 6) size = 2;
      else if (style == 7) size = 3;
   }

   glPointSize(size);

   glVertexPointer(3, GL_DOUBLE, 0, &points[0]);
   glEnableClientState(GL_VERTEX_ARRAY);

   // Circumvent bug in ATI's linux drivers.
   Int_t nleft = points.size() / 3;
   Int_t ndone = 0;
   const Int_t maxChunk = 8192;
   while (nleft > maxChunk)
   {
      glDrawArrays(GL_POINTS, ndone, maxChunk);
      nleft -= maxChunk;
      ndone += maxChunk;
   }

   if (nleft > 0)
      glDrawArrays(GL_POINTS, ndone, nleft);

   glDisableClientState(GL_VERTEX_ARRAY);
   glPointSize(1.f);
}

//______________________________________________________________________________
void TGLUtil::RenderCrosses(const TAttMarker& marker,
                            Float_t* op, Int_t n,
                            Bool_t sec_selection)
{
   // Render markers as crosses.
   // Color is never changed.

   if (marker.GetMarkerStyle() == 28)
   {
      glEnable(GL_BLEND);
      glEnable(GL_LINE_SMOOTH);
      TGLUtil::LineWidth(2);
   }
   else
   {
      glDisable(GL_LINE_SMOOTH);
      TGLUtil::LineWidth(1);
   }

   // cross dim
   const Float_t d = 2*marker.GetMarkerSize();
   Float_t* p = op;
   if (sec_selection)
   {
      glPushName(0);
      for (Int_t i=0; i<n; ++i, p+=3)
      {
         glLoadName(i);
         glBegin(GL_LINES);
         glVertex3f(p[0]-d, p[1],   p[2]);   glVertex3f(p[0]+d, p[1],   p[2]);
         glVertex3f(p[0],   p[1]-d, p[2]);   glVertex3f(p[0],   p[1]+d, p[2]);
         glVertex3f(p[0],   p[1],   p[2]-d); glVertex3f(p[0],   p[1],   p[2]+d);
         glEnd();
      }
      glPopName();
   }
   else
   {
      glBegin(GL_LINES);
      for (Int_t i=0; i<n; ++i, p+=3)
      {
         glVertex3f(p[0]-d, p[1],   p[2]);   glVertex3f(p[0]+d, p[1],   p[2]);
         glVertex3f(p[0],   p[1]-d, p[2]);   glVertex3f(p[0],   p[1]+d, p[2]);
         glVertex3f(p[0],   p[1],   p[2]-d); glVertex3f(p[0],   p[1],   p[2]+d);
      }
      glEnd();
   }

   // Anti-flickering -- when crosses get too small they
   // appear / disappear randomly.
   {
      glDisable(GL_POINT_SMOOTH);
      TGLUtil::PointSize(1);

      glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
      glVertexPointer(3, GL_FLOAT, 0, op);
      glEnableClientState(GL_VERTEX_ARRAY);
      { // Circumvent bug in ATI's linux drivers.
         Int_t nleft = n;
         Int_t ndone = 0;
         const Int_t maxChunk = 8192;
         while (nleft > maxChunk)
         {
            glDrawArrays(GL_POINTS, ndone, maxChunk);
            nleft -= maxChunk;
            ndone += maxChunk;
         }
         glDrawArrays(GL_POINTS, ndone, nleft);
      }
      glPopClientAttrib();
   }
}

//______________________________________________________________________________
void TGLUtil::RenderCrosses(const TAttMarker& marker, const std::vector<Double_t> &points,
                            Double_t dX, Double_t dY, Double_t dZ)
{
   // Render markers as crosses.
   // Color is never changed.
   if (marker.GetMarkerStyle() == 28)
   {
      glEnable(GL_BLEND);
      glEnable(GL_LINE_SMOOTH);
      glLineWidth(2.f);
   }
   else
   {
      glDisable(GL_LINE_SMOOTH);
      glLineWidth(1.f);
   }

   typedef std::vector<Double_t>::size_type size_type;

   glBegin(GL_LINES);

   for (size_type i = 0; i < points.size(); i += 3) {
      const Double_t *p = &points[i];
      glVertex3f(p[0] - dX, p[1], p[2]); glVertex3f(p[0] + dX, p[1], p[2]);
      glVertex3f(p[0], p[1] - dY, p[2]); glVertex3f(p[0], p[1] + dY, p[2]);
      glVertex3f(p[0], p[1], p[2] - dZ); glVertex3f(p[0], p[1], p[2] + dZ);
   }

   glEnd();

   if (marker.GetMarkerStyle() == 28) {
      glDisable(GL_LINE_SMOOTH);
      glDisable(GL_BLEND);
      glLineWidth(1.f);
   }
}

//______________________________________________________________________________
void TGLUtil::RenderPolyLine(const TAttLine& aline, Char_t transp,
                             Float_t* p, Int_t n,
                             Int_t pick_radius, Bool_t selection)
{
   // Render poly-line as specified by the p-array.

   if (n == 0) return;

   BeginAttLine(aline, transp, pick_radius, selection);

   Float_t* tp = p;
   glBegin(GL_LINE_STRIP);
   for (Int_t i=0; i<n; ++i, tp+=3)
      glVertex3fv(tp);
   glEnd();

   EndAttLine(pick_radius, selection);
}

//______________________________________________________________________________
void TGLUtil::BeginAttLine(const TAttLine& aline, Char_t transp,
                           Int_t pick_radius, Bool_t selection)
{
   // Setup drawing parrameters according to passed TAttLine.

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);

   glDisable(GL_LIGHTING);
   TGLUtil::ColorTransparency(aline.GetLineColor(), transp);
   TGLUtil::LineWidth(aline.GetLineWidth());
   if (aline.GetLineStyle() > 1)
   {
      // Int_t    fac = 1;
      UShort_t pat = 0xffff;
      switch (aline.GetLineStyle()) {
         case 2:  pat = 0x3333; break;
         case 3:  pat = 0x5555; break;
         case 4:  pat = 0xf040; break;
         case 5:  pat = 0xf4f4; break;
         case 6:  pat = 0xf111; break;
         case 7:  pat = 0xf0f0; break;
         case 8:  pat = 0xff11; break;
         case 9:  pat = 0x3fff; break;
         case 10: pat = 0x08ff; /* fac = 2; */ break;
      }

      glLineStipple(1, pat);
      glEnable(GL_LINE_STIPPLE);
   }

   // During selection extend picking region for large line-widths.
   if (selection && TGLUtil::LineWidth() > pick_radius)
      BeginExtendPickRegion((Float_t) pick_radius / TGLUtil::LineWidth());
}

//______________________________________________________________________________
void TGLUtil::EndAttLine(Int_t pick_radius, Bool_t selection)
{
   // Restore previous line drawing state.

   if (selection && TGLUtil::LineWidth() > pick_radius)
     EndExtendPickRegion();

   glPopAttrib(); 
}

/******************************************************************************/
// Rendering atoms used by TGLViewer / TGScene.
/******************************************************************************/

//______________________________________________________________________________
void TGLUtil::SetDrawColors(const UChar_t rgbai[4])
{
   // Set basic draw colors from 4 component 'rgba'
   // Used by other TGLUtil drawing routines
   //
   // Sets basic (unlit) color - glColor
   // and also GL materials (see OpenGL docs) thus:
   //
   // diffuse  : rgba
   // ambient  : 0.0 0.0 0.0 1.0
   // specular : 0.6 0.6 0.6 1.0
   // emission : rgba/4.0
   // shininess: 60.0
   //
   // emission is set so objects with no lights (but lighting still enabled)
   // are partially visible


   // Util function to setup GL color for both unlit and lit material
   Float_t rgba[4]     = {rgbai[0]/255.f, rgbai[1]/255.f, rgbai[2]/255.f, rgbai[3]/255.f};
   Float_t ambient[4]  = {0.0, 0.0, 0.0, 1.0};
   Float_t specular[4] = {0.6, 0.6, 0.6, 1.0};
   Float_t emission[4] = {rgba[0]/4.f, rgba[1]/4.f, rgba[2]/4.f, rgba[3]};

   glColor4fv(rgba);
   glMaterialfv(GL_FRONT, GL_DIFFUSE, rgba);
   glMaterialfv(GL_FRONT, GL_AMBIENT, ambient);
   glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
   glMaterialfv(GL_FRONT, GL_EMISSION, emission);
   glMaterialf(GL_FRONT, GL_SHININESS, 60.0);
}

//______________________________________________________________________________
void TGLUtil::DrawSphere(const TGLVertex3 & position, Double_t radius,
                         const UChar_t rgba[4])
{
   // Draw sphere, centered on vertex 'position', with radius 'radius',
   // color 'rgba'
   static TGLQuadric quad;
   SetDrawColors(rgba);
   glPushMatrix();
   glTranslated(position.X(), position.Y(), position.Z());
   gluSphere(quad.Get(), radius, fgDrawQuality, fgDrawQuality);
   glPopMatrix();
}

//______________________________________________________________________________
void TGLUtil::DrawLine(const TGLLine3 & line, ELineHeadShape head, Double_t size,
                       const UChar_t rgba[4])
{
   // Draw thick line (tube) defined by 'line', with head at end shape
   // 'head' - box/arrow/none, (head) size 'size', color 'rgba'
   DrawLine(line.Start(), line.Vector(), head, size, rgba);
}

//______________________________________________________________________________
void TGLUtil::DrawLine(const TGLVertex3 & start, const TGLVector3 & vector,
                       ELineHeadShape head, Double_t size, const UChar_t rgba[4])
{
   // Draw thick line (tube) running from 'start', length 'vector',
   // with head at end of shape 'head' - box/arrow/none,
   // (head) size 'size', color 'rgba'
   static TGLQuadric quad;

   // Draw 3D line (tube) with optional head shape
   SetDrawColors(rgba);
   glPushMatrix();
   TGLMatrix local(start, vector);
   glMultMatrixd(local.CArr());

   Double_t headHeight=0;
   if (head == kLineHeadNone) {
      headHeight = 0.0;
   } else if (head == kLineHeadArrow) {
      headHeight = size*2.0;
   } else if (head == kLineHeadBox) {
      headHeight = size*1.4;
   }

   // Line (tube) component
   gluCylinder(quad.Get(), 0.25*size, 0.25*size, vector.Mag() - headHeight, fgDrawQuality, 1);
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(quad.Get(), 0.0, 0.25*size, fgDrawQuality, 1);

   glTranslated(0.0, 0.0, vector.Mag() - headHeight); // Shift down local Z to end of line

   if (head == kLineHeadNone) {
      // Cap end of line
      gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
      gluDisk(quad.Get(), 0.0, size/4.0, fgDrawQuality, 1);
   }
   else if (head == kLineHeadArrow) {
      // Arrow base / end line cap
      gluDisk(quad.Get(), 0.0, size, fgDrawQuality, 1);
      // Arrow cone
      gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
      gluCylinder(quad.Get(), size, 0.0, headHeight, fgDrawQuality, 1);
   } else if (head == kLineHeadBox) {
      // Box
      // TODO: Drawing box should be simplier - maybe make
      // a static helper which BB + others use.
      // Single face tesselation - ugly lighting
      gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
      TGLBoundingBox box(TGLVertex3(-size*.7, -size*.7, 0.0),
                         TGLVertex3(size*.7, size*.7, headHeight));
      box.Draw(kTRUE);
   }
   glPopMatrix();
}

//______________________________________________________________________________
void TGLUtil::DrawRing(const TGLVertex3 & center, const TGLVector3 & normal,
                       Double_t radius, const UChar_t rgba[4])
{
   // Draw ring, centered on 'center', lying on plane defined by 'center' & 'normal'
   // of outer radius 'radius', color 'rgba'
   static TGLQuadric quad;

   // Draw a ring, round vertex 'center', lying on plane defined by 'normal' vector
   // Radius defines the outer radius
   TGLUtil::SetDrawColors(rgba);

   Double_t outer = radius;
   Double_t width = radius*0.05;
   Double_t inner = outer - width;

   // Shift into local system, looking down 'normal' vector, origin at center
   glPushMatrix();
   TGLMatrix local(center, normal);
   glMultMatrixd(local.CArr());

   // Shift half width so rings centered over center vertex
   glTranslated(0.0, 0.0, -width/2.0);

   // Inner and outer faces
   gluCylinder(quad.Get(), inner, inner, width, fgDrawQuality, 1);
   gluCylinder(quad.Get(), outer, outer, width, fgDrawQuality, 1);

   // Top/bottom
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(quad.Get(), inner, outer, fgDrawQuality, 1);
   glTranslated(0.0, 0.0, width);
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
   gluDisk(quad.Get(), inner, outer, fgDrawQuality, 1);

   glPopMatrix();
}

/**************************************************************************/

//______________________________________________________________________________
void TGLUtil::DrawReferenceMarker(const TGLCamera  & camera,
                                  const TGLVertex3 & pos,
                                        Float_t      radius,
                                  const UChar_t    * rgba)
{
   // Draw a sphere- marker on world-coordinate 'pos' with pixel
   // radius 'radius'. Color argument is optional.

   static const UChar_t defColor[4] = { 250, 110, 0, 255 }; // Orange

   radius = camera.ViewportDeltaToWorld(pos, radius, radius).Mag();
   DrawSphere(pos, radius, rgba ? rgba : defColor);

}

//______________________________________________________________________________
void TGLUtil::DrawSimpleAxes(const TGLCamera      & camera,
                             const TGLBoundingBox & bbox,
                                   Int_t            axesType)
{
   // Draw simple xyz-axes for given bounding-box.

   if (axesType == kAxesNone)
      return;

   static const UChar_t axesColors[][4] = {
      {128,   0,   0, 255},  // -ive X axis light red
      {255,   0,   0, 255},  // +ive X axis deep red
      {  0, 128,   0, 255},  // -ive Y axis light green
      {  0, 255,   0, 255},  // +ive Y axis deep green
      {  0,   0, 128, 255},  // -ive Z axis light blue
      {  0,   0, 255, 255}   // +ive Z axis deep blue
   };

   static const UChar_t xyz[][8] = {
      {0x44, 0x44, 0x28, 0x10, 0x10, 0x28, 0x44, 0x44},
      {0x10, 0x10, 0x10, 0x10, 0x10, 0x28, 0x44, 0x44},
      {0x7c, 0x20, 0x10, 0x10, 0x08, 0x08, 0x04, 0x7c}
   };

   // Axes draw at fixed screen size - back project to world
   TGLVector3 pixelVector = camera.ViewportDeltaToWorld(bbox.Center(), 1, 1);
   Double_t   pixelSize   = pixelVector.Mag();

   // Find x/y/z min/max values
   Double_t min[3] = { bbox.XMin(), bbox.YMin(), bbox.ZMin() };
   Double_t max[3] = { bbox.XMax(), bbox.YMax(), bbox.ZMax() };

   for (UInt_t i = 0; i < 3; i++) {
      TGLVertex3 start;
      TGLVector3 vector;

      if (axesType == kAxesOrigin) {
         // Through origin axes
         start[(i+1)%3] = 0.0;
         start[(i+2)%3] = 0.0;
      } else {
         // Side axes
         start[(i+1)%3] = min[(i+1)%3];
         start[(i+2)%3] = min[(i+2)%3];
      }
      vector[(i+1)%3] = 0.0;
      vector[(i+2)%3] = 0.0;

      // -ive axis?
      if (min[i] < 0.0) {
         // Runs from origin?
         if (max[i] > 0.0) {
            start[i] = 0.0;
            vector[i] = min[i];
         } else {
            start[i] = max[i];
            vector[i] = min[i] - max[i];
         }
         DrawLine(start, vector, kLineHeadNone, pixelSize*2.5, axesColors[i*2]);
      }
      // +ive axis?
      if (max[i] > 0.0) {
         // Runs from origin?
         if (min[i] < 0.0) {
            start[i] = 0.0;
            vector[i] = max[i];
         } else {
            start[i] = min[i];
            vector[i] = max[i] - min[i];
         }
         DrawLine(start, vector, kLineHeadNone, pixelSize*2.5, axesColors[i*2 + 1]);
      }
   }

   // Draw origin sphere(s)
   if (axesType == kAxesOrigin) {
      // Single white origin sphere at 0, 0, 0
      DrawSphere(TGLVertex3(0.0, 0.0, 0.0), pixelSize*2.0, fgWhite);
   } else {
      for (UInt_t j = 0; j < 3; j++) {
         if (min[j] <= 0.0 && max[j] >= 0.0) {
            TGLVertex3 zero;
            zero[j] = 0.0;
            zero[(j+1)%3] = min[(j+1)%3];
            zero[(j+2)%3] = min[(j+2)%3];
            DrawSphere(zero, pixelSize*2.0, axesColors[j*2 + 1]);
         }
      }
   }

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   // Labels
   Double_t padPixels = 25.0;

   glDisable(GL_LIGHTING);
   for (UInt_t k = 0; k < 3; k++) {
      SetDrawColors(axesColors[k*2+1]);
      TGLVertex3 minPos, maxPos;
      if (axesType == kAxesOrigin) {
         minPos[(k+1)%3] = 0.0;
         minPos[(k+2)%3] = 0.0;
      } else {
         minPos[(k+1)%3] = min[(k+1)%3];
         minPos[(k+2)%3] = min[(k+2)%3];
      }
      maxPos = minPos;
      minPos[k] = min[k];
      maxPos[k] = max[k];

      TGLVector3 axis = maxPos - minPos;
      TGLVector3 axisViewport = camera.WorldDeltaToViewport(minPos, axis);

      // Skip drawning if viewport projection of axis very small - labels will overlap
      // Occurs with orthographic cameras
      if (axisViewport.Mag() < 1) {
         continue;
      }

      minPos -= camera.ViewportDeltaToWorld(minPos, padPixels*axisViewport.X()/axisViewport.Mag(),
                                                    padPixels*axisViewport.Y()/axisViewport.Mag());
      axisViewport = camera.WorldDeltaToViewport(maxPos, -axis);
      maxPos -= camera.ViewportDeltaToWorld(maxPos, padPixels*axisViewport.X()/axisViewport.Mag(),
                                                    padPixels*axisViewport.Y()/axisViewport.Mag());

      DrawNumber(Form("%.0f", min[k]), minPos, kTRUE); // Min value
      DrawNumber(Form("%.0f", max[k]), maxPos, kTRUE); // Max value

      // Axis name beside max value
      TGLVertex3 namePos = maxPos -
         camera.ViewportDeltaToWorld(maxPos, padPixels*axisViewport.X()/axisViewport.Mag(),
                                     padPixels*axisViewport.Y()/axisViewport.Mag());
      glRasterPos3dv(namePos.CArr());
      glBitmap(8, 8, 0.0, 4.0, 0.0, 0.0, xyz[k]); // Axis Name
   }
}

//______________________________________________________________________________
void TGLUtil::DrawNumber(const TString    & num,
                         const TGLVertex3 & pos,
                               Bool_t       center)
{
   // Draw number in string 'num' via internal 8x8-pixel bitmap on
   // vertex 'pos'. If 'center' is true, the number is centered on 'pos'.
   // Only numbers, '.', '-' and ' ' are supported.

   static const UChar_t digits[][8] = {
      {0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38},//0
      {0x10, 0x10, 0x10, 0x10, 0x10, 0x70, 0x10, 0x10},//1
      {0x7c, 0x44, 0x20, 0x18, 0x04, 0x04, 0x44, 0x38},//2
      {0x38, 0x44, 0x04, 0x04, 0x18, 0x04, 0x44, 0x38},//3
      {0x04, 0x04, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x44},//4
      {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x40, 0x40, 0x7c},//5
      {0x7c, 0x44, 0x44, 0x44, 0x7c, 0x40, 0x40, 0x7c},//6
      {0x20, 0x20, 0x20, 0x10, 0x08, 0x04, 0x44, 0x7c},//7
      {0x38, 0x44, 0x44, 0x44, 0x38, 0x44, 0x44, 0x38},//8
      {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x7c},//9
      {0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},//.
      {0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00},//-
      {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00} //space
   };

   Double_t xOffset = 0, yOffset = 0;
   if (center)
   {
      xOffset = 3.5 * num.Length();
      yOffset = 4.0;
   }

   glRasterPos3dv(pos.CArr());
   for (Ssiz_t i = 0, e = num.Length(); i < e; ++i) {
      if (num[i] == '.') {
         glBitmap(8, 8, xOffset, yOffset, 7.0, 0.0, digits[10]);
      } else if (num[i] == '-') {
         glBitmap(8, 8, xOffset, yOffset, 7.0, 0.0, digits[11]);
      } else if (num[i] == ' ') {
         glBitmap(8, 8, xOffset, yOffset, 7.0, 0.0, digits[12]);
      } else if (num[i] >= '0' && num[i] <= '9') {
         glBitmap(8, 8, xOffset, yOffset, 7.0, 0.0, digits[num[i] - '0']);
      }
   }
}


/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________________
TGLCapabilitySwitch::TGLCapabilitySwitch(Int_t what, Bool_t state) :
   fWhat(what)
{
   // Constructor - change state only if necessary.

   fState = glIsEnabled(fWhat);
   fFlip  = (fState != state);
   if (fFlip)
      SetState(state);
}

//______________________________________________________________________________
TGLCapabilitySwitch::~TGLCapabilitySwitch()
{
   // Destructor - reset state if changed.

   if (fFlip)
      SetState(fState);
}

//______________________________________________________________________________
void TGLCapabilitySwitch::SetState(Bool_t s)
{
   if (s)
      glEnable(fWhat);
   else
      glDisable(fWhat);
}


//______________________________________________________________________________
TGLCapabilityEnabler::TGLCapabilityEnabler(Int_t what, Bool_t state) :
   fWhat(what)
{
   // Constructor - change state only if necessary.

   fFlip = ! glIsEnabled(fWhat) && state;
   if (fFlip)
      glEnable(fWhat);
}

//______________________________________________________________________________
TGLCapabilityEnabler::~TGLCapabilityEnabler()
{
   // Destructor - reset state if changed.

   if (fFlip)
      glDisable(fWhat);
}


//______________________________________________________________________________
TGLFloatHolder::TGLFloatHolder(Int_t what, Float_t state, void (*foo)(Float_t)) :
      fWhat(what), fState(0), fFlip(kFALSE), fFoo(foo)
   {
      glGetFloatv(fWhat, &fState);
      fFlip = (fState != state);
      if (fFlip) fFoo(state);
   }

//______________________________________________________________________________
TGLFloatHolder::~TGLFloatHolder()
   {
      if (fFlip) fFoo(fState);
   }


//______________________________________________________________________________
TGLEnableGuard::TGLEnableGuard(Int_t cap)
                  : fCap(cap)
{
   // TGLEnableGuard constructor.
   glEnable(GLenum(fCap));
}

//______________________________________________________________________________
TGLEnableGuard::~TGLEnableGuard()
{
   // TGLEnableGuard destructor.
   glDisable(GLenum(fCap));
}

//______________________________________________________________________________
TGLDisableGuard::TGLDisableGuard(Int_t cap)
                  : fCap(cap)
{
   // TGLDisableGuard constructor.
   glDisable(GLenum(fCap));
}

//______________________________________________________________________________
TGLDisableGuard::~TGLDisableGuard()
{
   // TGLDisableGuard destructor.
   glEnable(GLenum(fCap));
}

ClassImp(TGLSelectionBuffer);

//______________________________________________________________________________
TGLSelectionBuffer::TGLSelectionBuffer()
                        : fWidth(0), fHeight(0)
{
   // TGLSelectionBuffer constructor.
}

//______________________________________________________________________________
TGLSelectionBuffer::~TGLSelectionBuffer()
{
   // TGLSelectionBuffer destructor.
}

//______________________________________________________________________________
void TGLSelectionBuffer::ReadColorBuffer(Int_t w, Int_t h)
{
   // Read color buffer.
   fWidth = w;
   fHeight = h;
   fBuffer.resize(w * h * 4);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, &fBuffer[0]);
}

//______________________________________________________________________________
void TGLSelectionBuffer::ReadColorBuffer(Int_t x, Int_t y, Int_t w, Int_t h)
{
   // Read color buffer.
   fWidth = w;
   fHeight = h;
   fBuffer.resize(w * h * 4);
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(x, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, &fBuffer[0]);
}

//______________________________________________________________________________
const UChar_t *TGLSelectionBuffer::GetPixelColor(Int_t px, Int_t py)const
{
   // Get pixel color.
   if (px < 0)
      px = 0;
   if (py < 0)
      py = 0;

   if (UInt_t(px * fWidth * 4 + py * 4) > fBuffer.size())
      return &fBuffer[0];

   return &fBuffer[px * fWidth * 4 + py * 4];
}

namespace Rgl {

const Float_t gRedEmission[]    = {1.f, 0.f,  0.f, 1.f};
const Float_t gGreenEmission[]  = {0.f, 1.f,  0.f, 1.f};
const Float_t gBlueEmission[]   = {0.f, 0.f,  1.f, 1.f};
const Float_t gOrangeEmission[] = {1.f, 0.4f, 0.f, 1.f};
const Float_t gWhiteEmission[]  = {1.f, 1.f,  1.f, 1.f};
const Float_t gGrayEmission[]   = {0.3f,0.3f, 0.3f,1.f};
const Float_t gNullEmission[]   = {0.f, 0.f,  0.f, 1.f};

namespace {
   struct RGB_t {
      Int_t fRGB[3];
   };

   RGB_t gColorTriplets[] = {{{255, 0, 0}},
                              {{0, 255, 0}},
                              {{0, 0, 255}},
                              {{255, 255, 0}},
                              {{255, 0, 255}},
                              {{0, 255, 255}},
                              {{255, 255, 255}}};

   Bool_t operator < (const RGB_t &lhs, const RGB_t &rhs)
   {
      if (lhs.fRGB[0] < rhs.fRGB[0])
         return kTRUE;
      else if (lhs.fRGB[0] > rhs.fRGB[0])
         return kFALSE;
      else if (lhs.fRGB[1] < rhs.fRGB[1])
         return kTRUE;
      else if (lhs.fRGB[1] > rhs.fRGB[1])
         return kFALSE;
      else if (lhs.fRGB[2] < rhs.fRGB[2])
         return kTRUE;

      return kFALSE;
   }

   typedef std::map<Int_t, RGB_t> ColorLookupTable_t;
   typedef ColorLookupTable_t::const_iterator CLTCI_t;

   ColorLookupTable_t gObjectIDToColor;

   typedef std::map<RGB_t, Int_t> ObjectLookupTable_t;
   typedef ObjectLookupTable_t::const_iterator OLTCI_t;

   ObjectLookupTable_t gColorToObjectID;
}
//______________________________________________________________________________
void ObjectIDToColor(Int_t objectID, Bool_t highColor)
{
   //Object id encoded as rgb triplet.
   if (!highColor)
      glColor3ub(objectID & 0xff, (objectID & 0xff00) >> 8, (objectID & 0xff0000) >> 16);
   else {
      if (!gObjectIDToColor.size()) {
      //Initialize lookup tables.
         for (Int_t i = 0, id = 1; i < Int_t(sizeof gColorTriplets / sizeof(RGB_t)); ++i, ++id)
            gObjectIDToColor[id] = gColorTriplets[i];
         for (Int_t i = 0, id = 1; i < Int_t(sizeof gColorTriplets / sizeof(RGB_t)); ++i, ++id)
            gColorToObjectID[gColorTriplets[i]] = id;
      }

      CLTCI_t it = gObjectIDToColor.find(objectID);

      if (it != gObjectIDToColor.end())
         glColor3ub(it->second.fRGB[0], it->second.fRGB[1], it->second.fRGB[2]);
      else {
         Error("ObjectIDToColor", "No color for such object ID: %d", objectID);
         glColor3ub(0, 0, 0);
      }
   }
}

//______________________________________________________________________________
Int_t ColorToObjectID(const UChar_t *pixel, Bool_t highColor)
{
   if (!highColor)
      return pixel[0] | (pixel[1] << 8) | (pixel[2] << 16);
   else {
      if (!gObjectIDToColor.size())
         return 0;

      RGB_t triplet = {{pixel[0], pixel[1], pixel[2]}};
      OLTCI_t it = gColorToObjectID.find(triplet);

      if (it != gColorToObjectID.end())
         return it->second;
      else
         return 0;
   }
}


//______________________________________________________________________________
void DrawQuadOutline(const TGLVertex3 &v1, const TGLVertex3 &v2,
                     const TGLVertex3 &v3, const TGLVertex3 &v4)
{
   //Draw quad outline.
   glBegin(GL_LINE_LOOP);
   glVertex3dv(v1.CArr());
   glVertex3dv(v2.CArr());
   glVertex3dv(v3.CArr());
   glVertex3dv(v4.CArr());
   glEnd();
}

//______________________________________________________________________________
void DrawQuadFilled(const TGLVertex3 &v0, const TGLVertex3 &v1, const TGLVertex3 &v2,
                     const TGLVertex3 &v3, const TGLVector3 &normal)
{
   //Draw quad face.
   glBegin(GL_POLYGON);
   glNormal3dv(normal.CArr());
   glVertex3dv(v0.CArr());
   glVertex3dv(v1.CArr());
   glVertex3dv(v2.CArr());
   glVertex3dv(v3.CArr());
   glEnd();
}

//______________________________________________________________________________
void DrawQuadFilled(const Double_t *v0, const Double_t *v1, const Double_t *v2, const Double_t *v3,
                    const Double_t *normal)
{
   //Draw quad face.
   glBegin(GL_QUADS);
   glNormal3dv(normal);
   glVertex3dv(v0);
   glVertex3dv(v1);
   glVertex3dv(v2);
   glVertex3dv(v3);
   glEnd();
}

//______________________________________________________________________________
void DrawSmoothFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                  const TGLVector3 &norm1, const TGLVector3 &norm2, const TGLVector3 &norm3)
{
   //Draws triangle face, each vertex has its own averaged normal
   glBegin(GL_POLYGON);
   glNormal3dv(norm1.CArr());
   glVertex3dv(v1.CArr());
   glNormal3dv(norm2.CArr());
   glVertex3dv(v2.CArr());
   glNormal3dv(norm3.CArr());
   glVertex3dv(v3.CArr());
   glEnd();
}

const Int_t    gBoxFrontQuads[][4] = {{0, 1, 2, 3}, {4, 0, 3, 5}, {4, 5, 6, 7}, {7, 6, 2, 1}};
const Double_t gBoxFrontNormals[][3] = {{-1., 0., 0.}, {0., -1., 0.}, {1., 0., 0.}, {0., 1., 0.}};
const Int_t    gBoxFrontPlanes[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

const Int_t    gBoxBackQuads[][4] = {{7, 1, 2, 6}, {4, 7, 6, 5}, {0, 4, 5, 3}, {0, 3, 2, 1}};
const Double_t gBoxBackNormals[][3] = {{0., -1., 0.}, {-1., 0., 0.}, {0., 1., 0.}, {1., 0., 0.}};
const Int_t    gBoxBackPlanes[][2] = {{0, 1}, {3, 0}, {2, 3}, {1, 2}};

//______________________________________________________________________________
void DrawBoxFront(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax,
                  Double_t zMin, Double_t zMax, Int_t fp)
{
   //Draws lego's bar as a 3d box
   if (zMax < zMin)
      std::swap(zMax, zMin);

   //Bottom is always drawn.
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., -1.);
   glVertex3d(xMax, yMin, zMin);
   glVertex3d(xMin, yMin, zMin);
   glVertex3d(xMin, yMax, zMin);
   glVertex3d(xMax, yMax, zMin);
   glEnd();
   //Draw two visible front planes.
   const Double_t box[][3] = {{xMin, yMin, zMax}, {xMin, yMax, zMax}, {xMin, yMax, zMin}, {xMin, yMin, zMin},
                              {xMax, yMin, zMax}, {xMax, yMin, zMin}, {xMax, yMax, zMin}, {xMax, yMax, zMax}};
   const Int_t *verts = gBoxFrontQuads[gBoxFrontPlanes[fp][0]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][0]]);
   glVertex3dv(box[verts[0]]);
   glVertex3dv(box[verts[1]]);
   glVertex3dv(box[verts[2]]);
   glVertex3dv(box[verts[3]]);
   glEnd();

   verts = gBoxFrontQuads[gBoxFrontPlanes[fp][1]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][1]]);
   glVertex3dv(box[verts[0]]);
   glVertex3dv(box[verts[1]]);
   glVertex3dv(box[verts[2]]);
   glVertex3dv(box[verts[3]]);
   glEnd();

   //Top is always drawn.
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glVertex3d(xMax, yMin, zMax);
   glVertex3d(xMax, yMax, zMax);
   glVertex3d(xMin, yMax, zMax);
   glVertex3d(xMin, yMin, zMax);
   glEnd();
}

//______________________________________________________________________________
void DrawTransparentBox(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax,
                        Double_t zMin, Double_t zMax, Int_t fp)
{
   //Draws lego's bar as a 3d box
   if (zMax < zMin)
      std::swap(zMax, zMin);

   //The order is: 1) two back planes, 2) bottom plane, 3) two front planes,
   //4) top.

   //Bottom is always drawn.
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., -1.);
   glVertex3d(xMax, yMin, zMin);
   glVertex3d(xMin, yMin, zMin);
   glVertex3d(xMin, yMax, zMin);
   glVertex3d(xMax, yMax, zMin);
   glEnd();

   const Double_t box[][3] = {{xMin, yMin, zMax}, {xMin, yMax, zMax}, {xMin, yMax, zMin}, {xMin, yMin, zMin},
                              {xMax, yMin, zMax}, {xMax, yMin, zMin}, {xMax, yMax, zMin}, {xMax, yMax, zMax}};

   //Draw two back planes.
   const Int_t *verts = gBoxBackQuads[gBoxBackPlanes[fp][0]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxBackNormals[gBoxBackPlanes[fp][0]]);
   glVertex3dv(box[verts[0]]);
   glVertex3dv(box[verts[1]]);
   glVertex3dv(box[verts[2]]);
   glVertex3dv(box[verts[3]]);
   glEnd();

   verts = gBoxBackQuads[gBoxBackPlanes[fp][1]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxBackNormals[gBoxBackPlanes[fp][1]]);
   glVertex3dv(box[verts[0]]);
   glVertex3dv(box[verts[1]]);
   glVertex3dv(box[verts[2]]);
   glVertex3dv(box[verts[3]]);
   glEnd();

   //Draw two visible front planes.
   verts = gBoxFrontQuads[gBoxFrontPlanes[fp][0]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][0]]);
   glVertex3dv(box[verts[0]]);
   glVertex3dv(box[verts[1]]);
   glVertex3dv(box[verts[2]]);
   glVertex3dv(box[verts[3]]);
   glEnd();

   verts = gBoxFrontQuads[gBoxFrontPlanes[fp][1]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][1]]);
   glVertex3dv(box[verts[0]]);
   glVertex3dv(box[verts[1]]);
   glVertex3dv(box[verts[2]]);
   glVertex3dv(box[verts[3]]);
   glEnd();

   //Top is always drawn.
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glVertex3d(xMax, yMin, zMax);
   glVertex3d(xMax, yMax, zMax);
   glVertex3d(xMin, yMax, zMax);
   glVertex3d(xMin, yMin, zMax);
   glEnd();
}

//______________________________________________________________________________
void DrawBoxFrontTextured(Double_t xMin, Double_t xMax, Double_t yMin,
                           Double_t yMax, Double_t zMin, Double_t zMax,
                           Double_t texMin, Double_t texMax, Int_t fp)
{
   //Draws lego's bar as a 3d box
   //LULULULU
   if (zMax < zMin) {
      std::swap(zMax, zMin);
      std::swap(texMax, texMin);
   }

   //Top and bottom are always drawn.
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glTexCoord1d(texMax);
   glVertex3d(xMax, yMin, zMax);
   glVertex3d(xMax, yMax, zMax);
   glVertex3d(xMin, yMax, zMax);
   glVertex3d(xMin, yMin, zMax);
   glEnd();

   glBegin(GL_POLYGON);
   glTexCoord1d(texMin);
   glNormal3d(0., 0., -1.);
   glVertex3d(xMax, yMin, zMin);
   glVertex3d(xMin, yMin, zMin);
   glVertex3d(xMin, yMax, zMin);
   glVertex3d(xMax, yMax, zMin);
   glEnd();
   //Draw two visible front planes.
   const Double_t box[][3] = {{xMin, yMin, zMax}, {xMin, yMax, zMax}, {xMin, yMax, zMin}, {xMin, yMin, zMin},
                              {xMax, yMin, zMax}, {xMax, yMin, zMin}, {xMax, yMax, zMin}, {xMax, yMax, zMax}};

   const Double_t tex[] = {texMax, texMax, texMin, texMin, texMax, texMin, texMin, texMax};
   const Int_t *verts = gBoxFrontQuads[gBoxFrontPlanes[fp][0]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][0]]);
   glTexCoord1d(tex[verts[0]]);
   glVertex3dv(box[verts[0]]);
   glTexCoord1d(tex[verts[1]]);
   glVertex3dv(box[verts[1]]);
   glTexCoord1d(tex[verts[2]]);
   glVertex3dv(box[verts[2]]);
   glTexCoord1d(tex[verts[3]]);
   glVertex3dv(box[verts[3]]);
   glEnd();

   verts = gBoxFrontQuads[gBoxFrontPlanes[fp][1]];

   glBegin(GL_POLYGON);
   glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][1]]);
   glTexCoord1d(tex[verts[0]]);
   glVertex3dv(box[verts[0]]);
   glTexCoord1d(tex[verts[1]]);
   glVertex3dv(box[verts[1]]);
   glTexCoord1d(tex[verts[2]]);
   glVertex3dv(box[verts[2]]);
   glTexCoord1d(tex[verts[3]]);
   glVertex3dv(box[verts[3]]);
   glEnd();
}


//______________________________________________________________________________
void DrawCylinder(TGLQuadric *quadric, Double_t xMin, Double_t xMax, Double_t yMin,
                  Double_t yMax, Double_t zMin, Double_t zMax)
{
   //Cylinder for lego3.
   GLUquadric *quad = quadric->Get();

   if (quad) {
      if (zMin > zMax)
         std::swap(zMin, zMax);
      const Double_t xCenter = xMin + (xMax - xMin) / 2;
      const Double_t yCenter = yMin + (yMax - yMin) / 2;
      const Double_t radius = TMath::Min((xMax - xMin) / 2, (yMax - yMin) / 2);

      glPushMatrix();
      glTranslated(xCenter, yCenter, zMin);
      gluCylinder(quad, radius, radius, zMax - zMin, 40, 1);
      glPopMatrix();
      glPushMatrix();
      glTranslated(xCenter, yCenter, zMax);
      gluDisk(quad, 0., radius, 40, 1);
      glPopMatrix();
      glPushMatrix();
      glTranslated(xCenter, yCenter, zMin);
      glRotated(180., 0., 1., 0.);
      gluDisk(quad, 0., radius, 40, 1);
      glPopMatrix();
   }
}

//______________________________________________________________________________
void DrawSphere(TGLQuadric *quadric, Double_t xMin, Double_t xMax, Double_t yMin,
                  Double_t yMax, Double_t zMin, Double_t zMax)
{
   //Cylinder for lego3.
   GLUquadric *quad = quadric->Get();

   if (quad) {
      const Double_t xCenter = xMin + (xMax - xMin) / 2;
      const Double_t yCenter = yMin + (yMax - yMin) / 2;
      const Double_t zCenter = zMin + (zMax - zMin) / 2;

      const Double_t radius = TMath::Min((zMax - zMin) / 2,
                                          TMath::Min((xMax - xMin) / 2, (yMax - yMin) / 2));

      glPushMatrix();
      glTranslated(xCenter, yCenter, zCenter);
      gluSphere(quad, radius, 10, 10);
      glPopMatrix();
   }
}


//______________________________________________________________________________
void DrawError(Double_t xMin, Double_t xMax, Double_t yMin,
               Double_t yMax, Double_t zMin, Double_t zMax)
{
   const Double_t xWid = xMax - xMin;
   const Double_t yWid = yMax - yMin;

   glBegin(GL_LINES);
   glVertex3d(xMin + xWid / 2, yMin + yWid / 2, zMin);
   glVertex3d(xMin + xWid / 2, yMin + yWid / 2, zMax);
   glEnd();

   glBegin(GL_LINES);
   glVertex3d(xMin + xWid / 2, yMin, zMin);
   glVertex3d(xMin + xWid / 2, yMax, zMin);
   glEnd();

   glBegin(GL_LINES);
   glVertex3d(xMin, yMin + yWid / 2, zMin);
   glVertex3d(xMax, yMin + yWid / 2, zMin);
   glEnd();
}

void CylindricalNormal(const Double_t *v, Double_t *normal)
{
   const Double_t n = TMath::Sqrt(v[0] * v[0] + v[1] * v[1]);
   if (n > 0.) {
      normal[0] = v[0] / n;
      normal[1] = v[1] / n;
      normal[2] = 0.;
   } else {
      normal[0] = v[0];
      normal[1] = v[1];
      normal[2] = 0.;
   }
}

void CylindricalNormalInv(const Double_t *v, Double_t *normal)
{
   const Double_t n = TMath::Sqrt(v[0] * v[0] + v[1] * v[1]);
   if (n > 0.) {
      normal[0] = -v[0] / n;
      normal[1] = -v[1] / n;
      normal[2] = 0.;
   } else {
      normal[0] = -v[0];
      normal[1] = -v[1];
      normal[2] = 0.;
   }
}

void DrawTrapezoid(const Double_t ver[][2], Double_t zMin, Double_t zMax, Bool_t color)
{
   //In polar coordinates, box became trapezoid.
   //Four faces need normal calculations.
   if (zMin > zMax)
      std::swap(zMin, zMax);
   //top
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glVertex3d(ver[0][0], ver[0][1], zMax);
   glVertex3d(ver[1][0], ver[1][1], zMax);
   glVertex3d(ver[2][0], ver[2][1], zMax);
   glVertex3d(ver[3][0], ver[3][1], zMax);
   glEnd();
   //bottom
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., -1.);
   glVertex3d(ver[0][0], ver[0][1], zMin);
   glVertex3d(ver[3][0], ver[3][1], zMin);
   glVertex3d(ver[2][0], ver[2][1], zMin);
   glVertex3d(ver[1][0], ver[1][1], zMin);
   glEnd();
   //

   Double_t trapezoid[][3] = {{ver[0][0], ver[0][1], zMin}, {ver[1][0], ver[1][1], zMin},
                              {ver[2][0], ver[2][1], zMin}, {ver[3][0], ver[3][1], zMin},
                              {ver[0][0], ver[0][1], zMax}, {ver[1][0], ver[1][1], zMax},
                              {ver[2][0], ver[2][1], zMax}, {ver[3][0], ver[3][1], zMax}};
   Double_t normal[3] = {0.};
   glBegin(GL_POLYGON);
   CylindricalNormal(trapezoid[1], normal), glNormal3dv(normal), glVertex3dv(trapezoid[1]);
   CylindricalNormal(trapezoid[2], normal), glNormal3dv(normal), glVertex3dv(trapezoid[2]);
   CylindricalNormal(trapezoid[6], normal), glNormal3dv(normal), glVertex3dv(trapezoid[6]);
   CylindricalNormal(trapezoid[5], normal), glNormal3dv(normal), glVertex3dv(trapezoid[5]);
   glEnd();

   glBegin(GL_POLYGON);
   CylindricalNormalInv(trapezoid[0], normal), glNormal3dv(normal), glVertex3dv(trapezoid[0]);
   CylindricalNormalInv(trapezoid[4], normal), glNormal3dv(normal), glVertex3dv(trapezoid[4]);
   CylindricalNormalInv(trapezoid[7], normal), glNormal3dv(normal), glVertex3dv(trapezoid[7]);
   CylindricalNormalInv(trapezoid[3], normal), glNormal3dv(normal), glVertex3dv(trapezoid[3]);
   glEnd();

   glBegin(GL_POLYGON);
   if (color) {
      TMath::Normal2Plane(trapezoid[0], trapezoid[1], trapezoid[5], normal);
      glNormal3dv(normal);
   }
   glVertex3dv(trapezoid[0]);
   glVertex3dv(trapezoid[1]);
   glVertex3dv(trapezoid[5]);
   glVertex3dv(trapezoid[4]);
   glEnd();

   glBegin(GL_POLYGON);
   if (color) {
      TMath::Normal2Plane(trapezoid[3], trapezoid[7], trapezoid[6], normal);
      glNormal3dv(normal);
   }
   glVertex3dv(trapezoid[3]);
   glVertex3dv(trapezoid[7]);
   glVertex3dv(trapezoid[6]);
   glVertex3dv(trapezoid[2]);
   glEnd();
}

//______________________________________________________________________________
void DrawTrapezoidTextured(const Double_t ver[][2], Double_t zMin, Double_t zMax,
                           Double_t texMin, Double_t texMax)
{
   //In polar coordinates, box became trapezoid.
   //Four faces need normal calculations.
   if (zMin > zMax) {
      std::swap(zMin, zMax);
      std::swap(texMin, texMax);
   }

   //top
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glTexCoord1d(texMax);
   glVertex3d(ver[0][0], ver[0][1], zMax);
   glVertex3d(ver[1][0], ver[1][1], zMax);
   glVertex3d(ver[2][0], ver[2][1], zMax);
   glVertex3d(ver[3][0], ver[3][1], zMax);
   glEnd();
   //bottom
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., -1.);
   glTexCoord1d(texMin);
   glVertex3d(ver[0][0], ver[0][1], zMin);
   glVertex3d(ver[3][0], ver[3][1], zMin);
   glVertex3d(ver[2][0], ver[2][1], zMin);
   glVertex3d(ver[1][0], ver[1][1], zMin);
   glEnd();
   //

   Double_t trapezoid[][3] = {{ver[0][0], ver[0][1], zMin}, {ver[1][0], ver[1][1], zMin},
                              {ver[2][0], ver[2][1], zMin}, {ver[3][0], ver[3][1], zMin},
                              {ver[0][0], ver[0][1], zMax}, {ver[1][0], ver[1][1], zMax},
                              {ver[2][0], ver[2][1], zMax}, {ver[3][0], ver[3][1], zMax}};
   Double_t normal[3] = {0.};
   glBegin(GL_POLYGON);
   CylindricalNormal(trapezoid[1], normal), glNormal3dv(normal), glTexCoord1d(texMin), glVertex3dv(trapezoid[1]);
   CylindricalNormal(trapezoid[2], normal), glNormal3dv(normal), glTexCoord1d(texMin), glVertex3dv(trapezoid[2]);
   CylindricalNormal(trapezoid[6], normal), glNormal3dv(normal), glTexCoord1d(texMax), glVertex3dv(trapezoid[6]);
   CylindricalNormal(trapezoid[5], normal), glNormal3dv(normal), glTexCoord1d(texMax), glVertex3dv(trapezoid[5]);
   glEnd();

   glBegin(GL_POLYGON);
   CylindricalNormalInv(trapezoid[0], normal), glNormal3dv(normal), glTexCoord1d(texMin), glVertex3dv(trapezoid[0]);
   CylindricalNormalInv(trapezoid[4], normal), glNormal3dv(normal), glTexCoord1d(texMax), glVertex3dv(trapezoid[4]);
   CylindricalNormalInv(trapezoid[7], normal), glNormal3dv(normal), glTexCoord1d(texMax), glVertex3dv(trapezoid[7]);
   CylindricalNormalInv(trapezoid[3], normal), glNormal3dv(normal), glTexCoord1d(texMin), glVertex3dv(trapezoid[3]);
   glEnd();

   glBegin(GL_POLYGON);
   TMath::Normal2Plane(trapezoid[0], trapezoid[1], trapezoid[5], normal);
   glNormal3dv(normal);
   glTexCoord1d(texMin);
   glVertex3dv(trapezoid[0]);
   glTexCoord1d(texMin);
   glVertex3dv(trapezoid[1]);
   glTexCoord1d(texMax);
   glVertex3dv(trapezoid[5]);
   glTexCoord1d(texMax);
   glVertex3dv(trapezoid[4]);
   glEnd();

   glBegin(GL_POLYGON);
   TMath::Normal2Plane(trapezoid[3], trapezoid[7], trapezoid[6], normal);
   glNormal3dv(normal);
   glTexCoord1d(texMin);
   glVertex3dv(trapezoid[3]);
   glTexCoord1d(texMax);
   glVertex3dv(trapezoid[7]);
   glTexCoord1d(texMax);
   glVertex3dv(trapezoid[6]);
   glTexCoord1d(texMin);
   glVertex3dv(trapezoid[2]);
   glEnd();
}

//______________________________________________________________________________
void DrawTrapezoidTextured2(const Double_t ver[][2], Double_t zMin, Double_t zMax,
                              Double_t texMin, Double_t texMax)
{
   //In polar coordinates, box became trapezoid.
   if (zMin > zMax)
      std::swap(zMin, zMax);

   const Double_t trapezoid[][3] = {{ver[0][0], ver[0][1], zMin}, {ver[1][0], ver[1][1], zMin},
                                    {ver[2][0], ver[2][1], zMin}, {ver[3][0], ver[3][1], zMin},
                                    {ver[0][0], ver[0][1], zMax}, {ver[1][0], ver[1][1], zMax},
                                    {ver[2][0], ver[2][1], zMax}, {ver[3][0], ver[3][1], zMax}};
   const Double_t tex[] = {texMin, texMax, texMax, texMin, texMin, texMax, texMax, texMin};
   //top
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glTexCoord1d(tex[4]), glVertex3dv(trapezoid[4]);
   glTexCoord1d(tex[5]), glVertex3dv(trapezoid[5]);
   glTexCoord1d(tex[6]), glVertex3dv(trapezoid[6]);
   glTexCoord1d(tex[7]), glVertex3dv(trapezoid[7]);
   glEnd();
   //bottom
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., -1.);
   glTexCoord1d(tex[0]), glVertex3dv(trapezoid[0]);
   glTexCoord1d(tex[3]), glVertex3dv(trapezoid[3]);
   glTexCoord1d(tex[2]), glVertex3dv(trapezoid[2]);
   glTexCoord1d(tex[1]), glVertex3dv(trapezoid[1]);
   glEnd();
   //
   glBegin(GL_POLYGON);
   Double_t normal[3] = {};
   CylindricalNormal(trapezoid[1], normal), glNormal3dv(normal), glTexCoord1d(tex[1]), glVertex3dv(trapezoid[1]);
   CylindricalNormal(trapezoid[2], normal), glNormal3dv(normal), glTexCoord1d(tex[2]), glVertex3dv(trapezoid[2]);
   CylindricalNormal(trapezoid[6], normal), glNormal3dv(normal), glTexCoord1d(tex[6]), glVertex3dv(trapezoid[6]);
   CylindricalNormal(trapezoid[5], normal), glNormal3dv(normal), glTexCoord1d(tex[5]), glVertex3dv(trapezoid[5]);
   glEnd();

   glBegin(GL_POLYGON);
   CylindricalNormalInv(trapezoid[0], normal), glNormal3dv(normal), glTexCoord1d(tex[0]), glVertex3dv(trapezoid[0]);
   CylindricalNormalInv(trapezoid[4], normal), glNormal3dv(normal), glTexCoord1d(tex[4]), glVertex3dv(trapezoid[4]);
   CylindricalNormalInv(trapezoid[7], normal), glNormal3dv(normal), glTexCoord1d(tex[7]), glVertex3dv(trapezoid[7]);
   CylindricalNormalInv(trapezoid[3], normal), glNormal3dv(normal), glTexCoord1d(tex[3]), glVertex3dv(trapezoid[3]);
   glEnd();

   glBegin(GL_POLYGON);
   TMath::Normal2Plane(trapezoid[0], trapezoid[1], trapezoid[5], normal);
   glNormal3dv(normal);
   glTexCoord1d(tex[0]), glVertex3dv(trapezoid[0]);
   glTexCoord1d(tex[1]), glVertex3dv(trapezoid[1]);
   glTexCoord1d(tex[5]), glVertex3dv(trapezoid[5]);
   glTexCoord1d(tex[4]), glVertex3dv(trapezoid[4]);
   glEnd();

   glBegin(GL_POLYGON);
   TMath::Normal2Plane(trapezoid[3], trapezoid[7], trapezoid[6], normal);
   glNormal3dv(normal);
   glTexCoord1d(tex[3]), glVertex3dv(trapezoid[3]);
   glTexCoord1d(tex[7]), glVertex3dv(trapezoid[7]);
   glTexCoord1d(tex[6]), glVertex3dv(trapezoid[6]);
   glTexCoord1d(tex[2]), glVertex3dv(trapezoid[2]);
   glEnd();
}

//______________________________________________________________________________
void SphericalNormal(const Double_t *v, Double_t *normal)
{
   const Double_t n = TMath::Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
   if (n > 0.) {
      normal[0] = v[0] / n;
      normal[1] = v[1] / n;
      normal[2] = v[2] / n;
   } else {
      normal[0] = v[0];
      normal[1] = v[1];
      normal[2] = v[2];
   }
}

//______________________________________________________________________________
void SphericalNormalInv(const Double_t *v, Double_t *normal)
{
   const Double_t n = TMath::Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
   if (n > 0.) {
      normal[0] = -v[0] / n;
      normal[1] = -v[1] / n;
      normal[2] = -v[2] / n;
   } else {
      normal[0] = -v[0];
      normal[1] = -v[1];
      normal[2] = -v[2];
   }
}

//______________________________________________________________________________
void DrawTrapezoid(const Double_t ver[][3])
{
   Double_t normal[3] = {0.};

   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[1], ver[2], ver[3], normal);
   glNormal3dv(normal);
   glVertex3dv(ver[0]);
   glVertex3dv(ver[1]);
   glVertex3dv(ver[2]);
   glVertex3dv(ver[3]);
   glEnd();
   //bottom
   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[4], ver[7], ver[6], normal);
   glNormal3dv(normal);
   glVertex3dv(ver[4]);
   glVertex3dv(ver[7]);
   glVertex3dv(ver[6]);
   glVertex3dv(ver[5]);
   glEnd();
   //

   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[0], ver[3], ver[7], normal);
   glNormal3dv(normal);
   glVertex3dv(ver[0]);
   glVertex3dv(ver[3]);
   glVertex3dv(ver[7]);
   glVertex3dv(ver[4]);
   glEnd();

   glBegin(GL_POLYGON);
   SphericalNormal(ver[3], normal), glNormal3dv(normal), glVertex3dv(ver[3]);
   SphericalNormal(ver[2], normal), glNormal3dv(normal), glVertex3dv(ver[2]);
   SphericalNormal(ver[6], normal), glNormal3dv(normal), glVertex3dv(ver[6]);
   SphericalNormal(ver[7], normal), glNormal3dv(normal), glVertex3dv(ver[7]);
   glEnd();

   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[5], ver[6], ver[2], normal);
   glNormal3dv(normal);
   glVertex3dv(ver[5]);
   glVertex3dv(ver[6]);
   glVertex3dv(ver[2]);
   glVertex3dv(ver[1]);
   glEnd();

   glBegin(GL_POLYGON);
   SphericalNormalInv(ver[0], normal), glNormal3dv(normal), glVertex3dv(ver[0]);
   SphericalNormalInv(ver[4], normal), glNormal3dv(normal), glVertex3dv(ver[4]);
   SphericalNormalInv(ver[5], normal), glNormal3dv(normal), glVertex3dv(ver[5]);
   SphericalNormalInv(ver[1], normal), glNormal3dv(normal), glVertex3dv(ver[1]);
   glEnd();
}

//______________________________________________________________________________
void DrawTrapezoidTextured(const Double_t ver[][3], Double_t texMin, Double_t texMax)
{
   Double_t normal[3] = {};
   if (texMin > texMax)
      std::swap(texMin, texMax);

   const Double_t tex[] = {texMin, texMin, texMax, texMax, texMin, texMin, texMax, texMax};
   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[0], ver[1], ver[2], normal);
   glNormal3dv(normal);
   glTexCoord1d(tex[0]), glVertex3dv(ver[0]);
   glTexCoord1d(tex[1]), glVertex3dv(ver[1]);
   glTexCoord1d(tex[2]), glVertex3dv(ver[2]);
   glTexCoord1d(tex[3]), glVertex3dv(ver[3]);
   glEnd();
   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[4], ver[7], ver[6], normal);
   glNormal3dv(normal);
   glTexCoord1d(tex[4]), glVertex3dv(ver[4]);
   glTexCoord1d(tex[7]), glVertex3dv(ver[7]);
   glTexCoord1d(tex[6]), glVertex3dv(ver[6]);
   glTexCoord1d(tex[5]), glVertex3dv(ver[5]);
   glEnd();
   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[0], ver[3], ver[7], normal);
   glNormal3dv(normal);
   glTexCoord1d(tex[0]), glVertex3dv(ver[0]);
   glTexCoord1d(tex[3]), glVertex3dv(ver[3]);
   glTexCoord1d(tex[7]), glVertex3dv(ver[7]);
   glTexCoord1d(tex[4]), glVertex3dv(ver[4]);
   glEnd();
   glBegin(GL_POLYGON);
   SphericalNormal(ver[3], normal), glNormal3dv(normal), glTexCoord1d(tex[3]), glVertex3dv(ver[3]);
   SphericalNormal(ver[2], normal), glNormal3dv(normal), glTexCoord1d(tex[2]), glVertex3dv(ver[2]);
   SphericalNormal(ver[6], normal), glNormal3dv(normal), glTexCoord1d(tex[6]), glVertex3dv(ver[6]);
   SphericalNormal(ver[7], normal), glNormal3dv(normal), glTexCoord1d(tex[7]), glVertex3dv(ver[7]);
   glEnd();
   glBegin(GL_POLYGON);
   TMath::Normal2Plane(ver[5], ver[6], ver[2], normal);
   glNormal3dv(normal);
   glTexCoord1d(tex[5]), glVertex3dv(ver[5]);
   glTexCoord1d(tex[6]), glVertex3dv(ver[6]);
   glTexCoord1d(tex[2]), glVertex3dv(ver[2]);
   glTexCoord1d(tex[1]), glVertex3dv(ver[1]);
   glEnd();
   glBegin(GL_POLYGON);
   SphericalNormalInv(ver[0], normal), glNormal3dv(normal), glTexCoord1d(tex[0]), glVertex3dv(ver[0]);
   SphericalNormalInv(ver[4], normal), glNormal3dv(normal), glTexCoord1d(tex[4]), glVertex3dv(ver[4]);
   SphericalNormalInv(ver[5], normal), glNormal3dv(normal), glTexCoord1d(tex[5]), glVertex3dv(ver[5]);
   SphericalNormalInv(ver[1], normal), glNormal3dv(normal), glTexCoord1d(tex[1]), glVertex3dv(ver[1]);
   glEnd();
}


void Draw2DAxis(TAxis *axis, Double_t xMin, Double_t yMin, Double_t xMax, Double_t yMax,
               Double_t min, Double_t max, Bool_t log, Bool_t z = kFALSE)
{
   //Axes are drawn with help of TGaxis class
   std::string option;
   option.reserve(20);

   if (xMin > xMax || z) option += "SDH=+";
   else option += "SDH=-";

   if (log) option += 'G';

   Int_t nDiv = axis->GetNdivisions();

   if (nDiv < 0) {
      option += 'N';
      nDiv = -nDiv;
   }

   TGaxis axisPainter;
   axisPainter.SetLineWidth(1);

   static const Double_t zero = 0.001;

   if (TMath::Abs(xMax - xMin) >= zero || TMath::Abs(yMax - yMin) >= zero) {
      axisPainter.ImportAxisAttributes(axis);
      axisPainter.SetLabelOffset(axis->GetLabelOffset() + axis->GetTickLength());

      if (log) {
         min = TMath::Power(10, min);
         max = TMath::Power(10, max);
      }
      //Option time display is required ?
      if (axis->GetTimeDisplay()) {
         option += 't';

         if (!strlen(axis->GetTimeFormatOnly()))
            axisPainter.SetTimeFormat(axis->ChooseTimeFormat(max - min));
         else
            axisPainter.SetTimeFormat(axis->GetTimeFormat());
      }

      axisPainter.SetOption(option.c_str());
      axisPainter.PaintAxis(xMin, yMin, xMax, yMax, min, max, nDiv, option.c_str());
   }
}

const Int_t gFramePoints[][2] = {{3, 1}, {0, 2}, {1, 3}, {2, 0}};
//Each point has two "neighbouring axes" (left and right). Axes types are 1 (ordinata) and 0 (abscissa)
const Int_t gAxisType[][2]    = {{1, 0}, {0, 1}, {1, 0}, {0, 1}};

//______________________________________________________________________________
void DrawAxes(Int_t fp, const Int_t *vp, const TGLVertex3 *box, const TGLPlotCoordinates *coord,
               TAxis *xAxis, TAxis *yAxis, TAxis *zAxis)
{
   //Using front point, find, where to draw axes and which labels to use for them
   //gVirtualX->SelectWindow(gGLManager->GetVirtualXInd(fGLDevice));
   //gVirtualX->SetDrawMode(TVirtualX::kCopy);//TCanvas by default sets in kInverse

   const Int_t left  = gFramePoints[fp][0];
   const Int_t right = gFramePoints[fp][1];
   const Double_t xLeft = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw()
                                             + box[left].X() - vp[0]));
   const Double_t yLeft = gPad->AbsPixeltoY(Int_t(vp[3] - box[left].Y()
                                             + (1 - gPad->GetHNDC() - gPad->GetYlowNDC())
                                             * gPad->GetWh() + vp[1]));
   const Double_t xMid = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw()
                                             + box[fp].X()  - vp[0]));
   const Double_t yMid = gPad->AbsPixeltoY(Int_t(vp[3] - box[fp].Y()
                                             + (1 - gPad->GetHNDC() - gPad->GetYlowNDC())
                                             * gPad->GetWh() + vp[1]));
   const Double_t xRight = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC()
                                             * gPad->GetWw() + box[right].X() - vp[0]));
   const Double_t yRight = gPad->AbsPixeltoY(Int_t(vp[3] - box[right].Y()
                                             + (1 - gPad->GetHNDC() - gPad->GetYlowNDC())
                                             * gPad->GetWh() + vp[1]));
   const Double_t points[][2] = {{coord->GetXRange().first,  coord->GetYRange().first },
                                 {coord->GetXRange().second, coord->GetYRange().first },
                                 {coord->GetXRange().second, coord->GetYRange().second},
                                 {coord->GetXRange().first,  coord->GetYRange().second}};
   const Int_t    leftType      = gAxisType[fp][0];
   const Int_t    rightType     = gAxisType[fp][1];
   const Double_t leftLabel     = points[left][leftType];
   const Double_t leftMidLabel  = points[fp][leftType];
   const Double_t rightMidLabel = points[fp][rightType];
   const Double_t rightLabel    = points[right][rightType];

   if (xLeft - xMid || yLeft - yMid) {//To supress error messages from TGaxis
      TAxis *axis = leftType ? yAxis : xAxis;
      if (leftLabel < leftMidLabel)
         Draw2DAxis(axis, xLeft, yLeft, xMid, yMid, leftLabel, leftMidLabel,
                     leftType ? coord->GetYLog() : coord->GetXLog());
      else
         Draw2DAxis(axis, xMid, yMid, xLeft, yLeft, leftMidLabel, leftLabel,
                     leftType ? coord->GetYLog() : coord->GetXLog());
   }

   if (xRight - xMid || yRight - yMid) {//To supress error messages from TGaxis
      TAxis *axis = rightType ? yAxis : xAxis;

      if (rightMidLabel < rightLabel)
         Draw2DAxis(axis, xMid, yMid, xRight, yRight, rightMidLabel, rightLabel,
                     rightType ? coord->GetYLog() : coord->GetXLog());
      else
         Draw2DAxis(axis, xRight, yRight, xMid, yMid, rightLabel, rightMidLabel,
                     rightType ? coord->GetYLog() : coord->GetXLog());
   }

   const Double_t xUp = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw()
                                          + box[left + 4].X() - vp[0]));
   const Double_t yUp = gPad->AbsPixeltoY(Int_t(vp[3] - box[left + 4].Y()
                                          + (1 - gPad->GetHNDC() - gPad->GetYlowNDC())
                                          * gPad->GetWh() + vp[1]));
   Draw2DAxis(zAxis, xLeft, yLeft, xUp, yUp, coord->GetZRange().first,
               coord->GetZRange().second, coord->GetZLog(), kTRUE);
}

void SetZLevels(TAxis *zAxis, Double_t zMin, Double_t zMax,
                  Double_t zScale, std::vector<Double_t> &zLevels)
{
   Int_t nDiv = zAxis->GetNdivisions() % 100;
   Int_t nBins = 0;
   Double_t binLow = 0., binHigh = 0., binWidth = 0.;
   THLimitsFinder::Optimize(zMin, zMax, nDiv, binLow, binHigh, nBins, binWidth, " ");
   zLevels.resize(nBins + 1);

   for (Int_t i = 0; i < nBins + 1; ++i)
      zLevels[i] = (binLow + i * binWidth) * zScale;
}

//______________________________________________________________________________
void DrawFaceTextured(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                        Double_t t1, Double_t t2, Double_t t3, const TGLVector3 &norm1,
                        const TGLVector3 &norm2, const TGLVector3 &norm3)
{
   //Draw textured triangle

   glBegin(GL_POLYGON);
   glNormal3dv(norm1.CArr());
   glTexCoord1d(t1);
   glVertex3dv(v1.CArr());
   glNormal3dv(norm2.CArr());
   glTexCoord1d(t2);
   glVertex3dv(v2.CArr());
   glNormal3dv(norm3.CArr());
   glTexCoord1d(t3);
   glVertex3dv(v3.CArr());
   glEnd();
}

//______________________________________________________________________________
void DrawFaceTextured(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                        Double_t t1, Double_t t2, Double_t t3, Double_t z,
                        const TGLVector3 &normal)
{
   //Draw textured triangle on a plane
   glBegin(GL_POLYGON);
   glNormal3dv(normal.CArr());
   glTexCoord1d(t1);
   glVertex3d(v1.X(), v1.Y(), z);
   glTexCoord1d(t2);
   glVertex3d(v2.X(), v2.Y(), z);
   glTexCoord1d(t3);
   glVertex3d(v3.X(), v3.Y(), z);
   glEnd();
}

//______________________________________________________________________________
void GetColor(Float_t v, Float_t vmin, Float_t vmax, Int_t type, Float_t *rgba)
{
   //This function creates color for parametric surface's vertex,
   //using its 'u' value.
   //I've found it in one of Apple's Carbon tutorials , and it's based
   //on Paul Bourke work. Very nice colors!!! :)
   Float_t dv,vmid;
   //Float_t c[] = {1.f, 1.f, 1.f};
   Float_t c1[3] = {}, c2[3] = {}, c3[3] = {};
   Float_t ratio ;
   rgba[3] = 1.f;

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   dv = vmax - vmin;

   switch (type) {
   case 0:
      rgba[0] = 1.f;
      rgba[1] = 1.f;
      rgba[2] = 1.f;
   break;
   case 1:
   if (v < (vmin + 0.25 * dv)) {
      rgba[0] = 0;
      rgba[1] = 4 * (v - vmin) / dv;
      rgba[2] = 1;
   } else if (v < (vmin + 0.5 * dv)) {
      rgba[0] = 0;
      rgba[1] = 1;
      rgba[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      rgba[0] = 4 * (v - vmin - 0.5 * dv) / dv;
      rgba[1] = 1;
      rgba[2] = 0;
   } else {
      rgba[0] = 1;
      rgba[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      rgba[2] = 0;
   }
   break;
   case 2:
      rgba[0] = (v - vmin) / dv;
      rgba[1] = 0;
      rgba[2] = (vmax - v) / dv;
      break;
   case 3:
      rgba[0] = (v - vmin) / dv;
      rgba[1] = rgba[0];
      rgba[2] = rgba[0];
      break;
   case 4:
      if (v < (vmin + dv / 6.0)) {
         rgba[0] = 1;
         rgba[1] = 6 * (v - vmin) / dv;
         rgba[2] = 0;
      } else if (v < (vmin + 2.0 * dv / 6.0)) {
         rgba[0] = 1 + 6 * (vmin + dv / 6.0 - v) / dv;
         rgba[1] = 1;
         rgba[2] = 0;
      } else if (v < (vmin + 3.0 * dv / 6.0)) {
         rgba[0] = 0;
         rgba[1] = 1;
         rgba[2] = 6 * (v - vmin - 2.0 * dv / 6.0) / dv;
      } else if (v < (vmin + 4.0 * dv / 6.0)) {
         rgba[0] = 0;
         rgba[1] = 1 + 6 * (vmin + 3.0 * dv / 6.0 - v) / dv;
         rgba[2] = 1;
      } else if (v < (vmin + 5.0 * dv / 6.0)) {
         rgba[0] = 6 * (v - vmin - 4.0 * dv / 6.0) / dv;
         rgba[1] = 0;
         rgba[2] = 1;
      } else {
         rgba[0] = 1;
         rgba[1] = 0;
         rgba[2] = 1 + 6 * (vmin + 5.0 * dv / 6.0 - v) / dv;
      }
      break;
   case 5:
      rgba[0] = (v - vmin) / (vmax - vmin);
      rgba[1] = 1;
      rgba[2] = 0;
      break;
   case 6:
      rgba[0] = (v - vmin) / (vmax - vmin);
      rgba[1] = (vmax - v) / (vmax - vmin);
      rgba[2] = rgba[0];
      break;
   case 7:
      if (v < (vmin + 0.25 * dv)) {
         rgba[0] = 0;
         rgba[1] = 4 * (v - vmin) / dv;
         rgba[2] = 1 - rgba[1];
      } else if (v < (vmin + 0.5 * dv)) {
         rgba[0] = 4 * (v - vmin - 0.25 * dv) / dv;
         rgba[1] = 1 - rgba[0];
         rgba[2] = 0;
      } else if (v < (vmin + 0.75 * dv)) {
         rgba[1] = 4 * (v - vmin - 0.5 * dv) / dv;
         rgba[0] = 1 - rgba[1];
         rgba[2] = 0;
      } else {
         rgba[0] = 0;
         rgba[2] = 4 * (v - vmin - 0.75 * dv) / dv;
         rgba[1] = 1 - rgba[2];
      }
      break;
   case 8:
      if (v < (vmin + 0.5 * dv)) {
         rgba[0] = 2 * (v - vmin) / dv;
         rgba[1] = rgba[0];
         rgba[2] = rgba[0];
      } else {
         rgba[0] = 1 - 2 * (v - vmin - 0.5 * dv) / dv;
         rgba[1] = rgba[0];
         rgba[2] = rgba[0];
      }
      break;
   case 9:
      if (v < (vmin + dv / 3)) {
         rgba[2] = 3 * (v - vmin) / dv;
         rgba[1] = 0;
         rgba[0] = 1 - rgba[2];
      } else if (v < (vmin + 2 * dv / 3)) {
         rgba[0] = 0;
         rgba[1] = 3 * (v - vmin - dv / 3) / dv;
         rgba[2] = 1;
      } else {
         rgba[0] = 3 * (v - vmin - 2 * dv / 3) / dv;
         rgba[1] = 1 - rgba[0];
         rgba[2] = 1;
      }
      break;
   case 10:
      if (v < (vmin + 0.2 * dv)) {
         rgba[0] = 0;
         rgba[1] = 5 * (v - vmin) / dv;
         rgba[2] = 1;
      } else if (v < (vmin + 0.4 * dv)) {
         rgba[0] = 0;
         rgba[1] = 1;
         rgba[2] = 1 + 5 * (vmin + 0.2 * dv - v) / dv;
      } else if (v < (vmin + 0.6 * dv)) {
         rgba[0] = 5 * (v - vmin - 0.4 * dv) / dv;
         rgba[1] = 1;
         rgba[2] = 0;
      } else if (v < (vmin + 0.8 * dv)) {
         rgba[0] = 1;
         rgba[1] = 1 - 5 * (v - vmin - 0.6 * dv) / dv;
         rgba[2] = 0;
      } else {
         rgba[0] = 1;
         rgba[1] = 5 * (v - vmin - 0.8 * dv) / dv;
         rgba[2] = 5 * (v - vmin - 0.8 * dv) / dv;
      }
      break;
   case 11:
      c1[0] = 200 / 255.0; c1[1] =  60 / 255.0; c1[2] =   0 / 255.0;
      c2[0] = 250 / 255.0; c2[1] = 160 / 255.0; c2[2] = 110 / 255.0;
      rgba[0] = (c2[0] - c1[0]) * (v - vmin) / dv + c1[0];
      rgba[1] = (c2[1] - c1[1]) * (v - vmin) / dv + c1[1];
      rgba[2] = (c2[2] - c1[2]) * (v - vmin) / dv + c1[2];
      break;
   case 12:
      c1[0] =  55 / 255.0; c1[1] =  55 / 255.0; c1[2] =  45 / 255.0;
      c2[0] = 200 / 255.0; c2[1] =  60 / 255.0; c2[2] =   0 / 255.0;
      c3[0] = 250 / 255.0; c3[1] = 160 / 255.0; c3[2] = 110 / 255.0;
      ratio = 0.4;
      vmid = vmin + ratio * dv;
      if (v < vmid) {
         rgba[0] = (c2[0] - c1[0]) * (v - vmin) / (ratio*dv) + c1[0];
         rgba[1] = (c2[1] - c1[1]) * (v - vmin) / (ratio*dv) + c1[1];
         rgba[2] = (c2[2] - c1[2]) * (v - vmin) / (ratio*dv) + c1[2];
      } else {
         rgba[0] = (c3[0] - c2[0]) * (v - vmid) / ((1-ratio)*dv) + c2[0];
         rgba[1] = (c3[1] - c2[1]) * (v - vmid) / ((1-ratio)*dv) + c2[1];
         rgba[2] = (c3[2] - c2[2]) * (v - vmid) / ((1-ratio)*dv) + c2[2];
      }
      break;
   case 13:
      c1[0] =   0 / 255.0; c1[1] = 255 / 255.0; c1[2] =   0 / 255.0;
      c2[0] = 255 / 255.0; c2[1] = 150 / 255.0; c2[2] =   0 / 255.0;
      c3[0] = 255 / 255.0; c3[1] = 250 / 255.0; c3[2] = 240 / 255.0;
      ratio = 0.3;
      vmid = vmin + ratio * dv;
      if (v < vmid) {
         rgba[0] = (c2[0] - c1[0]) * (v - vmin) / (ratio*dv) + c1[0];
         rgba[1] = (c2[1] - c1[1]) * (v - vmin) / (ratio*dv) + c1[1];
         rgba[2] = (c2[2] - c1[2]) * (v - vmin) / (ratio*dv) + c1[2];
      } else {
         rgba[0] = (c3[0] - c2[0]) * (v - vmid) / ((1-ratio)*dv) + c2[0];
         rgba[1] = (c3[1] - c2[1]) * (v - vmid) / ((1-ratio)*dv) + c2[1];
         rgba[2] = (c3[2] - c2[2]) * (v - vmid) / ((1-ratio)*dv) + c2[2];
      }
      break;
   case 14:
      rgba[0] = 1;
      rgba[1] = 1 - (v - vmin) / dv;
      rgba[2] = 0;
      break;
   case 15:
      if (v < (vmin + 0.25 * dv)) {
         rgba[0] = 0;
         rgba[1] = 4 * (v - vmin) / dv;
         rgba[2] = 1;
      } else if (v < (vmin + 0.5 * dv)) {
         rgba[0] = 0;
         rgba[1] = 1;
         rgba[2] = 1 - 4 * (v - vmin - 0.25 * dv) / dv;
      } else if (v < (vmin + 0.75 * dv)) {
         rgba[0] = 4 * (v - vmin - 0.5 * dv) / dv;
         rgba[1] = 1;
         rgba[2] = 0;
      } else {
         rgba[0] = 1;
         rgba[1] = 1;
         rgba[2] = 4 * (v - vmin - 0.75 * dv) / dv;
      }
      break;
   case 16:
      if (v < (vmin + 0.5 * dv)) {
         rgba[0] = 0.0;
         rgba[1] = 2 * (v - vmin) / dv;
         rgba[2] = 1 - 2 * (v - vmin) / dv;
      } else {
         rgba[0] = 2 * (v - vmin - 0.5 * dv) / dv;
         rgba[1] = 1 - 2 * (v - vmin - 0.5 * dv) / dv;
         rgba[2] = 0.0;
      }
      break;
   case 17:
      if (v < (vmin + 0.5 * dv)) {
         rgba[0] = 1.0;
         rgba[1] = 1 - 2 * (v - vmin) / dv;
         rgba[2] = 2 * (v - vmin) / dv;
      } else {
         rgba[0] = 1 - 2 * (v - vmin - 0.5 * dv) / dv;
         rgba[1] = 2 * (v - vmin - 0.5 * dv) / dv;
         rgba[2] = 1.0;
      }
      break;
   case 18:
      rgba[0] = 0;
      rgba[1] = (v - vmin) / (vmax - vmin);
      rgba[2] = 1;
      break;
   case 19:
      rgba[0] = (v - vmin) / (vmax - vmin);
      rgba[1] = rgba[0];
      rgba[2] = 1;
      break;
   case 20:
      c1[0] =   0 / 255.0; c1[1] = 160 / 255.0; c1[2] =   0 / 255.0;
      c2[0] = 180 / 255.0; c2[1] = 220 / 255.0; c2[2] =   0 / 255.0;
      c3[0] = 250 / 255.0; c3[1] = 220 / 255.0; c3[2] = 170 / 255.0;
      ratio = 0.3;
      vmid = vmin + ratio * dv;
      if (v < vmid) {
         rgba[0] = (c2[0] - c1[0]) * (v - vmin) / (ratio*dv) + c1[0];
         rgba[1] = (c2[1] - c1[1]) * (v - vmin) / (ratio*dv) + c1[1];
         rgba[2] = (c2[2] - c1[2]) * (v - vmin) / (ratio*dv) + c1[2];
      } else {
         rgba[0] = (c3[0] - c2[0]) * (v - vmid) / ((1-ratio)*dv) + c2[0];
         rgba[1] = (c3[1] - c2[1]) * (v - vmid) / ((1-ratio)*dv) + c2[1];
         rgba[2] = (c3[2] - c2[2]) * (v - vmid) / ((1-ratio)*dv) + c2[2];
      }
      break;
   }
}

}

//______________________________________________________________________________
TGLLevelPalette::TGLLevelPalette()
                  : fContours(0),
                    fPaletteSize(0),
                    fTexture(0),
                    fMaxPaletteSize(0)
{
   //Ctor.
}

//______________________________________________________________________________
Bool_t TGLLevelPalette::GeneratePalette(UInt_t paletteSize, const Rgl::Range_t &zRange, Bool_t check)
{
   //Try to find colors for palette.
   if (!fMaxPaletteSize && check)
      glGetIntegerv(GL_MAX_TEXTURE_SIZE, &fMaxPaletteSize);

   if (!(zRange.second - zRange.first))
      return kFALSE;

   if (check && paletteSize > UInt_t(fMaxPaletteSize)) {
      Error("TGLLevelPalette::GeneratePalette",
            "Number of contours %d is too big for GL 1D texture, try to reduce it to %d",
            paletteSize, fMaxPaletteSize);
      return kFALSE;
   }

   UInt_t nearestPow2 = 2;
   while (nearestPow2 < paletteSize)
      nearestPow2 <<= 1;

   fTexels.resize(4 * nearestPow2);
   fPaletteSize = paletteSize;

   //Generate texels.
   const Int_t nColors = gStyle->GetNumberOfColors();

   //Map color index into index in real palette.

   for (UInt_t i = 0; i < paletteSize; ++i) {
      Int_t paletteInd = Int_t(nColors / Double_t(paletteSize) * i);
      if (paletteInd > nColors - 1)
         paletteInd = nColors - 1;
      Int_t colorInd = gStyle->GetColorPalette(paletteInd);

      if (const TColor *c = gROOT->GetColor(colorInd)) {
         Float_t rgb[3] = {};
         c->GetRGB(rgb[0], rgb[1], rgb[2]);
         fTexels[i * 4]     = UChar_t(rgb[0] * 255);
         fTexels[i * 4 + 1] = UChar_t(rgb[1] * 255);
         fTexels[i * 4 + 2] = UChar_t(rgb[2] * 255);
         fTexels[i * 4 + 3] = 200;//alpha
      }
   }

   fZRange = zRange;

   return kTRUE;
}

//______________________________________________________________________________
void TGLLevelPalette::SetContours(const std::vector<Double_t> *cont)
{
   //Clear :)
   fContours = cont;
}

//______________________________________________________________________________
void TGLLevelPalette::EnableTexture(Int_t mode)const
{
   //Enable 1D texture
   glEnable(GL_TEXTURE_1D);

   glGenTextures(1, &fTexture);
   
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glBindTexture(GL_TEXTURE_1D, fTexture);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, fTexels.size() / 4, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, &fTexels[0]);
   glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GLint(mode));
}

//______________________________________________________________________________
void TGLLevelPalette::DisableTexture()const
{
   //Disable 1D texture
   glDeleteTextures(1, &fTexture);
   glDisable(GL_TEXTURE_1D);
}

//______________________________________________________________________________
Int_t TGLLevelPalette::GetPaletteSize()const
{
   //Get. Palette. Size.
   return Int_t(fPaletteSize);
}

//______________________________________________________________________________
Double_t TGLLevelPalette::GetTexCoord(Double_t z)const
{
   //Get tex coordinate
   if (!fContours)
      return (z - fZRange.first) / (fZRange.second - fZRange.first) * fPaletteSize / (fTexels.size() / 4);

   /*
   //This part is wrong. To be fixed.
   std::vector<Double_t>::size_type i = 0, e = fContours->size();

   if (!e)
      return 0.;

   for (; i < e - 1; ++i) {
      if (z >= (*fContours)[i] && z <= (*fContours)[i + 1])
         return i / Double_t(fTexels.size() / 4);
   }
   */

   return 1.;
}

//______________________________________________________________________________
const UChar_t *TGLLevelPalette::GetColour(Double_t z)const
{
   //Get color.
   const Int_t ind = Int_t((z - fZRange.first) / (fZRange.second - fZRange.first) * fPaletteSize);
   return &fTexels[ind * 4];
}

//______________________________________________________________________________
const UChar_t *TGLLevelPalette::GetColour(Int_t ind)const
{
   //Get color.
   return &fTexels[ind * 4];
}
