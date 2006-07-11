// @(#)root/gl:$Name:  $:$Id: TGLUtil.cxx,v 1.27 2006/06/13 15:43:39 couet Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TGLUtil.h"
#include "TGLBoundingBox.h"
#include "TGLQuadric.h"
#include "TGLIncludes.h"

#include "TError.h"
#include "TMath.h"

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

ClassImp(TGLVertex3)

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

ClassImp(TGLVector3)

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

ClassImp(TGLLine3)

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

ClassImp(TGLRect)

//______________________________________________________________________________
TGLRect::TGLRect() :
      fX(0), fY(0), fWidth(0), fHeight(0)
{
   // Construct empty rect object, corner (0,0), width/height 0
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
EOverlap TGLRect::Overlap(const TGLRect & other) const
{
   // Return overlap result (kInside, kOutside, kPartial) of this
   // rect with 'other'
   if ((fX <= other.fX) && (fX + fWidth >= other.fX + other.fWidth) &&
        (fY <= other.fY) && (fY +fHeight >= other.fY + other.fHeight)) {
      return kInside;
   }
   else if ((fX >= other.fX + static_cast<Int_t>(other.fWidth)) ||
            (fX + static_cast<Int_t>(fWidth) <= other.fX) ||
            (fY >= other.fY + static_cast<Int_t>(other.fHeight)) ||
            (fY + static_cast<Int_t>(fHeight) <= other.fY)) {
      return kOutside;
   } else {
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

ClassImp(TGLPlane)

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
void TGLPlane::Dump() const
{
   // Output plane equation to std::out
   std::cout.precision(6);
   std::cout << "Plane : " << fVals[0] << "x + " << fVals[1] << "y + " << fVals[2] << "z + " << fVals[3] << std::endl;
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
   if (!extend && factor < 0.0 || factor > 1.0) {
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

ClassImp(TGLMatrix)

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
TGLMatrix::TGLMatrix(const TGLVertex3 & origin, const TGLVector3 & zAxis, const TGLVector3 * xAxis)
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
void TGLMatrix::Set(const TGLVertex3 & origin, const TGLVector3 & zAxis, const TGLVector3 * xAxis)
{
   // Set matrix which when applied puts local origin at 
   // 'origin' and the local Z axis in direction 'z'. Both
   // 'origin' and 'z' are expressed in the parent frame
   TGLVector3 zAxisInt(zAxis);
   zAxisInt.Normalise();

   TGLVector3 xAxisInt;
   if (xAxis) {
      xAxisInt = *xAxis;
   } else {
      TGLVector3 arbAxis;
      if (TMath::Abs(zAxisInt.X()) <= TMath::Abs(zAxisInt.Y()) && TMath::Abs(zAxisInt.X()) <= TMath::Abs(zAxisInt.Z())) {
         arbAxis.Set(1.0, 0.0, 0.0); 
      } else if (TMath::Abs(zAxisInt.Y()) <= TMath::Abs(zAxisInt.X()) && TMath::Abs(zAxisInt.Y()) <= TMath::Abs(zAxisInt.Z())) {
         arbAxis.Set(0.0, 1.0, 0.0); 
      } else { 
         arbAxis.Set(0.0, 0.0, 1.0);
      }
      xAxisInt = Cross(zAxisInt, arbAxis);
   }

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
   // ordering is maintained - i.e. should be column major 
   // as we are
   for (UInt_t i=0; i < 16; i++) {
      fVals[i] = vals[i];
   }
}

//______________________________________________________________________________
void TGLMatrix::SetIdentity()
{
   // Set matrix to identity:
   //
   // 1 0 0 0
   // 0 1 0 0
   // 0 0 1 0
   // 0 0 0 1
   fVals[0] = 1.0; fVals[4] = 0.0; fVals[8 ] = 0.0; fVals[12] = 0.0;
   fVals[1] = 0.0; fVals[5] = 1.0; fVals[9 ] = 0.0; fVals[13] = 0.0;
   fVals[2] = 0.0; fVals[6] = 0.0; fVals[10] = 1.0; fVals[14] = 0.0;
   fVals[3] = 0.0; fVals[7] = 0.0; fVals[11] = 0.0; fVals[15] = 1.0;
}

//______________________________________________________________________________
void TGLMatrix::SetTranslation(Double_t x, Double_t y, Double_t z)
{
   // Set matrix translation components x,y,z:
   //
   // . . . x
   // . . . y
   // . . . z
   // . . . . 
   //
   // The other components are NOT modified
   SetTranslation(TGLVertex3(x,y,z));
}

//______________________________________________________________________________
void TGLMatrix::SetTranslation(const TGLVertex3 & translation)
{
   // Set matrix translation components x,y,z:
   //
   // . . . translation.X()
   // . . . translation.Y()
   // . . . translation.Z()
   // . . . . 
   //
   // . = Exisiting component value - NOT modified
   fVals[12] = translation[0];
   fVals[13] = translation[1];
   fVals[14] = translation[2];
}

//______________________________________________________________________________
TGLVertex3 TGLMatrix::GetTranslation() const
{
   // Return the translation component of matrix
   //
   // . . . X()
   // . . . Y()
   // . . . Z()
   // . . . . 
      
   return TGLVertex3(fVals[12], fVals[13], fVals[14]);
}

//______________________________________________________________________________
void TGLMatrix::Translate(const TGLVector3 & vect)
{
   // Offset (shift) matrix translation components by 'vect'
   //
   // . . . . + vect.X()
   // . . . . + vect.Y()
   // . . . . + vect.Z()
   // . . . . 
   //
   // . = Exisiting component value - NOT modified
   fVals[12] += vect[0];
   fVals[13] += vect[1];
   fVals[14] += vect[2];
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
TGLVector3 TGLMatrix::GetScale() const
{
   // Get local axis scaling factors
   TGLVector3 x(fVals[0], fVals[1], fVals[2]);
   TGLVector3 y(fVals[4], fVals[5], fVals[6]);
   TGLVector3 z(fVals[8], fVals[9], fVals[10]);
   return TGLVector3(x.Mag(), y.Mag(), z.Mag());
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLUtil                                                              //
//                                                                      //
// Wrapper class for various misc static functions - error checking,    //
// draw helpers etc.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLUtil)

UInt_t TGLUtil::fgDrawQuality = 60;

//______________________________________________________________________________
void TGLUtil::CheckError(const char * loc)
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
}

//______________________________________________________________________________
void TGLUtil::SetDrawColors(const Float_t rgba[4])
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
   static Float_t ambient[4] = {0.0, 0.0, 0.0, 1.0};
   static Float_t specular[4] = {0.6, 0.6, 0.6, 1.0};
   Float_t emission[4] = {rgba[0]/4.0, rgba[1]/4.0, rgba[2]/4.0, rgba[3]};

   glColor3d(rgba[0], rgba[1], rgba[2]);
   glMaterialfv(GL_FRONT, GL_DIFFUSE, rgba);
   glMaterialfv(GL_FRONT, GL_AMBIENT, ambient);
   glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
   glMaterialfv(GL_FRONT, GL_EMISSION, emission);
   glMaterialf(GL_FRONT, GL_SHININESS, 60.0);
}

//______________________________________________________________________________
void TGLUtil::DrawSphere(const TGLVertex3 & position, Double_t radius, 
                         const Float_t rgba[4])
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
                       const Float_t rgba[4])
{
   // Draw thick line (tube) defined by 'line', with head at end shape 
   // 'head' - box/arrow/none, (head) size 'size', color 'rgba'
   DrawLine(line.Start(), line.Vector(), head, size, rgba);
}

//______________________________________________________________________________
void TGLUtil::DrawLine(const TGLVertex3 & start, const TGLVector3 & vector, 
                       ELineHeadShape head, Double_t size, const Float_t rgba[4])
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
   gluCylinder(quad.Get(), size/4.0, size/4.0, vector.Mag() - headHeight, fgDrawQuality, fgDrawQuality);
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(quad.Get(), 0.0, size/4.0, fgDrawQuality, fgDrawQuality); 

   glTranslated(0.0, 0.0, vector.Mag() - headHeight); // Shift down local Z to end of line

   if (head == kLineHeadNone) { 
      // Cap end of line
      gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
      gluDisk(quad.Get(), 0.0, size/4.0, fgDrawQuality, fgDrawQuality); 
   }
   else if (head == kLineHeadArrow) {
      // Arrow base / end line cap
      gluDisk(quad.Get(), 0.0, size, fgDrawQuality, fgDrawQuality); 
      // Arrow cone
      gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
      gluCylinder(quad.Get(), size, 0.0, headHeight, fgDrawQuality, fgDrawQuality);
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
                       Double_t radius, const Float_t rgba[4])
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
   gluCylinder(quad.Get(), inner, inner, width, fgDrawQuality, fgDrawQuality);
   gluCylinder(quad.Get(), outer, outer, width, fgDrawQuality, fgDrawQuality);
   
   // Top/bottom
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_INSIDE);
   gluDisk(quad.Get(), inner, outer, fgDrawQuality, fgDrawQuality); 
   glTranslated(0.0, 0.0, width);
   gluQuadricOrientation(quad.Get(), (GLenum)GLU_OUTSIDE);
   gluDisk(quad.Get(), inner, outer, fgDrawQuality, fgDrawQuality); 
   
   glPopMatrix();
}

namespace RootGL {

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

}

ClassImp(TGLSelectionBuffer)

//______________________________________________________________________________
TGLSelectionBuffer::TGLSelectionBuffer()
                        : fWidth(0)
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
const UChar_t *TGLSelectionBuffer::GetPixelColor(Int_t px, Int_t py)const
{
   // Get pixel color.
   if (px < 0) 
      px = 0; 
   if (py < 0)
      py = 0;

   if ((UInt_t)(px * fWidth * 4 + py * 4) > fBuffer.size())
      return &fBuffer[0];

   return &fBuffer[px * fWidth * 4 + py * 4];
}
