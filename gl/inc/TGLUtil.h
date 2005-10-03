// @(#)root/gl:$Name:  $:$Id: TGLUtil.h,v 1.11 2005/08/10 16:26:35 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLUtil
#define ROOT_TGLUtil

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include <vector>
#include <math.h>
#include <assert.h>

/*************************************************************************
 * TGLUtil - A collection of utility classes for GL. Vertex, vector, rect
 * matrix etc. These provide const and non-const accessors Arr() / CArr() to
 * a GL compatible internal field. This means they can be used directly
 * with OpenGL C API calls.
 *
 * They are not intended to be fully featured - just provide minimum
 * required.
 *
 * Also other various common types, error checking etc.
 *
 *************************************************************************/

// TODO: Split these into own h/cxx files

// TODO: Where should these enums live?
enum  ELODPresets {
   kLow = 20,
   kMed = 50,
   kHigh = 100
};

enum EPosition
{
   kInFront = 0,
   kBehind
};

enum EOverlap
{
   kInside = 0,
   kPartial,
   kOutside
};

// TODO: Put this into a proper draw style flag UInt_t
// seperated into viewer/scene/physical/logical sections
// modify TGLDrawable to cache on shape subset
enum EDrawStyle
{
   kFill = 0, 
   kOutline, 
   kWireFrame
};

enum EClipType 
{ 
   kClipNone = 0, 
   kClipPlane, 
   kClipBox
};


// TODO: Namespace
// TODO: Mark all headers as INTERNAL

/*************************************************************************
 * TGLVertex3 - TODO
 *
 *
 *
 *************************************************************************/

class TGLVector3; // Forward declare for Shift()

class TGLVertex3
{
protected:
   // Fields
   Bool_t ValidIndex(UInt_t index) const { return (index < 3); }
   Double_t fVals[3];

public:
   TGLVertex3();
   TGLVertex3(Double_t x, Double_t y, Double_t z);
   TGLVertex3(const TGLVertex3 & other);
   virtual ~TGLVertex3();

         TGLVertex3 & operator =  (const TGLVertex3 & rhs);
   const TGLVertex3 & operator -= (const TGLVector3 & val);
   const TGLVertex3 & operator += (const TGLVector3 & val);
         TGLVertex3   operator -  () const;

   void Fill(Double_t val);
   void Set(Double_t x, Double_t y, Double_t z);
   void Set(const TGLVertex3 & other);
   void Shift(TGLVector3 & shift);
   void Shift(Double_t xDelta, Double_t yDelta, Double_t zDelta);

   // Accessors
   Double_t & operator [] (Int_t index);
   const Double_t& operator [] (Int_t index) const;
   Double_t   X() const { return fVals[0]; }
   Double_t & X()       { return fVals[0]; }
   Double_t   Y() const { return fVals[1]; }
   Double_t & Y()       { return fVals[1]; }
   Double_t   Z() const { return fVals[2]; }
   Double_t & Z()       { return fVals[2]; }

   const Double_t * CArr() const { return fVals; }
   Double_t *       Arr()        { return fVals; }

   void Dump() const;

   ClassDef(TGLVertex3,0) // GL 3 component vertex helper/wrapper class
};

//______________________________________________________________________________
inline TGLVertex3 & TGLVertex3::operator = (const TGLVertex3 & rhs) 
{ 
   // Check for self-assignment
   if (this != &rhs) {
      Set(rhs); 
   }
   return *this; 
}

// operator -= & operator += inline needs to be defered until full TGLVector3 definition

//______________________________________________________________________________
inline TGLVertex3 TGLVertex3::operator - () const
{ 
   return TGLVertex3(-fVals[0], -fVals[1], -fVals[2]);
}

//______________________________________________________________________________
inline Double_t & TGLVertex3::operator [] (Int_t index)
{ 
   /*if (!ValidIndex(index)) { 
      assert(kFALSE); 
      return fVals[0]; 
   } else {*/
      return fVals[index]; 
   //} 
}

//______________________________________________________________________________
inline const Double_t& TGLVertex3::operator [] (Int_t index) const
{ 
   /*if (!ValidIndex(index)) { 
      assert(kFALSE); 
      return fVals[0]; 
   } else {*/
      return fVals[index]; 
   //} 
}

//______________________________________________________________________________
inline void TGLVertex3::Fill(Double_t val)
{ 
   Set(val,val,val); 
}

//______________________________________________________________________________
inline void TGLVertex3::Set(Double_t x, Double_t y, Double_t z) 
{ 
   fVals[0]=x; 
   fVals[1]=y; 
   fVals[2]=z; 
}

//______________________________________________________________________________
inline void TGLVertex3::Set(const TGLVertex3 & other)
{ 
   fVals[0]=other.fVals[0]; 
   fVals[1]=other.fVals[1]; 
   fVals[2]=other.fVals[2]; 
}

/*************************************************************************
 * TGLVector3 - TODO
 *
 *
 *
 *************************************************************************/
class TGLVector3 : public TGLVertex3
{
public:
   TGLVector3();
   TGLVector3(Double_t x, Double_t y, Double_t z);
   TGLVector3(const TGLVector3 & other);
   virtual ~TGLVector3();

   const TGLVector3 & operator /= (Double_t val);
   const TGLVector3 & operator *= (Double_t val);
         TGLVector3   operator -  () const;

   Double_t Mag() const;
   void     Normalise();

   ClassDef(TGLVector3,0) // GL 3 component vector helper/wrapper class
};

// Inline for TGLVertex3 requiring full TGLVector definition
//______________________________________________________________________________
inline const TGLVertex3 & TGLVertex3::operator -= (const TGLVector3 & vec)
{ 
   fVals[0] -= vec[0]; fVals[1] -= vec[1]; fVals[2] -= vec[2]; 
   return *this; 
}

// Inline for TGLVertex3 requiring full TGLVector definition
//______________________________________________________________________________
inline const TGLVertex3 & TGLVertex3::operator += (const TGLVector3 & vec)
{ 
   fVals[0] += vec[0]; fVals[1] += vec[1]; fVals[2] += vec[2]; 
   return *this; 
}

//______________________________________________________________________________
inline const TGLVector3 & TGLVector3::operator /= (Double_t val)
{ 
   fVals[0] /= val; 
   fVals[1] /= val; 
   fVals[2] /= val; 
   return *this; 
}

//______________________________________________________________________________
inline const TGLVector3 & TGLVector3::operator *= (Double_t val)
{ 
   fVals[0] *= val; 
   fVals[1] *= val; 
   fVals[2] *= val; 
   return *this; 
}

//______________________________________________________________________________
inline TGLVector3 TGLVector3::operator - () const
{ 
   return TGLVector3(-fVals[0], -fVals[1], -fVals[2]);
}

//______________________________________________________________________________
inline Double_t TGLVector3::Mag() const
{
   return sqrt(fVals[0]*fVals[0] + fVals[1]*fVals[1] + fVals[2]*fVals[2]);
}

//______________________________________________________________________________
inline void TGLVector3::Normalise()
{
   Double_t mag = Mag();
   if ( mag == 0.0 ) {
      assert( kFALSE );
      return;
   }
   fVals[0] /= mag;
   fVals[1] /= mag;
   fVals[2] /= mag;
}

//______________________________________________________________________________
inline Double_t Dot(const TGLVector3 & v1, const TGLVector3 & v2)
{
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

//______________________________________________________________________________
inline TGLVector3 Cross(const TGLVector3 & v1, const TGLVector3 & v2)
{
    return TGLVector3(v1[1]*v2[2] - v2[1]*v1[2],
                      v1[2]*v2[0] - v2[2]*v1[0],
                      v1[0]*v2[1] - v2[0]*v1[1]);
}

//______________________________________________________________________________
inline const TGLVector3 operator / (const TGLVector3 & vec, Double_t val)
{
   return TGLVector3(vec[0] / val, vec[1] / val, vec[2] / val);
}

//______________________________________________________________________________
inline const TGLVector3 operator * (const TGLVector3 & vec, Double_t val)
{
   return TGLVector3(vec[0] * val, vec[1] * val, vec[2] * val);
}

//______________________________________________________________________________
// Vertex + Vector => Vertex
inline TGLVertex3 operator + (const TGLVertex3 & vertex1, const TGLVector3 & vertex2)
{
   return TGLVertex3(vertex1[0] + vertex2[0], vertex1[1] + vertex2[1], vertex1[2] + vertex2[2]);
}

//______________________________________________________________________________
// Vertex - Vertex => Vector
inline TGLVector3 operator - (const TGLVertex3 & vertex1, const TGLVertex3 & vertex2)
{
   return TGLVector3(vertex1[0] - vertex2[0], vertex1[1] - vertex2[1], vertex1[2] - vertex2[2]);
}

//______________________________________________________________________________
// Vector + Vector => Vector
inline TGLVector3 operator + (const TGLVector3 & vector1, const TGLVector3 & vector2)
{
   return TGLVector3(vector1[0] + vector2[0], vector1[1] + vector2[1], vector1[2] + vector2[2]);
}

//______________________________________________________________________________
// Vector - Vector => Vector
inline TGLVector3 operator - (const TGLVector3 & vector1, const TGLVector3 & vector2)
{
   return TGLVector3(vector1[0] - vector2[0], vector1[1] - vector2[1], vector1[2] - vector2[2]);
}

/*************************************************************************
 * TGLRect - TODO
 *
 *
 *
 *************************************************************************/
class TGLRect
{
   private:
   // Fields
   Int_t    fX, fY;           //! Corner
   UInt_t   fWidth, fHeight;  //! Positive width/height

   public:
   TGLRect();
   TGLRect(Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TGLRect(); // ClassDef introduces virtual fns

   // Bitwise copy const & =op are ok at present
   void Set(Int_t x, Int_t y, UInt_t width, UInt_t height);
   void SetCorner(Int_t x, Int_t y);
   void Offset(Int_t dX, Int_t dY);
   void Expand(Int_t x, Int_t y);

   Int_t    X()       const { return fX; }
   Int_t &  X()             { return fX; }
   Int_t    Y()       const { return fY; }
   Int_t &  Y()             { return fY; }
   UInt_t   Width()   const { return fWidth; }
   UInt_t & Width()         { return fWidth; }
   UInt_t   Height()  const { return fHeight; }
   UInt_t & Height()        { return fHeight; }
   Int_t    CenterX() const { return fX + fWidth/2; }
   Int_t    CenterY() const { return fY + fHeight/2; }
   Int_t    Left()    const { return fX; }
   Int_t    Right()   const { return fX + fWidth; }
   Int_t    Top()     const { return fY; }
   Int_t    Bottom()  const { return fY + fHeight; }

   UInt_t Diagonal() const { return static_cast<UInt_t>(sqrt(static_cast<Double_t>(fWidth*fWidth + fHeight*fHeight))); }
   UInt_t Longest() const { return fWidth > fHeight ? fWidth:fHeight; }

   Double_t Aspect() const;
   EOverlap Overlap(const TGLRect & other) const;

   // TODO: Change fields
   //const Double_t * CArr() const { return fVals; }
   //Double_t * Arr() { return fVals; }

   ClassDef(TGLRect,0) // GL rect helper/wrapper class
};

//______________________________________________________________________________
inline void TGLRect::Set(Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   fX = x;
   fY = y;
   fWidth = width;
   fHeight = height;
}

//______________________________________________________________________________
inline void TGLRect::SetCorner(Int_t x, Int_t y)
{
   fX = x;
   fY = y;
}

//______________________________________________________________________________
inline void TGLRect::Offset(Int_t dX, Int_t dY)
{
   fX += dX;
   fY += dY;
}

/*************************************************************************
 * TGLPlane - TODO
 *
 *
 *
 *************************************************************************/
class TGLPlane
{
private:
   // Fields
   Double_t fVals[4];

   // Methods
   void Normalise();

public:
   TGLPlane();
   TGLPlane(const TGLPlane & other);
   TGLPlane(Double_t a, Double_t b, Double_t c, Double_t d);
   TGLPlane(Double_t eq[4]);
   TGLPlane(const TGLVector3 & norm, const TGLVertex3 & point);
   TGLPlane(const TGLVertex3 & p1, const TGLVertex3 & p2, const TGLVertex3 & p3);
   virtual ~TGLPlane(); // ClassDef introduces virtual fns

   void Set(const TGLPlane & other);
   void Set(Double_t a, Double_t b, Double_t c, Double_t d);
   void Set(Double_t eq[4]);
   void Set(const TGLVector3 & norm, const TGLVertex3 & point); 
   void Set(const TGLVertex3 & p1, const TGLVertex3 & p2, const TGLVertex3 & p3);

   Double_t A() const { return fVals[0]; }
   Double_t B() const { return fVals[1]; }
   Double_t C() const { return fVals[2]; }
   Double_t D() const { return fVals[3]; }

   TGLVector3 Norm() const { return TGLVector3( fVals[0], fVals[1], fVals[2]); }
   Double_t DistanceTo(const TGLVertex3 & vertex) const;
   TGLVertex3 NearestOn(const TGLVertex3 & point) const;

   void Negate();
   
   const Double_t * CArr() const { return fVals; }
   Double_t * Arr() { return fVals; }

   void Dump() const;

   ClassDef(TGLPlane,0) // GL plane helper/wrapper class
};

typedef std::vector<TGLPlane> TGLPlaneSet_t;

//______________________________________________________________________________
inline void TGLPlane::Set(const TGLPlane & other)
{
   fVals[0] = other.fVals[0];
   fVals[1] = other.fVals[1];
   fVals[2] = other.fVals[2];
   fVals[3] = other.fVals[3];
}

//______________________________________________________________________________
inline void TGLPlane::Set(Double_t a, Double_t b, Double_t c, Double_t d)
{
   fVals[0] = a;
   fVals[1] = b;
   fVals[2] = c;
   fVals[3] = d;
   Normalise();
}

//______________________________________________________________________________
inline void TGLPlane::Set(Double_t eq[4])
{
   fVals[0] = eq[0];
   fVals[1] = eq[1];
   fVals[2] = eq[2];
   fVals[3] = eq[3];
   Normalise();
}
   
//______________________________________________________________________________
inline void TGLPlane::Set(const TGLVector3 & norm, const TGLVertex3 & point)
{
   // Set plane from a normal vector and in-plane point pair
   fVals[0] = norm[0];
   fVals[1] = norm[1];
   fVals[2] = norm[2];
   fVals[3] = -(fVals[0]*point[0] + fVals[1]*point[1] + fVals[2]*point[2]);
   Normalise();
}

//______________________________________________________________________________
inline void TGLPlane::Set(const TGLVertex3 & p1, const TGLVertex3 & p2, const TGLVertex3 & p3)
{
   TGLVector3 norm = Cross(p2 - p1, p3 - p1);
   Set(norm, p2);
}

//______________________________________________________________________________
inline void TGLPlane::Negate()
{
   fVals[0] = -fVals[0];
   fVals[1] = -fVals[1];
   fVals[2] = -fVals[2];
   fVals[3] = -fVals[3];
}

//______________________________________________________________________________
inline void TGLPlane::Normalise()
{
   Double_t mag = sqrt( fVals[0]*fVals[0] + fVals[1]*fVals[1] + fVals[2]*fVals[2] );

   if ( mag == 0.0 ) {
      assert( kFALSE );
      return;
   }

   fVals[0] /= mag;
   fVals[1] /= mag;
   fVals[2] /= mag;
   fVals[3] /= mag;
}

//______________________________________________________________________________
inline Double_t TGLPlane::DistanceTo(const TGLVertex3 & vertex) const
{
   return (fVals[0]*vertex[0] + fVals[1]*vertex[1] + fVals[2]*vertex[2] + fVals[3]);
}

inline TGLVertex3 TGLPlane::NearestOn(const TGLVertex3 & point) const
{
   TGLVector3 o = Norm() * (Dot(Norm(), TGLVector3(point[0], point[1], point[2])) + D() / Dot(Norm(), Norm()));
   TGLVertex3 v = point - o;
   return v;
}

//______________________________________________________________________________
inline TGLVertex3 Intersection(const TGLPlane & p1, const TGLPlane & p2, const TGLPlane & p3)
{
   Double_t m = Dot(p1.Norm(), Cross(p2.Norm(), p3.Norm()));
   TGLVector3 v = (Cross(p2.Norm(),p3.Norm())* -p1.D()) - (Cross(p3.Norm(),p1.Norm())*p2.D()) - (Cross(p1.Norm(),p2.Norm())*p3.D());
   return v / m;
}

/*************************************************************************
 * TGLMatrix - TODO
 *
 *
 *
 *************************************************************************/
class TGLMatrix
{
private:
   // Fields
   Double_t fVals[16]; // Column MAJOR as per OGL

   // Methods
   Bool_t ValidIndex(UInt_t index) const { return (index < 16); }

public:
   TGLMatrix();
   TGLMatrix(Double_t x, Double_t y, Double_t z);
   TGLMatrix(const TGLVertex3 & translation);
   TGLMatrix(const TGLVertex3 & origin, const TGLVector3 & zAxis);
   TGLMatrix(const Double_t vals[16]);
   TGLMatrix(const TGLMatrix & other);
   virtual ~TGLMatrix(); // ClassDef introduces virtual fns

   const TGLMatrix & operator =(const TGLMatrix & rhs);

   Double_t & operator [] (Int_t index);

   void Set(Double_t x, Double_t y, Double_t z);
   void Set(const TGLVertex3 & translation);
   void Set(const TGLVertex3 & origin, const TGLVector3 & zAxis);
   void Set(const Double_t vals[16]);
   void SetIdentity();

   void        Shift(const TGLVector3 & shift);

   TGLVertex3  Translation() const;
   TGLVector3  Scale() const;
   void        SetScale(const TGLVector3 & scale);

   void TransformVertex(TGLVertex3 & vertex) const;
	void Transpose3x3();

   void Dump() const;

   const Double_t * CArr() const { return fVals; }
   Double_t * Arr() { return fVals; }

   ClassDef(TGLMatrix,0) // GL matrix helper/wrapper class
};

//______________________________________________________________________________
inline const TGLMatrix & TGLMatrix::operator =(const TGLMatrix & rhs) 
{
   // Check for self-assignment
   if (this != &rhs) {
      Set(rhs.fVals); 
   }
   return *this; 
}

//______________________________________________________________________________
inline Double_t & TGLMatrix::operator [] (Int_t index)
{ 
   /*if (!ValidIndex(index)) { 
      assert(kFALSE); 
      return fVals[0]; 
   } else {*/
      return fVals[index]; 
   //} 
}

/*************************************************************************
 * TGLUtil - TODO
 *
 *
 *
 *************************************************************************/
class TGLUtil
{
public:
   virtual ~TGLUtil() { }

   static void   CheckError();
   ClassDef(TGLUtil,0) // Wrapper class for misc GL pieces
};

#endif // ROOT_TGLUtil
