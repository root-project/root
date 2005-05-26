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

// TODO: Namespace
// TODO: Mark all headers as INTERNAL

/*************************************************************************
 * TGLVertex3 - TODO
 *
 *
 *
 *************************************************************************/
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
      
   TGLVertex3 & operator=(const TGLVertex3 & rhs) { Set(rhs); return *this; }
   const TGLVertex3 & operator - ()
   { fVals[0] = -fVals[0]; fVals[1] = -fVals[1]; fVals[2] = -fVals[2]; return *this; }
   
   void Fill(Double_t val) { Set(val,val,val); }
   void Set(Double_t x, Double_t y, Double_t z) { fVals[0]=x; fVals[1]=y; fVals[2]=z; }
   void Set(const TGLVertex3 & other) { fVals[0]=other.fVals[0]; fVals[1]=other.fVals[1]; fVals[2]=other.fVals[2]; }
   
   // Accessors
   Double_t & operator [] (Int_t index) 
   { if (!ValidIndex(index)) { assert(kFALSE); return fVals[0]; } else { return fVals[index]; } }
   const Double_t& operator [] (Int_t index) const
   { if (!ValidIndex(index)) { assert(kFALSE); return fVals[0]; } else { return fVals[index]; } }
   Double_t   X() const { return fVals[0]; }
   Double_t & X()       { return fVals[0]; }
   Double_t   Y() const { return fVals[1]; }
   Double_t & Y()       { return fVals[1]; }
   Double_t   Z() const { return fVals[2]; }
   Double_t & Z()       { return fVals[2]; }
   
   const Double_t * CArr() const { return fVals; }
   Double_t * Arr() { return fVals; }

   void Dump() const;

   ClassDef(TGLVertex3,0) // GL 3 component vertex helper/wrapper class
};

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
   
   const TGLVector3 & operator -= (const TGLVector3 & val)
   { fVals[0] -= val[0]; fVals[1] -= val[1]; fVals[2] -= val[2]; return *this; }
   const TGLVector3 & operator /= (Double_t val)
   { fVals[0] /= val; fVals[1] /= val; fVals[2] /= val; return *this; }
   const TGLVector3 & operator *= (Double_t val)
   { fVals[0] *= val; fVals[1] *= val; fVals[2] *= val; return *this; }

   inline Double_t Mag() const;
   inline void Normalise();

   ClassDef(TGLVector3,0) // GL 3 component vector helper/wrapper class
};

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
inline const TGLVector3 operator / (const TGLVector3 & vector, Double_t val)
{ 
   return TGLVector3(vector[0] / val, vector[1] / val, vector[2] / val);
}

//______________________________________________________________________________
inline const TGLVector3 operator * (const TGLVector3 & vector, Double_t val)
{    
   return TGLVector3(vector[0] * val, vector[1] * val, vector[2] * val);
}

//______________________________________________________________________________
// Vertex + Vector => Vertex
inline TGLVertex3 operator + (const TGLVertex3 & vertex, const TGLVector3 & vector)
{
   return TGLVertex3(vertex[0] + vector[0], vertex[1] + vector[1], vertex[2] + vector[2]);
}

//______________________________________________________________________________
// Vertex - Vertex => Vector
inline TGLVector3 operator - (const TGLVertex3 & vertex, const TGLVertex3 & vector)
{
   return TGLVector3(vertex[0] - vector[0], vertex[1] - vector[1], vertex[2] - vector[2]);
}

//______________________________________________________________________________
// Vector + Vector => Vector
inline TGLVector3 operator + (const TGLVector3 & first, const TGLVector3 & second)
{
   return TGLVector3(first[0] + second[0], first[1] + second[1], first[2] + second[2]);
}

//______________________________________________________________________________
// Vector - Vector => Vector
inline TGLVector3 operator - (const TGLVector3 & first, const TGLVector3 & second)
{
   return TGLVector3(first[0] - second[0], first[1] - second[1], first[2] - second[2]);
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
   inline void Set(Int_t x, Int_t y, UInt_t width, UInt_t height);
   inline void SetCorner(Int_t x, Int_t y);
   inline void Offset(Int_t dX, Int_t dY);
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
   TGLPlane(Double_t a, Double_t b, Double_t c, Double_t d, Bool_t norm = kTRUE);
   TGLPlane(Double_t eq[4], Bool_t norm = kTRUE);
   virtual ~TGLPlane(); // ClassDef introduces virtual fns
   
   //inline const TGLPlane& operator *= (Double_t val);
   inline void Set(Double_t a, Double_t b, Double_t c, Double_t d, Bool_t norm = kTRUE);
   inline void Set(Double_t eq[4], Bool_t norm = kTRUE);

   Double_t A() const { return fVals[0]; }
   Double_t B() const { return fVals[1]; }
   Double_t C() const { return fVals[2]; }
   Double_t D() const { return fVals[3]; }

   TGLVector3 Norm() const { return TGLVector3( fVals[0], fVals[1], fVals[2]); }
   inline Double_t DistanceTo(const TGLVertex3 & vertex) const;
   inline TGLVertex3 NearestOn(const TGLVertex3 & point) const;

   const Double_t * CArr() const { return fVals; }
   Double_t * Arr() { return fVals; }

   ClassDef(TGLPlane,0) // GL plane helper/wrapper class
};

//______________________________________________________________________________
inline void TGLPlane::Set(Double_t a, Double_t b, Double_t c, Double_t d, Bool_t norm)
{
   fVals[0] = a;
   fVals[1] = b;
   fVals[2] = c;
   fVals[3] = d;
   if (norm) {
      Normalise();
   }
}

//______________________________________________________________________________
inline void TGLPlane::Set(Double_t eq[4], Bool_t norm)
{
   fVals[0] = eq[0];
   fVals[1] = eq[1];
   fVals[2] = eq[2];
   fVals[3] = eq[3];
   if (norm) {
      Normalise();
   }
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
   TGLMatrix(const Double_t vals[16]);
   TGLMatrix(const TGLMatrix & other);
   virtual ~TGLMatrix(); // ClassDef introduces virtual fns

   const TGLMatrix & operator=(const TGLMatrix & rhs) { Set(rhs.fVals); return *this; }
   //const TGLMatrix & operator*(const TGLMatrix & rhs);

   Double_t & operator [] (Int_t index) 
   { if (!ValidIndex(index)) { assert(kFALSE); return fVals[0]; } else { return fVals[index]; } }

   void Set(const Double_t vals[16]);
   //void Fill(Double_t val);

   void SetIdentity();
   void SetTranslation(Double_t x, Double_t y, Double_t z);
   void TransformVertex(TGLVertex3 & vertex) const;
   
	void InvRot();
	
   void Dump() const;
      
   const Double_t * CArr() const { return fVals; }
   Double_t * Arr() { return fVals; }
   
   ClassDef(TGLMatrix,0) // GL matrix helper/wrapper class
};

/*************************************************************************
 * TGLUtil - TODO
 *
 *
 *
 *************************************************************************/
class TGLUtil
{
public:
   static void CheckError();

   ClassDef(TGLUtil,0) // Wrapper class for misc GL pieces 
};

#endif // ROOT_TGLUtil
