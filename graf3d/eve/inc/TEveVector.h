// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveVector
#define ROOT_TEveVector

#include "TMath.h"

class TVector3;

//==============================================================================
// TEveVector
//==============================================================================

class TEveVector
{
public:
   Float_t fX, fY, fZ; // Components of the vector.

   TEveVector() : fX(0), fY(0), fZ(0) {}
   TEveVector(const Float_t* v)  : fX(v[0]), fY(v[1]), fZ(v[2]) {}
   TEveVector(const Double_t* v) : fX(v[0]), fY(v[1]), fZ(v[2]) {}
   TEveVector(Float_t x, Float_t y, Float_t z) : fX(x), fY(y), fZ(z) {}
   ~TEveVector() {}

   void Dump() const;

   operator const Float_t*() const { return &fX; }
   operator       Float_t*()       { return &fX; }

   TEveVector& operator *=(Float_t s)           { fX *= s;    fY *= s;    fZ *= s;    return *this; }
   TEveVector& operator +=(const TEveVector& v) { fX += v.fX; fY += v.fY; fZ += v.fZ; return *this; }
   TEveVector& operator -=(const TEveVector& v) { fX -= v.fX; fY -= v.fY; fZ -= v.fZ; return *this; }

   TEveVector operator + (const TEveVector &) const;
   TEveVector operator - (const TEveVector &) const;
   TEveVector operator * (Float_t a) const;

   Float_t& operator [] (Int_t indx);
   Float_t  operator [] (Int_t indx) const;

   const Float_t* Arr() const { return &fX; }
   Float_t* Arr()       { return &fX; }

   void Set(const Float_t*  v) { fX = v[0]; fY = v[1]; fZ = v[2]; }
   void Set(const Double_t* v) { fX = v[0]; fY = v[1]; fZ = v[2]; }
   void Set(Float_t  x, Float_t  y, Float_t  z) { fX = x; fY = y; fZ = z; }
   void Set(Double_t x, Double_t y, Double_t z) { fX = x; fY = y; fZ = z; }
   void Set(const TEveVector& v) { fX = v.fX;  fY = v.fY;  fZ = v.fZ;  }
   void Set(const TVector3& v);

   void NegateXYZ() { fX = - fX; fY = -fY; fZ = -fZ; }
   void Normalize(Float_t length=1);

   Float_t Phi()      const;
   Float_t Theta()    const;
   Float_t CosTheta() const;
   Float_t Eta()      const;

   Float_t Mag()  const { return TMath::Sqrt(fX*fX + fY*fY + fZ*fZ);}
   Float_t Mag2() const { return fX*fX + fY*fY + fZ*fZ;}

   Float_t Perp()  const { return TMath::Sqrt(fX*fX + fY*fY);}
   Float_t Perp2() const { return fX*fX + fY*fY;}
   Float_t R()     const { return Perp(); }

   Float_t Distance(const TEveVector& v) const;
   Float_t SquareDistance(const TEveVector& v) const;

   Float_t    Dot(const TEveVector& a) const;
   TEveVector Cross(const TEveVector& a) const;

   TEveVector& Sub(const TEveVector& p, const TEveVector& q);

   TEveVector& Mult(const TEveVector& a, Float_t af);

   TEveVector Orthogonal() const;
   void       OrthoNormBase(TEveVector& a, TEveVector& b) const;

   ClassDefNV(TEveVector, 1); // Float three-vector; a minimal Float_t copy of TVector3 used to represent points and momenta (also used in VSD).
};

//______________________________________________________________________________
inline Float_t TEveVector::Phi() const
{
   return fX == 0.0 && fY == 0.0 ? 0.0 : TMath::ATan2(fY, fX);
}

//______________________________________________________________________________
inline Float_t TEveVector::Theta() const
{
   return fX == 0.0 && fY == 0.0 && fZ == 0.0 ? 0.0 : TMath::ATan2(Perp(), fZ);
}

//______________________________________________________________________________
inline Float_t TEveVector::CosTheta() const
{
   Float_t ptot = Mag(); return ptot == 0.0 ? 1.0 : fZ/ptot;
}

//______________________________________________________________________________
inline Float_t TEveVector::Distance( const TEveVector& b) const
{
   return TMath::Sqrt((fX - b.fX)*(fX - b.fX) +
                      (fY - b.fY)*(fY - b.fY) +
                      (fZ - b.fZ)*(fZ - b.fZ));
}

//______________________________________________________________________________
inline Float_t TEveVector::SquareDistance(const TEveVector& b) const
{
   return ((fX - b.fX) * (fX - b.fX) +
           (fY - b.fY) * (fY - b.fY) +
           (fZ - b.fZ) * (fZ - b.fZ));
}

//______________________________________________________________________________
inline Float_t TEveVector::Dot(const TEveVector& a) const
{
   return a.fX*fX + a.fY*fY + a.fZ*fZ;
}

//______________________________________________________________________________
inline TEveVector TEveVector::Cross(const TEveVector& a) const
{
   TEveVector r;
   r.fX = fY * a.fZ - fZ * a.fY;
   r.fY = fZ * a.fX - fX * a.fZ;
   r.fZ = fX * a.fY - fY * a.fX;
   return r;
}

//______________________________________________________________________________
inline TEveVector& TEveVector::Sub(const TEveVector& p, const TEveVector& q)
{
   fX = p.fX - q.fX;
   fY = p.fY - q.fY;
   fZ = p.fZ - q.fZ;
   return *this;
}

//______________________________________________________________________________
inline TEveVector& TEveVector::Mult(const TEveVector& a, Float_t af)
{
   fX = a.fX * af;
   fY = a.fY * af;
   fZ = a.fZ * af;
   return *this;
}

//______________________________________________________________________________
inline Float_t& TEveVector::operator [] (Int_t idx)
{
   return (&fX)[idx];
}

//______________________________________________________________________________
inline Float_t TEveVector::operator [] (Int_t idx) const
{
   return (&fX)[idx];
}


//==============================================================================
// TEveVector4
//==============================================================================

class TEveVector4 : public TEveVector
{
public:
   Float_t fT;

   TEveVector4()                    : TEveVector(),  fT(0) {}
   TEveVector4(const TEveVector& v) : TEveVector(v), fT(0) {}
   TEveVector4(Float_t x, Float_t y, Float_t z, Float_t t=0) :
      TEveVector(x, y, z), fT(t) {}
   ~TEveVector4() {}

   void Dump() const;

   TEveVector4 operator + (const TEveVector4 & b) const
   { return TEveVector4(fX + b.fX, fY + b.fY, fZ + b.fZ, fT + b.fT); }

   TEveVector4 operator - (const TEveVector4 & b) const
   { return TEveVector4(fX - b.fX, fY - b.fY, fZ - b.fZ, fT - b.fT); }

   TEveVector4 operator * (Float_t a) const
   { return TEveVector4(a*fX, a*fY, a*fZ, a*fT); }

   TEveVector4& operator += (const TEveVector4 & b)
   { fX += b.fX; fY += b.fY; fZ += b.fZ; fT += b.fT; return *this; }

   ClassDefNV(TEveVector4, 1); // Float four-vector.
};


//==============================================================================
// TEvePoint
//==============================================================================

class TEvePoint
{
public:
   Float_t fX, fY; // Components of the point.

   TEvePoint() : fX(0), fY(0) {}
   TEvePoint(const Float_t* v)  : fX(v[0]), fY(v[1]) {}
   TEvePoint(const Double_t* v) : fX(v[0]), fY(v[1]) {}
   TEvePoint(Float_t x, Float_t y) : fX(x), fY(y)    {}
   ~TEvePoint() {}

   void Dump() const;

   operator const Float_t*() const { return &fX; }
   operator       Float_t*()       { return &fX; }

   TEvePoint& operator *=(Float_t s)          { fX *= s;    fY *= s;    return *this; }
   TEvePoint& operator +=(const TEvePoint& v) { fX += v.fX; fY += v.fY; return *this; }
   TEvePoint& operator -=(const TEvePoint& v) { fX -= v.fX; fY -= v.fY; return *this; }

   TEvePoint operator + (const TEvePoint &) const;
   TEvePoint operator - (const TEvePoint &) const;
   TEvePoint operator * (Float_t a) const;

   Float_t& operator [] (Int_t indx);
   Float_t  operator [] (Int_t indx) const;

   const Float_t* Arr() const { return &fX; }
   Float_t* Arr()       { return &fX; }

   void Set(const Float_t*  v) { fX = v[0]; fY = v[1]; }
   void Set(const Double_t* v) { fX = v[0]; fY = v[1]; }
   void Set(Float_t  x, Float_t  y) { fX = x; fY = y; }
   void Set(Double_t x, Double_t y) { fX = x; fY = y; }
   void Set(const TEvePoint& v) { fX = v.fX;  fY = v.fY;  }

   void NegateXY() { fX = - fX; fY = -fY; }
   void Normalize(Float_t length=1);

   Float_t Phi()  const;

   Float_t Mag()  const { return TMath::Sqrt(fX*fX + fY*fY);}
   Float_t Mag2() const { return fX*fX + fY*fY;}

   Float_t Distance(const TEvePoint& v) const;
   Float_t SquareDistance(const TEvePoint& v) const;

   Float_t    Dot(const TEvePoint& a) const;
   Float_t    Cross(const TEvePoint& a) const;

   TEvePoint& Sub(const TEvePoint& p, const TEvePoint& q);

   TEvePoint& Mult(const TEvePoint& a, Float_t af);

   ClassDefNV(TEvePoint, 1); // Float two-vector.
};

//______________________________________________________________________________
inline Float_t TEvePoint::Phi() const
{
   return fX == 0.0 && fY == 0.0 ? 0.0 : TMath::ATan2(fY, fX);
}

//______________________________________________________________________________
inline Float_t TEvePoint::Distance( const TEvePoint& b) const
{
   return TMath::Sqrt((fX - b.fX)*(fX - b.fX) +
                      (fY - b.fY)*(fY - b.fY));
}

//______________________________________________________________________________
inline Float_t TEvePoint::SquareDistance(const TEvePoint& b) const
{
   return ((fX - b.fX) * (fX - b.fX) +
           (fY - b.fY) * (fY - b.fY));
}

//______________________________________________________________________________
inline Float_t TEvePoint::Dot(const TEvePoint& a) const
{
   return a.fX*fX + a.fY*fY;
}

//______________________________________________________________________________
inline Float_t TEvePoint::Cross(const TEvePoint& a) const
{
   return fX * a.fY - fY * a.fX;
}

//______________________________________________________________________________
inline TEvePoint& TEvePoint::Sub(const TEvePoint& p, const TEvePoint& q)
{
   fX = p.fX - q.fX;
   fY = p.fY - q.fY;
   return *this;
}

//______________________________________________________________________________
inline TEvePoint& TEvePoint::Mult(const TEvePoint& a, Float_t af)
{
   fX = a.fX * af;
   fY = a.fY * af;
   return *this;
}

//______________________________________________________________________________
inline Float_t& TEvePoint::operator [] (Int_t idx)
{
   return (&fX)[idx];
}

//______________________________________________________________________________
inline Float_t TEvePoint::operator [] (Int_t idx) const
{
   return (&fX)[idx];
}


//==============================================================================
// TEvePathMark
//==============================================================================

class TEvePathMark
{
public:
   enum EType_e   { kReference, kDaughter, kDecay, kCluster2D };

   EType_e     fType; // Mark-type.
   TEveVector  fV;    // Vertex.
   TEveVector  fP;    // Momentum.
   TEveVector  fE;    // Extra, meaning depends on fType.
   Float_t     fTime; // Time.

   TEvePathMark(EType_e type=kReference) :
      fType(type), fV(), fP(), fE(), fTime(0) {}

   TEvePathMark(EType_e type, const TEveVector& v, Float_t time=0) :
      fType(type), fV(v), fP(), fE(), fTime(time) {}

   TEvePathMark(EType_e type, const TEveVector& v, const TEveVector& p, Float_t time=0) :
      fType(type), fV(v), fP(p), fE(), fTime(time) {}

   TEvePathMark(EType_e type, const TEveVector& v, const TEveVector& p, const TEveVector& e, Float_t time=0) :
      fType(type), fV(v), fP(p), fE(e), fTime(time) {}

   ~TEvePathMark() {}

   const char* TypeName();

   ClassDefNV(TEvePathMark, 1); // Special-point on track: position/momentum reference, daughter creation or decay (also used in VSD).
};

#endif
