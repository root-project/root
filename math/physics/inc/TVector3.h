// @(#)root/physics:$Id$
// Author: Pasha Murat, Peter Malzacher   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVector3
#define ROOT_TVector3

#include "TError.h"
#include "TVector2.h"
#include "TMatrix.h"
#include "TMath.h"

class TRotation;


class TVector3 : public TObject {

public:

   typedef Double_t Scalar;   // to be able to use it with the ROOT::Math::VectorUtil functions

   TVector3();

   TVector3(Double_t x, Double_t y, Double_t z);
   // The constructor.

   TVector3(const Double_t *);
   TVector3(const Float_t *);
   // Constructors from an array

   TVector3(const TVector3 &);
   // The copy constructor.

   virtual ~TVector3() {};
   // Destructor

   Double_t operator () (int) const;
   inline Double_t operator [] (int) const;
   // Get components by index (Geant4).

   Double_t & operator () (int);
   inline Double_t & operator [] (int);
   // Set components by index.

   inline Double_t x()  const;
   inline Double_t y()  const;
   inline Double_t z()  const;
   inline Double_t X()  const;
   inline Double_t Y()  const;
   inline Double_t Z()  const;
   inline Double_t Px() const;
   inline Double_t Py() const;
   inline Double_t Pz() const;
   // The components in cartesian coordinate system.

   inline void SetX(Double_t);
   inline void SetY(Double_t);
   inline void SetZ(Double_t);
   inline void SetXYZ(Double_t x, Double_t y, Double_t z);
   void        SetPtEtaPhi(Double_t pt, Double_t eta, Double_t phi);
   void        SetPtThetaPhi(Double_t pt, Double_t theta, Double_t phi);

   inline void GetXYZ(Double_t *carray) const;
   inline void GetXYZ(Float_t *carray) const;
   // Get the components into an array
   // not checked!

   Double_t Phi() const;
   // The azimuth angle. returns phi from -pi to pi

   Double_t Theta() const;
   // The polar angle.
   
   void ToPolar();
   // Convert a Cartesian Vector to a Spherical Vector
   void ToCartesian();
   // Convert a Spherical Vector to a Cartesian Vector

   inline Double_t CosTheta() const;
   // Cosine of the polar angle.

   inline Double_t Mag2() const;
   // The magnitude squared (rho^2 in spherical coordinate system).

   Double_t Mag() const { return TMath::Sqrt(Mag2()); }
   // The magnitude (rho in spherical coordinate system).

   void SetPhi(Double_t);
   // Set phi keeping mag and theta constant (BaBar).

   void SetTheta(Double_t);
   // Set theta keeping mag and phi constant (BaBar).

   inline void SetMag(Double_t);
   // Set magnitude keeping theta and phi constant (BaBar).

   inline Double_t Perp2() const;
   // The transverse component squared (R^2 in cylindrical coordinate system).

   inline Double_t Pt() const;
   Double_t Perp() const;
   // The transverse component (R in cylindrical coordinate system).

   inline void SetPerp(Double_t);
   // Set the transverse component keeping phi and z constant.

   inline Double_t Perp2(const TVector3 &) const;
   // The transverse component w.r.t. given axis squared.

   inline Double_t Pt(const TVector3 &) const;
   Double_t Perp(const TVector3 &) const;
   // The transverse component w.r.t. given axis.

   inline Double_t DeltaPhi(const TVector3 &) const;
   Double_t DeltaR(const TVector3 &) const;
   inline Double_t DrEtaPhi(const TVector3 &) const;
   inline TVector2 EtaPhiVector() const;
   void SetMagThetaPhi(Double_t mag, Double_t theta, Double_t phi);

   inline TVector3 & operator = (const TVector3 &);
   // Assignment.

   inline Bool_t operator == (const TVector3 &) const;
   inline Bool_t operator != (const TVector3 &) const;
   // Comparisons (Geant4).

   inline TVector3 & operator += (const TVector3 &);
   // Addition.

   inline TVector3 & operator -= (const TVector3 &);
   // Subtraction.

   inline TVector3 operator - () const;
   // Unary minus.

   inline TVector3 & operator *= (Double_t);
   // Scaling with real numbers.

   TVector3 Unit() const;
   // Unit vector parallel to this.

   inline TVector3 Orthogonal() const;
   // Vector orthogonal to this (Geant4).

   inline Double_t Dot(const TVector3 &) const;
   // Scalar product.

   inline TVector3 Cross(const TVector3 &) const;
   // Cross product.

   Double_t Angle(const TVector3 &) const;
   // The angle w.r.t. another 3-vector.

   Double_t PseudoRapidity() const;
   // Returns the pseudo-rapidity, i.e. -ln(tan(theta/2))

   inline Double_t Eta() const;

   void RotateX(Double_t);
   // Rotates the Hep3Vector around the x-axis.

   void RotateY(Double_t);
   // Rotates the Hep3Vector around the y-axis.

   void RotateZ(Double_t);
   // Rotates the Hep3Vector around the z-axis.

   void RotateUz(const TVector3&);
   // Rotates reference frame from Uz to newUz (unit vector) (Geant4).

   void Rotate(Double_t, const TVector3 &);
   // Rotates around the axis specified by another Hep3Vector.

   TVector3 & operator *= (const TRotation &);
   TVector3 & Transform(const TRotation &);
   // Transformation with a Rotation matrix.

   inline TVector2 XYvector() const;

   void Print(Option_t* option="") const;

private:

   Double_t fX, fY, fZ;
   // The components.

   ClassDef(TVector3,3) // A 3D physics vector

   // make TLorentzVector a friend class
   friend class TLorentzVector;
};

TVector3 operator + (const TVector3 &, const TVector3 &);
// Addition of 3-vectors.

TVector3 operator - (const TVector3 &, const TVector3 &);
// Subtraction of 3-vectors.

Double_t operator * (const TVector3 &, const TVector3 &);
// Scalar product of 3-vectors.

TVector3 operator * (const TVector3 &, Double_t a);
TVector3 operator * (Double_t a, const TVector3 &);
// Scaling of 3-vectors with a real number

TVector3 operator * (const TMatrix &, const TVector3 &);


inline Double_t & TVector3::operator[] (int i)       { return operator()(i); }
inline Double_t   TVector3::operator[] (int i) const { return operator()(i); }

inline Double_t TVector3::x()  const { return fX; }
inline Double_t TVector3::y()  const { return fY; }
inline Double_t TVector3::z()  const { return fZ; }
inline Double_t TVector3::X()  const { return fX; }
inline Double_t TVector3::Y()  const { return fY; }
inline Double_t TVector3::Z()  const { return fZ; }
inline Double_t TVector3::Px() const { return fX; }
inline Double_t TVector3::Py() const { return fY; }
inline Double_t TVector3::Pz() const { return fZ; }

inline void TVector3::SetX(Double_t xx) { fX = xx; }
inline void TVector3::SetY(Double_t yy) { fY = yy; }
inline void TVector3::SetZ(Double_t zz) { fZ = zz; }

inline void TVector3::SetXYZ(Double_t xx, Double_t yy, Double_t zz) {
   fX = xx;
   fY = yy;
   fZ = zz;
}

inline void TVector3::GetXYZ(Double_t *carray) const {
   carray[0] = fX;
   carray[1] = fY;
   carray[2] = fZ;
}

inline void TVector3::GetXYZ(Float_t *carray) const {
   carray[0] = fX;
   carray[1] = fY;
   carray[2] = fZ;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructors
inline TVector3::TVector3()
: fX(0.0), fY(0.0), fZ(0.0) {}

inline TVector3::TVector3(const TVector3 & p) : TObject(p),
  fX(p.fX), fY(p.fY), fZ(p.fZ) {}

inline TVector3::TVector3(Double_t xx, Double_t yy, Double_t zz)
: fX(xx), fY(yy), fZ(zz) {}

inline TVector3::TVector3(const Double_t * x0)
: fX(x0[0]), fY(x0[1]), fZ(x0[2]) {}

inline TVector3::TVector3(const Float_t * x0)
: fX(x0[0]), fY(x0[1]), fZ(x0[2]) {}


inline Double_t TVector3::operator () (int i) const {
   switch(i) {
      case 0:
         return fX;
      case 1:
         return fY;
      case 2:
         return fZ;
      default:
         Error("operator()(i)", "bad index (%d) returning 0",i);
   }
   return 0.;
}

inline Double_t & TVector3::operator () (int i) {
   switch(i) {
      case 0:
         return fX;
      case 1:
         return fY;
      case 2:
         return fZ;
      default:
         Error("operator()(i)", "bad index (%d) returning &fX",i);
   }
   return fX;
}

inline TVector3 & TVector3::operator = (const TVector3 & p) {
   fX = p.fX;
   fY = p.fY;
   fZ = p.fZ;
   return *this;
}

inline Bool_t TVector3::operator == (const TVector3& v) const {
   return (v.fX==fX && v.fY==fY && v.fZ==fZ) ? kTRUE : kFALSE;
}

inline Bool_t TVector3::operator != (const TVector3& v) const {
   return (v.fX!=fX || v.fY!=fY || v.fZ!=fZ) ? kTRUE : kFALSE;
}

inline TVector3& TVector3::operator += (const TVector3 & p) {
   fX += p.fX;
   fY += p.fY;
   fZ += p.fZ;
   return *this;
}

inline TVector3& TVector3::operator -= (const TVector3 & p) {
   fX -= p.fX;
   fY -= p.fY;
   fZ -= p.fZ;
   return *this;
}

inline TVector3 TVector3::operator - () const {
   return TVector3(-fX, -fY, -fZ);
}

inline TVector3& TVector3::operator *= (Double_t a) {
   fX *= a;
   fY *= a;
   fZ *= a;
   return *this;
}

inline Double_t TVector3::Dot(const TVector3 & p) const {
   return fX*p.fX + fY*p.fY + fZ*p.fZ;
}

inline TVector3 TVector3::Cross(const TVector3 & p) const {
   return TVector3(fY*p.fZ-p.fY*fZ, fZ*p.fX-p.fZ*fX, fX*p.fY-p.fX*fY);
}

inline Double_t TVector3::Mag2() const { return fX*fX + fY*fY + fZ*fZ; }


inline TVector3 TVector3::Orthogonal() const {
   Double_t xx = fX < 0.0 ? -fX : fX;
   Double_t yy = fY < 0.0 ? -fY : fY;
   Double_t zz = fZ < 0.0 ? -fZ : fZ;
   if (xx < yy) {
      return xx < zz ? TVector3(0,fZ,-fY) : TVector3(fY,-fX,0);
   } else {
      return yy < zz ? TVector3(-fZ,0,fX) : TVector3(fY,-fX,0);
   }
}

inline Double_t TVector3::Perp2() const { return fX*fX + fY*fY; }


inline Double_t TVector3::Pt() const { return Perp(); }

inline Double_t TVector3::Perp2(const TVector3 & p)  const {
   Double_t tot = p.Mag2();
   Double_t ss  = Dot(p);
   Double_t per = Mag2();
   if (tot > 0.0) per -= ss*ss/tot;
   if (per < 0)   per = 0;
   return per;
}

inline Double_t TVector3::Pt(const TVector3 & p) const {
   return Perp(p);
}

inline Double_t TVector3::CosTheta() const {
   Double_t ptot = Mag();
   return ptot == 0.0 ? 1.0 : fZ/ptot;
}

inline void TVector3::SetMag(Double_t ma) {
   Double_t factor = Mag();
   if (factor == 0) {
      Warning("SetMag","zero vector can't be stretched");
   } else {
      factor = ma/factor;
      SetX(fX*factor);
      SetY(fY*factor);
      SetZ(fZ*factor);
   }
}

inline void TVector3::SetPerp(Double_t r) {
   Double_t p = Perp();
   if (p != 0.0) {
      fX *= r/p;
      fY *= r/p;
   }
}

inline Double_t TVector3::DeltaPhi(const TVector3 & v) const {
   return TVector2::Phi_mpi_pi(Phi()-v.Phi());
}

inline Double_t TVector3::Eta() const {
   return PseudoRapidity();
}

inline Double_t TVector3::DrEtaPhi(const TVector3 & v) const{
   return DeltaR(v);
}


inline TVector2 TVector3::EtaPhiVector() const {
   return TVector2 (Eta(),Phi());
}

inline TVector2 TVector3::XYvector() const {
   return TVector2(fX,fY);
}


#endif
