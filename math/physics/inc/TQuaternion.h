// @(#)root/physics:$Id$
// Author: Eric Anciant 28/06/2005

#ifndef ROOT_TQuaternion
#define ROOT_TQuaternion

#include "TVector3.h"
#include "TMath.h"


class TQuaternion : public TObject {

public:

   TQuaternion(Double_t real = 0, Double_t X = 0, Double_t Y = 0, Double_t Z = 0);
   TQuaternion(const TVector3 & vector, Double_t real = 0);
   TQuaternion(const Double_t *);
   TQuaternion(const Float_t *);
   // Constructors from an array : 0 to 2 = vector part, 3 = real part

   TQuaternion(const TQuaternion &);
   // The copy constructor.

   ~TQuaternion() override;
   // Destructor

   Double_t operator () (int) const;
   inline Double_t operator [] (int) const;
   // Get components by index. 0 to 2 = vector part, 3 = real part

   Double_t & operator () (int);
   inline Double_t & operator [] (int);
   // Set components by index. 0 to 2 = vector part, 3 = real part

   inline TQuaternion& SetRXYZ(Double_t r,Double_t x,Double_t y,Double_t z);
   inline TQuaternion& SetRV(Double_t r, TVector3& vect);
   // Sets components
   TQuaternion& SetAxisQAngle(TVector3& v,Double_t QAngle);
   // Set from vector direction and quaternion angle
   Double_t GetQAngle() const;
   TQuaternion& SetQAngle(Double_t angle);
   // set and get quaternion angle

   inline void GetRXYZ(Double_t *carray) const;
   inline void GetRXYZ(Float_t *carray) const;
   // Get the components into an array : 0 to 2 vector part, 3 real part
   // not checked!

   // ---------------  real to quaternion algebra
   inline TQuaternion& operator=(Double_t r);
   inline Bool_t operator == (Double_t r) const;
   inline Bool_t operator != (Double_t r) const;
   inline TQuaternion& operator+=(Double_t real);
   inline TQuaternion& operator-=(Double_t real);
   inline TQuaternion& operator*=(Double_t real);
   inline TQuaternion& operator/=(Double_t real);
   TQuaternion operator*(Double_t real) const;
   TQuaternion operator+(Double_t real) const;
   TQuaternion operator-(Double_t real) const;
   TQuaternion operator/(Double_t real) const;

   // ---------------- vector to quaternion algebra
   inline TQuaternion& operator=(const TVector3& );
   inline Bool_t operator == (const TVector3&) const;
   inline Bool_t operator != (const TVector3&) const;
   inline TQuaternion& operator+=(const TVector3 &vector);
   inline TQuaternion& operator-=(const TVector3 &vector);
   TQuaternion& MultiplyLeft(const TVector3 &vector);
   TQuaternion& operator*=(const TVector3 &vector);
   TQuaternion& DivideLeft(const TVector3 &vector);
   TQuaternion& operator/=(const TVector3 &vector);
   TQuaternion operator+(const TVector3 &vector) const;
   TQuaternion operator-(const TVector3 &vector) const;
   TQuaternion LeftProduct(const TVector3 &vector) const;
   TQuaternion operator*(const TVector3 &vector) const;
   TQuaternion LeftQuotient(const TVector3 &vector) const;
   TQuaternion operator/(const TVector3 &vector) const;

   // ----------------- quaternion algebra
   inline TQuaternion& operator=(const TQuaternion& );
   inline Bool_t operator == (const TQuaternion&) const;
   inline Bool_t operator != (const TQuaternion&) const;
   inline TQuaternion& operator+=(const TQuaternion &quaternion);
   inline TQuaternion& operator-=(const TQuaternion &quaternion);
   TQuaternion& MultiplyLeft(const TQuaternion &quaternion);
   TQuaternion& operator*=(const TQuaternion &quaternion);
   TQuaternion& DivideLeft(const TQuaternion &quaternion);
   TQuaternion& operator/=(const TQuaternion &quaternion);
   TQuaternion operator+(const TQuaternion &quaternion) const;
   TQuaternion operator-(const TQuaternion &quaternion) const;
   TQuaternion LeftProduct(const TQuaternion &quaternion) const;
   TQuaternion operator*(const TQuaternion &quaternion) const;
   TQuaternion LeftQuotient(const TQuaternion &quaternion) const;
   TQuaternion operator/(const TQuaternion &quaternion) const;

   // ------------------ general algebra
   inline Double_t Norm() const; // quaternion magnitude
   inline Double_t Norm2() const; // quaternion squared magnitude
   Double_t QMag() const { return Norm(); } // quaternion magnitude
   Double_t QMag2() const { return Norm2(); } // quaternion squared magnitude
   inline TQuaternion& Normalize();  // normalize quaternion
   inline TQuaternion operator - () const; // Unary minus.
   inline TQuaternion Conjugate() const;
   TQuaternion Invert() const;
   void Rotate(TVector3& vect) const;
   TVector3 Rotation(const TVector3& vect) const;

   void Print(Option_t* option="") const override;

   Double_t fRealPart;          // Real part
   TVector3 fVectorPart; // vector part

   ClassDefOverride(TQuaternion,1) // a quaternion class
};


// getters / setters

inline TQuaternion& TQuaternion::SetRXYZ(Double_t r,Double_t x,Double_t y,Double_t z) {
   fRealPart = r;
   fVectorPart.SetXYZ(x,y,z);
   return (*this);
}

inline TQuaternion& TQuaternion::SetRV(Double_t r, TVector3& vect) {
   fRealPart = r;
   fVectorPart= vect;
   return (*this);
}

inline void TQuaternion::GetRXYZ(Double_t *carray) const {
   fVectorPart.GetXYZ(carray+1);
   carray[0] = fRealPart;
}

inline void TQuaternion::GetRXYZ(Float_t *carray) const {
   fVectorPart.GetXYZ(carray+1);
   carray[0] = (Float_t) fRealPart;
}

inline Double_t & TQuaternion::operator[] (int i)       { return operator()(i); }
inline Double_t   TQuaternion::operator[] (int i) const { return operator()(i); }

// ------------------ real to quaternion algebra

inline Bool_t TQuaternion::operator == (Double_t r) const {
   return (fVectorPart.Mag2()==0 && fRealPart == r) ? kTRUE : kFALSE;
}

inline Bool_t TQuaternion::operator != (Double_t r) const {
   return (fVectorPart.Mag2()!=0 || fRealPart != r) ? kTRUE : kFALSE;
}

inline TQuaternion& TQuaternion::operator=(Double_t r) {
   fRealPart = r;
   fVectorPart.SetXYZ(0,0,0);
   return (*this);
}

inline TQuaternion& TQuaternion::operator+=(Double_t real) {
   fRealPart += real;
   return (*this);
}

inline TQuaternion& TQuaternion::operator-=(Double_t real) {
   fRealPart -= real;
   return (*this);
}

inline TQuaternion& TQuaternion::operator*=(Double_t real) {
   fRealPart *= real;
   fVectorPart *= real;
   return (*this);
}

inline TQuaternion& TQuaternion::operator/=(Double_t real) {
   if (real!=0) {
      fRealPart /= real;
      fVectorPart.SetX(fVectorPart.x()/real); // keep numericaly compliant with operator/(Double_t)
      fVectorPart.SetY(fVectorPart.y()/real);
      fVectorPart.SetZ(fVectorPart.z()/real);
   } else {
      Error("operator/=()(Double_t)", "bad value (%f) ignored",real);
   }
   return (*this);
}

TQuaternion operator + (Double_t r, const TQuaternion & q);
TQuaternion operator - (Double_t r, const TQuaternion & q);
TQuaternion operator * (Double_t r, const TQuaternion & q);
TQuaternion operator / (Double_t r, const TQuaternion & q);

// ------------------- vector to quaternion algebra

inline Bool_t TQuaternion::operator == (const TVector3& V) const {
   return (fVectorPart == V && fRealPart == 0) ? kTRUE : kFALSE;
}

inline Bool_t TQuaternion::operator != (const TVector3& V) const {
   return (fVectorPart != V || fRealPart != 0) ? kTRUE : kFALSE;
}

inline TQuaternion& TQuaternion::operator=(const TVector3& vect) {
   fRealPart = 0;
   fVectorPart.SetXYZ(vect.X(),vect.Y(),vect.Z());
   return *this;
}

inline TQuaternion& TQuaternion::operator+=(const TVector3 &vect) {
   fVectorPart += vect;
   return (*this);
}

inline TQuaternion& TQuaternion::operator-=(const TVector3 &vect) {
   fVectorPart -= vect;
   return (*this);
}

TQuaternion operator + (const TVector3 &V, const TQuaternion &Q);
TQuaternion operator - (const TVector3 &V, const TQuaternion &Q);
TQuaternion operator * (const TVector3 &V, const TQuaternion &Q);
TQuaternion operator / (const TVector3 &V, const TQuaternion &Q);

// --------------- quaternion algebra

inline Bool_t TQuaternion::operator == (const TQuaternion& Q) const {
   return (fVectorPart == Q.fVectorPart && fRealPart == Q.fRealPart) ? kTRUE : kFALSE;
}

inline Bool_t TQuaternion::operator != (const TQuaternion& Q) const {
   return (fVectorPart != Q.fVectorPart || fRealPart != Q.fRealPart) ? kTRUE : kFALSE;
}

inline TQuaternion& TQuaternion::operator=(const TQuaternion& quat) {
   if (&quat != this) {
      fRealPart = quat.fRealPart;
      fVectorPart.SetXYZ(quat.fVectorPart.X(),quat.fVectorPart.Y(),quat.fVectorPart.Z());
   }
   return (*this);
}

inline TQuaternion& TQuaternion::operator+=(const TQuaternion &quaternion) {
   fVectorPart += quaternion.fVectorPart;
   fRealPart += quaternion.fRealPart;
   return (*this);
}

inline TQuaternion& TQuaternion::operator-=(const TQuaternion &quaternion) {
   fVectorPart -= quaternion.fVectorPart;
   fRealPart   -= quaternion.fRealPart;
   return (*this);
}

inline TQuaternion TQuaternion::operator+(const TQuaternion &quaternion) const {

   return TQuaternion(fVectorPart+quaternion.fVectorPart, fRealPart+quaternion.fRealPart);
}

inline TQuaternion TQuaternion::operator-(const TQuaternion &quaternion) const {

   return TQuaternion(fVectorPart-quaternion.fVectorPart, fRealPart-quaternion.fRealPart);
}

// ---------------- general
inline Double_t TQuaternion::Norm() const {
   return TMath::Sqrt(Norm2());
}

inline Double_t TQuaternion::Norm2() const {
   return fRealPart*fRealPart + fVectorPart.Mag2();
}

inline TQuaternion& TQuaternion::Normalize() {

   (*this) /= Norm();
   return (*this);
}

inline TQuaternion TQuaternion::operator - () const {
   return TQuaternion(-fVectorPart,-fRealPart);
}

inline TQuaternion TQuaternion::Conjugate() const {
   return TQuaternion(-fVectorPart,fRealPart);
}

#endif

