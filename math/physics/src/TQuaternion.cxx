// @(#)root/physics:$Id$
// Author: Eric Anciant 28/06/2005


/** \class TQuaternion
    \ingroup Physics
 Quaternion is a 4-component mathematic object quite convenient when dealing with
space rotation (or reference frame transformation).

 In short, think of quaternion Q as a 3-vector augmented by a real number.
 \f$ Q = Q|_r + Q|_V \f$

 ####  Quaternion multiplication :

 Quaternion multiplication is given by :
 \f[
 Q.Q' = (Q|_r + Q|_V )*( Q'|_r + Q'|_V) = [ Q|_r*Q'|_r - Q|_V*Q'|_V ] + [ Q|_r*Q'|_V + Q'|_r*Q|_V + Q|_V X Q'|_V ]
\f]

 where :
  - \f$ Q|_r*Q'|_r \f$ is a real number product of real numbers
  - \f$ Q|_V*Q'|_V \f$ is a real number, scalar product of two 3-vectors
  - \f$ Q|_r*Q'|_V \f$ is a 3-vector, scaling of a 3-vector by a real number
  - \f$ Q|_VXQ'|_V \f$ is a 3-vector, cross product of two 3-vectors

Thus, quaternion product is a generalization of real number product and product of a
vector by a real number. Product of two pure vectors gives a quaternion whose real part
is the opposite of scalar product and the vector part the cross product.

The conjugate of a quaternion \f$ Q = Q|r + Q|V \f$ is \f$ \bar{Q} = Q|r - Q|V \f$

The magnitude of a quaternion \f$ Q \f$ is given by \f$ |Q|^2 = Q.\bar{Q} = \bar{Q}.Q = Q^2|r + |Q|V|^2 \f$

Therefore, the inverse of a quaternion is \f$ Q-1 = \bar{Q} /|Q|^2 \f$

"unit" quaternion is a quaternion of magnitude 1 : \f$ |Q|^2 = 1. \f$
Unit quaternions are a subset of the quaternions set.

 #### Quaternion and rotations :


 A rotation of angle \f$ f \f$ around a given axis, is represented by a unit quaternion Q :
 - The axis of the rotation is given by the vector part of Q.
 - The ratio between the magnitude of the vector part and the real part of Q equals \f$ tan(\frac{f}{2}) \f$.

 In other words : \f$ Q = Q|_r + Q|_V = cos(\frac{f}{2}) + sin(\frac{f}{2}) \f$.
 (where u is a unit vector // to the rotation axis,
\f$ cos(\frac{f}{2}) \f$ is the real part, \f$ sin(\frac{f}{2}) \f$ .u is the vector part)
 Note : The quaternion of identity is \f$ Q_I = cos(0) + sin(0)*(AnyVector) = 1\f$ .

 The composition of two rotations is described by the product of the two corresponding quaternions.
 As for 3-space rotations, quaternion multiplication is not commutative !

 \f$ Q = Q_1.Q_2 \f$ represents the composition of the successive rotation R1 and R2 expressed in the current frame (the axis of rotation hold by \f$ Q_2 \f$ is expressed in the frame as it is after R1 rotation).
 \f$ Q = Q_2.Q_1 \f$ represents the composition of the successive rotation R1 and R2 expressed in the initial reference frame.

 The inverse of a rotation is a rotation about the same axis but of opposite angle, thus if Q is a unit quaternion,
 \f$ Q = cos(\frac{f}{2}) + sin(\frac{f}{2}).u = Q|_r + Q|_V\f$ , then :
 \f$ Q^{-1} =cos(-\frac{f}{2}) + sin(-\frac{f}{2}).u = cos(\frac{f}{2}) - sin(\frac{f}{2}).u = Q|_r -Q|_V \f$ is its inverse quaternion.

 One verifies that :
 \f$ Q.Q^{-1} = Q^{-1}.Q = Q|_r*Q|_r + Q|_V*Q|_V + Q|_r*Q|_V -Q|_r*Q|_V + Q|_VXQ|_V = Q\leq|_r + Q\leq|_V = 1 \f$


 The rotation of a vector V by the rotation described by a unit quaternion Q is obtained by the following operation :
 \f$ V' = Q*V*Q^{-1} \f$, considering V as a quaternion whose real part is null.

 #### Numeric computation considerations :

 Numerically, the quaternion multiplication involves 12 additions and 16 multiplications.
 It is therefore faster than 3x3 matrixes multiplication involving 18 additions and 27 multiplications.

 On the contrary, rotation of a vector by the above formula ( \f$ Q*V*Q^{-1} \f$ ) involves 18 additions
 and 24 multiplications, whereas multiplication of a 3-vector by a 3x3 matrix involves only 6 additions
 and 9 multiplications.

 When dealing with numerous composition of space rotation, it is therefore faster to use quaternion product. On the other hand if a huge set of vectors must be rotated by a given quaternion, it is more optimized to convert the quaternion into a rotation matrix once, and then use that later to rotate the set of vectors.

 #### More information :

http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

http://en.wikipedia.org/wiki/Quaternion

 This Class represents all quaternions (unit or non-unit)
 It possesses a Normalize() method to make a given quaternion unit
 The Rotate(TVector3&) and Rotation(TVector3&) methods can be used even for a non-unit quaternion, in that case, the proper normalization is applied to perform the rotation.

 A TRotation constructor exists than takes a quaternion for parameter (even non-unit), in that cas the proper normalisation is applied.
*/

#include "TMath.h"
#include "TQuaternion.h"
#include "TString.h"

ClassImp(TQuaternion);

/****************** CONSTRUCTORS ****************************************************/
////////////////////////////////////////////////////////////////////////////////

TQuaternion::TQuaternion(const TQuaternion & q) : TObject(q),
  fRealPart(q.fRealPart), fVectorPart(q.fVectorPart) {}

TQuaternion::TQuaternion(const TVector3 & vect, Double_t real)
        : fRealPart(real), fVectorPart(vect)  {}

TQuaternion::TQuaternion(const Double_t * x0)
        : fRealPart(x0[3]), fVectorPart(x0) {}

TQuaternion::TQuaternion(const Float_t * x0)
        : fRealPart(x0[3]), fVectorPart(x0) {}

TQuaternion::TQuaternion(Double_t real, Double_t X, Double_t Y, Double_t Z)
        : fRealPart(real), fVectorPart(X,Y,Z) {}

TQuaternion::~TQuaternion() {}

////////////////////////////////////////////////////////////////////////////////
///dereferencing operator const

Double_t TQuaternion::operator () (int i) const {
   switch(i) {
      case 0:
      case 1:
      case 2:
         return fVectorPart(i);
      case 3:
         return fRealPart;
      default:
         Error("operator()(i)", "bad index (%d) returning 0",i);
   }
   return 0.;
}

////////////////////////////////////////////////////////////////////////////////
///dereferencing operator

Double_t & TQuaternion::operator () (int i) {
   switch(i) {
      case 0:
      case 1:
      case 2:
         return fVectorPart(i);
      case 3:
         return fRealPart;
      default:
         Error("operator()(i)", "bad index (%d) returning &fRealPart",i);
   }
   return fRealPart;
}
////////////////////////////////////////////////////////////////////////////////
/// Get angle of quaternion (rad)
/// N.B : this angle is half of the corresponding rotation angle

Double_t TQuaternion::GetQAngle() const {
   if (fRealPart == 0) return TMath::PiOver2();
   Double_t denominator = fVectorPart.Mag();
   return atan(denominator/fRealPart);
}

////////////////////////////////////////////////////////////////////////////////
/// Set angle of quaternion (rad) - keep quaternion norm
/// N.B : this angle is half of the corresponding rotation angle

TQuaternion& TQuaternion::SetQAngle(Double_t angle) {
   Double_t norm = Norm();
   Double_t normSinV = fVectorPart.Mag();
   if (normSinV != 0) fVectorPart *= (sin(angle)*norm/normSinV);
   fRealPart = cos(angle)*norm;

   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// set quaternion from vector and angle (rad)
/// quaternion is set as unitary
/// N.B : this angle is half of the corresponding rotation angle

TQuaternion& TQuaternion::SetAxisQAngle(TVector3& v,Double_t QAngle) {
   fVectorPart = v;
   double norm = v.Mag();
   if (norm>0) fVectorPart*=(1./norm);
   fVectorPart*=sin(QAngle);
   fRealPart = cos(QAngle);

   return (*this);
}

/**************** REAL TO QUATERNION ALGEBRA ****************************************/

////////////////////////////////////////////////////////////////////////////////
/// sum of quaternion with a real

TQuaternion TQuaternion::operator+(Double_t real) const {
   return TQuaternion(fVectorPart, fRealPart + real);
}

////////////////////////////////////////////////////////////////////////////////
/// substraction of real to quaternion

TQuaternion TQuaternion::operator-(Double_t real) const {
   return TQuaternion(fVectorPart, fRealPart - real);
}

////////////////////////////////////////////////////////////////////////////////
/// product of quaternion with a real

TQuaternion TQuaternion::operator*(Double_t real) const {
   return TQuaternion(fRealPart*real,fVectorPart.x()*real,fVectorPart.y()*real,fVectorPart.z()*real);
}


////////////////////////////////////////////////////////////////////////////////
/// division by a real

TQuaternion TQuaternion::operator/(Double_t real) const {
   if (real !=0 ) {
      return TQuaternion(fRealPart/real,fVectorPart.x()/real,fVectorPart.y()/real,fVectorPart.z()/real);
   } else {
      Error("operator/(Double_t)", "bad value (%f) ignored",real);
   }

   return (*this);
}

TQuaternion operator + (Double_t r, const TQuaternion & q) { return (q+r); }
TQuaternion operator - (Double_t r, const TQuaternion & q) { return (-q+r); }
TQuaternion operator * (Double_t r, const TQuaternion & q) { return (q*r); }
TQuaternion operator / (Double_t r, const TQuaternion & q) { return (q.Invert()*r); }

/**************** VECTOR TO QUATERNION ALGEBRA ****************************************/

////////////////////////////////////////////////////////////////////////////////
/// sum of quaternion with a real

TQuaternion TQuaternion::operator+(const TVector3 &vect) const {
   return TQuaternion(fVectorPart + vect, fRealPart);
}

////////////////////////////////////////////////////////////////////////////////
/// substraction of real to quaternion

TQuaternion TQuaternion::operator-(const TVector3 &vect) const {
   return TQuaternion(fVectorPart - vect, fRealPart);
}

////////////////////////////////////////////////////////////////////////////////
/// left multiplication

TQuaternion& TQuaternion::MultiplyLeft(const TVector3 &vect) {
   Double_t savedRealPart = fRealPart;
   fRealPart = - (fVectorPart * vect);
   fVectorPart = vect.Cross(fVectorPart);
   fVectorPart += (vect * savedRealPart);

   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// right multiplication

TQuaternion& TQuaternion::operator*=(const TVector3 &vect) {
   Double_t savedRealPart = fRealPart;
   fRealPart = -(fVectorPart * vect);
   fVectorPart = fVectorPart.Cross(vect);
   fVectorPart += (vect * savedRealPart );

   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// left product

TQuaternion TQuaternion::LeftProduct(const TVector3 &vect) const {
   return TQuaternion(vect * fRealPart + vect.Cross(fVectorPart), -(fVectorPart * vect));
}

////////////////////////////////////////////////////////////////////////////////
/// right product

TQuaternion TQuaternion::operator*(const TVector3 &vect) const {
   return TQuaternion(vect * fRealPart + fVectorPart.Cross(vect), -(fVectorPart * vect));
}

////////////////////////////////////////////////////////////////////////////////
/// left division

TQuaternion& TQuaternion::DivideLeft(const TVector3 &vect) {
   Double_t norm2 = vect.Mag2();
   MultiplyLeft(vect);
   if (norm2 > 0 ) {
      // use (1./nom2) to be numerically compliant with LeftQuotient(const TVector3 &)
      (*this) *= -(1./norm2); // minus <- using conjugate of vect
   } else {
      Error("DivideLeft(const TVector3)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// right division

TQuaternion& TQuaternion::operator/=(const TVector3 &vect) {
   Double_t norm2 = vect.Mag2();
   (*this) *= vect;
   if (norm2 > 0 ) {
      // use (1./real) to be numerically compliant with operator/(const TVector3 &)
      (*this) *= - (1./norm2); // minus <- using conjugate of vect
   } else {
      Error("operator/=(const TVector3 &)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// left quotient

TQuaternion TQuaternion::LeftQuotient(const TVector3 &vect) const {
   Double_t norm2 = vect.Mag2();

   if (norm2>0) {
      double invNorm2 = 1./norm2;
      return TQuaternion((vect * -fRealPart - vect.Cross(fVectorPart))*invNorm2,
                                                                                                        (fVectorPart * vect ) * invNorm2 );
   } else {
      Error("LeftQuotient(const TVector3 &)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
///  right quotient

TQuaternion TQuaternion::operator/(const TVector3 &vect) const {
   Double_t norm2 = vect.Mag2();

   if (norm2>0) {
      double invNorm2 = 1./norm2;
      return TQuaternion((vect * -fRealPart - fVectorPart.Cross(vect)) * invNorm2,
                                                                                                                (fVectorPart * vect) * invNorm2 );
   } else {
      Error("operator/(const TVector3 &)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

TQuaternion operator + (const TVector3 &V, const TQuaternion &Q) { return (Q+V); }
TQuaternion operator - (const TVector3 &V, const TQuaternion &Q) { return (-Q+V); }
TQuaternion operator * (const TVector3 &V, const TQuaternion &Q) { return Q.LeftProduct(V); }

TQuaternion operator / (const TVector3 &vect, const TQuaternion &quat) {
   //divide operator
   TQuaternion res(vect);
   res /= quat;
   return res;
}

/**************** QUATERNION ALGEBRA ****************************************/

////////////////////////////////////////////////////////////////////////////////
/// right multiplication

TQuaternion& TQuaternion::operator*=(const TQuaternion &quaternion) {
   Double_t saveRP = fRealPart;
   TVector3 cross(fVectorPart.Cross(quaternion.fVectorPart));

   fRealPart = fRealPart * quaternion.fRealPart - fVectorPart * quaternion.fVectorPart;

   fVectorPart *= quaternion.fRealPart;
   fVectorPart += quaternion.fVectorPart * saveRP;
   fVectorPart += cross;
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// left multiplication

TQuaternion& TQuaternion::MultiplyLeft(const TQuaternion &quaternion) {
   Double_t saveRP = fRealPart;
   TVector3 cross(quaternion.fVectorPart.Cross(fVectorPart));

   fRealPart = fRealPart * quaternion.fRealPart - fVectorPart * quaternion.fVectorPart;

   fVectorPart *= quaternion.fRealPart;
   fVectorPart += quaternion.fVectorPart * saveRP;
   fVectorPart += cross;

   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// left product

TQuaternion TQuaternion::LeftProduct(const TQuaternion &quaternion) const {
   return TQuaternion( fVectorPart*quaternion.fRealPart + quaternion.fVectorPart*fRealPart
                                 + quaternion.fVectorPart.Cross(fVectorPart),
                                   fRealPart*quaternion.fRealPart - fVectorPart*quaternion.fVectorPart );
}

////////////////////////////////////////////////////////////////////////////////
/// right product

TQuaternion TQuaternion::operator*(const TQuaternion &quaternion) const {
   return TQuaternion(fVectorPart*quaternion.fRealPart + quaternion.fVectorPart*fRealPart
                    + fVectorPart.Cross(quaternion.fVectorPart),
                      fRealPart*quaternion.fRealPart - fVectorPart*quaternion.fVectorPart );
}

////////////////////////////////////////////////////////////////////////////////
/// left division

TQuaternion& TQuaternion::DivideLeft(const TQuaternion &quaternion) {
   Double_t norm2 = quaternion.Norm2();

   if (norm2 > 0 ) {
      MultiplyLeft(quaternion.Conjugate());
      (*this) *= (1./norm2);
   } else {
      Error("DivideLeft(const TQuaternion &)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// right division

TQuaternion& TQuaternion::operator/=(const TQuaternion& quaternion) {
   Double_t norm2 = quaternion.Norm2();

   if (norm2 > 0 ) {
      (*this) *= quaternion.Conjugate();
      // use (1./norm2) top be numerically compliant with operator/(const TQuaternion&)
      (*this) *= (1./norm2);
   } else {
      Error("operator/=(const TQuaternion&)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// left quotient

TQuaternion TQuaternion::LeftQuotient(const TQuaternion& quaternion) const {
   Double_t norm2 = quaternion.Norm2();

   if (norm2 > 0 ) {
      double invNorm2 = 1./norm2;
      return TQuaternion(
             (fVectorPart*quaternion.fRealPart - quaternion.fVectorPart*fRealPart
                        - quaternion.fVectorPart.Cross(fVectorPart)) * invNorm2,
                        (fRealPart*quaternion.fRealPart + fVectorPart*quaternion.fVectorPart)*invNorm2 );
   } else {
      Error("LeftQuotient(const TQuaternion&)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// right quotient

TQuaternion TQuaternion::operator/(const TQuaternion &quaternion) const {
   Double_t norm2 = quaternion.Norm2();

   if (norm2 > 0 ) {
      double invNorm2 = 1./norm2;
      return TQuaternion(
             (fVectorPart*quaternion.fRealPart - quaternion.fVectorPart*fRealPart
                        - fVectorPart.Cross(quaternion.fVectorPart)) * invNorm2,
                         (fRealPart*quaternion.fRealPart + fVectorPart*quaternion.fVectorPart) * invNorm2 );
   } else {
      Error("operator/(const TQuaternion &)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// invert

TQuaternion TQuaternion::Invert() const {
   double norm2 = Norm2();
   if (norm2 > 0 ) {
      double invNorm2 = 1./norm2;
      return TQuaternion(fVectorPart*(-invNorm2), fRealPart*invNorm2 );
   } else {
      Error("Invert()", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// rotate vect by current quaternion

void TQuaternion::Rotate(TVector3 & vect) const {
   vect = Rotation(vect);
}

////////////////////////////////////////////////////////////////////////////////
/// rotation of vect by current quaternion

TVector3 TQuaternion::Rotation(const TVector3 & vect) const {
   // Vres = (*this) * vect * (this->Invert());
   double norm2 = Norm2();

   if (norm2>0) {
      TQuaternion quat(*this);
      quat *= vect;

      // only compute vect part : (real part has to be 0 ) :
      // VECT [ quat * ( this->Conjugate() ) ] = quat.fRealPart * -this->fVectorPart
      //                                                                                        + this->fRealPart * quat.fVectorPart
      //                                                                                        + quat.fVectorPart X (-this->fVectorPart)
      TVector3 cross(fVectorPart.Cross(quat.fVectorPart));
      quat.fVectorPart *=  fRealPart;
      quat.fVectorPart -= fVectorPart * quat.fRealPart;
      quat.fVectorPart += cross;

      return quat.fVectorPart*(1./norm2);
   } else {
      Error("Rotation()", "bad norm2 (%f) ignored",norm2);
   }
   return vect;
}

////////////////////////////////////////////////////////////////////////////////
///Print Quaternion parameters

void TQuaternion::Print(Option_t*) const
{
   Printf("%s %s (r,x,y,z)=(%f,%f,%f,%f) \n (alpha,rho,theta,phi)=(%f,%f,%f,%f)",GetName(),GetTitle(),
            fRealPart,fVectorPart.X(),fVectorPart.Y(),fVectorPart.Z(),
            GetQAngle()*TMath::RadToDeg(),fVectorPart.Mag(),fVectorPart.Theta()*TMath::RadToDeg(),fVectorPart.Phi()*TMath::RadToDeg());
}
