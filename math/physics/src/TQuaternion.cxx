// @(#)root/physics:$Id$
// Author: Eric Anciant 28/06/2005


//////////////////////////////////////////////////////////////////////////
//____________________
//
//  A Quaternion Class
// Begin_html
// <p> Quaternion is a 4-component mathematic object quite convenient when dealing with
// space rotation (or reference frame transformation). </p>
// </p>
// <p> In short, think of quaternion Q as a 3-vector augmented by a real number. Q = Q|<sub>r</sub> + Q|<sub>V</sub>
//
// <p> <u> Quaternion multiplication :</u>
// </p>
// <p> Quaternion multiplication is given by :
// <br> Q.Q'        = (Q|<sub>r</sub> + Q|<sub>V</sub> )*( Q'|<sub>r</sub> + Q'|<sub>V</sub>)
// <br>         = [ Q|<sub>r</sub>*Q'|<sub>r</sub> - Q|<sub>V</sub>*Q'|<sub>V</sub> ] + [ Q|<sub>r</sub>*Q'|<sub>V</sub> + Q'|<sub>r</sub>*Q|<sub>V</sub> + Q|<sub>V</sub> X Q'|<sub>V</sub> ]
// <br>
// <br> where :
// <br> Q|<sub>r</sub>*Q'|<sub>r</sub>  is a real number product of real numbers
// <br> Q|<sub>V</sub>*Q'|<sub>V</sub> is a real number, scalar product of two 3-vectors
// <br> Q|<sub>r</sub>*Q'|<sub>V</sub> is a 3-vector, scaling of a 3-vector by a real number
// <br> Q|<sub>V</sub>XQ'|<sub>V</sub> is a 3-vector, cross product of two 3-vectors
// <br>
// <br> Thus, quaternion product is a generalization of real number product and product of a vector by a real number. Product of two pure vectors gives a quaternion whose real part is the opposite of scalar product and the vector part the cross product.
// </p>
//
// <p> The conjugate of a quaternion Q = Q|<sub>r</sub> + Q|<sub>V</sub> is Q_bar = Q|<sub>r</sub> - Q|<sub>V</sub>
// </p>
// <p> The magnitude of a quaternion Q is given by |Q|² = Q.Q_bar = Q_bar.Q = Q²|<sub>r</sub> + |Q|<sub>V</sub>|²
// </p>
// <p> Therefore, the inverse of a quaternion is Q<sup>-1</sup> = Q_bar /|Q|²
// </p>
// <p> "unit" quaternion is a quaternion of magnitude 1 : |Q|² = 1.
// <br> Unit quaternions are a subset of the quaternions set.
// </p>
//
// <p> <u>Quaternion and rotations :</u>
// </p>
//
// <p> A rotation of angle <font face="Symbol">f</font> around a given axis, is represented by a unit quaternion Q :
// <br> -        The axis of the rotation is given by the vector part of Q.
// <br> -        The ratio between the magnitude of the vector part and the real part of Q equals tan(<font face="Symbol">f</font>/2).
// </p>
// <p> In other words : Q = Q|<sub>r</sub> + Q|<sub>V</sub> = cos(<font face="Symbol">f</font>/2) + sin(<font face="Symbol">f</font>/2).
// <br> (where u is a unit vector // to the rotation axis,
//                        cos(<font face="Symbol">f</font>/2) is the real part, sin(<font face="Symbol">f</font>/2).u is the vector part)
// <br> Note : The quaternion of identity is Q<sub>I</sub> = cos(0) + sin(0)*(any vector) = 1.
// </p>
// <p> The composition of two rotations is described by the product of the two corresponding quaternions.
// <br> As for 3-space rotations, quaternion multiplication is not commutative !
// <br>
// <br> Q = Q<sub>1</sub>.Q<sub>2</sub> represents the composition of the successive rotation R1 and R2 expressed in the <b>current</b> frame (the axis of rotation hold by Q<sub>2</sub> is expressed in the frame as it is after R1 rotation).
// <br> Q = Q<sub>2</sub>.Q<sub>1</sub> represents the composition of the successive rotation R1 and R2 expressed in the <b>initial</b> reference frame.
// </p>
// <p> The inverse of a rotation is a rotation about the same axis but of opposite angle, thus if Q is a unit quaternion,
// <br> Q = cos(<font face="Symbol">f</font>/2) + sin(<font face="Symbol">f</font>/2).u = Q|<sub>r</sub> + Q|<sub>V</sub>, then :
// <br> Q<sup>-1</sup> =cos(-<font face="Symbol">f</font>/2) + sin(-<font face="Symbol">f</font>/2).u = cos(<font face="Symbol">f</font>/2) - sin(<font face="Symbol">f</font>/2).u = Q|<sub>r</sub> -Q|<sub>V</sub> is its inverse quaternion.
// </p>
// <p> One verifies that :
// <br> Q.Q<sup>-1</sup> = Q<sup>-1</sup>.Q = Q|<sub>r</sub>*Q|<sub>r</sub> + Q|<sub>V</sub>*Q|<sub>V</sub> + Q|<sub>r</sub>*Q|<sub>V</sub> -Q|<sub>r</sub>*Q|<sub>V</sub> + Q|<sub>V</sub>XQ|<sub>V</sub>
// <br>                 = Q²|<sub>r</sub> + Q²|<sub>V</sub> = 1
// </p>
// <br>
// <p> The rotation of a vector V by the rotation described by a unit quaternion Q is obtained by the following operation : V' = Q*V*Q<sup>-1</sup>, considering V as a quaternion whose real part is null.
// </p>
// <p> <u>Numeric computation considerations :</u>
// </p>
// <p> Numerically, the quaternion multiplication involves 12 additions and 16 multiplications.
// <br> It is therefore faster than 3x3 matrixes multiplication involving 18 additions and 27 multiplications.
// <br>
// <br> On the contrary, rotation of a vector by the above formula ( Q*V*Q<sup>-1</sup> ) involves 18 additions and 24 multiplications, whereas multiplication of a 3-vector by a 3x3 matrix involves only 6 additions and 9 multiplications.
// <br>
// <br> When dealing with numerous composition of space rotation, it is therefore faster to use quaternion product. On the other hand if a huge set of vectors must be rotated by a given quaternion, it is more optimized to convert the quaternion into a rotation matrix once, and then use that later to rotate the set of vectors.
// </p>
// <p> <u>More information :</u>
// </p>
// <p>
// <A HREF="http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation">
//  en.wikipedia.org/wiki/Quaternions_and_spatial_rotation </A>.
// <br> <br>
// <A HREF="http://en.wikipedia.org/wiki/Quaternion">
//  en.wikipedia.org/wiki/Quaternion </A>.
// </p>
// <p> _______________________________________________
// <br>
// <p> This Class represents all quaternions (unit or non-unit)
// <br> It possesses a Normalize() method to make a given quaternion unit
// <br> The Rotate(TVector3&) and Rotation(TVector3&) methods can be used even for a non-unit quaternion, in that case, the proper normalization is applied to perform the rotation.
// <br>
// <br> A TRotation constructor exists than takes a quaternion for parameter (even non-unit), in that cas the proper normalisation is applied.
// </p>
// End_html

#include "TMath.h"
#include "TQuaternion.h"

ClassImp(TQuaternion)

/****************** CONSTRUCTORS ****************************************************/
//______________________________________________________________________________
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

//______________________________________________________________________________
Double_t TQuaternion::operator () (int i) const {
   //dereferencing operator const
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

//______________________________________________________________________________
Double_t & TQuaternion::operator () (int i) {
   //dereferencing operator
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
//_____________________________________
Double_t TQuaternion::GetQAngle() const {
   // Get angle of quaternion (rad)
   // N.B : this angle is half of the corresponding rotation angle

   if (fRealPart == 0) return TMath::PiOver2();
   Double_t denominator = fVectorPart.Mag();
   return atan(denominator/fRealPart);
}

//_____________________________________
TQuaternion& TQuaternion::SetQAngle(Double_t angle) {
   // Set angle of quaternion (rad) - keep quaternion norm
   // N.B : this angle is half of the corresponding rotation angle

   Double_t norm = Norm();
   Double_t normSinV = fVectorPart.Mag();
   if (normSinV != 0) fVectorPart *= (sin(angle)*norm/normSinV);
   fRealPart = cos(angle)*norm;

   return (*this);
}

//_____________________________________
TQuaternion& TQuaternion::SetAxisQAngle(TVector3& v,Double_t QAngle) {
   // set quaternion from vector and angle (rad)
   // quaternion is set as unitary
   // N.B : this angle is half of the corresponding rotation angle

   fVectorPart = v;
   double norm = v.Mag();
   if (norm>0) fVectorPart*=(1./norm);
   fVectorPart*=sin(QAngle);
   fRealPart = cos(QAngle);

   return (*this);
}

/**************** REAL TO QUATERNION ALGEBRA ****************************************/

//_____________________________________
TQuaternion TQuaternion::operator+(Double_t real) const {
   // sum of quaternion with a real

   return TQuaternion(fVectorPart, fRealPart + real);
}

//_____________________________________
TQuaternion TQuaternion::operator-(Double_t real) const {
   // substraction of real to quaternion

   return TQuaternion(fVectorPart, fRealPart - real);
}

//_____________________________________
TQuaternion TQuaternion::operator*(Double_t real) const {
   // product of quaternion with a real

   return TQuaternion(fRealPart*real,fVectorPart.x()*real,fVectorPart.y()*real,fVectorPart.z()*real);
}


//_____________________________________
TQuaternion TQuaternion::operator/(Double_t real) const {
   // division by a real

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

//_____________________________________
TQuaternion TQuaternion::operator+(const TVector3 &vect) const {
   // sum of quaternion with a real

   return TQuaternion(fVectorPart + vect, fRealPart);
}

//_____________________________________
TQuaternion TQuaternion::operator-(const TVector3 &vect) const {
   // substraction of real to quaternion

   return TQuaternion(fVectorPart - vect, fRealPart);
}

//_____________________________________
TQuaternion& TQuaternion::MultiplyLeft(const TVector3 &vect) {
   // left multitplication

   Double_t savedRealPart = fRealPart;
   fRealPart = - (fVectorPart * vect);
   fVectorPart = vect.Cross(fVectorPart);
   fVectorPart += (vect * savedRealPart);

   return (*this);
}

//_____________________________________
TQuaternion& TQuaternion::operator*=(const TVector3 &vect) {
   // right multiplication

   Double_t savedRealPart = fRealPart;
   fRealPart = -(fVectorPart * vect);
   fVectorPart = fVectorPart.Cross(vect);
   fVectorPart += (vect * savedRealPart );

   return (*this);
}

//_____________________________________
TQuaternion TQuaternion::LeftProduct(const TVector3 &vect) const {
   // left product

   return TQuaternion(vect * fRealPart + vect.Cross(fVectorPart), -(fVectorPart * vect));
}

//_____________________________________
TQuaternion TQuaternion::operator*(const TVector3 &vect) const {
   // right product

   return TQuaternion(vect * fRealPart + fVectorPart.Cross(vect), -(fVectorPart * vect));
}

//_____________________________________
TQuaternion& TQuaternion::DivideLeft(const TVector3 &vect) {
   // left division

   Double_t norm2 = vect.Mag2();
   MultiplyLeft(vect);
   if (norm2 > 0 ) {
      // use (1./nom2) to be numericaly compliant with LeftQuotient(const TVector3 &)
      (*this) *= -(1./norm2); // minus <- using conjugate of vect
   } else {
      Error("DivideLeft(const TVector3)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

//_____________________________________
TQuaternion& TQuaternion::operator/=(const TVector3 &vect) {
   // right division

   Double_t norm2 = vect.Mag2();
   (*this) *= vect;
   if (norm2 > 0 ) {
      // use (1./real) to be numericaly compliant with operator/(const TVector3 &)
      (*this) *= - (1./norm2); // minus <- using conjugate of vect
   } else {
      Error("operator/=(const TVector3 &)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

//_____________________________________
TQuaternion TQuaternion::LeftQuotient(const TVector3 &vect) const {
   // left quotient

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

//_____________________________________
TQuaternion TQuaternion::operator/(const TVector3 &vect) const {
   //  right quotient

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

//_____________________________________
TQuaternion& TQuaternion::operator*=(const TQuaternion &quaternion) {
   // right multiplication

   Double_t saveRP = fRealPart;
   TVector3 cross(fVectorPart.Cross(quaternion.fVectorPart));

   fRealPart = fRealPart * quaternion.fRealPart - fVectorPart * quaternion.fVectorPart;

   fVectorPart *= quaternion.fRealPart;
   fVectorPart += quaternion.fVectorPart * saveRP;
   fVectorPart += cross;
   return (*this);
}

//_____________________________________
TQuaternion& TQuaternion::MultiplyLeft(const TQuaternion &quaternion) {
   // left multiplication

   Double_t saveRP = fRealPart;
   TVector3 cross(quaternion.fVectorPart.Cross(fVectorPart));

   fRealPart = fRealPart * quaternion.fRealPart - fVectorPart * quaternion.fVectorPart;

   fVectorPart *= quaternion.fRealPart;
   fVectorPart += quaternion.fVectorPart * saveRP;
   fVectorPart += cross;

   return (*this);
}

//_____________________________________
TQuaternion TQuaternion::LeftProduct(const TQuaternion &quaternion) const {
   // left product

   return TQuaternion( fVectorPart*quaternion.fRealPart + quaternion.fVectorPart*fRealPart
                                 + quaternion.fVectorPart.Cross(fVectorPart),
                                   fRealPart*quaternion.fRealPart - fVectorPart*quaternion.fVectorPart );
}

//_____________________________________
TQuaternion TQuaternion::operator*(const TQuaternion &quaternion) const {
   // right product

   return TQuaternion(fVectorPart*quaternion.fRealPart + quaternion.fVectorPart*fRealPart
                    + fVectorPart.Cross(quaternion.fVectorPart),
                      fRealPart*quaternion.fRealPart - fVectorPart*quaternion.fVectorPart );
}

//_____________________________________
TQuaternion& TQuaternion::DivideLeft(const TQuaternion &quaternion) {
   // left division

   Double_t norm2 = quaternion.Norm2();

   if (norm2 > 0 ) {
      MultiplyLeft(quaternion.Conjugate());
      (*this) *= (1./norm2);
   } else {
      Error("DivideLeft(const TQuaternion &)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

//_____________________________________
TQuaternion& TQuaternion::operator/=(const TQuaternion& quaternion) {
   // right division

   Double_t norm2 = quaternion.Norm2();

   if (norm2 > 0 ) {
      (*this) *= quaternion.Conjugate();
      // use (1./norm2) top be numericaly compliant with operator/(const TQuaternion&)
      (*this) *= (1./norm2);
   } else {
      Error("operator/=(const TQuaternion&)", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

//_____________________________________
TQuaternion TQuaternion::LeftQuotient(const TQuaternion& quaternion) const {
   // left quotient

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

//_____________________________________
TQuaternion TQuaternion::operator/(const TQuaternion &quaternion) const {
   // right quotient

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

//_____________________________________
TQuaternion TQuaternion::Invert() const {
   // invert

   double norm2 = Norm2();
   if (norm2 > 0 ) {
      double invNorm2 = 1./norm2;
      return TQuaternion(fVectorPart*(-invNorm2), fRealPart*invNorm2 );
   } else {
      Error("Invert()", "bad norm2 (%f) ignored",norm2);
   }
   return (*this);
}

//_____________________________________
void TQuaternion::Rotate(TVector3 & vect) const {
   // rotate vect by current quaternion

   vect = Rotation(vect);
}

//_____________________________________
TVector3 TQuaternion::Rotation(const TVector3 & vect) const {
   // rotation of vect by current quaternion

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

//_____________________________________
void TQuaternion::Print(Option_t*) const
{
   //Print Quaternion parameters
   Printf("%s %s (r,x,y,z)=(%f,%f,%f,%f) \n (alpha,rho,theta,phi)=(%f,%f,%f,%f)",GetName(),GetTitle(),
            fRealPart,fVectorPart.X(),fVectorPart.Y(),fVectorPart.Z(),
            GetQAngle()*TMath::RadToDeg(),fVectorPart.Mag(),fVectorPart.Theta()*TMath::RadToDeg(),fVectorPart.Phi()*TMath::RadToDeg());
}
