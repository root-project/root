// @(#)root/physics:$Id$
// Author: Pasha Murat, Peter Malzacher   12/02/99
//    Aug 11 1999: added Pt == 0 guard to Eta()
//    Oct  8 1999: changed Warning to Error and
//                 return fX in Double_t & operator()
//    Oct 20 1999: Bug fix: sign in PseudoRapidity
//                 Warning-> Error in Double_t operator()

/** \class TVector3
    \ingroup Physics

TVector3 is a general three vector class, which can be used for
the description of different vectors in 3D.

### Declaration / Access to the components

TVector3 has been implemented as a vector of three Double_t
variables, representing the cartesian coordinates. By default all components
are initialized to zero:

~~~
 TVector3 v1;        // v1 = (0,0,0)
 TVector3 v3(1,2,3); // v3 = (1,2,3)
 TVector3 v4(v2);    // v4 = v2
~~~

It is also possible (but not recommended) to initialize a TVector3
with a Double_t or Float_t C array.

You can get the basic components either by name or by index using operator():

~~~
 xx = v1.X(); or xx = v1(0);
 yy = v1.Y();    yy = v1(1);
 zz = v1.Z();    zz = v1(2);
~~~

The member functions SetX(), SetY(), SetZ() and SetXYZ() allow to set the components:

~~~
 v1.SetX(1.); v1.SetY(2.); v1.SetZ(3.);
 v1.SetXYZ(1.,2.,3.);
~~~

### Non-cartesian coordinates

To get information on the TVector3 in spherical (rho,phi,theta)
or cylindrical (z,r,theta) coordinates, the

the member functions Mag() (=magnitude=rho in spherical coordinates),
Mag2(), Theta(), CosTheta(), Phi(), Perp() (the transverse component=r in
cylindrical coordinates), Perp2() can be used:


~~~
 Double_t m = v.Mag();       // get magnitude (=rho=Sqrt(x*x+y*y+z*z)))
 Double_t m2 = v.Mag2();     // get magnitude squared
 Double_t t = v.Theta();     // get polar angle
 Double_t ct = v.CosTheta(); // get cos of theta
 Double_t p = v.Phi();       // get azimuth angle
 Double_t pp = v.Perp();     // get transverse component
 Double_t pp2= v.Perp2();    // get transvers component squared
~~~

It is also possible to get the transverse component with respect to
another vector:

~~~
 Double_t ppv1 = v.Perp(v1);
 Double_t pp2v1 = v.Perp2(v1);
~~~

The pseudo-rapidity ( eta=-ln (tan (theta/2)) ) can be obtained by Eta()
or PseudoRapidity():

~~~
 Double_t eta = v.PseudoRapidity();
~~~

There are set functions to change one of the non-cartesian coordinates:

~~~
 v.SetTheta(.5); // keeping rho and phi
 v.SetPhi(.8);   // keeping rho and theta
 v.SetMag(10.);  // keeping theta and phi
 v.SetPerp(3.);  // keeping z and phi
~~~

### Arithmetic / Comparison

The TVector3 class provides the operators to add, subtract, scale and compare
vectors:

~~~
 v3  = -v1;
 v1  = v2+v3;
 v1 += v3;
 v1  = v1 - v3
 v1 -= v3;
 v1 *= 10;
 v1  = 5*v2;
 if (v1==v2) {...}
 if (v1!=v2) {...}
~~~

### Related Vectors

~~~
 v2 = v1.Unit(); // get unit vector parallel to v1
 v2 = v1.Orthogonal(); // get vector orthogonal to v1
~~~

### Scalar and vector products

~~~
 s = v1.Dot(v2);   // scalar product
 s = v1 * v2;      // scalar product
 v = v1.Cross(v2); // vector product
~~~

### Angle between two vectors

~~~
 Double_t a = v1.Angle(v2);
~~~

### Rotations

#### Rotation around axes

~~~
 v.RotateX(.5);
 v.RotateY(TMath::Pi());
 v.RotateZ(angle);
~~~

#### Rotation around a vector

~~~
 v1.Rotate(TMath::Pi()/4, v2); // rotation around v2
~~~

#### Rotation by TRotation
TVector3 objects can be rotated by objects of the TRotation
class using the Transform() member functions,

the operator *= or the operator * of the TRotation class:

~~~
 TRotation m;
 ...
 v1.transform(m);
 v1  = m*v1;
 v1 *= m; // Attention v1 = m*v1
~~~

#### Transformation from rotated frame

~~~
 TVector3 direction = v.Unit()
 v1.RotateUz(direction); // direction must be TVector3 of unit length
~~~

transforms v1 from the rotated frame (z' parallel to direction, x' in
the theta plane and y' in the xy plane as well as perpendicular to the
theta plane) to the (x,y,z) frame.
*/


#include "TMatrix.h"
#include "TVector3.h"

#include "TBuffer.h"
#include "TRotation.h"
#include "TMath.h"

ClassImp(TVector3);


////////////////////////////////////////////////////////////////////////////////
/// Convert two angles Theta and Phi into a Cartesian Transformation Vector

TVector3 TVector3::Cartesian(Double_t phi, Double_t theta)
{
   return TVector3 spherical(TMath::Sin(theta) * TMath::Cos(phi), TMath::Sin(theta) * TMath::Sin(phi),
                             TMath::Cos(theta));
}

////////////////////////////////////////////////////////////////////////////////
/// Multiplication operator

TVector3 & TVector3::operator *= (const TRotation & m){
   return *this = m * (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Transform this vector with a TRotation

TVector3 & TVector3::Transform(const TRotation & m) {
   return *this = m * (*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the angle w.r.t. another 3-vector.

Double_t TVector3::Angle(const TVector3 & q) const
{
   Double_t ptot2 = Mag2()*q.Mag2();
   if(ptot2 <= 0) {
      return 0.0;
   } else {
      Double_t arg = Dot(q)/TMath::Sqrt(ptot2);
      if(arg >  1.0) arg =  1.0;
      if(arg < -1.0) arg = -1.0;
      return TMath::ACos(arg);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the transverse component  (R in cylindrical coordinate system)

Double_t TVector3::Perp() const
{
   return TMath::Sqrt(Perp2());
}


////////////////////////////////////////////////////////////////////////////////
/// Return the transverse component (R in cylindrical coordinate system)

Double_t TVector3::Perp(const TVector3 & p) const
{
   return TMath::Sqrt(Perp2(p));
}

////////////////////////////////////////////////////////////////////////////////
/// Return the azimuth angle. Returns phi from -pi to pi.

Double_t TVector3::Phi() const
{
   return fX == 0.0 && fY == 0.0 ? 0.0 : TMath::ATan2(fY,fX);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the polar angle

Double_t TVector3::Theta() const
{
   return fX == 0.0 && fY == 0.0 && fZ == 0.0 ? 0.0 : TMath::ATan2(Perp(),fZ);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert a Cartesian Vector into a Spherical Vector
/// Spherical Vector: (r, theta, phi)

TVector3 TVector3::ToSpherical(const TVector3 &c)
{
   return TVector3 sphericalVector(TMath::Sqrt(c.X() * c.X() + c.Y() * c.Y() + c.Z() * c.Z()),
                                   TMath::ACos(c.Z() / TMath::Sqrt(c.X() * c.X() + c.Y() * c.Y() + c.Z() * c.Z())),
                                   TMath::ATan(c.Y() / c.X()));
}

////////////////////////////////////////////////////////////////////////////////
/// Return unit vector parallel to this.

TVector3 TVector3::Unit() const
{
   Double_t  tot2 = Mag2();
   Double_t tot = (tot2 > 0) ?  1.0/TMath::Sqrt(tot2) : 1.0;
   TVector3 p(fX*tot,fY*tot,fZ*tot);
   return p;
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate vector around X.

void TVector3::RotateX(Double_t angle) {
   Double_t s = TMath::Sin(angle);
   Double_t c = TMath::Cos(angle);
   Double_t yy = fY;
   fY = c*yy - s*fZ;
   fZ = s*yy + c*fZ;
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate vector around Y.

void TVector3::RotateY(Double_t angle) {
   Double_t s = TMath::Sin(angle);
   Double_t c = TMath::Cos(angle);
   Double_t zz = fZ;
   fZ = c*zz - s*fX;
   fX = s*zz + c*fX;
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate vector around Z.

void TVector3::RotateZ(Double_t angle) {
   Double_t s = TMath::Sin(angle);
   Double_t c = TMath::Cos(angle);
   Double_t xx = fX;
   fX = c*xx - s*fY;
   fY = s*xx + c*fY;
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate vector.

void TVector3::Rotate(Double_t angle, const TVector3 & axis){
   TRotation trans;
   trans.Rotate(angle, axis);
   operator*=(trans);
}

////////////////////////////////////////////////////////////////////////////////
/// NewUzVector must be normalized !

void TVector3::RotateUz(const TVector3& NewUzVector) {
   Double_t u1 = NewUzVector.fX;
   Double_t u2 = NewUzVector.fY;
   Double_t u3 = NewUzVector.fZ;
   Double_t up = u1*u1 + u2*u2;

   if (up) {
      up = TMath::Sqrt(up);
      Double_t px = fX,  py = fY,  pz = fZ;
      fX = (u1*u3*px - u2*py + u1*up*pz)/up;
      fY = (u2*u3*px + u1*py + u2*up*pz)/up;
      fZ = (u3*u3*px -    px + u3*up*pz)/up;
   } else if (u3 < 0.) { fX = -fX; fZ = -fZ; }      // phi=0  teta=pi
   else {};
}

////////////////////////////////////////////////////////////////////////////////
/// Double_t m = Mag();
/// return 0.5*log( (m+fZ)/(m-fZ) );
/// guard against Pt=0

Double_t TVector3::PseudoRapidity() const {
   double cosTheta = CosTheta();
   if (cosTheta*cosTheta < 1) return -0.5* TMath::Log( (1.0-cosTheta)/(1.0+cosTheta) );
   if (fZ == 0) return 0;
   //Warning("PseudoRapidity","transvers momentum = 0! return +/- 10e10");
   if (fZ > 0) return 10e10;
   else        return -10e10;
}

////////////////////////////////////////////////////////////////////////////////
/// Set Pt, Eta and Phi

void TVector3::SetPtEtaPhi(Double_t pt, Double_t eta, Double_t phi) {
   Double_t apt = TMath::Abs(pt);
   SetXYZ(apt*TMath::Cos(phi), apt*TMath::Sin(phi), apt/TMath::Tan(2.0*TMath::ATan(TMath::Exp(-eta))) );
}

////////////////////////////////////////////////////////////////////////////////
/// Set Pt, Theta and Phi

void TVector3::SetPtThetaPhi(Double_t pt, Double_t theta, Double_t phi) {
   fX = pt * TMath::Cos(phi);
   fY = pt * TMath::Sin(phi);
   Double_t tanTheta = TMath::Tan(theta);
   fZ = tanTheta ? pt / tanTheta : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set theta keeping mag and phi constant (BaBar).

void TVector3::SetTheta(Double_t th)
{
   Double_t ma   = Mag();
   Double_t ph   = Phi();
   SetX(ma*TMath::Sin(th)*TMath::Cos(ph));
   SetY(ma*TMath::Sin(th)*TMath::Sin(ph));
   SetZ(ma*TMath::Cos(th));
}

////////////////////////////////////////////////////////////////////////////////
/// Set phi keeping mag and theta constant (BaBar).

void TVector3::SetPhi(Double_t ph)
{
   Double_t xy   = Perp();
   SetX(xy*TMath::Cos(ph));
   SetY(xy*TMath::Sin(ph));
}

////////////////////////////////////////////////////////////////////////////////
/// Return deltaR with respect to v.

Double_t TVector3::DeltaR(const TVector3 & v) const
{
   Double_t deta = Eta()-v.Eta();
   Double_t dphi = TVector2::Phi_mpi_pi(Phi()-v.Phi());
   return TMath::Sqrt( deta*deta+dphi*dphi );
}

////////////////////////////////////////////////////////////////////////////////
/// Setter with mag, theta, phi.

void TVector3::SetMagThetaPhi(Double_t mag, Double_t theta, Double_t phi)
{
   Double_t amag = TMath::Abs(mag);
   fX = amag * TMath::Sin(theta) * TMath::Cos(phi);
   fY = amag * TMath::Sin(theta) * TMath::Sin(phi);
   fZ = amag * TMath::Cos(theta);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TVector3.

void TVector3::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         R__b.ReadClassBuffer(TVector3::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) TObject::Streamer(R__b);
      R__b >> fX;
      R__b >> fY;
      R__b >> fZ;
      R__b.CheckByteCount(R__s, R__c, TVector3::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TVector3::Class(),this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Operator +

TVector3 operator + (const TVector3 & a, const TVector3 & b) {
   return TVector3(a.X() + b.X(), a.Y() + b.Y(), a.Z() + b.Z());
}

////////////////////////////////////////////////////////////////////////////////
/// Operator -

TVector3 operator - (const TVector3 & a, const TVector3 & b) {
   return TVector3(a.X() - b.X(), a.Y() - b.Y(), a.Z() - b.Z());
}

////////////////////////////////////////////////////////////////////////////////
/// Operator *

TVector3 operator * (const TVector3 & p, Double_t a) {
   return TVector3(a*p.X(), a*p.Y(), a*p.Z());
}

////////////////////////////////////////////////////////////////////////////////
/// Operator *

TVector3 operator * (Double_t a, const TVector3 & p) {
   return TVector3(a*p.X(), a*p.Y(), a*p.Z());
}

////////////////////////////////////////////////////////////////////////////////
Double_t operator * (const TVector3 & a, const TVector3 & b) {
   return a.Dot(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Operator *

TVector3 operator * (const TMatrix & m, const TVector3 & v ) {
   return TVector3( m(0,0)*v.X()+m(0,1)*v.Y()+m(0,2)*v.Z(),
                    m(1,0)*v.X()+m(1,1)*v.Y()+m(1,2)*v.Z(),
                    m(2,0)*v.X()+m(2,1)*v.Y()+m(2,2)*v.Z());
}

////////////////////////////////////////////////////////////////////////////////
/// Print vector parameters.

void TVector3::Print(Option_t*)const
{
   Printf("%s %s (x,y,z)=(%f,%f,%f) (rho,theta,phi)=(%f,%f,%f)",GetName(),GetTitle(),X(),Y(),Z(),
                                          Mag(),Theta()*TMath::RadToDeg(),Phi()*TMath::RadToDeg());
}
