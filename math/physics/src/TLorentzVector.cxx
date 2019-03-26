// @(#)root/physics:$Id$
// Author: Pasha Murat , Peter Malzacher  12/02/99
//    Oct  8 1999: changed Warning to Error and
//                 return fX in Double_t & operator()
//    Oct 20 1999: dito in Double_t operator()
//    Jan 25 2000: implemented as (fP,fE) instead of (fX,fY,fZ,fE)

/** \class TLorentzVector
    \ingroup Physics

## Disclaimer
In order to represent 4-vectors, TLorentzVector shall not be used.
ROOT provides specialisations of the ROOT::Math::LorentzVector template which
are superior from the runtime performance offered, i.e.:
  - ROOT::Math::XYZTVector vector based on x,y,z,t coordinates (cartesian) in double precision
  - ROOT::Math::XYZTVectorF vector based on x,y,z,t coordinates (cartesian) in float precision
  - ROOT::Math::PtEtaPhiEVector vector based on pt (rho),eta,phi and E (t) coordinates in double precision
  - ROOT::Math::PtEtaPhiMVector vector based on pt (rho),eta,phi and M (t) coordinates in double precision
  - ROOT::Math::PxPyPzMVector vector based on px,py,pz and M (mass) coordinates in double precision

More details about the GenVector package can be found [here](Vector.html).

### Description
TLorentzVector is a general four-vector class, which can be used
either for the description of position and time (x,y,z,t) or momentum and
energy (px,py,pz,E).

### Declaration
TLorentzVector has been implemented as a set a TVector3 and a Double_t variable.
By default all components are initialized by zero.

~~~ {.cpp}
  TLorentzVector v1;      // initialized by (0., 0., 0., 0.)
  TLorentzVector v2(1., 1., 1., 1.);
  TLorentzVector v3(v1);
  TLorentzVector v4(TVector3(1., 2., 3.),4.);
~~~

For backward compatibility there are two constructors from an Double_t
and Float_t  C array.


### Access to the components
There are two sets of access functions to the components of a LorentzVector:
X(), Y(), Z(), T() and Px(),
Py(), Pz() and E(). Both sets return the same values
but the first set is more relevant for use where TLorentzVector
describes a combination of position and time and the second set is more
relevant where TLorentzVector describes momentum and energy:

~~~ {.cpp}
  Double_t xx =v.X();
  ...
  Double_t tt = v.T();

  Double_t px = v.Px();
  ...
  Double_t ee = v.E();
~~~

The components of TLorentzVector can also accessed by index:

~~~ {.cpp}
  xx = v(0);       or     xx = v[0];
  yy = v(1);              yy = v[1];
  zz = v(2);              zz = v[2];
  tt = v(3);              tt = v[3];
~~~

You can use the Vect() member function to get the vector component
of TLorentzVector:

~~~ {.cpp}
  TVector3 p = v.Vect();
~~~

For setting components also two sets of member functions can be used:

~~~ {.cpp}
  v.SetX(1.);        or    v.SetPx(1.);
  ...                               ...
  v.SetT(1.);              v.SetE(1.);
~~~

To set more the one component by one call you can use the SetVect()
function for the TVector3 part or SetXYZT(), SetPxPyPzE(). For convenience there is

also a SetXYZM():

~~~ {.cpp}
  v.SetVect(TVector3(1,2,3));
  v.SetXYZT(x,y,z,t);
  v.SetPxPyPzE(px,py,pz,e);
  v.SetXYZM(x,y,z,m);   //   ->  v=(x,y,z,e=Sqrt(x*x+y*y+z*z+m*m))
~~~

### Vector components in non-cartesian coordinate systems
There are a couple of member functions to get and set the TVector3
part of the parameters in
spherical coordinate systems:

~~~ {.cpp}
  Double_t m, theta, cost, phi, pp, pp2, ppv2, pp2v2;
  m = v.Rho();
  t = v.Theta();
  cost = v.CosTheta();
  phi = v.Phi();

  v.SetRho(10.);
  v.SetTheta(TMath::Pi()*.3);
  v.SetPhi(TMath::Pi());
~~~

or get information about the r-coordinate in cylindrical systems:

~~~ {.cpp}
  Double_t pp, pp2, ppv2, pp2v2;
  pp = v.Perp();         // get transvers component
  pp2 = v.Perp2();       // get transverse component squared
  ppv2 = v.Perp(v1);     // get transvers component with
                         // respect to another vector
  pp2v2 = v.Perp(v1);
~~~

for convenience there are two more set functions SetPtEtaPhiE(pt,eta,phi,e);
and SetPtEtaPhiM(pt,eta,phi,m);

### Arithmetic and comparison operators
The TLorentzVector class provides operators to add, subtract or
compare four-vectors:

~~~ {.cpp}
  v3 = -v1;
  v1 = v2+v3;
  v1+= v3;
  v1 = v2 + v3;
  v1-= v3;

  if (v1 == v2) {...}
  if(v1 != v3) {...}
~~~

### Magnitude/Invariant mass, beta, gamma, scalar product
The scalar product of two four-vectors is calculated with the (-,-,-,+)
metric,

   i.e.   `s = v1*v2 = t1*t2-x1*x2-y1*y2-z1*z2`
The magnitude squared mag2 of a four-vector is therefore:

~~~ {.cpp}
          mag2 = v*v = t*t-x*x-y*y-z*z
~~~
It mag2 is negative mag = -Sqrt(-mag*mag). The member
functions are:

~~~ {.cpp}
  Double_t s, s2;
  s  = v1.Dot(v2);     // scalar product
  s  = v1*v2;          // scalar product
  s2 = v.Mag2();   or    s2 = v.M2();
  s  = v.Mag();          s  = v.M();
~~~

Since in case of momentum and energy the magnitude has the meaning of
invariant mass TLorentzVector provides the more meaningful aliases
M2() and M();
The member functions Beta() and Gamma() returns
beta and gamma = 1/Sqrt(1-beta*beta).
### Lorentz boost
A boost in a general direction can be parameterised with three parameters
which can be taken as the components of a three vector b = (bx,by,bz).
With x = (x,y,z) and gamma = 1/Sqrt(1-beta*beta) (beta being the module of vector b),
an arbitrary active Lorentz boost transformation (from the rod frame
to the original frame) can be written as:

~~~ {.cpp}
          x = x' + (gamma-1)/(beta*beta) * (b*x') * b + gamma * t' * b
          t = gamma (t'+ b*x').
~~~

The member function Boost() performs a boost transformation
from the rod frame to the original frame. BoostVector() returns
a TVector3 of the spatial components divided by the time component:

~~~ {.cpp}
  TVector3 b;
  v.Boost(bx,by,bz);
  v.Boost(b);
  b = v.BoostVector();   // b=(x/t,y/t,z/t)
~~~

### Rotations
There are four sets of functions to rotate the TVector3 component
of a TLorentzVector:

#### rotation around axes

~~~ {.cpp}
  v.RotateX(TMath::Pi()/2.);
  v.RotateY(.5);
  v.RotateZ(.99);
~~~

#### rotation around an arbitrary axis
  v.Rotate(TMath::Pi()/4., v1); // rotation around v1

#### transformation from rotated frame

~~~ {.cpp}
  v.RotateUz(direction); //  direction must be a unit TVector3
~~~

#### by TRotation (see TRotation)

~~~ {.cpp}
  TRotation r;
  v.Transform(r);    or     v *= r; // Attention v=M*v
~~~

### Misc

#### Angle between two vectors

~~~ {.cpp}
  Double_t a = v1.Angle(v2.Vect());  // get angle between v1 and v2
~~~

#### Light-cone components
Member functions Plus() and Minus() return the positive
and negative light-cone components:

~~~ {.cpp}
  Double_t pcone = v.Plus();
  Double_t mcone = v.Minus();
~~~

CAVEAT: The values returned are T{+,-}Z. It is known that some authors
find it easier to define these components as (T{+,-}Z)/sqrt(2). Thus
check what definition is used in the physics you're working in and adapt
your code accordingly.

#### Transformation by TLorentzRotation
A general Lorentz transformation see class TLorentzRotation can
be used by the Transform() member function, the *= or
* operator of the TLorentzRotation class:

~~~ {.cpp}
  TLorentzRotation l;
  v.Transform(l);
  v = l*v;     or     v *= l;  // Attention v = l*v
~~~
*/

#include "TLorentzVector.h"

#include "TBuffer.h"
#include "TString.h"
#include "TLorentzRotation.h"

ClassImp(TLorentzVector);


void TLorentzVector::Boost(Double_t bx, Double_t by, Double_t bz)
{
   //Boost this Lorentz vector
   Double_t b2 = bx*bx + by*by + bz*bz;
   Double_t gamma = 1.0 / TMath::Sqrt(1.0 - b2);
   Double_t bp = bx*X() + by*Y() + bz*Z();
   Double_t gamma2 = b2 > 0 ? (gamma - 1.0)/b2 : 0.0;

   SetX(X() + gamma2*bp*bx + gamma*bx*T());
   SetY(Y() + gamma2*bp*by + gamma*by*T());
   SetZ(Z() + gamma2*bp*bz + gamma*bz*T());
   SetT(gamma*(T() + bp));
}

Double_t TLorentzVector::Rapidity() const
{
   //return rapidity
   return 0.5*log( (E()+Pz()) / (E()-Pz()) );
}

TLorentzVector &TLorentzVector::operator *= (const TLorentzRotation & m)
{
   //multiply this Lorentzvector by m
   return *this = m.VectorMultiplication(*this);
}

TLorentzVector &TLorentzVector::Transform(const TLorentzRotation & m)
{
   //Transform this Lorentzvector
   return *this = m.VectorMultiplication(*this);
}

void TLorentzVector::Streamer(TBuffer &R__b)
{
   // Stream an object of class TLorentzVector.
   Double_t x, y, z;
   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 3) {
         R__b.ReadClassBuffer(TLorentzVector::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v != 2) TObject::Streamer(R__b);
      R__b >> x;
      R__b >> y;
      R__b >> z;
      fP.SetXYZ(x,y,z);
      R__b >> fE;
      R__b.CheckByteCount(R__s, R__c, TLorentzVector::IsA());
   } else {
      R__b.WriteClassBuffer(TLorentzVector::Class(),this);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Print the TLorentz vector components as (x,y,z,t) and (P,eta,phi,E) representations

void TLorentzVector::Print(Option_t *) const
{
  Printf("(x,y,z,t)=(%f,%f,%f,%f) (P,eta,phi,E)=(%f,%f,%f,%f)",
    fP.x(),fP.y(),fP.z(),fE,
    P(),Eta(),Phi(),fE);
}
