// @(#)root/physics:$Id$
// Author: Pasha Murat, Peter Malzacher   12/02/99
//    Aug 11 1999: added Pt == 0 guard to Eta()
//    Oct  8 1999: changed Warning to Error and
//                 return fX in Double_t & operator()
//    Oct 20 1999: Bug fix: sign in PseudoRapidity
//                 Warning-> Error in Double_t operator()

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*The Physics Vector package *-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==========================                       *
//*-* The Physics Vector package consists of five classes:                *
//*-*   - TVector2                                                        *
//*-*   - TVector3                                                        *
//*-*   - TRotation                                                       *
//*-*   - TLorentzVector                                                  *
//*-*   - TLorentzRotation                                                *
//*-* It is a combination of CLHEPs Vector package written by             *
//*-* Leif Lonnblad, Andreas Nilsson and Evgueni Tcherniaev               *
//*-* and a ROOT package written by Pasha Murat.                          *
//*-* for CLHEP see:  http://wwwinfo.cern.ch/asd/lhc++/clhep/             *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//BEGIN_HTML <!--
/* -->
<H2>
TVector3</H2>
<TT>TVector3</TT> is a general three vector class, which can be used for
the description of different vectors in 3D.
<H3>
Declaration / Access to the components</H3>
<TT>TVector3</TT> has been implemented as a vector of three <TT>Double_t</TT>
variables, representing the cartesian coordinates. By default all components
are initialized to zero:

<P><TT>&nbsp; TVector3 v1;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; //
v1 = (0,0,0)</TT>
<BR><TT>&nbsp; TVector3 v2(1);&nbsp;&nbsp;&nbsp;&nbsp; // v2 = (1,0,0)</TT>
<BR><TT>&nbsp; TVector3 v3(1,2,3); // v3 = (1,2,3)</TT>
<BR><TT>&nbsp; TVector3 v4(v2);&nbsp;&nbsp;&nbsp; // v4 = v2</TT>

<P>It is also possible (but not recommended) to initialize a <TT>TVector3</TT>
with a <TT>Double_t</TT> or <TT>Float_t</TT> C array.

<P>You can get the basic components either by name or by index using <TT>operator()</TT>:

<P><TT>&nbsp; xx = v1.X();&nbsp;&nbsp;&nbsp; or&nbsp;&nbsp;&nbsp; xx =
v1(0);</TT>
<BR><TT>&nbsp; yy = v1.Y();&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
yy = v1(1);</TT>
<BR><TT>&nbsp; zz = v1.Z();&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
zz = v1(2);</TT>

<P>The memberfunctions <TT>SetX()</TT>, <TT>SetY()</TT>, <TT>SetZ()</TT>
and<TT> SetXYZ()</TT> allow to set the components:

<P><TT>&nbsp; v1.SetX(1.); v1.SetY(2.); v1.SetZ(3.);</TT>
<BR><TT>&nbsp; v1.SetXYZ(1.,2.,3.);</TT>
<BR>&nbsp;
<H3>
Noncartesian coordinates</H3>
To get information on the <TT>TVector3</TT> in spherical (rho,phi,theta)
or cylindrical (z,r,theta) coordinates, the
<BR>the member functions <TT>Mag()</TT> (=magnitude=rho in spherical coordinates),
<TT>Mag2()</TT>, <TT>Theta()</TT>, <TT>CosTheta()</TT>, <TT>Phi()</TT>,
<TT>Perp()</TT> (the transverse component=r in cylindrical coordinates),
<TT>Perp2()</TT> can be used:

<P><TT>&nbsp; Double_t m&nbsp; = v.Mag();&nbsp;&nbsp;&nbsp; // get magnitude
(=rho=Sqrt(x*x+y*y+z*z)))</TT>
<BR><TT>&nbsp; Double_t m2 = v.Mag2();&nbsp;&nbsp; // get magnitude squared</TT>
<BR><TT>&nbsp; Double_t t&nbsp; = v.Theta();&nbsp; // get polar angle</TT>
<BR><TT>&nbsp; Double_t ct = v.CosTheta();// get cos of theta</TT>
<BR><TT>&nbsp; Double_t p&nbsp; = v.Phi();&nbsp;&nbsp;&nbsp; // get azimuth
angle</TT>
<BR><TT>&nbsp; Double_t pp = v.Perp();&nbsp;&nbsp; // get transverse component</TT>
<BR><TT>&nbsp; Double_t pp2= v.Perp2();&nbsp; // get transvers component
squared</TT>

<P>It is also possible to get the transverse component with respect to
another vector:

<P><TT>&nbsp; Double_t ppv1 = v.Perp(v1);</TT>
<BR><TT>&nbsp; Double_t pp2v1 = v.Perp2(v1);</TT>

<P>The pseudo-rapidity ( eta=-ln (tan (theta/2)) ) can be obtained by <TT>Eta()</TT>
or <TT>PseudoRapidity()</TT>:
<BR>&nbsp;
<BR><TT>&nbsp; Double_t eta = v.PseudoRapidity();</TT>

<P>There are set functions to change one of the noncartesian coordinates:

<P><TT>&nbsp; v.SetTheta(.5); // keeping rho and phi</TT>
<BR><TT>&nbsp; v.SetPhi(.8);&nbsp;&nbsp; // keeping rho and theta</TT>
<BR><TT>&nbsp; v.SetMag(10.);&nbsp; // keeping theta and phi</TT>
<BR><TT>&nbsp; v.SetPerp(3.);&nbsp; // keeping z and phi</TT>
<BR>&nbsp;
<H3>
Arithmetic / Comparison</H3>
The <TT>TVector3</TT> class provides the operators to add, subtract, scale and compare
vectors:

<P><TT>&nbsp; v3&nbsp; = -v1;</TT>
<BR><TT>&nbsp; v1&nbsp; = v2+v3;</TT>
<BR><TT>&nbsp; v1 += v3;</TT>
<BR><TT>&nbsp; v1&nbsp; = v1 - v3</TT>
<BR><TT>&nbsp; v1 -= v3;</TT>
<BR><TT>&nbsp; v1 *= 10;</TT>
<BR><TT>&nbsp; v1&nbsp; = 5*v2;</TT>

<P><TT>&nbsp; if(v1==v2) {...}</TT>
<BR><TT>&nbsp; if(v1!=v2) {...}</TT>
<BR>&nbsp;
<H3>
Related Vectors</H3>
<TT>&nbsp; v2 = v1.Unit();&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // get unit
vector parallel to v1</TT>
<BR><TT>&nbsp; v2 = v1.Orthogonal(); // get vector orthogonal to v1</TT>
<H3>
Scalar and vector products</H3>
<TT>&nbsp; s = v1.Dot(v2);&nbsp;&nbsp; // scalar product</TT>
<BR><TT>&nbsp; s = v1 * v2;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // scalar product</TT>
<BR><TT>&nbsp; v = v1.Cross(v2); // vector product</TT>
<H3>
&nbsp;Angle between two vectors</H3>
<TT>&nbsp; Double_t a = v1.Angle(v2);</TT>
<H3>
Rotations</H3>

<H5>
Rotation around axes</H5>
<TT>&nbsp; v.RotateX(.5);</TT>
<BR><TT>&nbsp; v.RotateY(TMath::Pi());</TT>
<BR><TT>&nbsp; v.RotateZ(angle);</TT>
<H5>
Rotation around a vector</H5>
<TT>&nbsp; v1.Rotate(TMath::Pi()/4, v2); // rotation around v2</TT>
<H5>
Rotation by TRotation</H5>
<TT>TVector3</TT> objects can be rotated by objects of the <TT>TRotation</TT>
class using the <TT>Transform()</TT> member functions,
<BR>the <TT>operator *=</TT> or the <TT>operator *</TT> of the TRotation
class:

<P><TT>&nbsp; TRotation m;</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; v1.transform(m);</TT>
<BR><TT>&nbsp; v1 = m*v1;</TT>
<BR><TT>&nbsp; v1 *= m; // Attention v1 = m*v1</TT>
<H5>
Transformation from rotated frame</H5>
<TT>&nbsp; TVector3 direction = v.Unit()</TT>
<BR><TT>&nbsp; v1.RotateUz(direction); // direction must be TVector3 of
unit length</TT>

<P>transforms v1 from the rotated frame (z' parallel to direction, x' in
the theta plane and y' in the xy plane as well as perpendicular to the
theta plane) to the (x,y,z) frame.

<!--*/
// -->END_HTML
//

#include "TVector3.h"
#include "TRotation.h"
#include "TMath.h"
#include "TClass.h"

ClassImp(TVector3)

//______________________________________________________________________________
TVector3::TVector3(const TVector3 & p) : TObject(p),
  fX(p.fX), fY(p.fY), fZ(p.fZ) {}

TVector3::TVector3(Double_t xx, Double_t yy, Double_t zz)
: fX(xx), fY(yy), fZ(zz) {}

TVector3::TVector3(const Double_t * x0)
: fX(x0[0]), fY(x0[1]), fZ(x0[2]) {}

TVector3::TVector3(const Float_t * x0)
: fX(x0[0]), fY(x0[1]), fZ(x0[2]) {}

TVector3::~TVector3() {}

//______________________________________________________________________________
Double_t TVector3::operator () (int i) const {
   //dereferencing operator const
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

//______________________________________________________________________________
Double_t & TVector3::operator () (int i) {
   //dereferencing operator
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

//______________________________________________________________________________
TVector3 & TVector3::operator *= (const TRotation & m){
   //multiplication operator
   return *this = m * (*this);
}

//______________________________________________________________________________
TVector3 & TVector3::Transform(const TRotation & m) {
   //transform this vector with a TRotation
   return *this = m * (*this);
}

//______________________________________________________________________________
Double_t TVector3::Angle(const TVector3 & q) const 
{
   // return the angle w.r.t. another 3-vector
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

//______________________________________________________________________________
Double_t TVector3::Mag() const 
{ 
   // return the magnitude (rho in spherical coordinate system)
   
   return TMath::Sqrt(Mag2()); 
}

//______________________________________________________________________________
Double_t TVector3::Perp() const 
{ 
   //return the transverse component  (R in cylindrical coordinate system)

   return TMath::Sqrt(Perp2()); 
}


//______________________________________________________________________________
Double_t TVector3::Perp(const TVector3 & p) const
{ 
   //return the transverse component (R in cylindrical coordinate system)

   return TMath::Sqrt(Perp2(p)); 
}

//______________________________________________________________________________
Double_t TVector3::Phi() const 
{
   //return the  azimuth angle. returns phi from -pi to pi
   return fX == 0.0 && fY == 0.0 ? 0.0 : TMath::ATan2(fY,fX);
}

//______________________________________________________________________________
Double_t TVector3::Theta() const 
{
   //return the polar angle
   return fX == 0.0 && fY == 0.0 && fZ == 0.0 ? 0.0 : TMath::ATan2(Perp(),fZ);
}

//______________________________________________________________________________
TVector3 TVector3::Unit() const 
{
   // return unit vector parallel to this.
   Double_t  tot = Mag2();
   TVector3 p(fX,fY,fZ);
   return tot > 0.0 ? p *= (1.0/TMath::Sqrt(tot)) : p;
}

//______________________________________________________________________________
void TVector3::RotateX(Double_t angle) {
   //rotate vector around X
   Double_t s = TMath::Sin(angle);
   Double_t c = TMath::Cos(angle);
   Double_t yy = fY;
   fY = c*yy - s*fZ;
   fZ = s*yy + c*fZ;
}

//______________________________________________________________________________
void TVector3::RotateY(Double_t angle) {
   //rotate vector around Y
   Double_t s = TMath::Sin(angle);
   Double_t c = TMath::Cos(angle);
   Double_t zz = fZ;
   fZ = c*zz - s*fX;
   fX = s*zz + c*fX;
}

//______________________________________________________________________________
void TVector3::RotateZ(Double_t angle) {
   //rotate vector around Z
   Double_t s = TMath::Sin(angle);
   Double_t c = TMath::Cos(angle);
   Double_t xx = fX;
   fX = c*xx - s*fY;
   fY = s*xx + c*fY;
}

//______________________________________________________________________________
void TVector3::Rotate(Double_t angle, const TVector3 & axis){
   //rotate vector
   TRotation trans;
   trans.Rotate(angle, axis);
   operator*=(trans);
}

//______________________________________________________________________________
void TVector3::RotateUz(const TVector3& NewUzVector) {
   // NewUzVector must be normalized !

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

//______________________________________________________________________________
Double_t TVector3::PseudoRapidity() const {
   //Double_t m = Mag();
   //return 0.5*log( (m+fZ)/(m-fZ) );
   // guard against Pt=0
   double cosTheta = CosTheta();
   if (cosTheta*cosTheta < 1) return -0.5* TMath::Log( (1.0-cosTheta)/(1.0+cosTheta) );
   Warning("PseudoRapidity","transvers momentum = 0! return +/- 10e10");
   if (fZ > 0) return 10e10;
   else        return -10e10;
}

//______________________________________________________________________________
void TVector3::SetPtEtaPhi(Double_t pt, Double_t eta, Double_t phi) {
   //set Pt, Eta and Phi
   Double_t apt = TMath::Abs(pt);
   SetXYZ(apt*TMath::Cos(phi), apt*TMath::Sin(phi), apt/TMath::Tan(2.0*TMath::ATan(TMath::Exp(-eta))) );
}

//______________________________________________________________________________
void TVector3::SetPtThetaPhi(Double_t pt, Double_t theta, Double_t phi) {
   //set Pt, Theta and Phi
   fX = pt * TMath::Cos(phi);
   fY = pt * TMath::Sin(phi); 
   Double_t tanTheta = TMath::Tan(theta);
   fZ = tanTheta ? pt / tanTheta : 0;
}

//______________________________________________________________________________
void TVector3::SetTheta(Double_t th) 
{
   // Set theta keeping mag and phi constant (BaBar).
   Double_t ma   = Mag();
   Double_t ph   = Phi();
   SetX(ma*TMath::Sin(th)*TMath::Cos(ph));
   SetY(ma*TMath::Sin(th)*TMath::Sin(ph));
   SetZ(ma*TMath::Cos(th));
}

//______________________________________________________________________________
void TVector3::SetPhi(Double_t ph) 
{
   // Set phi keeping mag and theta constant (BaBar).
   Double_t xy   = Perp();
   SetX(xy*TMath::Cos(ph));
   SetY(xy*TMath::Sin(ph));
}

//______________________________________________________________________________
Double_t TVector3::DeltaR(const TVector3 & v) const 
{
   //return deltaR with respect to v
   Double_t deta = Eta()-v.Eta();
   Double_t dphi = TVector2::Phi_mpi_pi(Phi()-v.Phi());
   return TMath::Sqrt( deta*deta+dphi*dphi );
}

//______________________________________________________________________________
void TVector3::SetMagThetaPhi(Double_t mag, Double_t theta, Double_t phi) 
{
   //setter with mag, theta, phi
   Double_t amag = TMath::Abs(mag);
   fX = amag * TMath::Sin(theta) * TMath::Cos(phi);
   fY = amag * TMath::Sin(theta) * TMath::Sin(phi);
   fZ = amag * TMath::Cos(theta);
}

//______________________________________________________________________________
void TVector3::Streamer(TBuffer &R__b)
{
   // Stream an object of class TVector3.

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

TVector3 operator + (const TVector3 & a, const TVector3 & b) {
   return TVector3(a.X() + b.X(), a.Y() + b.Y(), a.Z() + b.Z());
}

TVector3 operator - (const TVector3 & a, const TVector3 & b) {
   return TVector3(a.X() - b.X(), a.Y() - b.Y(), a.Z() - b.Z());
}

TVector3 operator * (const TVector3 & p, Double_t a) {
   return TVector3(a*p.X(), a*p.Y(), a*p.Z());
}

TVector3 operator * (Double_t a, const TVector3 & p) {
   return TVector3(a*p.X(), a*p.Y(), a*p.Z());
}

Double_t operator * (const TVector3 & a, const TVector3 & b) {
   return a.Dot(b);
}

TVector3 operator * (const TMatrix & m, const TVector3 & v ) {
   return TVector3( m(0,0)*v.X()+m(0,1)*v.Y()+m(0,2)*v.Z(),
                    m(1,0)*v.X()+m(1,1)*v.Y()+m(1,2)*v.Z(),
                    m(2,0)*v.X()+m(2,1)*v.Y()+m(2,2)*v.Z());
}


//const TVector3 kXHat(1.0, 0.0, 0.0);
//const TVector3 kYHat(0.0, 1.0, 0.0);
//const TVector3 kZHat(0.0, 0.0, 1.0);

void TVector3::Print(Option_t*)const
{
   //print vector parameters
   Printf("%s %s (x,y,z)=(%f,%f,%f) (rho,theta,phi)=(%f,%f,%f)",GetName(),GetTitle(),X(),Y(),Z(),
                                          Mag(),Theta()*TMath::RadToDeg(),Phi()*TMath::RadToDeg());
}
