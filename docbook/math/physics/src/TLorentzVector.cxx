// @(#)root/physics:$Id$
// Author: Pasha Murat , Peter Malzacher  12/02/99
//    Oct  8 1999: changed Warning to Error and
//                 return fX in Double_t & operator()
//    Oct 20 1999: dito in Double_t operator()
//    Jan 25 2000: implemented as (fP,fE) instead of (fX,fY,fZ,fE)

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
//*-* Adaption to ROOT by Peter Malzacher                                 *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//BEGIN_HTML <!--
/* -->
<H2>TLorentzVector</H2>
<TT>TLorentzVector</TT> is a general four-vector class, which can be used
either for the description of position and time (x,y,z,t) or momentum and
energy (px,py,pz,E).
<BR>&nbsp;
<H3>
Declaration</H3>
<TT>TLorentzVector</TT> has been implemented as a set a <TT>TVector3</TT> and a <TT>Double_t</TT> variable.
By default all components are initialized by zero.

<P><TT>&nbsp; TLorentzVector v1;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // initialized
by (0., 0., 0., 0.)</TT>
<BR><TT>&nbsp; TLorentzVector v2(1., 1., 1., 1.);</TT>
<BR><TT>&nbsp; TLorentzVector v3(v1);</TT>
<BR><TT>&nbsp; TLorentzVector v4(TVector3(1., 2., 3.),4.);</TT>

<P>For backward compatibility there are two constructors from an <TT>Double_t</TT>
and <TT>Float_t</TT>&nbsp; C array.
<BR>&nbsp;

<H3>
Access to the components</H3>
There are two sets of access functions to the components of a<TT> LorentzVector</TT>:
<TT>X(</TT>), <TT>Y()</TT>, <TT>Z()</TT>, <TT>T()</TT> and <TT>Px()</TT>,<TT>
Py()</TT>, <TT>Pz()</TT> and <TT>E()</TT>. Both sets return the same values
but the first set is more relevant for use where <TT>TLorentzVector</TT>
describes a combination of position and time and the second set is more
relevant where <TT>TLorentzVector</TT> describes momentum and energy:

<P><TT>&nbsp; Double_t xx =v.X();</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; Double_t tt = v.T();</TT>

<P><TT>&nbsp; Double_t px = v.Px();</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; Double_t ee = v.E();</TT>

<P>The components of TLorentzVector can also accessed by index:

<P><TT>&nbsp; xx = v(0);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; or&nbsp;&nbsp;&nbsp;&nbsp;
xx = v[0];</TT>
<BR><TT>&nbsp; yy = v(1);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
yy = v[1];</TT>
<BR><TT>&nbsp; zz = v(2);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
zz = v[2];</TT>
<BR><TT>&nbsp; tt = v(3);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
tt = v[3];</TT>

<P>You can use the <TT>Vect()</TT> member function to get the vector component
of <TT>TLorentzVector</TT>:

<P>&nbsp; <TT>TVector3 p = v.Vect();</TT>

<P>For setting components also two sets of member functions can be used:
SetX(),.., SetPx(),..:
<BR>&nbsp;
<BR><TT>&nbsp; v.SetX(1.);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; or&nbsp;&nbsp;&nbsp;
v.SetPx(1.);</TT>
<BR><TT>&nbsp; ...&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
...</TT>
<BR><TT>&nbsp; v.SetT(1.);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
v.SetE(1.);</TT>

<P>To set more the one component by one call you can use the <TT>SetVect()</TT>
function for the <TT>TVector3</TT> part or <TT>SetXYZT()</TT>, <TT>SetPxPyPzE()</TT>. For convenience there is

also a <TT>SetXYZM()</TT>:

<P><TT>&nbsp; v.SetVect(TVector3(1,2,3));</TT>
<BR><TT>&nbsp; v.SetXYZT(x,y,z,t);</TT>
<BR><TT>&nbsp; v.SetPxPyPzE(px,py,pz,e);</TT>
<BR><TT>&nbsp; v.SetXYZM(x,y,z,m);&nbsp;&nbsp; //&nbsp;&nbsp; ->&nbsp;
v=(x,y,z,e=Sqrt(x*x+y*y+z*z+m*m))</TT>
<H3>
Vector components in noncartesian coordinate systems</H3>
There are a couple of memberfunctions to get and set the <TT>TVector3</TT>
part of the parameters in
<BR><B>sherical</B> coordinate systems:

<P><TT>&nbsp; Double_t m, theta, cost, phi, pp, pp2, ppv2, pp2v2;</TT>
<BR><TT>&nbsp; m = v.Rho();</TT>
<BR><TT>&nbsp; t = v.Theta();</TT>
<BR><TT>&nbsp; cost = v.CosTheta();</TT>
<BR><TT>&nbsp; phi = v.Phi();</TT>

<P><TT>&nbsp; v.SetRho(10.);</TT>
<BR><TT>&nbsp; v.SetTheta(TMath::Pi()*.3);</TT>
<BR><TT>&nbsp; v.SetPhi(TMath::Pi());</TT>

<P>or get infoormation about the r-coordinate in <B>cylindrical</B> systems:

<P><TT>&nbsp; Double_t pp, pp2, ppv2, pp2v2;</TT>
<BR><TT>&nbsp; pp = v.Perp();&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// get transvers component</TT>
<BR><TT>&nbsp; pp2 = v.Perp2();&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// get transverse component squared</TT>
<BR><TT>&nbsp; ppv2 = v.Perp(v1);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // get
transvers component with</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// respect to another vector</TT>
<BR><TT>&nbsp; pp2v2 = v.Perp(v1);</TT>

<P>for convenience there are two more set functions <TT>SetPtEtaPhiE(pt,eta,phi,e);
and SetPtEtaPhiM(pt,eta,phi,m);</TT>
<H3>
Arithmetic and comparison operators</H3>
The <TT>TLorentzVecto</TT>r class provides operators to add, subtract or
compare four-vectors:

<P><TT>&nbsp; v3 = -v1;</TT>
<BR><TT>&nbsp; v1 = v2+v3;</TT>
<BR><TT>&nbsp; v1+= v3;</TT>
<BR><TT>&nbsp; v1 = v2 + v3;</TT>
<BR><TT>&nbsp; v1-= v3;</TT>

<P><TT>&nbsp; if (v1 == v2) {...}</TT>
<BR><TT>&nbsp; if(v1 != v3) {...}</TT>
<H3>
Magnitude/Invariant mass, beta, gamma, scalar product</H3>
The scalar product of two four-vectors is calculated with the (-,-,-,+)
metric,
<BR>&nbsp;&nbsp; i.e.&nbsp;&nbsp; <TT><B>s = v1*v2 = </B>t1*t2-x1*x2-y1*y2-z1*z2</TT>
<BR>The magnitude squared <B><TT>mag2</TT></B> of a four-vector is therfore:
<BR>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <TT>mag2<B>
= v*v = </B>t*t-x*x-y*y-z*z</TT>
<BR>It <TT>mag2</TT> is negative <TT>mag = -Sqrt(-mag*mag)</TT>. The member
functions are:

<P><TT>&nbsp; Double_t s, s2;</TT>
<BR><TT>&nbsp; s&nbsp; = v1.Dot(v2);&nbsp;&nbsp;&nbsp;&nbsp; // scalar
product</TT>
<BR><TT>&nbsp; s&nbsp; = v1*v2;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// scalar product</TT>
<BR><TT>&nbsp; s2 = v.Mag2();&nbsp;&nbsp; or&nbsp;&nbsp;&nbsp; s2 = v.M2();</TT>
<BR><TT>&nbsp; s&nbsp; = v.Mag();&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
s&nbsp; = v.M();</TT>

<P>Since in case of momentum and energy the magnitude has the meaning of
invariant mass <TT>TLorentzVector</TT> provides the more meaningful aliases
<TT>M2()</TT> and <TT>M()</TT>;
<P>The member functions <TT>Beta()</TT> and <TT>Gamma()</TT> returns
<TT>beta</TT> and <tt>gamma = 1/Sqrt(1-beta*beta)</tt>.
<H3>
Lorentz boost</H3>
A boost in a general direction can be parameterized with three parameters
which can be taken as the components of a three vector <TT><B>b </B>= (bx,by,bz)</TT>.
With
<BR><TT><B>&nbsp; x</B> = (x,y,z) and gamma = 1/Sqrt(1-beta*beta)</TT> (beta being the module of vector b),
an arbitary active Lorentz boost transformation (from the rod frame
to the original frame) can be written as:
<BR>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <TT><B>x</B>
= <B>x'</B> + (gamma-1)/(beta*beta) * (<B>b</B>*<B>x'</B>) * <B>b</B> +
gamma * t' *<B> b</B></TT>
<BR><B>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </B><TT>t
= gamma (t'+ <B>b</B>*<B>x'</B>).</TT>

<P>The member function <TT>Boost()</TT> performs a boost transformation
from the rod frame to the original frame. <TT>BoostVector()</TT> returns
a <TT>TVector3</TT> of the spatial components divided by the time component:

<P><TT>&nbsp; TVector3 b;</TT>
<BR><TT>&nbsp; v.Boost(bx,by,bz);</TT>
<BR><TT>&nbsp; v.Boost(b);</TT>
<BR><TT>&nbsp; b = v.BoostVector();&nbsp;&nbsp; // b=(x/t,y/t,z/t)</TT>
<H3>
Rotations</H3>
There are four sets of functions to rotate the <TT>TVector3</TT> component
of a <TT>TLorentzVector</TT>:
<H5>
rotation around axes</H5>
<TT>&nbsp; v.RotateX(TMath::Pi()/2.);</TT>
<BR><TT>&nbsp; v.RotateY(.5);</TT>
<BR><TT>&nbsp; v.RotateZ(.99);</TT>
<H5>
rotation around an arbitary axis</H5>
<TT>&nbsp; v.Rotate(TMath::Pi()/4., v1); // rotation around v1</TT>
<H5>
transformation from rotated frame</H5>
<TT>&nbsp; v.RotateUz(direction); //&nbsp; direction must be a unit TVector3</TT>
<H5>
by TRotation (see TRotation)</H5>
<TT>&nbsp; TRotation r;</TT>
<BR><TT>&nbsp; v.Transform(r);&nbsp;&nbsp;&nbsp; or&nbsp;&nbsp;&nbsp;&nbsp;
v *= r; // Attention v=M*v</TT>
<H3>
Misc</H3>

<H5>
Angle between two vectors</H5>
<TT>&nbsp; Double_t a = v1.Angle(v2.Vect());&nbsp; // get angle between v1 and
v2</TT>
<H5>
Light-cone components</H5>
Member functions <TT>Plus()</TT> and <TT>Minus(</TT>) return the positive
and negative light-cone components:<TT></TT>

<P><TT>&nbsp; Double_t pcone = v.Plus();</TT>
<BR><TT>&nbsp; Double_t mcone = v.Minus();</TT>
<P>CAVEAT: The values returned are T{+,-}Z. It is known that some authors
find it easier to define these components as (T{+,-}Z)/sqrt(2). Thus
check what definition is used in the physics you're working in and adapt
your code accordingly.
   
<H5>
Transformation by TLorentzRotation</H5>
A general Lorentz transformation see class <TT>TLorentzRotation</TT> can
be used by the <TT>Transform()</TT> member function, the <TT>*=</TT> or
<TT>*</TT> operator of the <TT>TLorentzRotation</TT> class:<TT></TT>

<P><TT>&nbsp; TLorentzRotation l;</TT>
<BR><TT>&nbsp; v.Transform(l);</TT>
<BR><TT>&nbsp; v = l*v;&nbsp;&nbsp;&nbsp;&nbsp; or&nbsp;&nbsp;&nbsp;&nbsp;
v *= l;&nbsp; // Attention v = l*v</TT>

<P>
<!--*/
// -->END_HTML

#include "TClass.h"
#include "TError.h"
#include "TLorentzVector.h"
#include "TLorentzRotation.h"

ClassImp(TLorentzVector)

TLorentzVector::TLorentzVector(Double_t x, Double_t y, Double_t z, Double_t t)
               : fP(x,y,z), fE(t) {}

TLorentzVector::TLorentzVector(const Double_t * x0)
               : fP(x0), fE(x0[3]) {}

TLorentzVector::TLorentzVector(const Float_t * x0)
               : fP(x0), fE(x0[3]) {}

TLorentzVector::TLorentzVector(const TVector3 & p, Double_t e)
               : fP(p), fE(e) {}

TLorentzVector::TLorentzVector(const TLorentzVector & p) : TObject(p)
               , fP(p.Vect()), fE(p.T()) {}

TLorentzVector::~TLorentzVector()  {}

Double_t TLorentzVector::operator () (int i) const
{
   //dereferencing operator const
   switch(i) {
      case kX:
      case kY:
      case kZ:
         return fP(i);
      case kT:
         return fE;
      default:
         Error("operator()()", "bad index (%d) returning 0",i);
   }
   return 0.;
}

Double_t & TLorentzVector::operator () (int i)
{
   //dereferencing operator
   switch(i) {
      case kX:
      case kY:
      case kZ:
         return fP(i);
      case kT:
         return fE;
      default:
         Error("operator()()", "bad index (%d) returning &fE",i);
   }
   return fE;
}

void TLorentzVector::Boost(Double_t bx, Double_t by, Double_t bz)
{
   //Boost this Lorentz vector
   Double_t b2 = bx*bx + by*by + bz*bz;
   register Double_t gamma = 1.0 / TMath::Sqrt(1.0 - b2);
   register Double_t bp = bx*X() + by*Y() + bz*Z();
   register Double_t gamma2 = b2 > 0 ? (gamma - 1.0)/b2 : 0.0;

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


//______________________________________________________________________________
void TLorentzVector::Print(Option_t *) const
{
  // Print the TLorentz vector components as (x,y,z,t) and (P,eta,phi,E) representations
  Printf("(x,y,z,t)=(%f,%f,%f,%f) (P,eta,phi,E)=(%f,%f,%f,%f)",
	 fP.x(),fP.y(),fP.z(),fE,
	 P(),Eta(),Phi(),fE);
}
