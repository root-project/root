// @(#)root/physics:$Id$
// Author: Peter Malzacher   19/06/99

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
TRotation</H2>
The TRotation class describes a rotation of objects of the TVector3 class.
It is a 3*3 matrix of Double_t:

<P><TT>| xx&nbsp; xy&nbsp; xz |</TT>
<BR><TT>| yx&nbsp; yy&nbsp; yz |</TT>
<BR><TT>| zx&nbsp; zy&nbsp; zz |</TT>

<P>It describes a so called active rotation, i.e. rotation of objects inside
a static system of coordinates. In case you want to rotate the frame and
want to know the coordinates of objects in the rotated system, you should
apply the inverse rotation to the objects. If you want to transform coordinates
from the rotated frame to the original frame you have to apply the direct
transformation.

<P>A rotation around a specified axis means counterclockwise rotation around
the positive direction of the axis.
<BR>&nbsp;
<H3>
Declaration, Access, Comparisons</H3>
<TT>&nbsp; TRotation r;&nbsp;&nbsp;&nbsp; // r initialized as identity</TT>
<BR><TT>&nbsp; TRotation m(r); // m = r</TT>

<P>There is no direct way to to set the matrix elements - to ensure that
a <TT>TRotation</TT> object always describes a real rotation. But you can get the
values by the member functions <TT>XX()..ZZ()</TT> or the<TT> (,)</TT>
operator:

<P><TT>&nbsp; Double_t xx = r.XX();&nbsp;&nbsp;&nbsp;&nbsp; //&nbsp; the
same as xx=r(0,0)</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; xx
= r(0,0);</TT>

<P><TT>&nbsp; if (r==m) {...}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// test for equality</TT>
<BR><TT>&nbsp; if (r!=m) {..}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// test for inequality</TT>
<BR><TT>&nbsp; if (r.IsIdentity()) {...} // test for identity</TT>
<BR>&nbsp;
<H3>
Rotation around axes</H3>
The following matrices desrcibe counterclockwise rotations around coordinate
axes

<P><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 1&nbsp;&nbsp; 0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
0&nbsp;&nbsp;&nbsp; |</TT>
<BR><TT>Rx(a) = | 0 cos(a) -sin(a) |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 0 sin(a) cos(a)&nbsp;
|</TT>

<P><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | cos(a)&nbsp; 0 sin(a)
|</TT>
<BR><TT>Ry(a) = |&nbsp;&nbsp; 0&nbsp;&nbsp;&nbsp;&nbsp; 1&nbsp;&nbsp;&nbsp;
0&nbsp;&nbsp; |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | -sin(a) 0 cos(a) |</TT>

<P><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | cos(a) -sin(a) 0 |</TT>
<BR><TT>Rz(a) = | sin(a) cos(a) 0 |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp; 0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
0&nbsp;&nbsp;&nbsp;&nbsp; 1 |</TT>
<BR>and are implemented as member functions <TT>RotateX()</TT>, <TT>RotateY()</TT>
and <TT>RotateZ()</TT>:

<P><TT>&nbsp; r.RotateX(TMath::Pi()); // rotation around the x-axis</TT>
<H3>
Rotation around arbitary axis</H3>
The member function <TT>Rotate()</TT> allows to rotate around an arbitary vector
(not neccessary a unit one) and returns the result.

<P><TT>&nbsp; r.Rotate(TMath::Pi()/3,TVector3(3,4,5));</TT>

<P>It is possible to find a unit vector and an angle, which describe the
same rotation as the current one:

<P><TT>&nbsp; Double_t angle;</TT>
<BR><TT>&nbsp; TVector3 axis;</TT>
<BR><TT>&nbsp; r.GetAngleAxis(angle,axis);</TT>
<H3>
Rotation of local axes</H3>
Member function <TT>RotateAxes()</TT> adds a rotation of local axes to
the current rotation and returns the result:

<P><TT>&nbsp; TVector3 newX(0,1,0);</TT>
<BR><TT>&nbsp; TVector3 newY(0,0,1);</TT>
<BR><TT>&nbsp; TVector3 newZ(1,0,0);</TT>
<BR><TT>&nbsp; a.RotateAxes(newX,newY,newZ);</TT>

<P>Member functions <TT>ThetaX()</TT>, <TT>ThetaY()</TT>, <TT>ThetaZ()</TT>,
<TT>PhiX()</TT>, <TT>PhiY()</TT>,<TT>PhiZ()</TT> return azimuth and polar
angles of the rotated axes:

<P><TT>&nbsp; Double_t tx,ty,tz,px,py,pz;</TT>
<BR><TT>&nbsp; tx= a.ThetaX();</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; pz= a.PhiZ();</TT>

<H3>
Setting The Rotations</H3>
The member function <TT>SetToIdentity()</TT> will set the rotation object 
to the identity (no rotation).

With a minor caveat, the Euler angles of the rotation may be set using 
<TT>SetXEulerAngles()</TT> or individually set with <TT>SetXPhi()</TT>, 
<TT>SetXTheta()</TT>, and <TT>SetXPsi()</TT>.  These routines set the Euler 
angles using the X-convention which is defined by a rotation about the Z-axis,
about the new X-axis, and about the new Z-axis.  This is the convention used
in Landau and Lifshitz, Goldstein and other common physics texts.  The 
Y-convention euler angles can be set with <TT>SetYEulerAngles()</TT>,
<TT>SetYPhi()</TT>, <TT>SetYTheta()</TT>, and <TT>SetYPsi()</TT>.  The caveat 
is that Euler angles usually define the rotation of the new coordinate system 
with respect to the original system, however, the TRotation class specifies 
the rotation of the object in the original system (an active rotation).  To 
recover the usual Euler rotations (ie. rotate the system not the object), you 
must take the inverse of the rotation.

The member functions <TT>SetXAxis()</TT>, <TT>SetYAxis()</TT>, and 
<TT>SetZAxis()</TT> will create a rotation which rotates the requested axis
of the object to be parallel to a vector.  If used with one argument, the 
rotation about that axis is arbitrary.  If used with two arguments, the
second variable defines the <TT>XY</TT>, <TT>YZ</TT>, or <TT>ZX</TT> 
respectively.

<H3>
Inverse rotation</H3>
<TT>&nbsp; TRotation a,b;</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; b = a.Inverse();&nbsp; // b is inverse of a, a is unchanged</TT>
<BR><TT>&nbsp; b = a.Invert();&nbsp;&nbsp; // invert a and set b = a</TT>
<H3>
Compound Rotations</H3>
The <TT>operator *</TT> has been implemented in a way that follows the 
mathematical notation of a product of the two matrices which describe the 
two consecutive rotations. Therefore the second rotation should be placed 
first:

<P><TT>&nbsp; r = r2 * r1;</TT>
<H3>
Rotation of TVector3</H3>
The TRotation class provides an <TT>operator *</TT> which allows to express
a rotation of a <TT>TVector3</TT> analog to the mathematical notation

<P><TT>&nbsp; | x' |&nbsp;&nbsp; | xx xy xz | | x |</TT>
<BR><TT>&nbsp; | y' | = | yx yy yz | | y |</TT>
<BR><TT>&nbsp; | z' |&nbsp;&nbsp; | zx zy zz | | z |</TT><TT></TT>

<P>e.g.:

<P><TT>&nbsp; TVector3 v(1,1,1);</TT>
<BR><TT>&nbsp; v = r * v;</TT><TT></TT>

<P>You can also use the <TT>Transform()</TT> member function or the o<TT>perator
*=</TT> of the
<BR>TVector3 class:<TT></TT>

<P><TT>&nbsp; TVector3 v;</TT>
<BR><TT>&nbsp; TRotation r;</TT>
<BR><TT>&nbsp; v.Transform(r);</TT>
<BR><TT>&nbsp; v *= r;&nbsp; //Attention v = r * v</TT>
<!--*/
// -->END_HTML
//

#include "TRotation.h"
#include "TMath.h"
#include "TQuaternion.h"
#include "TError.h"

ClassImp(TRotation)

#define TOLERANCE (1.0E-6)

TRotation::TRotation()
: fxx(1.0), fxy(0.0), fxz(0.0), fyx(0.0), fyy(1.0), fyz(0.0),
  fzx(0.0), fzy(0.0), fzz(1.0) {}

TRotation::TRotation(const TRotation & m) : TObject(m),
  fxx(m.fxx), fxy(m.fxy), fxz(m.fxz), fyx(m.fyx), fyy(m.fyy), fyz(m.fyz),
  fzx(m.fzx), fzy(m.fzy), fzz(m.fzz) {}

TRotation::TRotation(Double_t mxx, Double_t mxy, Double_t mxz,
                         Double_t myx, Double_t myy, Double_t myz,
                         Double_t mzx, Double_t mzy, Double_t mzz)
: fxx(mxx), fxy(mxy), fxz(mxz), fyx(myx), fyy(myy), fyz(myz),
  fzx(mzx), fzy(mzy), fzz(mzz) {}


Double_t TRotation::operator() (int i, int j) const {
   //dereferencing operator const
   if (i == 0) {
      if (j == 0) { return fxx; }
      if (j == 1) { return fxy; }
      if (j == 2) { return fxz; }
   } else if (i == 1) {
      if (j == 0) { return fyx; }
      if (j == 1) { return fyy; }
      if (j == 2) { return fyz; }
   } else if (i == 2) {
      if (j == 0) { return fzx; }
      if (j == 1) { return fzy; }
      if (j == 2) { return fzz; }
   }
  
   Warning("operator()(i,j)", "bad indices (%d , %d)",i,j);

   return 0.0;
}

TRotation TRotation::operator* (const TRotation & b) const {
   //multiplication operator
   return TRotation(fxx*b.fxx + fxy*b.fyx + fxz*b.fzx,
                    fxx*b.fxy + fxy*b.fyy + fxz*b.fzy,
                    fxx*b.fxz + fxy*b.fyz + fxz*b.fzz,
                    fyx*b.fxx + fyy*b.fyx + fyz*b.fzx,
                    fyx*b.fxy + fyy*b.fyy + fyz*b.fzy,
                    fyx*b.fxz + fyy*b.fyz + fyz*b.fzz,
                    fzx*b.fxx + fzy*b.fyx + fzz*b.fzx,
                    fzx*b.fxy + fzy*b.fyy + fzz*b.fzy,
                    fzx*b.fxz + fzy*b.fyz + fzz*b.fzz);
}

//_____________________________________
TRotation::TRotation(const TQuaternion & Q) {
   // Constructor for a rotation based on a Quaternion
   // if magnitude of quaternion is null, creates identity rotation
   // if quaternion is non-unit, creates rotation corresponding to the normalized (unit) quaternion


   double two_r2 = 2 * Q.fRealPart * Q.fRealPart;
   double two_x2 = 2 * Q.fVectorPart.X() * Q.fVectorPart.X();
   double two_y2 = 2 * Q.fVectorPart.Y() * Q.fVectorPart.Y();
   double two_z2 = 2 * Q.fVectorPart.Z() * Q.fVectorPart.Z();
   double two_xy = 2 * Q.fVectorPart.X() * Q.fVectorPart.Y();
   double two_xz = 2 * Q.fVectorPart.X() * Q.fVectorPart.Z();
   double two_xr = 2 * Q.fVectorPart.X() * Q.fRealPart;
   double two_yz = 2 * Q.fVectorPart.Y() * Q.fVectorPart.Z();
   double two_yr = 2 * Q.fVectorPart.Y() * Q.fRealPart;
   double two_zr = 2 * Q.fVectorPart.Z() * Q.fRealPart;

   // protect agains zero quaternion
   double mag2 = Q.QMag2();
   if (mag2 > 0) {

      // diago + identity
      fxx = two_r2 + two_x2;
      fyy = two_r2 + two_y2;
      fzz = two_r2 + two_z2;

      //        line 0 column 1 and conjugate
      fxy = two_xy - two_zr;
      fyx = two_xy + two_zr;

      //        line 0 column 2 and conjugate
      fxz = two_xz + two_yr;
      fzx = two_xz - two_yr;

      //        line 1 column 2 and conjugate
      fyz = two_yz - two_xr;
      fzy = two_yz + two_xr;

      // protect agains non-unit quaternion 
      if (TMath::Abs(mag2-1) > 1e-10) {
         fxx /= mag2;
         fyy /= mag2;
         fzz /= mag2;
         fxy /= mag2;
         fyx /= mag2;
         fxz /= mag2;
         fzx /= mag2;
         fyz /= mag2;
         fzy /= mag2;
      }

      // diago : remove identity
      fxx -= 1;
      fyy -= 1;
      fzz -= 1;


   } else {
      // Identity

      fxx = fyy = fzz = 1;
      fxy = fyx = fxz = fzx = fyz = fzy = 0;

   }

}

TRotation & TRotation::Rotate(Double_t a, const TVector3& axis) {
   //rotate along an axis
   if (a != 0.0) {
      Double_t ll = axis.Mag();
      if (ll == 0.0) {
         Warning("Rotate(angle,axis)"," zero axis");
      } else {
         Double_t sa = TMath::Sin(a), ca = TMath::Cos(a);
         Double_t dx = axis.X()/ll, dy = axis.Y()/ll, dz = axis.Z()/ll;
         TRotation m(
             ca+(1-ca)*dx*dx,          (1-ca)*dx*dy-sa*dz,    (1-ca)*dx*dz+sa*dy,
             (1-ca)*dy*dx+sa*dz, ca+(1-ca)*dy*dy,          (1-ca)*dy*dz-sa*dx,
             (1-ca)*dz*dx-sa*dy,    (1-ca)*dz*dy+sa*dx, ca+(1-ca)*dz*dz );
         Transform(m);
      }
   }
   return *this;
}

TRotation & TRotation::RotateX(Double_t a) {
   //rotate around x
   Double_t c = TMath::Cos(a);
   Double_t s = TMath::Sin(a);
   Double_t x = fyx, y = fyy, z = fyz;
   fyx = c*x - s*fzx;
   fyy = c*y - s*fzy;
   fyz = c*z - s*fzz;
   fzx = s*x + c*fzx;
   fzy = s*y + c*fzy;
   fzz = s*z + c*fzz;
   return *this;
}

TRotation & TRotation::RotateY(Double_t a){
   //rotate around y
   Double_t c = TMath::Cos(a);
   Double_t s = TMath::Sin(a);
   Double_t x = fzx, y = fzy, z = fzz;
   fzx = c*x - s*fxx;
   fzy = c*y - s*fxy;
   fzz = c*z - s*fxz;
   fxx = s*x + c*fxx;
   fxy = s*y + c*fxy;
   fxz = s*z + c*fxz;
   return *this;
}

TRotation & TRotation::RotateZ(Double_t a) {
   //rotate around z
   Double_t c = TMath::Cos(a);
   Double_t s = TMath::Sin(a);
   Double_t x = fxx, y = fxy, z = fxz;
   fxx = c*x - s*fyx;
   fxy = c*y - s*fyy;
   fxz = c*z - s*fyz;
   fyx = s*x + c*fyx;
   fyy = s*y + c*fyy;
   fyz = s*z + c*fyz;
   return *this;
}

TRotation & TRotation::RotateAxes(const TVector3 &newX,
                                  const TVector3 &newY,
                                  const TVector3 &newZ) {
   //rotate axes
   Double_t del = 0.001;
   TVector3 w = newX.Cross(newY);

   if (TMath::Abs(newZ.X()-w.X()) > del ||
       TMath::Abs(newZ.Y()-w.Y()) > del ||
       TMath::Abs(newZ.Z()-w.Z()) > del ||
       TMath::Abs(newX.Mag2()-1.) > del ||
       TMath::Abs(newY.Mag2()-1.) > del ||
       TMath::Abs(newZ.Mag2()-1.) > del ||
       TMath::Abs(newX.Dot(newY)) > del ||
       TMath::Abs(newY.Dot(newZ)) > del ||
       TMath::Abs(newZ.Dot(newX)) > del) {
      Warning("RotateAxes","bad axis vectors");
      return *this;
   } else {
      return Transform(TRotation(newX.X(), newY.X(), newZ.X(),
                                 newX.Y(), newY.Y(), newZ.Y(),
                                 newX.Z(), newY.Z(), newZ.Z()));
   }
}

Double_t TRotation::PhiX() const {
   //return Phi
   return (fyx == 0.0 && fxx == 0.0) ? 0.0 : TMath::ATan2(fyx,fxx);
}

Double_t TRotation::PhiY() const {
   //return Phi
   return (fyy == 0.0 && fxy == 0.0) ? 0.0 : TMath::ATan2(fyy,fxy);
}

Double_t TRotation::PhiZ() const {
   //return Phi
   return (fyz == 0.0 && fxz == 0.0) ? 0.0 : TMath::ATan2(fyz,fxz);
}

Double_t TRotation::ThetaX() const {
   //return Phi
   return TMath::ACos(fzx);
}

Double_t TRotation::ThetaY() const {
   //return Theta
   return TMath::ACos(fzy);
}

Double_t TRotation::ThetaZ() const {
   //return Theta
   return TMath::ACos(fzz);
}

void TRotation::AngleAxis(Double_t &angle, TVector3 &axis) const {
   //rotation defined by an angle and a vector
   Double_t cosa  = 0.5*(fxx+fyy+fzz-1);
   Double_t cosa1 = 1-cosa;
   if (cosa1 <= 0) {
      angle = 0;
      axis  = TVector3(0,0,1);
   } else {
      Double_t x=0, y=0, z=0;
      if (fxx > cosa) x = TMath::Sqrt((fxx-cosa)/cosa1);
      if (fyy > cosa) y = TMath::Sqrt((fyy-cosa)/cosa1);
      if (fzz > cosa) z = TMath::Sqrt((fzz-cosa)/cosa1);
      if (fzy < fyz)  x = -x;
      if (fxz < fzx)  y = -y;
      if (fyx < fxy)  z = -z;
      angle = TMath::ACos(cosa);
      axis  = TVector3(x,y,z);
   }
}

TRotation & TRotation::SetXEulerAngles(Double_t phi,
                                      Double_t theta,
                                      Double_t psi) {
   // Rotate using the x-convention (Landau and Lifshitz, Goldstein, &c) by 
   // doing the explicit rotations.  This is slightly less efficient than 
   // directly applying the rotation, but makes the code much clearer.  My
   // presumption is that this code is not going to be a speed bottle neck.

   SetToIdentity();
   RotateZ(phi);
   RotateX(theta);
   RotateZ(psi);
  
   return *this;
}

TRotation & TRotation::SetYEulerAngles(Double_t phi,
                                       Double_t theta,
                                       Double_t psi) {
   // Rotate using the y-convention.
    
   SetToIdentity();
   RotateZ(phi);
   RotateY(theta);
   RotateZ(psi);
   return *this;
}

TRotation & TRotation::RotateXEulerAngles(Double_t phi,
                                         Double_t theta,
                                         Double_t psi) {
   // Rotate using the x-convention.
   TRotation euler;
   euler.SetXEulerAngles(phi,theta,psi);
   return Transform(euler);
}

TRotation & TRotation::RotateYEulerAngles(Double_t phi,
                                          Double_t theta,
                                          Double_t psi) {
   // Rotate using the y-convention.
   TRotation euler;
   euler.SetYEulerAngles(phi,theta,psi);
   return Transform(euler);
}

void TRotation::SetXPhi(Double_t phi) {
   //set XPhi
   SetXEulerAngles(phi,GetXTheta(),GetXPsi());
}

void TRotation::SetXTheta(Double_t theta) {
   //set XTheta
   SetXEulerAngles(GetXPhi(),theta,GetXPsi());
}

void TRotation::SetXPsi(Double_t psi) {
   //set XPsi
   SetXEulerAngles(GetXPhi(),GetXTheta(),psi);
}

void TRotation::SetYPhi(Double_t phi) {
   //set YPhi
   SetYEulerAngles(phi,GetYTheta(),GetYPsi());
}

void TRotation::SetYTheta(Double_t theta) {
   //set YTheta
   SetYEulerAngles(GetYPhi(),theta,GetYPsi());
}

void TRotation::SetYPsi(Double_t psi) {
   //set YPsi
   SetYEulerAngles(GetYPhi(),GetYTheta(),psi);
}

Double_t TRotation::GetXPhi(void) const {
   //return phi angle
   Double_t finalPhi;

   Double_t s2 =  1.0 - fzz*fzz;
   if (s2 < 0) {
      Warning("GetPhi()"," |fzz| > 1 ");
      s2 = 0;
   }
   const Double_t sinTheta = TMath::Sqrt(s2);

   if (sinTheta != 0) {
      const Double_t cscTheta = 1/sinTheta;
      Double_t cosAbsPhi =  fzy * cscTheta;
      if ( TMath::Abs(cosAbsPhi) > 1 ) {        // NaN-proofing
         Warning("GetPhi()","finds | cos phi | > 1");
         cosAbsPhi = 1;
      }
      const Double_t absPhi = TMath::ACos(cosAbsPhi);
      if (fzx > 0) {
         finalPhi = absPhi;
      } else if (fzx < 0) {
         finalPhi = -absPhi;
      } else if (fzy > 0) {
         finalPhi = 0.0;
      } else {
         finalPhi = TMath::Pi();
      }
   } else {              // sinTheta == 0 so |Fzz| = 1
      const Double_t absPhi = .5 * TMath::ACos (fxx);
      if (fxy > 0) {
         finalPhi =  -absPhi;
      } else if (fxy < 0) {
         finalPhi =   absPhi;
      } else if (fxx>0) {
         finalPhi = 0.0;
      } else {
         finalPhi = fzz * TMath::PiOver2();
      }
   }
   return finalPhi;
}

Double_t TRotation::GetYPhi(void) const {
   //return YPhi
   return GetXPhi() + TMath::Pi()/2.0;
}

Double_t TRotation::GetXTheta(void) const {
   //return XTheta
   return  ThetaZ();
}

Double_t TRotation::GetYTheta(void) const {
   //return YTheta
   return  ThetaZ();
}

Double_t TRotation::GetXPsi(void) const {
   //Get psi angle
   double finalPsi = 0.0;

   Double_t s2 =  1.0 - fzz*fzz;
   if (s2 < 0) {
      Warning("GetPsi()"," |fzz| > 1 ");
      s2 = 0;
   }
   const Double_t sinTheta = TMath::Sqrt(s2);

   if (sinTheta != 0) {
      const Double_t cscTheta = 1/sinTheta;
      Double_t cosAbsPsi =  - fyz * cscTheta;
      if ( TMath::Abs(cosAbsPsi) > 1 ) {        // NaN-proofing
         Warning("GetPsi()","| cos psi | > 1 ");
         cosAbsPsi = 1;
      }
      const Double_t absPsi = TMath::ACos(cosAbsPsi);
      if (fxz > 0) {
         finalPsi = absPsi;
      } else if (fxz < 0) {
         finalPsi = -absPsi;
      } else {
         finalPsi = (fyz < 0) ? 0 : TMath::Pi();
      }
   } else {              // sinTheta == 0 so |Fzz| = 1
      Double_t absPsi = fxx;
      if ( TMath::Abs(fxx) > 1 ) {        // NaN-proofing
         Warning("GetPsi()","| fxx | > 1 ");
         absPsi = 1;
      }
      absPsi = .5 * TMath::ACos (absPsi);
      if (fyx > 0) {
         finalPsi = absPsi;
      } else if (fyx < 0) {
         finalPsi = -absPsi;
      } else {
         finalPsi = (fxx > 0) ? 0 : TMath::PiOver2();
      }
   }
   return finalPsi;
}

Double_t TRotation::GetYPsi(void) const {
   //return YPsi
   return GetXPsi() - TMath::Pi()/2;
}

TRotation & TRotation::SetXAxis(const TVector3& axis, 
                                const TVector3& xyPlane) {
   //set X axis
   TVector3 xAxis(xyPlane);
   TVector3 yAxis;
   TVector3 zAxis(axis);
   MakeBasis(xAxis,yAxis,zAxis);
   fxx = zAxis.X();  fyx = zAxis.Y();  fzx = zAxis.Z();
   fxy = xAxis.X();  fyy = xAxis.Y();  fzy = xAxis.Z();
   fxz = yAxis.X();  fyz = yAxis.Y();  fzz = yAxis.Z();
   return *this;
}

TRotation & TRotation::SetXAxis(const TVector3& axis) {
   //set X axis
   TVector3 xyPlane(0.0,1.0,0.0);
   return SetXAxis(axis,xyPlane);
}

TRotation & TRotation::SetYAxis(const TVector3& axis, 
                                const TVector3& yzPlane) {
   //set Y axis
   TVector3 xAxis(yzPlane);
   TVector3 yAxis;
   TVector3 zAxis(axis);
   MakeBasis(xAxis,yAxis,zAxis);
   fxx = yAxis.X();  fyx = yAxis.Y();  fzx = yAxis.Z();
   fxy = zAxis.X();  fyy = zAxis.Y();  fzy = zAxis.Z();
   fxz = xAxis.X();  fyz = xAxis.Y();  fzz = xAxis.Z();
   return *this;
}

TRotation & TRotation::SetYAxis(const TVector3& axis) {
   //set Y axis
   TVector3 yzPlane(0.0,0.0,1.0);
   return SetYAxis(axis,yzPlane);
}

TRotation & TRotation::SetZAxis(const TVector3& axis, 
                                const TVector3& zxPlane) {
   //set Z axis
   TVector3 xAxis(zxPlane);
   TVector3 yAxis;
   TVector3 zAxis(axis);
   MakeBasis(xAxis,yAxis,zAxis);
   fxx = xAxis.X();  fyx = xAxis.Y();  fzx = xAxis.Z();
   fxy = yAxis.X();  fyy = yAxis.Y();  fzy = yAxis.Z();
   fxz = zAxis.X();  fyz = zAxis.Y();  fzz = zAxis.Z();
   return *this;
}

TRotation & TRotation::SetZAxis(const TVector3& axis) {
   //set Z axis
   TVector3 zxPlane(1.0,0.0,0.0);
   return SetZAxis(axis,zxPlane);
}

void TRotation::MakeBasis(TVector3& xAxis,
                          TVector3& yAxis,
                          TVector3& zAxis) const {
   // Make the Z axis into a unit variable. 
   Double_t zmag = zAxis.Mag();
   if (zmag<TOLERANCE) {
      Warning("MakeBasis(X,Y,Z)","non-zero Z Axis is required");
   }
   zAxis *= (1.0/zmag);

   Double_t xmag = xAxis.Mag();
   if (xmag<TOLERANCE*zmag) {
      xAxis = zAxis.Orthogonal();
      xmag = 1.0;
   }

   // Find the Y axis
   yAxis = zAxis.Cross(xAxis)*(1.0/xmag);
   Double_t ymag = yAxis.Mag();
   if (ymag<TOLERANCE*zmag) {
      yAxis = zAxis.Orthogonal();
   } else {
      yAxis *= (1.0/ymag);
   }

   xAxis = yAxis.Cross(zAxis);
}
