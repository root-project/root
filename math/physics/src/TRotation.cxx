// @(#)root/physics:$Id$
// Author: Peter Malzacher   19/06/99

/** \class TRotation
    \ingroup Physics

The TRotation class describes a rotation of objects of the TVector3 class.
It is a 3*3 matrix of Double_t:

~~~
| xx  xy  xz |
| yx  yy  yz |
| zx  zy  zz |
~~~

It describes a so called active rotation, i.e. rotation of objects inside
a static system of coordinates. In case you want to rotate the frame and
want to know the coordinates of objects in the rotated system, you should
apply the inverse rotation to the objects. If you want to transform coordinates
from the rotated frame to the original frame you have to apply the direct
transformation.

A rotation around a specified axis means counterclockwise rotation around
the positive direction of the axis.


### Declaration, Access, Comparisons

~~~
  TRotation r;    // r initialized as identity
  TRotation m(r); // m = r
~~~

There is no direct way to set the matrix elements - to ensure that
a TRotation object always describes a real rotation. But you can get the
values by the member functions XX()..ZZ() or the (,)
operator:

~~~
  Double_t xx = r.XX();     //  the same as xx=r(0,0)
           xx = r(0,0);

  if (r==m) {...}          // test for equality
  if (r!=m) {..}           // test for inequality
  if (r.IsIdentity()) {...} // test for identity
~~~

### Rotation around axes
The following matrices describe counterclockwise rotations around coordinate
axes

~~~
        | 1   0      0     |
Rx(a) = | 0 cos(a) -sin(a) |
        | 0 sin(a) cos(a)  |

        | cos(a)  0 sin(a) |
Ry(a) = |   0     1   0    |
        | -sin(a) 0 cos(a) |

        | cos(a) -sin(a) 0 |
Rz(a) = | sin(a) cos(a) 0  |
        |   0     0     1  |
~~~

and are implemented as member functions RotateX(), RotateY() and RotateZ():

~~~
  r.RotateX(TMath::Pi()); // rotation around the x-axis
~~~

### Rotation around arbitrary axis
The member function Rotate() allows to rotate around an arbitrary vector
(not necessary a unit one) and returns the result.

~~~
  r.Rotate(TMath::Pi()/3,TVector3(3,4,5));
~~~

It is possible to find a unit vector and an angle, which describe the
same rotation as the current one:

~~~
  Double_t angle;
  TVector3 axis;
  r.GetAngleAxis(angle,axis);
~~~

### Rotation of local axes
Member function RotateAxes() adds a rotation of local axes to
the current rotation and returns the result:

~~~
  TVector3 newX(0,1,0);
  TVector3 newY(0,0,1);
  TVector3 newZ(1,0,0);
  a.RotateAxes(newX,newY,newZ);
~~~

Member functions ThetaX(), ThetaY(), ThetaZ(),
PhiX(), PhiY(),PhiZ() return azimuth and polar
angles of the rotated axes:

~~~
  Double_t tx,ty,tz,px,py,pz;
  tx= a.ThetaX();
  ...
  pz= a.PhiZ();
~~~

### Setting The Rotations
The member function SetToIdentity() will set the rotation object
to the identity (no rotation).

With a minor caveat, the Euler angles of the rotation may be set using
SetXEulerAngles() or individually set with SetXPhi(),
SetXTheta(), and SetXPsi().  These routines set the Euler
angles using the X-convention which is defined by a rotation about the Z-axis,
about the new X-axis, and about the new Z-axis.  This is the convention used
in Landau and Lifshitz, Goldstein and other common physics texts.  The
Y-convention Euler angles can be set with SetYEulerAngles(),
SetYPhi(), SetYTheta(), and SetYPsi().  The caveat
is that Euler angles usually define the rotation of the new coordinate system
with respect to the original system, however, the TRotation class specifies
the rotation of the object in the original system (an active rotation).  To
recover the usual Euler rotations (ie. rotate the system not the object), you
must take the inverse of the rotation.

The member functions SetXAxis(), SetYAxis(), and
SetZAxis() will create a rotation which rotates the requested axis
of the object to be parallel to a vector.  If used with one argument, the
rotation about that axis is arbitrary.  If used with two arguments, the
second variable defines the XY, YZ, or ZX
respectively.


### Inverse rotation

~~~
  TRotation a,b;
  ...
  b = a.Inverse();  // b is inverse of a, a is unchanged
  b = a.Invert();   // invert a and set b = a
~~~

### Compound Rotations
The operator * has been implemented in a way that follows the
mathematical notation of a product of the two matrices which describe the
two consecutive rotations. Therefore the second rotation should be placed
first:

~~~
  r = r2 * r1;
~~~

### Rotation of TVector3
The TRotation class provides an operator * which allows to express
a rotation of a TVector3 analog to the mathematical notation

~~~
  | x' |   | xx xy xz | | x |
  | y' | = | yx yy yz | | y |
  | z' |   | zx zy zz | | z |
~~~

e.g.:

~~~
  TVector3 v(1,1,1);
  v = r * v;
~~~

You can also use the Transform() member function or the operator *= of the
TVector3 class:

~~~
  TVector3 v;
  TRotation r;
  v.Transform(r);
  v *= r;  //Attention v = r * v
~~~
*/

#include "TRotation.h"
#include "TMath.h"
#include "TQuaternion.h"

ClassImp(TRotation);

#define TOLERANCE (1.0E-6)

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TRotation::TRotation()
: fxx(1.0), fxy(0.0), fxz(0.0), fyx(0.0), fyy(1.0), fyz(0.0),
  fzx(0.0), fzy(0.0), fzz(1.0) {}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TRotation::TRotation(const TRotation & m) : TObject(m),
  fxx(m.fxx), fxy(m.fxy), fxz(m.fxz), fyx(m.fyx), fyy(m.fyy), fyz(m.fyz),
  fzx(m.fzx), fzy(m.fzy), fzz(m.fzz) {}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TRotation::TRotation(Double_t mxx, Double_t mxy, Double_t mxz,
                         Double_t myx, Double_t myy, Double_t myz,
                         Double_t mzx, Double_t mzy, Double_t mzz)
: fxx(mxx), fxy(mxy), fxz(mxz), fyx(myx), fyy(myy), fyz(myz),
  fzx(mzx), fzy(mzy), fzz(mzz) {}

////////////////////////////////////////////////////////////////////////////////
/// Dereferencing operator const.

Double_t TRotation::operator() (int i, int j) const {
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

////////////////////////////////////////////////////////////////////////////////
/// Multiplication operator.

TRotation TRotation::operator* (const TRotation & b) const {
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

////////////////////////////////////////////////////////////////////////////////
/// Constructor for a rotation based on a Quaternion
/// if magnitude of quaternion is null, creates identity rotation
/// if quaternion is non-unit, creates rotation corresponding to the normalized (unit) quaternion

TRotation::TRotation(const TQuaternion & Q) {

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

////////////////////////////////////////////////////////////////////////////////
/// Rotate along an axis.

TRotation & TRotation::Rotate(Double_t a, const TVector3& axis) {
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

////////////////////////////////////////////////////////////////////////////////
/// Rotate around x.

TRotation & TRotation::RotateX(Double_t a) {
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

////////////////////////////////////////////////////////////////////////////////
/// Rotate around y.

TRotation & TRotation::RotateY(Double_t a){
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

////////////////////////////////////////////////////////////////////////////////
/// Rotate around z.

TRotation & TRotation::RotateZ(Double_t a) {
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

////////////////////////////////////////////////////////////////////////////////
/// Rotate axes.

TRotation & TRotation::RotateAxes(const TVector3 &newX,
                                  const TVector3 &newY,
                                  const TVector3 &newZ) {
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

////////////////////////////////////////////////////////////////////////////////
/// Return Phi.

Double_t TRotation::PhiX() const {
   return (fyx == 0.0 && fxx == 0.0) ? 0.0 : TMath::ATan2(fyx,fxx);
}

////////////////////////////////////////////////////////////////////////////////
/// Return Phi.

Double_t TRotation::PhiY() const {
   return (fyy == 0.0 && fxy == 0.0) ? 0.0 : TMath::ATan2(fyy,fxy);
}

////////////////////////////////////////////////////////////////////////////////
/// Return Phi.

Double_t TRotation::PhiZ() const {
   return (fyz == 0.0 && fxz == 0.0) ? 0.0 : TMath::ATan2(fyz,fxz);
}

////////////////////////////////////////////////////////////////////////////////
/// Return Theta.

Double_t TRotation::ThetaX() const {
   return TMath::ACos(fzx);
}

////////////////////////////////////////////////////////////////////////////////
/// Return Theta.

Double_t TRotation::ThetaY() const {
   return TMath::ACos(fzy);
}

////////////////////////////////////////////////////////////////////////////////
/// Return Theta.

Double_t TRotation::ThetaZ() const {
   return TMath::ACos(fzz);
}

////////////////////////////////////////////////////////////////////////////////
/// Rotation defined by an angle and a vector.

void TRotation::AngleAxis(Double_t &angle, TVector3 &axis) const {
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

////////////////////////////////////////////////////////////////////////////////
/// Rotate using the x-convention (Landau and Lifshitz, Goldstein, &c) by
/// doing the explicit rotations.  This is slightly less efficient than
/// directly applying the rotation, but makes the code much clearer.  My
/// presumption is that this code is not going to be a speed bottle neck.

TRotation & TRotation::SetXEulerAngles(Double_t phi,
                                      Double_t theta,
                                      Double_t psi) {
   SetToIdentity();
   RotateZ(phi);
   RotateX(theta);
   RotateZ(psi);

   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate using the y-convention.

TRotation & TRotation::SetYEulerAngles(Double_t phi,
                                       Double_t theta,
                                       Double_t psi) {
   SetToIdentity();
   RotateZ(phi);
   RotateY(theta);
   RotateZ(psi);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate using the x-convention.

TRotation & TRotation::RotateXEulerAngles(Double_t phi,
                                         Double_t theta,
                                         Double_t psi) {
   TRotation euler;
   euler.SetXEulerAngles(phi,theta,psi);
   return Transform(euler);
}

////////////////////////////////////////////////////////////////////////////////
/// Rotate using the y-convention.

TRotation & TRotation::RotateYEulerAngles(Double_t phi,
                                          Double_t theta,
                                          Double_t psi) {
   TRotation euler;
   euler.SetYEulerAngles(phi,theta,psi);
   return Transform(euler);
}

////////////////////////////////////////////////////////////////////////////////
/// Set XPhi.

void TRotation::SetXPhi(Double_t phi) {
   SetXEulerAngles(phi,GetXTheta(),GetXPsi());
}

////////////////////////////////////////////////////////////////////////////////
/// Set XTheta.

void TRotation::SetXTheta(Double_t theta) {
   SetXEulerAngles(GetXPhi(),theta,GetXPsi());
}

////////////////////////////////////////////////////////////////////////////////
/// Set XPsi.

void TRotation::SetXPsi(Double_t psi) {
   SetXEulerAngles(GetXPhi(),GetXTheta(),psi);
}

////////////////////////////////////////////////////////////////////////////////
/// Set YPhi.

void TRotation::SetYPhi(Double_t phi) {
   SetYEulerAngles(phi,GetYTheta(),GetYPsi());
}

////////////////////////////////////////////////////////////////////////////////
/// Set YTheta.

void TRotation::SetYTheta(Double_t theta) {
   SetYEulerAngles(GetYPhi(),theta,GetYPsi());
}

////////////////////////////////////////////////////////////////////////////////
/// Set YPsi.

void TRotation::SetYPsi(Double_t psi) {
   SetYEulerAngles(GetYPhi(),GetYTheta(),psi);
}

////////////////////////////////////////////////////////////////////////////////
/// Return phi angle.

Double_t TRotation::GetXPhi(void) const {
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

////////////////////////////////////////////////////////////////////////////////
/// Return YPhi.

Double_t TRotation::GetYPhi(void) const {
   return GetXPhi() + TMath::Pi()/2.0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return XTheta.

Double_t TRotation::GetXTheta(void) const {
   return  ThetaZ();
}

////////////////////////////////////////////////////////////////////////////////
/// Return YTheta.

Double_t TRotation::GetYTheta(void) const {
   return  ThetaZ();
}

////////////////////////////////////////////////////////////////////////////////
/// Get psi angle.

Double_t TRotation::GetXPsi(void) const {
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

////////////////////////////////////////////////////////////////////////////////
/// Return YPsi.

Double_t TRotation::GetYPsi(void) const {
   return GetXPsi() - TMath::Pi()/2;
}

////////////////////////////////////////////////////////////////////////////////
/// Set X axis.

TRotation & TRotation::SetXAxis(const TVector3& axis,
                                const TVector3& xyPlane) {
   TVector3 xAxis(xyPlane);
   TVector3 yAxis;
   TVector3 zAxis(axis);
   MakeBasis(xAxis,yAxis,zAxis);
   fxx = zAxis.X();  fyx = zAxis.Y();  fzx = zAxis.Z();
   fxy = xAxis.X();  fyy = xAxis.Y();  fzy = xAxis.Z();
   fxz = yAxis.X();  fyz = yAxis.Y();  fzz = yAxis.Z();
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Set X axis.

TRotation & TRotation::SetXAxis(const TVector3& axis) {
   TVector3 xyPlane(0.0,1.0,0.0);
   return SetXAxis(axis,xyPlane);
}

////////////////////////////////////////////////////////////////////////////////
/// Set Y axis.

TRotation & TRotation::SetYAxis(const TVector3& axis,
                                const TVector3& yzPlane) {
   TVector3 xAxis(yzPlane);
   TVector3 yAxis;
   TVector3 zAxis(axis);
   MakeBasis(xAxis,yAxis,zAxis);
   fxx = yAxis.X();  fyx = yAxis.Y();  fzx = yAxis.Z();
   fxy = zAxis.X();  fyy = zAxis.Y();  fzy = zAxis.Z();
   fxz = xAxis.X();  fyz = xAxis.Y();  fzz = xAxis.Z();
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Set Y axis.

TRotation & TRotation::SetYAxis(const TVector3& axis) {
   TVector3 yzPlane(0.0,0.0,1.0);
   return SetYAxis(axis,yzPlane);
}

////////////////////////////////////////////////////////////////////////////////
/// Set Z axis.

TRotation & TRotation::SetZAxis(const TVector3& axis,
                                const TVector3& zxPlane) {
   TVector3 xAxis(zxPlane);
   TVector3 yAxis;
   TVector3 zAxis(axis);
   MakeBasis(xAxis,yAxis,zAxis);
   fxx = xAxis.X();  fyx = xAxis.Y();  fzx = xAxis.Z();
   fxy = yAxis.X();  fyy = yAxis.Y();  fzy = yAxis.Z();
   fxz = zAxis.X();  fyz = zAxis.Y();  fzz = zAxis.Z();
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Set Z axis.

TRotation & TRotation::SetZAxis(const TVector3& axis) {
   TVector3 zxPlane(1.0,0.0,0.0);
   return SetZAxis(axis,zxPlane);
}

////////////////////////////////////////////////////////////////////////////////
/// Make the Z axis into a unit variable.

void TRotation::MakeBasis(TVector3& xAxis,
                          TVector3& yAxis,
                          TVector3& zAxis) const {
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
