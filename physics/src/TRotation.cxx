// @(#)root/physics:$Name:  $:$Id: TRotation.cxx,v 1.1.1.1 2000/05/16 17:00:45 rdm Exp $
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
//BEGIN_HTML
/*
</pre>
<H2>
TRotation</H2>
The TRotation class describes a rotation of objects of the TVector3 class.
It is a 3*3 matrix of Double_t:

<P><TT>| xx&nbsp; xy&nbsp; xz |</TT>
<BR><TT>| yx&nbsp; yy&nbsp; yz |</TT>
<BR><TT>| zx&nbsp; zy&nbsp; zz |</TT>

<P>It describes a socalled active rotation, i.e. rotation of objects inside
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
<BR><TT>Rx(a) = | 1 cos(a) -sin(a) |</TT>
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
<BR><TT>&nbsp; a.RotateAxes(newX,newX,newZ);</TT>

<P>Memberfunctions <TT>ThetaX()</TT>, <TT>ThetaY()</TT>, <TT>ThetaZ()</TT>,
<TT>PhiX()</TT>, <TT>PhiY()</TT>,<TT>PhiZ()</TT> return azimuth and polar
angles of the rotated axes:

<P><TT>&nbsp; Double_t tx,ty,tz,px,py,pz;</TT>
<BR><TT>&nbsp; tx= a.ThetaX();</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; pz= a.PhiZ();</TT>
<H3>
Inverse rotation</H3>
<TT>&nbsp; TRotation a,b;</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; b = a.Inverse();&nbsp; // b is inverse of a, a is unchanged</TT>
<BR><TT>&nbsp; b = a.Invert();&nbsp;&nbsp; // invert a and set b = a</TT>
<H3>
Compound Rotations</H3>
The <TT>operator *</TT> has been implemented in a way that follows the mathematical
notation of a product of the two matrices which describe the two consecutiv
rotations. Therefore the second rotation should be placed first:

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
<pre>
*/
//END_HTML
//

#include "TRotation.h"
#include "TError.h"

ClassImp(TRotation)

TRotation::TRotation()
: fxx(1.0), fxy(0.0), fxz(0.0), fyx(0.0), fyy(1.0), fyz(0.0),
  fzx(0.0), fzy(0.0), fzz(1.0) {}

TRotation::TRotation(const TRotation & m)
: fxx(m.fxx), fxy(m.fxy), fxz(m.fxz), fyx(m.fyx), fyy(m.fyy), fyz(m.fyz),
  fzx(m.fzx), fzy(m.fzy), fzz(m.fzz) {}

TRotation::TRotation(Double_t mxx, Double_t mxy, Double_t mxz,
			 Double_t myx, Double_t myy, Double_t myz,
			 Double_t mzx, Double_t mzy, Double_t mzz)
: fxx(mxx), fxy(mxy), fxz(mxz), fyx(myx), fyy(myy), fyz(myz),
  fzx(mzx), fzy(mzy), fzz(mzz) {}


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

  Warning("operator()(i,j)", "bad indeces (%d , %d)",i,j);

  return 0.0;
}

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

TRotation & TRotation::Rotate(Double_t a, const TVector3& axis) {
  if (a != 0.0) {
    Double_t ll = axis.Mag();
    if (ll == 0.0) {
      Warning("Rotate(angle,axis)"," zero axis");
    }else{
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
  }else{
    return Transform(TRotation(newX.X(), newY.X(), newZ.X(),
                                 newX.Y(), newY.Y(), newZ.Y(),
                                 newX.Z(), newY.Z(), newZ.Z()));
  }
}

Double_t TRotation::PhiX() const {
  return (fyx == 0.0 && fxx == 0.0) ? 0.0 : TMath::ATan2(fyx,fxx);
}

Double_t TRotation::PhiY() const {
  return (fyy == 0.0 && fxy == 0.0) ? 0.0 : TMath::ATan2(fyy,fxy);
}

Double_t TRotation::PhiZ() const {
  return (fyz == 0.0 && fxz == 0.0) ? 0.0 : TMath::ATan2(fyz,fxz);
}

Double_t TRotation::ThetaX() const {
  return TMath::ACos(fzx);
}


Double_t TRotation::ThetaY() const {
  return TMath::ACos(fzy);
}

Double_t TRotation::ThetaZ() const {
  return TMath::ACos(fzz);
}

void TRotation::AngleAxis(Double_t &angle, TVector3 &axis) const {
  Double_t cosa  = 0.5*(fxx+fyy+fzz-1);
  Double_t cosa1 = 1-cosa;
  if (cosa1 <= 0) {
    angle = 0;
    axis  = TVector3(0,0,1);
  }else{
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
