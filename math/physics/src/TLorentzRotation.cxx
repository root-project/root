// @(#)root/physics:$Id$
// Author: Peter Malzacher   19/06/99


/** \class TLorentzRotation
    \ingroup Physics
The TLorentzRotation class describes Lorentz transformations including
Lorentz boosts and rotations (see TRotation)

~~~
            | xx  xy  xz  xt |
            |                |
            | yx  yy  yz  yt |
   lambda = |                |
            | zx  zy  zz  zt |
            |                |
            | tx  ty  tz  tt |
~~~

### Declaration
By default it is initialized to the identity matrix, but it may also be
intialized by an other TLorentzRotation,
by a pure TRotation or by a boost:

 TLorentzRotation l; // l is
initialized as identity
 TLorentzRotation m(l); // m = l
 TRotation r;
 TLorentzRotation lr(r);
 TLorentzRotation lb1(bx,by,bz);
 TVector3 b;
 TLorentzRotation lb2(b);

The Matrix for a Lorentz boosts is:

~~~
 | 1+gamma'*bx*bx  gamma'*bx*by   gamma'*bx*bz  gamma*bx |
 |  gamma'*by*bx  1+gamma'*by*by  gamma'*by*bz  gamma*by |
 |  gamma'*bz*bx   gamma'*bz*by  1+gamma'*bz*bz gamma*bz |
 |    gamma*bx       gamma*by       gamma*bz     gamma   |
~~~

with the boost vector b=(bx,by,bz) and gamma=1/Sqrt(1-beta*beta)
and gamma'=(gamma-1)/beta*beta.
### Access to the matrix components/Comparisons
Access to the matrix components is possible through the member functions
XX(), XY() .. TT(),
through the operator (int,int):

~~~
 Double_t xx;
 TLorentzRotation l;
 xx = l.XX(); // gets the xx component
 xx = l(0,0); // gets the xx component

 if (l==m) {...} // test for equality
 if (l !=m) {...} // test for inequality
 if (l.IsIdentity()) {...} // test for identity
~~~

### Transformations of a LorentzRotation

#### Compound transformations
There are four possibilities to find the product of two TLorentzRotation
transformations:

~~~
 TLorentzRotation a,b,c;
 c = b*a;// product
 c = a.MatrixMultiplication(b); // a is unchanged
 a *= b;// Attention: a=a*b
 c = a.Transform(b)// a=b*a then c=a
~~~

#### Lorentz boosts

~~~
 Double_t bx, by, bz;
 TVector3 v(bx,by,bz);
 TLorentzRotation l;
 l.Boost(v);
 l.Boost(bx,by,bz);
~~~

#### Rotations

~~~
 TVector3 axis;
 l.RotateX(TMath::Pi()); // rotation around x-axis
 l.Rotate(.5,axis);// rotation around specified vector
~~~

#### Inverse transformation
The matrix for the inverse transformation of a TLorentzRotation is as follows:

~~~
            | xx  yx  zx -tx |
            |                |
            | xy  yy  zy -ty |
            |                |
            | xz  yz  zz -tz |
            |                |
            |-xt -yt -zt  tt |
~~~

To return the inverse transformation keeping the current unchanged
use the member function Inverse().
Invert() inverts the current TLorentzRotation:

~~~
 l1 = l2.Inverse(); // l1 is inverse of l2, l2 unchanged
 l1 = l2.Invert(); // invert l2, then l1=l2
~~~

### Transformation of a TLorentzVector
To apply TLorentzRotation to TLorentzVector you can use
either the VectorMultiplication() member function or the *
operator. You can also use the Transform() function and the *=
operator of the TLorentzVector class.:

~~~
 TLorentzVector v;
 ...
 v=l.VectorMultiplication(v);
 v = l * v;

 v.Transform(l);
 v *= l; // Attention v = l*v
~~~
*/

#include "TLorentzRotation.h"

ClassImp(TLorentzRotation);

TLorentzRotation::TLorentzRotation()
  : fxx(1.0), fxy(0.0), fxz(0.0), fxt(0.0),
    fyx(0.0), fyy(1.0), fyz(0.0), fyt(0.0),
    fzx(0.0), fzy(0.0), fzz(1.0), fzt(0.0),
    ftx(0.0), fty(0.0), ftz(0.0), ftt(1.0) {}

TLorentzRotation::TLorentzRotation(const TRotation & r)
  : fxx(r.XX()), fxy(r.XY()), fxz(r.XZ()), fxt(0.0),
    fyx(r.YX()), fyy(r.YY()), fyz(r.YZ()), fyt(0.0),
    fzx(r.ZX()), fzy(r.ZY()), fzz(r.ZZ()), fzt(0.0),
    ftx(0.0),    fty(0.0),    ftz(0.0),    ftt(1.0) {}

TLorentzRotation::TLorentzRotation(const TLorentzRotation & r) : TObject(r),
    fxx(r.fxx), fxy(r.fxy), fxz(r.fxz), fxt(r.fxt),
    fyx(r.fyx), fyy(r.fyy), fyz(r.fyz), fyt(r.fyt),
    fzx(r.fzx), fzy(r.fzy), fzz(r.fzz), fzt(r.fzt),
    ftx(r.ftx), fty(r.fty), ftz(r.ftz), ftt(r.ftt) {}

TLorentzRotation::TLorentzRotation(
  Double_t rxx, Double_t rxy, Double_t rxz, Double_t rxt,
  Double_t ryx, Double_t ryy, Double_t ryz, Double_t ryt,
  Double_t rzx, Double_t rzy, Double_t rzz, Double_t rzt,
  Double_t rtx, Double_t rty, Double_t rtz, Double_t rtt)
  : fxx(rxx), fxy(rxy), fxz(rxz), fxt(rxt),
    fyx(ryx), fyy(ryy), fyz(ryz), fyt(ryt),
    fzx(rzx), fzy(rzy), fzz(rzz), fzt(rzt),
    ftx(rtx), fty(rty), ftz(rtz), ftt(rtt) {}

TLorentzRotation::TLorentzRotation(Double_t bx,
                                   Double_t by,
                                   Double_t bz)
{
   //constructor
   SetBoost(bx, by, bz);
}

TLorentzRotation::TLorentzRotation(const TVector3 & p) {
   //copy constructor
   SetBoost(p.X(), p.Y(), p.Z());
}

Double_t TLorentzRotation::operator () (int i, int j) const {
   //derefencing operator
   if (i == 0) {
      if (j == 0) { return fxx; }
      if (j == 1) { return fxy; }
      if (j == 2) { return fxz; }
      if (j == 3) { return fxt; }
   } else if (i == 1) {
      if (j == 0) { return fyx; }
      if (j == 1) { return fyy; }
      if (j == 2) { return fyz; }
      if (j == 3) { return fyt; }
   } else if (i == 2) {
      if (j == 0) { return fzx; }
      if (j == 1) { return fzy; }
      if (j == 2) { return fzz; }
      if (j == 3) { return fzt; }
   } else if (i == 3) {
      if (j == 0) { return ftx; }
      if (j == 1) { return fty; }
      if (j == 2) { return ftz; }
      if (j == 3) { return ftt; }
   }
   Warning("operator()(i,j)","subscripting: bad indices(%d,%d)",i,j);
   return 0.0;
}

void TLorentzRotation::SetBoost(Double_t bx, Double_t by, Double_t bz) {
   //boost this Lorentz vector
   Double_t bp2 = bx*bx + by*by + bz*bz;
   Double_t gamma = 1.0 / TMath::Sqrt(1.0 - bp2);
   Double_t bgamma = gamma * gamma / (1.0 + gamma);
   fxx = 1.0 + bgamma * bx * bx;
   fyy = 1.0 + bgamma * by * by;
   fzz = 1.0 + bgamma * bz * bz;
   fxy = fyx = bgamma * bx * by;
   fxz = fzx = bgamma * bx * bz;
   fyz = fzy = bgamma * by * bz;
   fxt = ftx = gamma * bx;
   fyt = fty = gamma * by;
   fzt = ftz = gamma * bz;
   ftt = gamma;
}

TLorentzRotation TLorentzRotation::MatrixMultiplication(const TLorentzRotation & b) const {
   //multiply this vector by a matrix
   return TLorentzRotation(
    fxx*b.fxx + fxy*b.fyx + fxz*b.fzx + fxt*b.ftx,
    fxx*b.fxy + fxy*b.fyy + fxz*b.fzy + fxt*b.fty,
    fxx*b.fxz + fxy*b.fyz + fxz*b.fzz + fxt*b.ftz,
    fxx*b.fxt + fxy*b.fyt + fxz*b.fzt + fxt*b.ftt,
    fyx*b.fxx + fyy*b.fyx + fyz*b.fzx + fyt*b.ftx,
    fyx*b.fxy + fyy*b.fyy + fyz*b.fzy + fyt*b.fty,
    fyx*b.fxz + fyy*b.fyz + fyz*b.fzz + fyt*b.ftz,
    fyx*b.fxt + fyy*b.fyt + fyz*b.fzt + fyt*b.ftt,
    fzx*b.fxx + fzy*b.fyx + fzz*b.fzx + fzt*b.ftx,
    fzx*b.fxy + fzy*b.fyy + fzz*b.fzy + fzt*b.fty,
    fzx*b.fxz + fzy*b.fyz + fzz*b.fzz + fzt*b.ftz,
    fzx*b.fxt + fzy*b.fyt + fzz*b.fzt + fzt*b.ftt,
    ftx*b.fxx + fty*b.fyx + ftz*b.fzx + ftt*b.ftx,
    ftx*b.fxy + fty*b.fyy + ftz*b.fzy + ftt*b.fty,
    ftx*b.fxz + fty*b.fyz + ftz*b.fzz + ftt*b.ftz,
    ftx*b.fxt + fty*b.fyt + ftz*b.fzt + ftt*b.ftt);
}
