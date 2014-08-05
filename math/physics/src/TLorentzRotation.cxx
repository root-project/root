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
TLorentzRotation</H2>
The TLorentzRotation class describes Lorentz transformations including
Lorentz boosts and rotations (see TRotation)

<P><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| xx&nbsp; xy&nbsp; xz&nbsp; xt |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| yx&nbsp; yy&nbsp; yz&nbsp; yt |</TT>
<BR><TT>&nbsp;&nbsp; lambda = |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| zx&nbsp; zy&nbsp; zz&nbsp; zt |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| tx&nbsp; ty&nbsp; tz&nbsp; tt |</TT>
<BR>&nbsp;
<H3>
Declaration</H3>
By default it is initialized to the identity matrix, but it may also be
intialized by an other <TT>TLorentzRotation</TT>,
<BR>by a pure TRotation or by a boost:

<P><TT>&nbsp; TLorentzRotation l;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; // l is
initialized as identity</TT>
<BR><TT>&nbsp; TLorentzRotation m(l);&nbsp;&nbsp; // m = l</TT>
<BR><TT>&nbsp; TRotation r;</TT>
<BR><TT>&nbsp; TLorentzRotation lr(r);</TT>
<BR><TT>&nbsp; TLorentzRotation lb1(bx,by,bz);</TT>
<BR><TT>&nbsp; TVector3 b;</TT>
<BR><TT>&nbsp; TLorentzRotation lb2(b);</TT>

<P>The Matrix for a Lorentz boosts is:

<P><TT>&nbsp;| 1+gamma'*bx*bx&nbsp; gamma'*bx*by&nbsp;&nbsp; gamma'*bx*bz&nbsp;
gamma*bx |</TT>
<BR><TT>&nbsp;|&nbsp; gamma'*by*bx&nbsp; 1+gamma'*by*by&nbsp; gamma'*by*bz&nbsp;
gamma*by |</TT>
<BR><TT>&nbsp;|&nbsp; gamma'*bz*bx&nbsp;&nbsp; gamma'*bz*by&nbsp; 1+gamma'*bz*bz
gamma*bz |</TT>
<BR><TT>&nbsp;|&nbsp;&nbsp;&nbsp; gamma*bx&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
gamma*by&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; gamma*bz&nbsp;&nbsp;&nbsp;&nbsp;
gamma&nbsp;&nbsp; |</TT>

<P>with the boost vector <TT><B>b</B>=(bx,by,bz) </TT>and<TT> gamma=1/Sqrt(1-beta*beta)
</TT>and<TT> gamma'=(gamma-1)/beta*beta.</TT>
<H3>
Access to the matrix components/Comparisons</H3>
Access to the matrix components is possible through the member functions
XX(), XY() .. TT(),
<BR>through the operator (int,int):

<P><TT>&nbsp; Double_t xx;</TT>
<BR><TT>&nbsp; TLorentzRotation l;</TT>
<BR><TT>&nbsp; xx = l.XX();&nbsp;&nbsp;&nbsp; // gets the xx component</TT>
<BR><TT>&nbsp; xx = l(0,0);&nbsp;&nbsp;&nbsp; // gets the xx component</TT>

<P><TT>&nbsp; if (l==m) {...}&nbsp; // test for equality</TT>
<BR><TT>&nbsp; if (l !=m) {...} // test for inequality</TT>
<BR><TT>&nbsp; if (l.IsIdentity()) {...} // test for identity</TT>
<BR>&nbsp;
<H3>
Transformations of a LorentzRotation</H3>

<H5>
Compound transformations</H5>
There are four possibilities to find the product of two <TT>TLorentzRotation</TT>
transformations:

<P><TT>&nbsp; TLorentzRotation a,b,c;</TT>
<BR><TT>&nbsp; c = b*a;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// product</TT>
<BR><TT>&nbsp; c = a.MatrixMultiplication(b);&nbsp; // a is unchanged</TT>
<BR><TT>&nbsp; a *= b;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// Attention: a=a*b</TT>
<BR><TT>&nbsp; c = a.Transform(b)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// a=b*a then c=a</TT>
<BR>&nbsp;
<H5>
Lorentz boosts</H5>
<TT>&nbsp; Double_t bx, by, bz;</TT>
<BR><TT>&nbsp; TVector3 v(bx,by,bz);</TT>
<BR><TT>&nbsp; TLorentzRotation l;</TT>
<BR><TT>&nbsp; l.Boost(v);</TT>
<BR><TT>&nbsp; l.Boost(bx,by,bz);</TT>
<BR>&nbsp;
<H5>
Rotations</H5>
<TT>&nbsp; TVector3 axis;</TT>
<BR><TT>&nbsp; l.RotateX(TMath::Pi());&nbsp;&nbsp; //&nbsp; rotation around
x-axis</TT>
<BR><TT>&nbsp; l.Rotate(.5,axis);&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;//&nbsp; rotation around specified vector</TT>
<H5>
Inverse transformation</H5>
The matrix for the inverse transformation of a TLorentzRotation is as follows:
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| xx&nbsp; yx&nbsp; zx -tx |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| xy&nbsp; yy&nbsp; zy -ty |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
| xz&nbsp; yz&nbsp; zz -tz |</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|</TT>
<BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|-xt -yt -zt&nbsp; tt |</TT>
<BR>To return the inverse transformation keeping the current unchanged
use the memberfunction <TT>Inverse()</TT>.
<BR><TT>Invert()</TT> inverts the current <TT>TLorentzRotation</TT>:

<P><TT>&nbsp; l1 = l2.Inverse();&nbsp; // l1 is inverse of l2, l2 unchanged</TT>
<BR><TT>&nbsp; l1 = l2.Invert();&nbsp;&nbsp; // invert l2, then&nbsp; l1=l2</TT>
<H3>
Transformation of a TLorentzVector</H3>
To apply <TT>TLorentzRotation</TT> to <TT>TLorentzVector</TT> you can use
either the <TT>VectorMultiplication()</TT> member function or the <TT>*</TT>
operator. You can also use the <TT>Transform()</TT> function and the <TT>*=</TT>
operator of the <TT>TLorentzVector</TT> class.:

<P><TT>&nbsp; TLorentzVector v;</TT>
<BR><TT>&nbsp; ...</TT>
<BR><TT>&nbsp; v=l.VectorMultiplication(v);</TT>
<BR><TT>&nbsp; v = l * v;</TT>

<P><TT>&nbsp; v.Transform(l);</TT>
<BR><TT>&nbsp; v *= l;&nbsp; // Attention v = l*v</TT>
<!--*/
// -->END_HTML
//

#include "TError.h"
#include "TLorentzRotation.h"

ClassImp(TLorentzRotation)

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
   Warning("operator()(i,j)","subscripting: bad indeces(%d,%d)",i,j);
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
